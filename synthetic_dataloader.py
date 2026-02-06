from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig


class SyntheticTokenDataset(IterableDataset, Stateful):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        seed: int | None,
        dp_rank: int,
        dp_world_size: int,
        infinite: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        self.seed = 0 if seed is None else seed
        self._sample_idx = 0
        self._rng = self._build_generator()

    def _build_generator(self) -> torch.Generator:
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.dp_rank)
        return rng

    def __iter__(self):
        vocab_size = self.vocab_size
        seq_len = self.seq_len
        rng = self._rng

        while True:
            tokens = torch.randint(
                0,
                vocab_size,
                (seq_len + 1,),
                dtype=torch.long,
                generator=rng,
            )
            self._sample_idx += 1
            yield {"input": tokens[:-1]}, tokens[1:]

            if not self.infinite:
                break

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_idx": self._sample_idx,
            "rng_state": self._rng.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return

        self._sample_idx = state_dict.get("sample_idx", 0)
        rng_state = state_dict.get("rng_state")
        if rng_state is not None:
            self._rng.set_state(rng_state)


def build_synthetic_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer | None,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    if tokenizer is None:
        raise ValueError("Synthetic dataloader requires a tokenizer for vocab size.")

    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    vocab_size = tokenizer.get_vocab_size()
    seed = job_config.debug.seed

    dataset = SyntheticTokenDataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        seed=seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
