from __future__ import annotations

try:
    from benchmarks.torchtitan_transformer.venv_torchtitan import ensure_venv_torchtitan
except ImportError:
    from venv_torchtitan import ensure_venv_torchtitan

ensure_venv_torchtitan()

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

try:
    from benchmarks.torchtitan_transformer.model import (
        Transformer,
        llama3_args,
        parallelize_llama,
    )
    from benchmarks.torchtitan_transformer.model.state_dict_adapter import (
        Llama3StateDictAdapter,
    )
    from benchmarks.torchtitan_transformer.synthetic_dataloader import (
        build_synthetic_dataloader,
    )
except ImportError:
    from model import Transformer, llama3_args, parallelize_llama
    from model.state_dict_adapter import Llama3StateDictAdapter
    from synthetic_dataloader import build_synthetic_dataloader


def _build_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=llama3_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_synthetic_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )


register_train_spec("llama3_local", _build_train_spec())
