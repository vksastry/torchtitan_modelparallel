from __future__ import annotations

try:
    from benchmarks.torchtitan_transformer.venv_torchtitan import ensure_venv_torchtitan
except ImportError:
    from venv_torchtitan import ensure_venv_torchtitan

ensure_venv_torchtitan()

from torchtitan.distributed.pipeline_parallel import pipeline_llm

from .args import TransformerModelArgs
from .infra.parallelize import parallelize_llama
from .model import Transformer

__all__ = [
    "parallelize_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_args",
    "pipeline_llm",
]


llama3_args = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
    "debugmodel_varlen_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        attn_type="varlen",
        attn_mask_type="block_causal",
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "8B_flex": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
    "8B_varlen": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        attn_type="varlen",
        attn_mask_type="block_causal",
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}
