import argparse

import pytest

from prefill_DSE.run_prefill_dse import Trial as BfclTrial
from prefill_DSE.run_prefill_dse import _base_command as bfcl_base_command
from prefill_DSE.run_prefill_ppl import Trial as PplTrial
from prefill_DSE.run_prefill_ppl import _base_command as ppl_base_command


def _args(**overrides):
    values = {
        "limit": 1,
        "bfcl_max_new_tokens": 64,
        "gptq_max_layers": "1",
        "gpu_memory_reserve_mb": None,
        "gpu_memory_reserve_wait_sec": None,
        "gpu_memory_reserve_poll_sec": None,
        "gpu_memory_reserve_chunk_mb": None,
        "gpu_memory_reserve_disable": False,
        "dataset": None,
        "subset": None,
        "split": None,
        "ppl_seqlen": None,
        "ppl_max_samples": "1",
        "gptq_cache_mode": "require",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _cfg(parallel_mode="multiworker", reserve_enabled=False):
    return {
        "model": {
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "model_family": "qwen3_moe",
            "bfcl_model_alias": "Qwen/Qwen3-30B-A3B-Instruct-2507-FC",
            "bfcl_adapter": "qwen3_fc",
            "dtype": "bfloat16",
            "quant_config": "quant_eval/configs/qwen3_moe_mxint16.toml",
        },
        "bfcl": {"test_categories": "multiple", "tool_mode": "return", "num_threads": 1},
        "ppl": {"max_samples": 1, "seqlen": 1024},
        "gptq": {
            "dataset": "calib/qwen3_moe_30b_a3b_bfcl_official_wrapped_s1024_n32.pt",
            "nsamples": 32,
            "seqlen": 1024,
            "cache_mode": "auto",
        },
        "runtime": {
            "parallel_mode": parallel_mode,
            "decode_weight_mode": "fp",
            "gpu_memory_reserve_enabled": reserve_enabled,
            "gpu_memory_reserve_mb": 1024 if reserve_enabled else 0,
            "gpu_memory_reserve_disable": not reserve_enabled,
        },
    }


def test_bfcl_pp_command_enables_model_parallel_and_disables_reserve(tmp_path):
    trial = BfclTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    cmd = bfcl_base_command(_cfg(parallel_mode="pp"), trial, tmp_path, 9000, _args())
    joined = " ".join(cmd)

    assert "--model_parallel true" in joined
    assert "--gpu_memory_reserve_mb 0" in joined
    assert "--gpu_memory_reserve_disable true" in joined


def test_ppl_multiworker_command_keeps_single_worker_semantics(tmp_path):
    trial = PplTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    cmd = ppl_base_command(_cfg(parallel_mode="multiworker"), trial, tmp_path, _args())
    joined = " ".join(cmd)

    assert "--model_parallel" not in joined
    assert "--gpu_memory_reserve_mb 0" in joined
    assert "--gpu_memory_reserve_disable true" in joined


def test_pp_mode_rejects_enabled_gpu_reserve(tmp_path):
    trial = BfclTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    with pytest.raises(ValueError, match="parallel_mode='pp'"):
        bfcl_base_command(_cfg(parallel_mode="pp", reserve_enabled=True), trial, tmp_path, 9000, _args())
