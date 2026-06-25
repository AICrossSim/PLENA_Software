import argparse

import pytest

from prefill_DSE.run_prefill_dse import Trial as BfclTrial
from prefill_DSE.run_prefill_dse import _base_command as bfcl_base_command
from prefill_DSE.run_prefill_dse import _decode_weight_residency
from prefill_DSE.run_prefill_dse import _read_persistent_progress
from prefill_DSE.run_prefill_dse import _trial_weight_reuse
from prefill_DSE.run_prefill_ppl import Trial as PplTrial
from prefill_DSE.run_prefill_ppl import _base_command as ppl_base_command


def _args(**overrides):
    values = {
        "limit": 1,
        "bfcl_max_new_tokens": 64,
        "bfcl_generate_mode": None,
        "bfcl_batch_size": None,
        "gptq_max_layers": "1",
        "dataset": None,
        "subset": None,
        "split": None,
        "ppl_seqlen": None,
        "ppl_max_samples": "1",
        "gptq_cache_mode": "require",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _cfg(parallel_mode="multiworker"):
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
        },
    }


def test_bfcl_pp_command_enables_model_parallel_without_reserve_args(tmp_path):
    trial = BfclTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    cmd = bfcl_base_command(_cfg(parallel_mode="pp"), trial, tmp_path, 9000, _args())
    joined = " ".join(cmd)

    assert "--model_parallel true" in joined
    assert "gpu_memory_reserve" not in joined
    assert "--decode_weight_residency disk_reload" in joined


def test_ppl_multiworker_command_keeps_single_worker_semantics(tmp_path):
    trial = PplTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    cmd = ppl_base_command(_cfg(parallel_mode="multiworker"), trial, tmp_path, _args())
    joined = " ".join(cmd)

    assert "--model_parallel" not in joined
    assert "gpu_memory_reserve" not in joined


def test_gpu_dual_and_memory_cache_are_passed_to_bfcl_eval(tmp_path):
    cfg = _cfg(parallel_mode="pp")
    cfg["runtime"]["model_lifecycle"] = "persistent"
    cfg["runtime"]["decode_weight_residency"] = "gpu_dual"
    cfg["gptq"]["cache_mode"] = "memory"
    cfg["gptq"]["device_map_aware"] = True
    trial = BfclTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    cmd = bfcl_base_command(cfg, trial, tmp_path, 9000, _args())
    joined = " ".join(cmd)

    assert "--decode_weight_residency gpu_dual" in joined
    assert "--gptq_cache_mode memory" in joined
    assert "--gptq_device_map_aware true" in joined


def test_gpu_dual_requires_persistent_lifecycle():
    cfg = _cfg(parallel_mode="multiworker")
    cfg["runtime"]["decode_weight_residency"] = "gpu_dual"

    with pytest.raises(ValueError, match="model_lifecycle='persistent'"):
        _decode_weight_residency(cfg)


def test_gpu_dual_allowed_for_persistent_multiworker():
    cfg = _cfg(parallel_mode="multiworker")
    cfg["runtime"]["model_lifecycle"] = "persistent"
    cfg["runtime"]["trial_weight_reuse"] = True
    cfg["runtime"]["decode_weight_residency"] = "gpu_dual"

    assert _decode_weight_residency(cfg) == "gpu_dual"
    assert _trial_weight_reuse(cfg) is True


def test_bfcl_batched_generate_mode_is_passed_to_eval(tmp_path):
    cfg = _cfg(parallel_mode="pp")
    cfg["bfcl"]["generate_mode"] = "batched"
    cfg["bfcl"]["batch_size"] = 16
    trial = BfclTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    cmd = bfcl_base_command(cfg, trial, tmp_path, 9000, _args())
    joined = " ".join(cmd)

    assert "--bfcl_generate_mode batched" in joined
    assert "--bfcl_batch_size 16" in joined
    assert "--bfcl_batch_length_bucket true" in joined


def test_ppl_passes_device_map_aware_gptq_flag(tmp_path):
    cfg = _cfg(parallel_mode="pp")
    cfg["gptq"]["device_map_aware"] = True
    trial = PplTrial("trial", "MXINT_8", "MXINT_8", "FP_E8M5", 0)
    cmd = ppl_base_command(cfg, trial, tmp_path, _args())
    joined = " ".join(cmd)

    assert "--model_parallel true" in joined
    assert "--gptq_device_map_aware true" in joined


def test_persistent_progress_reader_deduplicates_and_skips_bad_lines(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        '{"trial_id": "a", "status": "done"}\n'
        'not-json\n'
        '{"trial_id": "b", "status": "done"}\n'
        '{"trial_id": "a", "status": "done"}\n',
        encoding="utf-8",
    )
    completed: set[str] = set()

    assert _read_persistent_progress(progress, completed) == ["a", "b"]
    assert _read_persistent_progress(progress, completed) == []
    assert completed == {"a", "b"}
