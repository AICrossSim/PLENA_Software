import json
import os
from pathlib import Path
import socket

import torch
from safetensors.torch import save_file

from prefill_DSE.run_prefill_dse import _cfg_for_trial_weight, build_trials
from quant_eval.cli.eval_phase_bfcl import (
    _GptqWeightCache,
    _gptq_cache_fingerprint,
    _stable_json_hash,
)


def test_weight_search_space_maps_mxfp_format_and_cache() -> None:
    cfg = {
        "gptq": {
            "cache_dirs": {"MXFP_E4M3": "/cache/e4m3"},
            "weight_block_size": 32,
        },
        "search_space": {
            "WEIGHT": ["MXFP_E4M3"],
            "ACT": ["MXFP_E4M3"],
            "KV": ["MXFP_E4M3"],
            "FP_SETTING": ["FP_E8M5"],
        },
    }

    trial = build_trials(cfg)[0]
    patched = _cfg_for_trial_weight(cfg, trial)

    assert trial.weight_precision == "MXFP_E4M3"
    assert "w-MXFP_E4M3" in trial.trial_id
    assert patched["gptq"] == {
        "cache_dirs": {"MXFP_E4M3": "/cache/e4m3"},
        "weight_block_size": 32,
        "format": "mxfp",
        "weight_width": 8,
        "weight_exponent_width": 4,
        "weight_frac_width": 3,
        "dse_weight_precision": "MXFP_E4M3",
        "dse_weight_block_size": 32,
        "cache_dir": "/cache/e4m3",
    }


def test_gptq_cache_load_targets_decoder_layers(tmp_path: Path) -> None:
    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(3, 2, bias=False) for _ in range(2)]
            )

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Decoder()

        def load_state_dict(self, *args, **kwargs):
            raise AssertionError("whole-model load_state_dict must not be called")

    model = Model()
    cache = object.__new__(_GptqWeightCache)
    cache.cache_path = tmp_path
    cache.expected_layers = [0, 1]
    cache.key = "synthetic"
    cache.hit = False
    cache.loaded_layers = 0

    expected = []
    for layer_idx, layer in enumerate(model.model.layers):
        weight = torch.full_like(layer.weight, layer_idx + 3.0)
        save_file({"weight": weight}, cache._layer_path(layer_idx))
        expected.append(weight)

    assert cache.load(model) == 2
    assert cache.hit is True
    assert all(
        torch.equal(layer.weight, expected[layer_idx])
        for layer_idx, layer in enumerate(model.model.layers)
    )


def test_gptq_cache_accepts_same_snapshot_at_migrated_local_path(tmp_path: Path) -> None:
    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False)])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Decoder()

    old_calibration = tmp_path / "old-calibration.pt"
    new_calibration = tmp_path / "migrated" / "new-calibration.pt"
    new_calibration.parent.mkdir()
    old_calibration.write_bytes(b"same calibration tokens")
    new_calibration.write_bytes(old_calibration.read_bytes())
    common = {
        "nsamples": 32,
        "seqlen": 1024,
        "format": "mxint",
        "weight_config": {"weight_width": 4, "weight_block_size": 32},
        "max_layers": None,
    }
    old_config = {
        **common,
        "dataset": f"file:{old_calibration}",
        "model_name": "/old/cache/models--Qwen--Qwen3-32B/snapshots/revision-a",
    }
    old_fingerprint = _gptq_cache_fingerprint(old_config, 1)
    old_key = _stable_json_hash(old_fingerprint)
    old_cache = tmp_path / old_key
    old_cache.mkdir()
    expected = torch.full((2, 3), 7.0)
    save_file({"weight": expected}, old_cache / "quantized_model_layer_0.safetensors")
    (old_cache / "metadata.json").write_text(
        json.dumps({
            "complete": True,
            "cache_key": old_key,
            "fingerprint": old_fingerprint,
            "expected_layers": [0],
        }),
        encoding="utf-8",
    )

    new_config = {
        **common,
        "dataset": f"file:{new_calibration}",
        "model_name": "/data/models/models--Qwen--Qwen3-32B/snapshots/revision-a",
    }
    model = Model()
    cache = _GptqWeightCache(
        cache_dir=tmp_path,
        mode="require",
        gptq_config=new_config,
        total_layers=1,
    )
    try:
        assert cache.prepare(model) is True
    finally:
        cache.release()
    assert cache.model_path_alias_hit is True
    assert cache.cache_path == old_cache
    assert torch.equal(model.model.layers[0].weight, expected)


def test_gptq_cache_only_hit_verifies_without_loading_weights(tmp_path: Path) -> None:
    config = {
        "model_name": "synthetic",
        "dataset": "synthetic",
        "nsamples": 1,
        "seqlen": 8,
        "format": "mxint",
        "weight_config": {"weight_width": 4, "weight_block_size": 4},
    }
    cache = _GptqWeightCache(
        cache_dir=tmp_path,
        mode="require",
        gptq_config=config,
        total_layers=1,
    )
    cache.cache_path.mkdir(parents=True)
    save_file({"weight": torch.ones(2, 2)}, cache._layer_path(0))
    (cache.cache_path / "metadata.json").write_text(
        json.dumps({
            "complete": True,
            "cache_key": cache.key,
            "fingerprint": cache.fingerprint,
            "expected_layers": cache.expected_layers,
        }),
        encoding="utf-8",
    )

    class Model:
        def load_state_dict(self, *_args, **_kwargs):
            raise AssertionError("cache-only hit must not load model weights")

    assert cache.prepare(Model(), load_on_hit=False) is True
    assert cache.hit is True
    assert cache.loaded_layers == 0


def test_gptq_cache_removes_lock_owned_by_dead_local_process(tmp_path: Path) -> None:
    config = {
        "model_name": "synthetic",
        "dataset": "synthetic",
        "nsamples": 1,
        "seqlen": 8,
        "format": "mxint",
        "weight_config": {"weight_width": 4, "weight_block_size": 4},
    }
    cache = _GptqWeightCache(
        cache_dir=tmp_path,
        mode="auto",
        gptq_config=config,
        total_layers=1,
    )
    cache.lock_path.mkdir(parents=True)
    (cache.lock_path / "owner.json").write_text(
        json.dumps({"hostname": socket.gethostname(), "pid": 999_999_999}),
        encoding="utf-8",
    )
    cache._poll_sec = 0.001
    try:
        cache._acquire_lock()
        owner = json.loads((cache.lock_path / "owner.json").read_text(encoding="utf-8"))
        assert owner["pid"] == os.getpid()
        assert cache._lock_acquired is True
    finally:
        cache.release()
    assert not cache.lock_path.exists()
