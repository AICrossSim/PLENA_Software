from pathlib import Path

import torch
from safetensors.torch import save_file

from prefill_DSE.run_prefill_dse import _cfg_for_trial_weight, build_trials
from quant_eval.cli.eval_phase_bfcl import _GptqWeightCache


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
