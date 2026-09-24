from types import SimpleNamespace

import pytest
import torch

from quant_eval import utils
from quant_eval.utils import create_layer_balanced_device_map


def _layer_counts(device_map: dict[str, int], n_devices: int) -> list[int]:
    counts = [0] * n_devices
    for name, device in device_map.items():
        if name.startswith("model.layers."):
            counts[device] += 1
    return counts


def test_layer_balanced_device_map_reserves_edge_devices_for_extra_modules():
    device_map = create_layer_balanced_device_map(
        SimpleNamespace(num_hidden_layers=94),
        n_devices=6,
    )

    assert _layer_counts(device_map, 6) == [15, 16, 16, 16, 16, 15]
    assert device_map["model.embed_tokens"] == 0
    assert device_map["model.rotary_emb"] == 0
    assert device_map["model.norm"] == 5
    assert device_map["lm_head"] == 5
    assert [device_map[f"model.layers.{idx}"] for idx in range(94)] == sorted(
        device_map[f"model.layers.{idx}"] for idx in range(94)
    )


def test_layer_balanced_device_map_handles_single_device():
    device_map = create_layer_balanced_device_map(
        SimpleNamespace(num_hidden_layers=3),
        n_devices=1,
    )

    assert _layer_counts(device_map, 1) == [3]
    assert set(device_map.values()) == {0}


@pytest.mark.parametrize(
    ("num_hidden_layers", "n_devices", "message"),
    [(0, 2, "num_hidden_layers"), (4, 0, "visible CUDA device")],
)
def test_layer_balanced_device_map_rejects_invalid_inputs(
    num_hidden_layers: int,
    n_devices: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        create_layer_balanced_device_map(
            SimpleNamespace(num_hidden_layers=num_hidden_layers),
            n_devices=n_devices,
        )


def test_setup_model_cpu_staged_uses_low_memory_cpu_load(monkeypatch):
    captured = {}
    model = torch.nn.Linear(2, 2)

    monkeypatch.setattr(
        utils.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )

    def fake_from_pretrained(*_args, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(utils.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    _tokenizer, loaded = utils.setup_model(
        "synthetic",
        model_parallel=False,
        dtype=torch.bfloat16,
        device="cuda:0",
        cpu_staged=True,
    )

    assert loaded is model
    assert captured["low_cpu_mem_usage"] is True
    assert "device_map" not in captured
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())


def test_setup_model_cpu_staged_rejects_model_parallel():
    with pytest.raises(ValueError, match="incompatible with model_parallel"):
        utils.setup_model(
            "synthetic",
            model_parallel=True,
            dtype=torch.bfloat16,
            device=None,
            cpu_staged=True,
        )
