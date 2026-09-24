import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi import HTTPException
from starlette.requests import Request

from quant_eval.bfcl_adapters import resolve_bfcl_adapter
from quant_eval.cli.eval_phase_bfcl import (
    _build_server_app,
    _bfcl_model_name,
    _extract_first_turn_messages,
    _load_bfcl_multiple_entries,
    _render_qwen3_bfcl_prompt,
    _write_bfcl_multiple_results,
    resolve_attention_backend,
)


class _ContextLimitedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.config = SimpleNamespace(max_position_embeddings=4)
        self.generate_called = False

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.generate_called = True
        return kwargs["input_ids"]


class _LongPromptTokenizer:
    name_or_path = "Qwen/Qwen3-test"
    eos_token_id = 0
    model_max_length = 4

    def __call__(self, text, *, return_tensors, add_special_tokens=False):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=torch.ones((1, 5), dtype=torch.long))


def test_load_bfcl_multiple_entries_respects_limit_and_order():
    rows = _load_bfcl_multiple_entries(limit=2)

    assert [row["id"] for row in rows] == ["multiple_0", "multiple_1"]
    assert rows[0]["function"]


def test_qwen3_batched_prompt_matches_official_markers():
    row = _load_bfcl_multiple_entries(limit=1)[0]
    prompt = _render_qwen3_bfcl_prompt(row)

    assert prompt.startswith("<|im_start|>system\n# Tools")
    assert "<tools>" in prompt
    assert "<tool_call>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_qwen3_batched_prompt_matches_installed_official_handler():
    row = _load_bfcl_multiple_entries(limit=1)[0]
    messages = _extract_first_turn_messages(row)
    bfcl_python = (
        Path(__file__).resolve().parents[1]
        / ".conda"
        / "envs"
        / "plena-bfcl"
        / "bin"
        / "python"
    )
    if not bfcl_python.is_file():
        pytest.skip("The isolated plena-bfcl environment is not installed.")
    script = (
        "import json,sys; "
        "from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler; "
        "payload=json.load(sys.stdin); "
        "handler=object.__new__(QwenFCHandler); "
        "sys.stdout.write(handler._format_prompt(payload['messages'], payload['function']))"
    )
    completed = subprocess.run(
        [str(bfcl_python), "-c", script],
        input=json.dumps({"messages": messages, "function": row["function"]}),
        text=True,
        capture_output=True,
        check=True,
    )

    assert _render_qwen3_bfcl_prompt(row) == completed.stdout


def test_qwen3_batched_prompt_can_disable_thinking():
    row = _load_bfcl_multiple_entries(limit=1)[0]
    prompt = _render_qwen3_bfcl_prompt(row, disable_thinking=True)

    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen3_quantized_auto_attention_uses_eager():
    assert (
        resolve_attention_backend(
            "auto",
            qwen_model_family=True,
            quant_config_is_none=False,
            codesign_tokens_enabled=True,
        )
        == "eager"
    )


def test_fp_auto_attention_keeps_sdpa():
    assert (
        resolve_attention_backend(
            "auto",
            qwen_model_family=True,
            quant_config_is_none=True,
            codesign_tokens_enabled=False,
        )
        == "sdpa"
    )


def test_qwen_completion_response_does_not_repair_nonofficial_json():
    adapter = resolve_bfcl_adapter("qwen3_fc")
    raw = '<think>reasoning</think>\n{"name":"weather","arguments":{"city":"London"}}'

    assert adapter.completion_response_text(raw) == raw


def test_serial_server_rejects_prompt_beyond_model_context():
    model = _ContextLimitedModel()
    tokenizer = _LongPromptTokenizer()
    app = _build_server_app(
        model,
        tokenizer,
        device="cpu",
        tool_mode="return",
        max_new_tokens=16,
        bfcl_adapter=resolve_bfcl_adapter("qwen3_fc"),
    )

    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v1/completions"
    )
    body = json.dumps({"prompt": "too long", "max_tokens": 16}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/completions",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )

    with pytest.raises(HTTPException, match="exceeds the model context window") as exc_info:
        asyncio.run(endpoint(request))

    assert exc_info.value.status_code == 400
    assert model.generate_called is False


def test_write_bfcl_multiple_results_uses_official_layout(tmp_path):
    path = _write_bfcl_multiple_results(
        tmp_path,
        _bfcl_model_name("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507-FC"),
        [
            {
                "id": "multiple_0",
                "result": "<tool_call>{}</tool_call>",
                "input_token_count": 10,
                "output_token_count": 2,
                "latency": 1.25,
            }
        ],
    )

    assert path.relative_to(tmp_path).as_posix() == (
        "Qwen_Qwen3-30B-A3B-Instruct-2507-FC/non_live/BFCL_v4_multiple_result.json"
    )
    record = json.loads(path.read_text().strip())
    assert record["id"] == "multiple_0"
    assert record["input_token_count"] == 10
    assert record["output_token_count"] == 2
