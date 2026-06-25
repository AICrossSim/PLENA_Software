import json

from quant_eval.cli.eval_phase_bfcl import (
    _bfcl_model_name,
    _load_bfcl_multiple_entries,
    _render_qwen3_bfcl_prompt,
    _write_bfcl_multiple_results,
    resolve_attention_backend,
)


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
