import json
import sys

from prefill_DSE import collect_bfcl_multiple_lengths as lengths


def test_load_length_rows_from_jsonl_and_build_summary(tmp_path):
    result_file = tmp_path / "BFCL_v4_multiple_result.json"
    records = [
        {"id": "multiple_10", "input_token_count": 30, "output_token_count": 7, "latency": 1.5},
        {"id": "multiple_2", "input_token_count": 20, "output_token_count": 5, "latency": 1.0},
    ]
    result_file.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    rows = lengths.load_length_rows(result_file)
    summary = lengths.build_summary(rows, max_new_tokens=7)

    assert [row["id"] for row in rows] == ["multiple_2", "multiple_10"]
    assert rows[0]["total_tokens"] == 25
    assert summary["count"] == 2
    assert summary["isl"]["mean"] == 25.0
    assert summary["osl"]["max"] == 7
    assert summary["hit_max_new_tokens_count"] == 1


def test_load_length_rows_from_single_json_usage_payload(tmp_path):
    result_file = tmp_path / "BFCL_v4_multiple_result.json"
    result_file.write_text(
        json.dumps(
            {
                "id": "multiple_0",
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "latency": 0.25,
            }
        ),
        encoding="utf-8",
    )

    rows = lengths.load_length_rows(result_file)

    assert rows == [
        {
            "id": "multiple_0",
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15,
            "latency": 0.25,
        }
    ]


def test_dry_run_defaults_to_three_gpus_and_1024_tokens(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect_bfcl_multiple_lengths.py", "--dry-run"],
    )

    lengths.main()

    output = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES: 2,3,4" in output
    assert "Max new tokens      : 1024" in output
    assert "Qwen/Qwen3-235B-A22B-Instruct-2507-FC" in output
