from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from quant_eval.cli import eval_phase_bfcl


eval_phase_bfcl._ensure_bfcl_eval_importable()
from quant_eval import bfcl_web_backend  # noqa: E402


def _configure(
    monkeypatch,
    tmp_path: Path,
    *,
    mode: str,
) -> Path:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("PLENA_BFCL_WEB_CACHE_MODE", mode)
    monkeypatch.setenv("PLENA_BFCL_WEB_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv(
        "PLENA_BFCL_WEB_STATS_PATH", str(tmp_path / "backend_stats.json")
    )
    monkeypatch.setenv("PLENA_BFCL_WEB_DDGS_RETRY_BASE_SECONDS", "0")
    return cache_dir


def test_ddgs_backend_preserves_bfcl_schema_and_replays(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path, mode="auto")
    live_calls = []
    backend = bfcl_web_backend.WebSearchAPI()
    backend._load_scenario({"show_snippet": True})
    monkeypatch.setattr(
        backend,
        "_ddgs_text",
        lambda keywords, max_results, region: live_calls.append(
            (keywords, max_results, region)
        )
        or [
            {"title": "Result", "href": "https://example.com", "body": "Text"},
            {"title": "Ignored", "href": "https://ignored.example", "body": "More"},
        ],
    )

    expected = [
        {"title": "Result", "href": "https://example.com", "body": "Text"}
    ]
    assert backend.search_engine_query("qwen", max_results=1, region="uk-en") == expected
    assert live_calls == [("qwen", 1, "uk-en")]

    monkeypatch.setenv("PLENA_BFCL_WEB_CACHE_MODE", "replay")
    replay = bfcl_web_backend.WebSearchAPI()
    replay._load_scenario({"show_snippet": True})
    monkeypatch.setattr(
        replay,
        "_ddgs_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live DDGS must not run on a replay hit")
        ),
    )
    assert replay.search_engine_query("qwen", max_results=1, region="uk-en") == expected

    stats = json.loads((tmp_path / "backend_stats.json").read_text())
    assert stats["live_calls"] == 1
    assert stats["live_call_seconds_total"] >= 0.0
    assert stats["operations"]["search_engine_query"]["live_call_seconds_total"] >= 0.0
    assert stats["cache_hits"] == 1
    assert stats["errors"] == 0


def test_ddgs_backend_stats_are_shared_across_request_threads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, mode="auto")

    def record_one_miss(_index: int) -> None:
        backend = bfcl_web_backend.WebSearchAPI()
        backend._increment("cache_misses", "search_engine_query")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record_one_miss, range(32)))

    stats = json.loads((tmp_path / "backend_stats.json").read_text())
    assert stats["cache_misses"] == 32
    assert stats["operations"]["search_engine_query"]["cache_misses"] == 32
    assert not list(tmp_path.glob(".backend_stats.json.*.tmp"))


def test_ddgs_backend_hides_snippets_when_scenario_requests_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, mode="record")
    backend = bfcl_web_backend.WebSearchAPI()
    backend._load_scenario({"show_snippet": False})
    monkeypatch.setattr(
        backend,
        "_ddgs_text",
        lambda *_args: [
            {"title": "Result", "url": "https://example.com", "body": "Hidden"}
        ],
    )

    assert backend.search_engine_query("qwen") == [
        {"title": "Result", "href": "https://example.com"}
    ]


def test_ddgs_backend_caches_official_url_fetch(monkeypatch, tmp_path: Path) -> None:
    _configure(monkeypatch, tmp_path, mode="auto")
    calls = []
    monkeypatch.setattr(
        bfcl_web_backend._OfficialWebSearchAPI,
        "fetch_url_content",
        lambda _self, url, mode="raw": calls.append((url, mode))
        or {"content": "stable page"},
    )
    backend = bfcl_web_backend.WebSearchAPI()
    assert backend.fetch_url_content("https://example.com", "truncate") == {
        "content": "stable page"
    }

    monkeypatch.setenv("PLENA_BFCL_WEB_CACHE_MODE", "replay")
    replay = bfcl_web_backend.WebSearchAPI()
    assert replay.fetch_url_content("https://example.com", "truncate") == {
        "content": "stable page"
    }
    assert calls == [("https://example.com", "truncate")]


def test_ddgs_backend_bounds_and_replays_fetched_page_content(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, mode="auto")
    monkeypatch.setenv("PLENA_BFCL_WEB_FETCH_MAX_CHARS", "8000")
    calls = []
    oversized = "x" * 20_000
    monkeypatch.setattr(
        bfcl_web_backend._OfficialWebSearchAPI,
        "fetch_url_content",
        lambda _self, url, mode="raw": calls.append((url, mode))
        or {"content": oversized},
    )

    backend = bfcl_web_backend.WebSearchAPI()
    expected = {"content": "x" * 8_000}
    assert backend.fetch_url_content("https://example.com/large", "truncate") == expected

    monkeypatch.setenv("PLENA_BFCL_WEB_CACHE_MODE", "replay")
    replay = bfcl_web_backend.WebSearchAPI()
    assert replay.fetch_url_content("https://example.com/large", "truncate") == expected
    assert calls == [("https://example.com/large", "truncate")]

    stats = json.loads((tmp_path / "backend_stats.json").read_text())
    assert stats["fetch_content_max_chars"] == 8_000
    assert stats["fetch_truncations"] == 1
    assert stats["fetch_original_chars_total"] == 20_000
    assert stats["fetch_returned_chars_total"] == 8_000


def test_ddgs_backend_keeps_official_fetch_content_unbounded_by_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure(monkeypatch, tmp_path, mode="off")
    monkeypatch.delenv("PLENA_BFCL_WEB_FETCH_MAX_CHARS", raising=False)
    content = "x" * 25_000
    monkeypatch.setattr(
        bfcl_web_backend._OfficialWebSearchAPI,
        "fetch_url_content",
        lambda *_args, **_kwargs: {"content": content},
    )

    backend = bfcl_web_backend.WebSearchAPI()
    assert backend.fetch_url_content("https://example.com/large", "truncate") == {
        "content": content
    }


def test_bfcl_launcher_injects_overlay_only_for_direct_ddgs(tmp_path: Path) -> None:
    env = {"PYTHONPATH": "/existing"}
    stats_path = tmp_path / "stats.json"
    eval_phase_bfcl._configure_bfcl_web_backend_env(
        env,
        search_backend="bfcl_harness_ddgs_duckduckgo",
        cache_mode="auto",
        cache_dir=tmp_path / "cache",
        stats_path=stats_path,
        max_search_concurrency=3,
    )

    entries = env["PYTHONPATH"].split(":")
    assert entries[0].endswith("quant_eval/bfcl_runtime")
    assert str(eval_phase_bfcl._REPO_ROOT) in entries
    assert env["PLENA_BFCL_WEB_SEARCH_BACKEND"] == (
        "bfcl_harness_ddgs_duckduckgo"
    )
    assert env["PLENA_BFCL_WEB_CACHE_MODE"] == "auto"
    assert env["PLENA_BFCL_WEB_DDGS_MAX_CONCURRENCY"] == "3"
    assert Path(env["PLENA_BFCL_WEB_CACHE_DIR"]).is_dir()

    official_env = {"PYTHONPATH": "/existing"}
    eval_phase_bfcl._configure_bfcl_web_backend_env(
        official_env,
        search_backend="bfcl_web_search_api_serpapi_duckduckgo",
        cache_mode="off",
        cache_dir=None,
        stats_path=None,
    )
    assert official_env == {"PYTHONPATH": "/existing"}


def test_bfcl_generate_passes_ddgs_runtime_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    class Completed:
        returncode = 0

    monkeypatch.setattr(
        eval_phase_bfcl.subprocess,
        "run",
        lambda cmd, env: captured.update({"cmd": cmd, "env": env}) or Completed(),
    )
    rc = eval_phase_bfcl._run_bfcl_generate(
        model_name="synthetic",
        test_categories=["web_search_base"],
        host="127.0.0.1",
        port=8915,
        result_dir=tmp_path / "results",
        num_threads=1,
        limit=None,
        model_alias="Qwen/Qwen3-32B-FC",
        web_search_backend="bfcl_harness_ddgs_duckduckgo",
        web_search_cache_mode="auto",
        web_search_cache_dir=tmp_path / "cache",
        web_search_stats_path=tmp_path / "stats.json",
        web_search_max_concurrency=2,
    )

    assert rc == 0
    assert captured["env"]["PLENA_BFCL_WEB_SEARCH_BACKEND"] == (
        "bfcl_harness_ddgs_duckduckgo"
    )
    assert captured["env"]["PLENA_BFCL_WEB_CACHE_MODE"] == "auto"
    assert captured["env"]["PLENA_BFCL_WEB_DDGS_MAX_CONCURRENCY"] == "2"
    assert captured["env"]["PLENA_BFCL_WEB_STATS_PATH"] == str(
        (tmp_path / "stats.json").resolve()
    )


def test_direct_ddgs_protocol_requires_serial_harness() -> None:
    workload = {
        "tool_execution_owner": "bfcl_harness",
        "generate_mode": "cli",
        "tool_mode": "return",
        "search_backend": "bfcl_harness_ddgs_duckduckgo",
        "search_cache_mode": "auto",
        "search_cache_dir": "/tmp/bfcl-ddgs-cache",
        "num_threads": 1,
        "max_turns": 20,
        "max_new_tokens": 4096,
        "require_serpapi_api_key": False,
    }
    eval_phase_bfcl._validate_bfcl_web_environment(workload)

    invalid_threads = {**workload, "num_threads": 2}
    try:
        eval_phase_bfcl._validate_bfcl_web_environment(invalid_threads)
    except ValueError as error:
        assert "num_threads=1" in str(error)
    else:
        raise AssertionError("DDGS protocol accepted concurrent BFCL threads")

    invalid_cache = {**workload, "search_cache_dir": None}
    try:
        eval_phase_bfcl._validate_bfcl_web_environment(invalid_cache)
    except ValueError as error:
        assert "search_cache_dir" in str(error)
    else:
        raise AssertionError("DDGS protocol accepted a missing cache directory")
