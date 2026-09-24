"""BFCL-Web tool backend using DuckDuckGo directly through ``ddgs``.

The class intentionally preserves BFCL's public ``WebSearchAPI`` contract and
inherits its URL-fetch implementation.  Only the SerpAPI search transport is
replaced.  Optional record/replay caching freezes successful tool responses so
baseline and quantized profiles can observe the same web content for identical
tool calls.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Optional

from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import (
    WebSearchAPI as _OfficialWebSearchAPI,
)


BACKEND_ID = "bfcl_harness_ddgs_duckduckgo"
BACKEND_SCHEMA_VERSION = 1
CACHE_MODES = {"off", "record", "replay", "auto"}
FETCH_CONTENT_LIMIT_ENV = "PLENA_BFCL_WEB_FETCH_MAX_CHARS"
_DDGS_MAX_CONCURRENCY = max(
    1, int(os.getenv("PLENA_BFCL_WEB_DDGS_MAX_CONCURRENCY", "1"))
)
_DDGS_LIVE_SEMAPHORE = threading.BoundedSemaphore(_DDGS_MAX_CONCURRENCY)
_STATS_REGISTRY_LOCK = threading.Lock()
_STATS_REGISTRY: dict[str, tuple[threading.RLock, dict[str, Any]]] = {}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _ddgs_version() -> str:
    try:
        return version("ddgs")
    except PackageNotFoundError:
        return "not-installed"


def _brotli_version() -> str:
    try:
        return version("Brotli")
    except PackageNotFoundError:
        return "not-installed"


class _ResponseCache:
    def __init__(self, root: str | None, mode: str):
        self.mode = str(mode or "off").strip().lower()
        if self.mode not in CACHE_MODES:
            raise ValueError(
                f"PLENA_BFCL_WEB_CACHE_MODE must be one of {sorted(CACHE_MODES)}, "
                f"got {mode!r}."
            )
        self.root = Path(root).expanduser().resolve() if root else None
        if self.mode != "off" and self.root is None:
            raise ValueError(
                "PLENA_BFCL_WEB_CACHE_DIR is required when the BFCL-Web "
                f"cache mode is {self.mode!r}."
            )
        if self.root is not None and self.mode in {"record", "auto"}:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(operation: str, request: dict[str, Any]) -> str:
        payload = {
            "schema_version": BACKEND_SCHEMA_VERSION,
            "backend": BACKEND_ID,
            "operation": operation,
            "request": request,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _path(self, operation: str, request: dict[str, Any]) -> Path:
        assert self.root is not None
        return self.root / operation / f"{self._digest(operation, request)}.json.gz"

    def load(self, operation: str, request: dict[str, Any]) -> Any | None:
        if self.mode not in {"replay", "auto"} or self.root is None:
            return None
        path = self._path(operation, request)
        if not path.is_file():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            envelope = json.load(handle)
        if (
            envelope.get("schema_version") != BACKEND_SCHEMA_VERSION
            or envelope.get("backend") != BACKEND_ID
            or envelope.get("operation") != operation
            or envelope.get("request") != request
        ):
            raise RuntimeError(f"Invalid BFCL-Web cache entry: {path}")
        return deepcopy(envelope["response"])

    def store(self, operation: str, request: dict[str, Any], response: Any) -> None:
        if self.mode not in {"record", "auto"} or self.root is None:
            return
        path = self._path(operation, request)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": BACKEND_SCHEMA_VERSION,
            "backend": BACKEND_ID,
            "ddgs_version": _ddgs_version(),
            "operation": operation,
            "request": request,
            "response": response,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(envelope, handle, ensure_ascii=False, sort_keys=True)
        os.replace(temporary, path)


class WebSearchAPI(_OfficialWebSearchAPI):
    """Drop-in BFCL WebSearchAPI backed by direct DuckDuckGo requests."""

    def __init__(self):
        if _brotli_version() == "not-installed":
            raise RuntimeError(
                "The BFCL-Web environment requires the 'Brotli' package. "
                "The official fetch_url_content() advertises br encoding; "
                "without a decoder some sites are returned as compressed "
                "garbage. Install it with: pip install Brotli"
            )
        super().__init__()
        raw_fetch_limit = os.getenv(FETCH_CONTENT_LIMIT_ENV, "").strip()
        self._fetch_content_limit = (
            None if not raw_fetch_limit else int(raw_fetch_limit)
        )
        if self._fetch_content_limit is not None and self._fetch_content_limit <= 0:
            raise ValueError(f"{FETCH_CONTENT_LIMIT_ENV} must be positive when set.")
        self._cache = _ResponseCache(
            os.getenv("PLENA_BFCL_WEB_CACHE_DIR"),
            os.getenv("PLENA_BFCL_WEB_CACHE_MODE", "off"),
        )
        self._max_attempts = max(
            1, int(os.getenv("PLENA_BFCL_WEB_DDGS_MAX_ATTEMPTS", "6"))
        )
        self._retry_base_seconds = max(
            0.0, float(os.getenv("PLENA_BFCL_WEB_DDGS_RETRY_BASE_SECONDS", "2"))
        )
        self._stats_path = (
            Path(os.environ["PLENA_BFCL_WEB_STATS_PATH"]).expanduser().resolve()
            if os.getenv("PLENA_BFCL_WEB_STATS_PATH")
            else None
        )
        empty_stats = {
            "backend": BACKEND_ID,
            "ddgs_version": _ddgs_version(),
            "brotli_version": _brotli_version(),
            "cache_mode": self._cache.mode,
            "cache_dir": str(self._cache.root) if self._cache.root else None,
            "cache_hits": 0,
            "cache_misses": 0,
            "live_calls": 0,
            "live_call_seconds_total": 0.0,
            "errors": 0,
            "ddgs_max_concurrency": _DDGS_MAX_CONCURRENCY,
            "fetch_content_max_chars": self._fetch_content_limit,
            "fetch_truncations": 0,
            "fetch_original_chars_total": 0,
            "fetch_returned_chars_total": 0,
            "operations": {},
        }
        if self._stats_path is None:
            self._stats_lock = threading.RLock()
            self._stats = empty_stats
        else:
            registry_key = str(self._stats_path)
            with _STATS_REGISTRY_LOCK:
                state = _STATS_REGISTRY.get(registry_key)
                if state is None:
                    if self._stats_path.is_file():
                        try:
                            prior_stats = json.loads(
                                self._stats_path.read_text(encoding="utf-8")
                            )
                            if prior_stats.get("backend") == BACKEND_ID:
                                empty_stats = prior_stats
                        except (OSError, json.JSONDecodeError):
                            # The launcher removes stale stats before each run.
                            # This only protects an interrupted atomic update.
                            pass
                    state = (threading.RLock(), empty_stats)
                    _STATS_REGISTRY[registry_key] = state
                self._stats_lock, self._stats = state
        self._write_stats()

    def _increment(self, metric: str, operation: str) -> None:
        with self._stats_lock:
            self._stats[metric] = int(self._stats.get(metric, 0)) + 1
            operations = self._stats.setdefault("operations", {})
            operation_stats = operations.setdefault(
                operation,
                {
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "live_calls": 0,
                    "live_call_seconds_total": 0.0,
                    "errors": 0,
                },
            )
            operation_stats[metric] = int(operation_stats.get(metric, 0)) + 1
            self._write_stats_locked()

    def _write_stats(self) -> None:
        with self._stats_lock:
            self._write_stats_locked()

    def _write_stats_locked(self) -> None:
        if self._stats_path is None:
            return
        self._stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._stats_path.with_name(
            f".{self._stats_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(self._stats, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self._stats_path)

    def _record_live_duration(self, operation: str, duration: float) -> None:
        with self._stats_lock:
            self._stats["live_call_seconds_total"] = float(
                self._stats.get("live_call_seconds_total", 0.0)
            ) + float(duration)
            operations = self._stats.setdefault("operations", {})
            operation_stats = operations.setdefault(operation, {})
            operation_stats["live_call_seconds_total"] = float(
                operation_stats.get("live_call_seconds_total", 0.0)
            ) + float(duration)
            self._write_stats_locked()

    def _bound_fetch_response(self, response: Any) -> Any:
        if not isinstance(response, dict):
            return response
        content = response.get("content")
        if not isinstance(content, str):
            return response

        limit = self._fetch_content_limit
        if limit is None or len(content) <= limit:
            return response

        bounded = dict(response)
        bounded["content"] = content[:limit]
        with self._stats_lock:
            self._stats["fetch_truncations"] = (
                int(self._stats.get("fetch_truncations", 0)) + 1
            )
            self._stats["fetch_original_chars_total"] = int(
                self._stats.get("fetch_original_chars_total", 0)
            ) + len(content)
            self._stats["fetch_returned_chars_total"] = int(
                self._stats.get("fetch_returned_chars_total", 0)
            ) + limit
            self._write_stats_locked()
        return bounded

    @staticmethod
    def _is_success(response: Any) -> bool:
        return not (isinstance(response, dict) and "error" in response)

    def _cached_call(
        self,
        operation: str,
        request: dict[str, Any],
        live_call: Callable[[], Any],
    ) -> Any:
        cached = self._cache.load(operation, request)
        if cached is not None:
            self._increment("cache_hits", operation)
            return cached

        if self._cache.mode in {"auto", "replay"}:
            self._increment("cache_misses", operation)
        if self._cache.mode == "replay":
            self._increment("errors", operation)
            raise RuntimeError(
                "BFCL-Web replay cache miss for "
                f"{operation}: {_canonical_json(request)}"
            )

        self._increment("live_calls", operation)
        live_started = time.perf_counter()
        response = live_call()
        self._record_live_duration(
            operation, time.perf_counter() - live_started
        )
        if self._is_success(response):
            self._cache.store(operation, request, response)
        else:
            self._increment("errors", operation)
        return response

    def _ddgs_text(
        self,
        keywords: str,
        max_results: int,
        region: str,
    ) -> list[dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError as error:
            raise RuntimeError(
                "The direct BFCL-Web backend requires the 'ddgs' package in "
                "the environment that runs the bfcl CLI."
            ) from error

        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                with DDGS() as client:
                    return list(
                        client.text(
                            keywords,
                            region=region,
                            max_results=max_results,
                            backend="duckduckgo",
                        )
                    )
            except Exception as error:  # DDGS exposes backend-specific failures.
                last_error = error
                if attempt == self._max_attempts:
                    break
                delay = min(
                    self._retry_base_seconds * (2 ** (attempt - 1)),
                    120.0,
                )
                print(
                    "[PLENA BFCL-Web] DDGS attempt "
                    f"{attempt}/{self._max_attempts} failed: {error}; "
                    f"retrying in {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list | dict:
        count = 10 if max_results is None else max(0, int(max_results))
        selected_region = "wt-wt" if region is None else str(region)
        request = {
            "keywords": str(keywords),
            "max_results": count,
            "region": selected_region,
            "show_snippet": bool(self.show_snippet),
        }

        def live() -> list | dict:
            try:
                # BFCL episodes may run concurrently so their model requests
                # can form GPU microbatches. Keep direct DDGS calls separately
                # bounded to avoid turning that model-side concurrency into a
                # burst against the public search endpoint.
                with _DDGS_LIVE_SEMAPHORE:
                    raw_results = self._ddgs_text(
                        str(keywords),
                        count,
                        selected_region,
                    )
            except Exception as error:
                return {"error": f"Direct DuckDuckGo search failed: {error}"}

            results = []
            for item in raw_results[:count]:
                title = item.get("title")
                href = item.get("href") or item.get("url") or item.get("link")
                if title is None or href is None:
                    continue
                converted = {"title": str(title), "href": str(href)}
                if self.show_snippet:
                    converted["body"] = str(
                        item.get("body") or item.get("snippet") or ""
                    )
                results.append(converted)
            return results

        return self._cached_call("search_engine_query", request, live)

    def fetch_url_content(self, url: str, mode: str = "raw") -> str | dict:
        selected_mode = str(mode)
        # Include the policy in the key so entries captured before bounded
        # fetching was introduced cannot replay an unbounded web page.
        request = {
            "url": str(url),
            "mode": selected_mode,
            # This extra key also prevents entries recorded by the old
            # unversioned backend from replaying into the corrected runtime.
            "max_content_chars": self._fetch_content_limit,
        }
        return self._cached_call(
            "fetch_url_content",
            request,
            lambda: self._bound_fetch_response(
                super(WebSearchAPI, self).fetch_url_content(url, selected_mode)
            ),
        )
