"""Optional startup overlay for the standalone BFCL CLI process."""

from __future__ import annotations

import os


if os.getenv("PLENA_BFCL_WEB_SEARCH_BACKEND") == (
    "bfcl_harness_ddgs_duckduckgo"
):
    from bfcl_eval.constants import executable_backend_config

    executable_backend_config.CLASS_FILE_PATH_MAPPING["WebSearchAPI"] = (
        "quant_eval.bfcl_web_backend"
    )
