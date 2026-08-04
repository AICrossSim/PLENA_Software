"""Seal a pinned publication chat-template asset from the local tokenizer.

Extracts the chat template from the locally cached tokenizer snapshot at the
config's pinned revision (no network access), writes the versioned asset file
next to the sweep configs, and pins ``publication.chat_template_asset`` and
``publication.chat_template_sha256`` in the config so every later pipeline
run validates against the sealed asset instead of re-executing the tokenizer.

The tokenizer files must already exist in the pinned local snapshot. For a
gated repository, fetch them once with an authenticated download, e.g.:

    hf download meta-llama/Llama-3.1-8B-Instruct \
        "tokenizer*" "special_tokens_map.json" \
        --revision <model_revision> --cache-dir /data/models

Then seal:

    python -m decode_dse.software.seal_chat_template_asset \
        --config decode_dse/configs/llama3_1_8b.json \
        --asset decode_dse/configs/publication_chat_template_llama3_1_8b.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from decode_dse.software.benchmark_runner import (
    PUBLICATION_CHAT_TEMPLATE_SCHEMA,
    seal_publication_chat_template,
)

_ASSET_FIELDS = (
    "schema_version",
    "model_name",
    "model_revision",
    "tokenizer_revision",
    "enable_thinking",
    "chat_template_sha256",
    "chat_template",
)


def _repo_relative(path: Path, repository: Path) -> str:
    return path.resolve().relative_to(repository.resolve()).as_posix()


def _pin_config_fields(
    config_path: Path, *, asset_reference: str, template_sha256: str
) -> None:
    """Set the two publication pin fields with minimal formatting churn."""

    text = config_path.read_text(encoding="utf-8")
    for key, value in (
        ("chat_template_asset", asset_reference),
        ("chat_template_sha256", template_sha256),
    ):
        pattern = re.compile(rf'("({key})":\s*)"[^"]*"')
        replacement = rf'\g<1>"{value}"'
        if pattern.search(text):
            text = pattern.sub(replacement, text, count=1)
        else:
            anchor = '"publication": {'
            if anchor not in text:
                raise SystemExit(
                    f"{config_path} has no publication object to pin"
                )
            text = text.replace(
                anchor,
                f'{anchor}\n    "{key}": "{value}",',
                1,
            )
    config_path.write_text(text, encoding="utf-8")
    reloaded = json.loads(config_path.read_text(encoding="utf-8"))
    publication = reloaded["publication"]
    if (
        publication.get("chat_template_asset") != asset_reference
        or publication.get("chat_template_sha256") != template_sha256
    ):
        raise SystemExit(f"failed to pin the chat template in {config_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--asset", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    extraction_config = copy.deepcopy(config)
    publication = extraction_config.get("publication")
    if not isinstance(publication, dict):
        raise SystemExit("config.publication is required")
    publication.pop("chat_template_asset", None)
    publication.pop("chat_template_sha256", None)

    sealed = seal_publication_chat_template(
        extraction_config, config_path=config_path
    )
    if sealed["schema_version"] != PUBLICATION_CHAT_TEMPLATE_SCHEMA:
        raise SystemExit("sealed chat template has an unexpected schema")
    asset = {field: sealed[field] for field in _ASSET_FIELDS}

    asset_path = args.asset.resolve()
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_text(
        json.dumps(asset, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    repository = Path(__file__).resolve().parents[2]
    asset_reference = _repo_relative(asset_path, repository)
    _pin_config_fields(
        config_path,
        asset_reference=asset_reference,
        template_sha256=str(sealed["chat_template_sha256"]),
    )
    print(f"asset: {asset_path}")
    print(f"chat_template_sha256: {sealed['chat_template_sha256']}")
    print(f"pinned in: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
