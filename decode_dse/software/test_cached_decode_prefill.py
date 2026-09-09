"""Tests for selective-logit BF16 prefill capture."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from types import SimpleNamespace
import json
from pathlib import Path

import pytest

from decode_dse.software.cache_artifacts import (
    ArtifactProvenance,
    BF16CacheConverter,
    admit_prefill_cache,
    load_prefill_artifact,
    load_prefill_decode_metadata,
    save_prefill_artifact,
)
from decode_dse.software.cached_decode import (
    ContinuationExample,
    DecodeStep,
    capture_bf16_prefill,
    evaluate_teacher_forced_cached,
)


class _SelectiveLogitModel:
    def __init__(self, vocabulary: int = 11) -> None:
        self.vocabulary = vocabulary
        self.requested_positions: tuple[int, ...] | None = None

    def eval(self) -> "_SelectiveLogitModel":
        return self

    def full_logits(self, input_ids):
        import torch

        batch, sequence = input_ids.shape
        rows = torch.arange(batch, device=input_ids.device)[:, None, None]
        positions = torch.arange(sequence, device=input_ids.device)[None, :, None]
        tokens = torch.arange(self.vocabulary, device=input_ids.device)[None, None, :]
        return (rows * 0.25 + positions * 0.5 - tokens * 0.125).to(torch.bfloat16)

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        cache_position,
        use_cache,
        logits_to_keep,
    ):
        import torch

        assert use_cache is True
        positions = tuple(int(value) for value in logits_to_keep.tolist())
        self.requested_positions = positions
        logits = self.full_logits(input_ids)[:, logits_to_keep, :]
        batch, sequence = input_ids.shape
        key = torch.arange(
            batch * sequence * 2,
            device=input_ids.device,
            dtype=torch.bfloat16,
        ).reshape(batch, 1, sequence, 2)
        return SimpleNamespace(
            logits=logits,
            past_key_values=((key, key.clone()),),
        )


def _provenance() -> ArtifactProvenance:
    return ArtifactProvenance(
        producer="selective-logit-test",
        code_revision="test",
        created_at_utc="1970-01-01T00:00:00Z",
    )


def _reseal_manifest(manifest: dict) -> None:
    descriptor = dict(manifest)
    descriptor.pop("artifact_id", None)
    payload = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    manifest["artifact_id"] = hashlib.sha256(payload).hexdigest()


class _DeterministicDecodeBackend:
    """Tiny backend whose score depends only on the decode inputs."""

    def materialize_cache(self, artifact):
        return {"batch": artifact.batch_size, "length": artifact.cache_length}

    @staticmethod
    def cache_batch_size(cache) -> int:
        return int(cache["batch"])

    @staticmethod
    def cache_length(cache) -> int:
        return int(cache["length"])

    @staticmethod
    def decode_step(
        model,
        *,
        input_token_id,
        cache,
        attention_mask,
        position_id,
        cache_position,
    ):
        del model
        assert cache_position == cache["length"]
        assert len(attention_mask) == cache_position + 1
        return DecodeStep(
            logits=(int(input_token_id), int(position_id)),
            cache={"batch": cache["batch"], "length": cache["length"] + 1},
        )

    @staticmethod
    def commit_append(cache, *, previous_length, artifact):
        del artifact
        assert cache["length"] == previous_length + 1
        return cache

    @staticmethod
    def token_nll(logits, target_token_id) -> float:
        return float(abs(logits[0] - target_token_id) + logits[1] / 1000.0)


def _artifact_and_admission(tmp_path):
    import torch

    artifact = capture_bf16_prefill(
        _SelectiveLogitModel(),
        input_ids=torch.tensor([[7, 8, 9]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        model_revision="model",
        tokenizer_revision="tokenizer",
        provenance=_provenance(),
    )
    root = tmp_path / "prefill"
    save_prefill_artifact(artifact, root)
    admitted = admit_prefill_cache(
        artifact,
        precision_id="BF16",
        layout_id="test-layout",
        converter=BF16CacheConverter(),
        provenance=_provenance(),
    )
    return artifact, admitted, root


def test_selective_prefill_logits_match_full_logits_for_padded_prompts():
    import torch

    model = _SelectiveLogitModel()
    input_ids = torch.tensor([[7, 8, 0, 0], [0, 0, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
    artifact = capture_bf16_prefill(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        model_revision="model",
        tokenizer_revision="tokenizer",
        provenance=_provenance(),
    )

    full = model.full_logits(input_ids)
    expected_last = torch.stack((full[0, 1], full[1, 3]))
    expected_tokens = tuple(
        int(value) for value in expected_last.argmax(dim=-1).tolist()
    )
    expected_log_probs = torch.log_softmax(expected_last.float(), dim=-1).gather(
        -1,
        torch.tensor(expected_tokens)[:, None],
    )[:, 0]

    assert model.requested_positions == (1, 3)
    assert artifact.first_token.token_ids == expected_tokens
    assert artifact.first_token.log_probabilities == pytest.approx(
        tuple(float(value) for value in expected_log_probs.tolist()),
        abs=0.0,
    )
    assert artifact.cache_length == input_ids.shape[1]


def test_selective_prefill_rejects_an_empty_padded_row_before_forward():
    import torch

    model = _SelectiveLogitModel()
    with pytest.raises(ValueError, match="at least one active token"):
        capture_bf16_prefill(
            model,
            input_ids=torch.tensor([[1, 2], [0, 0]]),
            attention_mask=torch.tensor([[1, 1], [0, 0]]),
            model_revision="model",
            tokenizer_revision="tokenizer",
            provenance=_provenance(),
        )
    assert model.requested_positions is None


def test_manifest_only_decode_metadata_retains_exact_prefill_identity(tmp_path):
    artifact, _, root = _artifact_and_admission(tmp_path)

    metadata = load_prefill_decode_metadata(root)
    assert metadata.artifact_id == artifact.artifact_id
    assert metadata.prompt_hash == artifact.prompt_hash
    assert metadata.attention_mask == artifact.attention_mask
    assert metadata.position_ids == artifact.position_ids
    assert metadata.first_token == artifact.first_token
    assert metadata.batch_size == artifact.batch_size
    assert metadata.cache_length == artifact.cache_length
    assert metadata.layer_count == len(artifact.layers)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["position_ids"][0][-1] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest identity mismatch"):
        load_prefill_decode_metadata(root)


@pytest.mark.parametrize(
    ("field", "value"),
    (("dtype", "float16"), ("shape", [1, 1, 2, 3])),
)
def test_manifest_only_decode_metadata_rejects_kv_descriptor_mismatch(
    tmp_path,
    field,
    value,
):
    _, _, root = _artifact_and_admission(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"][0]["key"][field] = value
    _reseal_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="K/V geometry mismatch"):
        load_prefill_decode_metadata(root)


def test_manifest_only_decode_metadata_rejects_prompt_hash_mismatch(tmp_path):
    _, _, root = _artifact_and_admission(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_hash"] = "0" * 64
    _reseal_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt hash mismatch"):
        load_prefill_decode_metadata(root)


def test_metadata_view_rejects_admitted_cache_from_another_prefill(tmp_path):
    artifact, admitted, root = _artifact_and_admission(tmp_path)
    metadata = load_prefill_decode_metadata(root)
    wrong_source = replace(
        admitted,
        source_artifact_id=("0" * 64 if artifact.artifact_id != "0" * 64 else "1" * 64),
    )

    with pytest.raises(ValueError, match="not derived from this prefill artifact"):
        ContinuationExample(
            document_id="document",
            prefill=metadata,
            decode_cache=wrong_source,
            continuation_ids=(artifact.first_token.token_ids[0], 4),
        )


def test_metadata_view_is_nll_identical_and_never_opens_source_planes(
    tmp_path,
    monkeypatch,
):
    artifact, admitted, root = _artifact_and_admission(tmp_path)
    fully_loaded = load_prefill_artifact(root)
    original_open = Path.open
    plane_opens: list[Path] = []

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".bin" and "layers" in path.parts:
            plane_opens.append(path)
            raise AssertionError(f"source BF16 plane was reopened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    metadata = load_prefill_decode_metadata(root)
    continuation = (artifact.first_token.token_ids[0], 4, 6, 3)
    full_result = evaluate_teacher_forced_cached(
        object(),
        ContinuationExample(
            document_id="document",
            prefill=fully_loaded,
            decode_cache=admitted,
            continuation_ids=continuation,
        ),
        _DeterministicDecodeBackend(),
    )
    metadata_result = evaluate_teacher_forced_cached(
        object(),
        ContinuationExample(
            document_id="document",
            prefill=metadata,
            decode_cache=admitted,
            continuation_ids=continuation,
        ),
        _DeterministicDecodeBackend(),
    )

    assert metadata_result == full_result
    assert plane_opens == []
