import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file, save_file
import tqdm


def _model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


_REFERENCE_MODES = {"none", "write", "compare"}


def _token_hash(tokens: torch.Tensor) -> str:
    values = tokens.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _read_progress(path: Path | None, profile_id: str) -> dict[int, dict]:
    if path is None or not path.is_file():
        return {}
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("profile_id") != profile_id:
            continue
        try:
            rows[int(row["chunk_index"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def compute_kl_stats(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    position_block_size: int = 8,
) -> dict[str, float | int]:
    """Return exact token-averaged KL components without a full FP32 copy."""
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            "Reference/candidate logits shape mismatch: "
            f"{tuple(reference_logits.shape)} != {tuple(candidate_logits.shape)}"
        )
    if reference_logits.ndim != 3 or reference_logits.shape[0] != 1:
        raise ValueError("KL logits must have shape [1, sequence, vocabulary].")
    if position_block_size <= 0:
        raise ValueError("position_block_size must be positive.")

    candidate_device = candidate_logits.device
    sequence = int(candidate_logits.shape[1])
    kl_sum = 0.0
    top1_matches = 0
    for start in range(0, sequence, position_block_size):
        stop = min(start + position_block_size, sequence)
        ref = reference_logits[:, start:stop].to(candidate_device).float()
        cand = candidate_logits[:, start:stop].float()
        ref_logp = F.log_softmax(ref, dim=-1)
        cand_logp = F.log_softmax(cand, dim=-1)
        ref_prob = ref_logp.exp()
        kl_sum += float((ref_prob * (ref_logp - cand_logp)).sum().item())
        top1_matches += int((ref.argmax(dim=-1) == cand.argmax(dim=-1)).sum().item())
        del ref, cand, ref_logp, cand_logp, ref_prob
    return {
        "kl_sum": kl_sum,
        "kl_tokens": sequence,
        "top1_matches": top1_matches,
    }


def _load_token_stream(
    tokenizer,
    dataset_name: str,
    subset: str | None,
    split: str,
) -> torch.Tensor:
    resolved_name = "Salesforce/wikitext" if dataset_name == "wikitext" else dataset_name
    resolved_subset = subset
    if dataset_name == "wikitext" and resolved_subset is None:
        resolved_subset = "wikitext-2-raw-v1"
    if resolved_subset:
        test_data = load_dataset(resolved_name, resolved_subset, split=split)
    else:
        test_data = load_dataset(resolved_name, split=split)
    return tokenizer("\n\n".join(test_data["text"]), return_tensors="pt").input_ids


def _reference_metadata(
    *,
    model_fingerprint: str,
    tokenizer_fingerprint: str,
    dataset_name: str,
    subset: str | None,
    split: str,
    max_length: int,
    kl_samples: int,
    token_hashes: list[str],
    complete: bool,
) -> dict:
    return {
        "schema_version": 1,
        "complete": bool(complete),
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "dataset": dataset_name,
        "subset": subset,
        "split": split,
        "max_length": int(max_length),
        "kl_samples": int(kl_samples),
        "token_hashes": token_hashes,
    }


def _validate_reference_metadata(actual: dict, expected: dict, *, require_complete: bool) -> None:
    for key, value in expected.items():
        if key == "complete":
            continue
        if actual.get(key) != value:
            raise ValueError(
                f"Reference metadata mismatch for {key}: {actual.get(key)!r} != {value!r}."
            )
    if require_complete and not actual.get("complete"):
        raise ValueError("Reference logits are incomplete; finish the BF16 reference run first.")


def evaluate_perplexity_with_kl(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    subset: str | None = None,
    split: str = "test",
    max_length: int = 2048,
    max_samples: int | None = None,
    verbose: bool = False,
    *,
    reference_mode: str = "none",
    reference_dir: str | Path | None = None,
    kl_samples: int = 64,
    model_fingerprint: str = "",
    tokenizer_fingerprint: str = "",
    profile_id: str = "default",
    progress_path: str | Path | None = None,
    kl_position_block_size: int = 8,
):
    reference_mode = str(reference_mode or "none").lower()
    if reference_mode not in _REFERENCE_MODES:
        raise ValueError(
            f"reference_mode must be one of {sorted(_REFERENCE_MODES)}, got {reference_mode!r}."
        )
    if reference_mode != "none" and reference_dir is None:
        raise ValueError("reference_dir is required for reference write/compare modes.")
    if kl_samples <= 0:
        raise ValueError("kl_samples must be positive.")

    input_ids = _load_token_stream(tokenizer, dataset_name, subset, split)
    model.seqlen = max_length
    model.eval()
    nsamples = input_ids.numel() // max_length
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive when set.")
        nsamples = min(nsamples, max_samples)
    if nsamples <= 0:
        raise ValueError(
            f"Not enough tokens ({input_ids.numel()}) for max_length={max_length}."
        )
    resolved_kl_samples = min(int(kl_samples), nsamples)
    chunk_hashes = [
        _token_hash(input_ids[:, i * max_length : (i + 1) * max_length])
        for i in range(resolved_kl_samples)
    ]

    reference_path = Path(reference_dir).expanduser().resolve() if reference_dir else None
    metadata_path = reference_path / "metadata.json" if reference_path else None
    expected_metadata = _reference_metadata(
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        max_length=max_length,
        kl_samples=resolved_kl_samples,
        token_hashes=chunk_hashes,
        complete=False,
    )
    if reference_mode == "write":
        reference_path.mkdir(parents=True, exist_ok=True)
        if metadata_path.is_file():
            _validate_reference_metadata(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                expected_metadata,
                require_complete=False,
            )
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
    elif reference_mode == "compare":
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing reference metadata: {metadata_path}")
        _validate_reference_metadata(
            json.loads(metadata_path.read_text(encoding="utf-8")),
            expected_metadata,
            require_complete=True,
        )

    progress = Path(progress_path).expanduser().resolve() if progress_path else None
    completed = _read_progress(progress, profile_id)
    device = _model_input_device(model)
    nll_total = 0.0
    num_tokens = 0
    kl_sum = 0.0
    kl_tokens = 0
    top1_matches = 0

    iterator = tqdm.tqdm(range(nsamples), desc="Evaluating PPL/KL", disable=not verbose)
    for i in iterator:
        batch_cpu = input_ids[:, i * max_length : (i + 1) * max_length]
        input_hash = _token_hash(batch_cpu)
        cached = completed.get(i)
        cached_is_usable = cached is not None and cached.get("input_hash") == input_hash
        if cached_is_usable and reference_mode == "write" and i < resolved_kl_samples:
            cached_is_usable = (
                reference_path / f"chunk_{i:04d}.safetensors"
            ).is_file()
        if cached_is_usable:
            nll_total += float(cached["nll"])
            num_tokens += int(cached["predicted_tokens"])
            kl_sum += float(cached.get("kl_sum", 0.0))
            kl_tokens += int(cached.get("kl_tokens", 0))
            top1_matches += int(cached.get("top1_matches", 0))
            continue

        batch = batch_cpu.to(device)
        with torch.inference_mode():
            outputs = model(batch, use_cache=True)
            shift_logits = outputs.logits[:, :-1, :]
            shift_labels = batch[:, 1:]
            nll = F.cross_entropy(
                shift_logits.float().reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="sum",
            )
        row = {
            "profile_id": profile_id,
            "chunk_index": i,
            "input_hash": input_hash,
            "nll": float(nll.item()),
            "predicted_tokens": int(max_length - 1),
        }

        if i < resolved_kl_samples and reference_mode == "write":
            chunk_path = reference_path / f"chunk_{i:04d}.safetensors"
            save_file(
                {"logits": shift_logits.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()},
                str(chunk_path),
            )
            row["reference_written"] = True
        elif i < resolved_kl_samples and reference_mode == "compare":
            chunk_path = reference_path / f"chunk_{i:04d}.safetensors"
            if not chunk_path.is_file():
                raise FileNotFoundError(f"Missing reference logits: {chunk_path}")
            reference_logits = load_file(str(chunk_path), device="cpu")["logits"]
            stats = compute_kl_stats(
                reference_logits,
                shift_logits,
                position_block_size=kl_position_block_size,
            )
            row.update(stats)
            del reference_logits

        nll_total += row["nll"]
        num_tokens += row["predicted_tokens"]
        kl_sum += float(row.get("kl_sum", 0.0))
        kl_tokens += int(row.get("kl_tokens", 0))
        top1_matches += int(row.get("top1_matches", 0))
        if progress is not None:
            _append_jsonl(progress, row)
        del batch, outputs, shift_logits, shift_labels, nll

    if reference_mode == "write":
        missing_chunks = [
            str(reference_path / f"chunk_{i:04d}.safetensors")
            for i in range(resolved_kl_samples)
            if not (reference_path / f"chunk_{i:04d}.safetensors").is_file()
        ]
        if missing_chunks:
            raise RuntimeError(
                "Refusing to mark BF16 reference complete; missing logits chunks: "
                + ", ".join(missing_chunks[:3])
            )
        complete_metadata = dict(expected_metadata)
        complete_metadata["complete"] = True
        metadata_path.write_text(
            json.dumps(complete_metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    ppl = float(torch.exp(torch.tensor(nll_total / num_tokens, dtype=torch.float64)).item())
    results = {
        "ppl": ppl,
        "nll": nll_total,
        "num_tokens": int(num_tokens),
        "nsamples": int(nsamples),
        "reference_mode": reference_mode,
        "kl_samples": int(resolved_kl_samples),
        "kl_tokens": int(kl_tokens),
        "kl_nats_per_token": (kl_sum / kl_tokens) if kl_tokens else None,
        "top1_agreement": (top1_matches / kl_tokens) if kl_tokens else None,
    }
    print(f"Perplexity: {ppl:.4f}")
    if kl_tokens:
        print(f"KL(BF16||candidate): {results['kl_nats_per_token']:.8f} nats/token")
        print(f"Top-1 agreement: {results['top1_agreement']:.4%}")
    return results


def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext",
    subset: str | None = None,
    split: str = "test",
    max_length: int = 2048,
    max_samples: int | None = None,
    verbose: bool = False,
):
    return evaluate_perplexity_with_kl(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        max_length=max_length,
        max_samples=max_samples,
        verbose=verbose,
    )
