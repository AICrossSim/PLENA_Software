from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quant_eval.eval import eval_ppl


class _ToyModel(nn.Module):
    def __init__(self, *, perturb: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.perturb = perturb
        self.calls = 0

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, use_cache=True):
        self.calls += 1
        vocab = 8
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab)
        logits.scatter_(2, input_ids.unsqueeze(-1), 4.0)
        if self.perturb:
            logits[..., 0] += self.perturb
        return SimpleNamespace(logits=logits)


def test_compute_kl_stats_matches_direct_formula():
    torch.manual_seed(3)
    reference = torch.randn(1, 5, 7)
    candidate = reference + 0.2 * torch.randn_like(reference)

    stats = eval_ppl.compute_kl_stats(reference, candidate, position_block_size=2)
    ref_logp = reference.float().log_softmax(dim=-1)
    cand_logp = candidate.float().log_softmax(dim=-1)
    expected = (ref_logp.exp() * (ref_logp - cand_logp)).sum()

    assert stats["kl_sum"] == pytest.approx(expected.item(), rel=1e-6, abs=1e-7)
    assert stats["kl_tokens"] == 5


def test_ppl_reference_compare_is_resumable_and_validates_fingerprint(tmp_path, monkeypatch):
    token_stream = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    monkeypatch.setattr(eval_ppl, "_load_token_stream", lambda *args, **kwargs: token_stream)
    reference_dir = tmp_path / "reference"

    baseline = _ToyModel()
    baseline_result = eval_ppl.evaluate_perplexity_with_kl(
        baseline,
        tokenizer=object(),
        max_length=4,
        max_samples=None,
        reference_mode="write",
        reference_dir=reference_dir,
        kl_samples=1,
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        profile_id="bf16",
        progress_path=tmp_path / "baseline.jsonl",
    )
    assert baseline_result["nsamples"] == 2
    assert (reference_dir / "metadata.json").is_file()
    assert (reference_dir / "chunk_0000.safetensors").is_file()

    # A progress row alone must not make a reference complete after its tensor
    # was removed. Resume should recompute and restore the missing chunk.
    (reference_dir / "chunk_0000.safetensors").unlink()
    repaired = _ToyModel()
    eval_ppl.evaluate_perplexity_with_kl(
        repaired,
        tokenizer=object(),
        max_length=4,
        max_samples=None,
        reference_mode="write",
        reference_dir=reference_dir,
        kl_samples=1,
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        profile_id="bf16",
        progress_path=tmp_path / "baseline.jsonl",
    )
    assert repaired.calls == 1
    assert (reference_dir / "chunk_0000.safetensors").is_file()

    candidate = _ToyModel(perturb=0.5)
    candidate_result = eval_ppl.evaluate_perplexity_with_kl(
        candidate,
        tokenizer=object(),
        max_length=4,
        max_samples=None,
        reference_mode="compare",
        reference_dir=reference_dir,
        kl_samples=1,
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        profile_id="quant",
        progress_path=tmp_path / "quant.jsonl",
    )
    assert candidate_result["kl_tokens"] == 3
    assert candidate_result["kl_nats_per_token"] > 0

    resumed = _ToyModel(perturb=0.5)
    resumed_result = eval_ppl.evaluate_perplexity_with_kl(
        resumed,
        tokenizer=object(),
        max_length=4,
        max_samples=None,
        reference_mode="compare",
        reference_dir=reference_dir,
        kl_samples=1,
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        profile_id="quant",
        progress_path=tmp_path / "quant.jsonl",
    )
    assert resumed.calls == 0
    assert resumed_result == candidate_result

    with pytest.raises(ValueError, match="model_fingerprint"):
        eval_ppl.evaluate_perplexity_with_kl(
            _ToyModel(),
            tokenizer=object(),
            max_length=4,
            reference_mode="compare",
            reference_dir=reference_dir,
            kl_samples=1,
            model_fingerprint="wrong-model",
            tokenizer_fingerprint="tokenizer-a",
        )
