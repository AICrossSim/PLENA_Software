"""Contracts for immutable route-summary model/config overlays."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from decode_dse.manifest import validate_sweep_config
from decode_dse.simulator_bridge import DecodeSimulator
from decode_dse.software.router_trace_repricing import (
    load_hashed_json,
    materialize_repricing_bundle,
    verify_repricing_bundle,
)


SOFTWARE_ROOT = Path(__file__).resolve().parents[2]
SIMULATOR = SOFTWARE_ROOT.parent / "PLENA_Simulator"
CONFIG = (
    SOFTWARE_ROOT
    / "decode_dse"
    / "configs"
    / "qwen3_30b_a3b_thinking_2507.json"
)
if str(SIMULATOR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR))

from analytic_models.disagg_serve.expert_placement import (  # noqa: E402
    EvidenceReceipt,
    RouteRecord,
    RoutingStep,
    RoutingTrace,
    trace_content_hash,
)
from analytic_models.disagg_serve.router_trace_summary import (  # noqa: E402
    summarize_trace,
)


def _trace() -> RoutingTrace:
    steps = []
    for step_index in range(5):
        records = []
        for layer in range(48):
            experts = (
                tuple(range(8, 16))
                if layer == 1 and step_index % 2
                else tuple(range(8))
            )
            records.append(
                RouteRecord(
                    token_id=f"token-{step_index}",
                    layer=layer,
                    source_chip=step_index % 4,
                    expert_ids=experts,
                )
            )
        steps.append(RoutingStep(step_index=step_index, records=tuple(records)))
    receipt = EvidenceReceipt(
        artifact_path="/trace/evidence.json",
        artifact_sha256="1" * 64,
        subject_sha256="2" * 64,
        command=("collect",),
        tool_revision="3" * 64,
        recorded_at_utc="2026-08-20T00:00:00Z",
        sample_count=5,
    )
    return RoutingTrace(
        source_kind="measured",
        steps=tuple(steps),
        receipt=receipt,
    )


def _summary(trace: RoutingTrace) -> dict:
    source_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    return summarize_trace(
        trace,
        [1, 4, 512],
        source_binding={
            "collector_verified": True,
            "router_index_path": str(CONFIG),
            "router_index_sha256": source_sha,
            "router_index_content_hash": "5" * 64,
            "placement_input_path": str(CONFIG),
            "placement_input_sha256": source_sha,
            "router_trace_evidence_path": str(CONFIG),
            "router_trace_evidence_sha256": source_sha,
            "trace_content_hash": trace_content_hash(trace),
        },
    )


def test_bundle_is_restartable_batch_bound_and_directly_loadable(tmp_path, monkeypatch):
    monkeypatch.setenv("PLENA_SIMULATOR_PATH", str(SIMULATOR))
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(_trace()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "overlays"
    index_path = tmp_path / "bundle.json"

    bundle = materialize_repricing_bundle(
        summary_path=summary_path,
        base_config_path=CONFIG,
        simulator_root=SIMULATOR,
        output_dir=output_dir,
        index_path=index_path,
    )

    assert bundle["materializable_batches"] == [1, 4]
    assert bundle["unsupported_batches"] == [512]
    assert bundle["classification"]["publication_rankable"] is False
    batch_four = next(row for row in bundle["entries"] if row["batch_size"] == 4)
    model_path = Path(batch_four["model_overlay"]["path"])
    config_path = Path(batch_four["study_config_overlay"]["path"])
    model = load_hashed_json(model_path)
    config = load_hashed_json(config_path)

    assert model["num_experts"] == 128
    assert model["num_experts_per_tok"] == 8
    assert model["norm_topk_prob"] is True
    assert model["moe_route_repricing"]["resident_expert_storage"] == {
        "num_experts": 128,
        "policy": "all_experts_remain_resident",
        "changed_by_overlay": False,
    }
    assert set(model["moe_route_repricing"]["injected_fields"]) == {
        "moe_unique_experts_per_step",
        "moe_routing_imbalance_factor",
    }
    assert config["sim_model"] == str(model_path)
    assert config["hardware_space"]["BATCH"] == [4]
    validate_sweep_config(config)

    simulator = DecodeSimulator(str(model_path))
    precision = simulator.make_precision(attn_w=4, ffn_w=4, kv=4, act_w=4)
    metrics = simulator.evaluate(
        precision,
        batch=4,
        input_seq=16,
        output_seq=2,
        hw_over=simulator.shipped_over(precision),
        n_chips=1,
        stride=1,
        hbm_gen="HBM2",
        hbm_channels=8,
    )
    workload = metrics.moe_workload
    assert workload is not None
    assert workload["num_experts"] == 128
    assert workload["norm_topk_prob"] is True
    assert workload["route_repricing"]["trace_content_hash"] == bundle[
        "trace_content_hash"
    ]
    assert workload["route_repricing"]["summary_content_hash"] == _summary(
        _trace()
    )["content_hash"]
    assert workload["expert_batch_ledger_source"] == (
        "balanced_integer_distribution_over_charged_active_experts"
    )
    assert workload["provenance"]["publication_rankable"] is False

    with pytest.raises(ValueError, match="overlay batch"):
        simulator.evaluate(
            precision,
            batch=1,
            input_seq=16,
            output_seq=1,
            hw_over=simulator.shipped_over(precision),
            n_chips=1,
            stride=1,
            hbm_gen="HBM2",
            hbm_channels=8,
        )

    reused = materialize_repricing_bundle(
        summary_path=summary_path,
        base_config_path=CONFIG,
        simulator_root=SIMULATOR,
        output_dir=output_dir,
        index_path=index_path,
    )
    assert reused["reused"] is True
    assert verify_repricing_bundle(
        index_path=index_path,
        simulator_root=SIMULATOR,
    )["verified"] is True

    summary_path.write_bytes(summary_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="summary file hash mismatch"):
        DecodeSimulator(str(model_path))


def test_bundle_verification_rejects_changed_overlay(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(_trace()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "bundle.json"
    bundle = materialize_repricing_bundle(
        summary_path=summary_path,
        base_config_path=CONFIG,
        simulator_root=SIMULATOR,
        output_dir=tmp_path / "overlays",
        index_path=index_path,
    )
    artifact = Path(bundle["entries"][0]["model_overlay"]["path"])
    artifact.write_bytes(artifact.read_bytes() + b" ")

    with pytest.raises(ValueError, match="missing or changed"):
        verify_repricing_bundle(index_path=index_path, simulator_root=SIMULATOR)
