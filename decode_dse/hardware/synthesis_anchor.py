"""Content-addressed full-chip DC/SAIF anchors for selected candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from decode_dse.hardware.design_space import (
    CalibratedEnergy,
    HardwareCandidate,
)
from decode_dse.profiles import DecodePrecisionProfile, format_descriptor

EXACT_DC_ANCHOR_SCHEMA = "decode-exact-dc-anchor-index"
EXACT_DC_SCOPE = "decode_chip_through_final_rmsnorm"
REQUIRED_ACTIVITY = frozenset({"linear", "qk", "pv", "vector", "selector"})
REQUIRED_ARTIFACT_KINDS = frozenset(
    {
        "area_report",
        "compiler_precision_binding",
        "constraints",
        "decode_trace",
        "library_manifest",
        "power_report",
        "rtl_source_manifest",
        "rtl_specialization",
        "saif",
        "synthesis_log",
        "synthesized_netlist",
        "timing_report",
    }
)
RTL_SOURCE_MANIFEST_SCHEMA = "decode-rtl-source-manifest"
RTL_SPECIALIZATION_SCHEMA = "decode-rtl-dc-specialization"
COMPILER_BINDING_SCHEMA = "plena-compiler-precision-binding"
_NUMBER = r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
_POWER_SCALE = {
    "W": 1.0,
    "mW": 1e-3,
    "uW": 1e-6,
    "nW": 1e-9,
}
ANCHOR_ERROR_LIMITS_PCT = {
    "area": 15.0,
    "dynamic_power": 25.0,
    "leakage_power": 25.0,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_confined(base: Path, raw: object) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("DC anchor paths must be confined and relative")
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError("DC anchor path escapes its artifact root") from exc
    return resolved


def _one_match(pattern: str, text: str, label: str) -> tuple[str, ...]:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"{label} report field must occur exactly once")
    value = matches[0]
    return (value,) if isinstance(value, str) else tuple(value)


def _parse_area(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="strict")
    value = float(
        _one_match(
            rf"^\s*Total\s+cell\s+area\s*:\s*{_NUMBER}\s*$",
            text,
            "area",
        )[0]
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("area report value must be finite and positive")
    return value / 1e6


def _parse_power(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="strict")
    dynamic_value, dynamic_unit = _one_match(
        rf"^\s*Total\s+Dynamic\s+Power\s*=\s*{_NUMBER}\s*"
        rf"(W|mW|uW|nW)(?:\s+\([^)]*\))?\s*$",
        text,
        "dynamic power",
    )
    leakage_value, leakage_unit = _one_match(
        rf"^\s*Cell\s+Leakage\s+Power\s*=\s*{_NUMBER}\s*"
        rf"(W|mW|uW|nW)(?:\s+\([^)]*\))?\s*$",
        text,
        "leakage power",
    )
    dynamic = float(dynamic_value) * _POWER_SCALE[dynamic_unit]
    leakage = float(leakage_value) * _POWER_SCALE[leakage_unit]
    if any(
        not math.isfinite(value) or value <= 0
        for value in (dynamic, leakage)
    ):
        raise ValueError("power report values must be finite and positive")
    return dynamic, leakage


def _parse_timing(path: Path) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8", errors="strict")
    period = float(
        _one_match(
            rf"^\s*Clock\s+period\s*:\s*{_NUMBER}\s*ns\s*$",
            text,
            "clock period",
        )[0]
    )
    slack = float(
        _one_match(
            rf"^\s*slack\s+\(MET\)\s+{_NUMBER}\s*$",
            text,
            "timing slack",
        )[0]
    )
    if (
        not math.isfinite(period)
        or period <= 0
        or not math.isfinite(slack)
        or slack < 0
    ):
        raise ValueError("timing report must meet a positive clock constraint")
    return period, slack


def _load_artifacts(
    raw: object,
    *,
    root: Path,
) -> dict[str, tuple[Path, str]]:
    if not isinstance(raw, list):
        raise TypeError("DC anchor artifacts must be a list")
    artifacts: dict[str, tuple[Path, str]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
            "kind",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("DC anchor artifact fields differ from the schema")
        kind = str(item["kind"])
        if kind in artifacts:
            raise ValueError(f"DC anchor repeats artifact kind {kind}")
        path = _resolve_confined(root, item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"DC anchor artifact is missing: {path}")
        size = item["size_bytes"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or path.stat().st_size != size
        ):
            raise ValueError(f"DC anchor artifact size mismatch: {path}")
        digest = str(item["sha256"])
        if _sha256_file(path) != digest:
            raise ValueError(f"DC anchor artifact checksum mismatch: {path}")
        artifacts[kind] = (path, digest)
    if set(artifacts) != REQUIRED_ARTIFACT_KINDS:
        raise ValueError("DC anchor artifact coverage is incomplete")
    return artifacts


def _load_hashed_json(path: Path, schema_version: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    body = dict(raw)
    content_hash = str(body.pop("content_hash", ""))
    if body.get("schema_version") != schema_version:
        raise ValueError(f"{path} uses an unsupported schema")
    if _content_hash(body) != content_hash:
        raise ValueError(f"{path} content hash mismatch")
    return {**body, "content_hash": content_hash}


def _load_source_manifest(path: Path) -> dict[str, Any]:
    value = _load_hashed_json(path, RTL_SOURCE_MANIFEST_SCHEMA)
    if set(value) != {
        "schema_version",
        "files",
        "source_tree_sha256",
        "content_hash",
    }:
        raise ValueError("RTL source manifest fields differ from the schema")
    files = value["files"]
    if not isinstance(files, Mapping) or not files:
        raise ValueError("RTL source manifest file hashes are invalid")
    canonical_files: dict[str, str] = {}
    for raw_name, raw_digest in files.items():
        name = str(raw_name)
        digest = str(raw_digest)
        relative = Path(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError("RTL source manifest file hashes are invalid")
        canonical_files[name] = digest
    expected = _content_hash({"files": dict(sorted(canonical_files.items()))})
    if value["source_tree_sha256"] != expected:
        raise ValueError("RTL source-tree hash mismatch")
    return value


def _load_compiler_binding(
    path: Path,
    *,
    profile_id: str,
    candidate: HardwareCandidate,
) -> dict[str, Any]:
    value = _load_hashed_json(path, COMPILER_BINDING_SCHEMA)
    if set(value) != {
        "schema_version",
        "profile_id",
        "profile",
        "target",
        "evidence_target",
        "matrix_binding_mode",
        "format_descriptors",
        "format_binding_ids",
        "runtime_precision_contract",
        "binding_id",
        "content_hash",
    }:
        raise ValueError("compiler precision binding fields differ from the schema")
    binding_body = {
        key: value[key]
        for key in value
        if key not in {"binding_id", "content_hash"}
    }
    expected_binding_id = "cpb-" + _content_hash(binding_body)
    if value["binding_id"] != expected_binding_id:
        raise ValueError("compiler precision binding identity mismatch")
    if value["profile_id"] != profile_id:
        raise ValueError("compiler precision binding profile mismatch")
    profile = value["profile"]
    if (
        not isinstance(profile, Mapping)
        or "schema_version" not in profile
        or profile_id != "dqp-" + _content_hash(profile)
    ):
        raise ValueError("compiler precision profile identity mismatch")
    target = value["target"]
    if not isinstance(target, Mapping) or set(target) != {
        "mlen",
        "blen",
        "hlen",
        "batch",
        "kv_heads",
        "head_dim",
        "block_size",
        "selector_bits",
        "packed_kv",
        "batched_attention",
    }:
        raise ValueError("compiler precision target fields differ from the schema")
    integer_target = {
        name: target[name]
        for name in (
            "mlen",
            "blen",
            "hlen",
            "batch",
            "kv_heads",
            "head_dim",
            "block_size",
            "selector_bits",
        )
    }
    if any(
        isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0
        for raw in integer_target.values()
    ):
        raise ValueError("compiler precision target dimensions are invalid")
    if (
        target["mlen"] != candidate.mlen
        or target["blen"] != candidate.blen
        or target["hlen"] != candidate.hlen
        or target["batch"] != 1
        or target["block_size"] != 8
        or target["kv_heads"] * target["head_dim"] != target["mlen"]
        or target["head_dim"] != target["hlen"]
        or 2 ** target["selector_bits"] < target["kv_heads"]
        or target["packed_kv"] is not True
        or target["batched_attention"] is not True
    ):
        raise ValueError("compiler precision target does not bind the anchor")
    evidence_target = value["evidence_target"]
    if not isinstance(evidence_target, Mapping) or set(evidence_target) != {
        "schema_version",
        "target_mode",
        "capability_scope",
        "source_tree_sha256",
        "mxint2_activation_scope",
        "rtl_deployment_supports_mxint2_activation",
        "common_deployment_valid",
    }:
        raise ValueError("compiler evidence target fields differ from the schema")
    compiler_source_hash = str(evidence_target["source_tree_sha256"])
    if len(compiler_source_hash) != 64 or any(
        character not in "0123456789abcdef"
        for character in compiler_source_hash
    ):
        raise ValueError("compiler source-tree identity is invalid")
    if (
        evidence_target["schema_version"] != "plena-evidence-target"
        or evidence_target["target_mode"]
        != "simulator_compiler_emulator"
        or evidence_target["capability_scope"] != "numerical_and_emulator"
        or evidence_target["mxint2_activation_scope"] != "emulator_only"
        or evidence_target[
            "rtl_deployment_supports_mxint2_activation"
        ]
        is not False
        or evidence_target["common_deployment_valid"] is not False
    ):
        raise ValueError("compiler evidence target is unsupported")
    role_tokens = {
        "weight": profile.get("weight_format"),
        "activation": profile.get("activation_format"),
        "key": profile.get("key_format"),
        "value": profile.get("value_format"),
        "vector": profile.get("vector_format"),
    }
    expected_descriptors: dict[str, dict[str, Any]] = {}
    for role, token in role_tokens.items():
        descriptor = format_descriptor(str(token))
        expected_descriptors[role] = {
            "token": descriptor.token,
            "family": descriptor.family,
            "element_bits": descriptor.element_bits,
            "exponent_bits": descriptor.exponent_bits,
            "mantissa_bits": descriptor.mantissa_bits,
            "signed": descriptor.signed,
            "block_scaled": descriptor.block_scaled,
        }
    if value["format_descriptors"] != expected_descriptors:
        raise ValueError("compiler format descriptors do not bind the profile")
    expected_format_ids = {
        role: "fmt-" + _content_hash(descriptor)
        for role, descriptor in expected_descriptors.items()
    }
    if value["format_binding_ids"] != expected_format_ids:
        raise ValueError("compiler format-binding identities mismatch")
    if value["matrix_binding_mode"] != "static_hardware_signature":
        raise ValueError("compiler precision binding mode is unsupported")
    runtime = value["runtime_precision_contract"]
    if not isinstance(runtime, Mapping):
        raise ValueError("compiler runtime precision contract is missing")
    runtime_body = dict(runtime)
    runtime_hash = str(runtime_body.pop("content_hash", ""))
    if _content_hash(runtime_body) != runtime_hash:
        raise ValueError("compiler runtime precision contract hash mismatch")
    parameters = runtime.get("rtl_precision_parameters")
    semantics = runtime.get("matrix_semantics")
    if not isinstance(parameters, Mapping) or not isinstance(
        semantics, Mapping
    ):
        raise ValueError("compiler runtime precision bindings are incomplete")
    semantics_body = dict(semantics)
    semantics_hash = str(semantics_body.pop("content_hash", ""))
    if _content_hash(semantics_body) != semantics_hash:
        raise ValueError("compiler matrix semantics hash mismatch")
    return value


def _load_specialization(
    path: Path,
    *,
    profile_id: str,
    candidate: HardwareCandidate,
    artifacts: Mapping[str, tuple[Path, str]],
    source_manifest: Mapping[str, Any],
    compiler_binding: Mapping[str, Any],
) -> dict[str, Any]:
    value = _load_hashed_json(path, RTL_SPECIALIZATION_SCHEMA)
    if set(value) != {
        "schema_version",
        "profile_id",
        "binding_id",
        "target",
        "format_bindings",
        "rtl_precision_parameters",
        "matrix_semantics_sha256",
        "rtl_source_tree_sha256",
        "artifact_sha256",
        "build_command",
        "selector_enabled",
        "specialization_id",
        "content_hash",
    }:
        raise ValueError("RTL specialization fields differ from the schema")
    if value["profile_id"] != profile_id:
        raise ValueError("RTL specialization profile mismatch")
    if value["binding_id"] != compiler_binding["binding_id"]:
        raise ValueError("RTL specialization compiler binding mismatch")
    if value["target"] != candidate.to_dict():
        raise ValueError("RTL specialization target mismatch")
    if value["selector_enabled"] is not True:
        raise ValueError("RTL specialization must enable PackedKV selection")
    bindings = value["format_bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "weight",
        "activation",
        "key",
        "value",
        "vector",
    }:
        raise ValueError("RTL specialization format bindings are invalid")
    parameters = value["rtl_precision_parameters"]
    if (
        not isinstance(parameters, Mapping)
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(raw, bool)
            or not isinstance(raw, int)
            or raw < 0
            for name, raw in parameters.items()
        )
    ):
        raise ValueError("RTL specialization parameter map is invalid")
    if value["rtl_source_tree_sha256"] != source_manifest[
        "source_tree_sha256"
    ]:
        raise ValueError("RTL specialization source tree mismatch")
    hashes = value["artifact_sha256"]
    expected_hashes = {
        kind: artifacts[kind][1]
        for kind in (
            "compiler_precision_binding",
            "constraints",
            "decode_trace",
            "library_manifest",
            "saif",
            "synthesis_log",
            "synthesized_netlist",
        )
    }
    if hashes != expected_hashes:
        raise ValueError("RTL specialization artifact hashes mismatch")
    command = value["build_command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("RTL specialization build command is invalid")
    semantics = str(value["matrix_semantics_sha256"])
    if len(semantics) != 64 or any(
        character not in "0123456789abcdef" for character in semantics
    ):
        raise ValueError("RTL specialization matrix semantics hash is invalid")
    runtime = compiler_binding["runtime_precision_contract"]
    compiler_semantics = runtime["matrix_semantics"]["content_hash"]
    if semantics != compiler_semantics:
        raise ValueError("RTL specialization matrix semantics mismatch")
    if value["rtl_precision_parameters"] != runtime[
        "rtl_precision_parameters"
    ]:
        raise ValueError("RTL specialization parameter map differs from compiler")
    specialization_body = {
        key: value[key]
        for key in value
        if key not in {"content_hash", "specialization_id"}
    }
    expected_id = "rtl-specialization-" + _content_hash(
        specialization_body
    )
    if value["specialization_id"] != expected_id:
        raise ValueError("RTL specialization identity mismatch")
    return value


def _validate_synthesis_log_bindings(
    path: Path,
    *,
    artifacts: Mapping[str, tuple[Path, str]],
    source_tree_sha256: str,
) -> None:
    text = path.read_text(encoding="utf-8", errors="strict")
    expected = {
        "RTL_SOURCE_TREE_SHA256": source_tree_sha256,
        "COMPILER_BINDING_SHA256": artifacts[
            "compiler_precision_binding"
        ][1],
        "CONSTRAINTS_SHA256": artifacts["constraints"][1],
        "LIBRARY_MANIFEST_SHA256": artifacts["library_manifest"][1],
        "SAIF_SHA256": artifacts["saif"][1],
        "DECODE_TRACE_SHA256": artifacts["decode_trace"][1],
        "NETLIST_SHA256": artifacts["synthesized_netlist"][1],
    }
    for label, digest in expected.items():
        observed = _one_match(
            rf"^\s*{label}\s*:\s*([0-9a-f]{{64}})\s*$",
            text,
            label,
        )[0]
        if observed != digest:
            raise ValueError(f"synthesis log binding mismatch: {label}")


@dataclass(frozen=True)
class ExactDCAnchor:
    """One measured full-chip implementation and decode activity point."""

    profile_id: str
    candidate: HardwareCandidate
    workload: Mapping[str, Any]
    timing_evidence_id: str
    layout_id: str
    traffic_ledger_id: str
    area_mm2: float
    dynamic_power_w: float
    leakage_power_w: float
    clock_period_ns: float
    worst_slack_ns: float
    record_hash: str
    artifact_sha256: tuple[tuple[str, str], ...]
    synthesis_context: Mapping[str, Any]
    specialization: Mapping[str, Any]
    compiler_binding: Mapping[str, Any]

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def anchor_id(self) -> str:
        return f"exact-dc-{self.record_hash}"

    def match(
        self,
        *,
        profile: DecodePrecisionProfile,
        candidate: HardwareCandidate,
        observation: Any,
    ) -> None:
        if profile.profile_id != self.profile_id:
            raise ValueError("DC anchor profile identity mismatch")
        if candidate.to_dict() != self.candidate.to_dict():
            raise ValueError("DC anchor hardware candidate mismatch")
        if observation.timing_evidence_id != self.timing_evidence_id:
            raise ValueError("DC anchor timing evidence mismatch")
        if observation.layout_id != self.layout_id:
            raise ValueError("DC anchor PackedKV layout mismatch")
        if observation.traffic_ledger_id != self.traffic_ledger_id:
            raise ValueError("DC anchor traffic ledger mismatch")
        bindings = self.specialization.get("format_bindings")
        expected_bindings = {
            "weight": profile.weight_format,
            "activation": profile.activation_format,
            "key": profile.key_format,
            "value": profile.value_format,
            "vector": profile.vector_format,
        }
        if bindings != expected_bindings:
            raise ValueError("DC anchor precision format binding mismatch")
        if self.compiler_binding.get("profile") != profile.to_dict():
            raise ValueError("DC anchor compiler profile binding mismatch")
        if self.compiler_binding.get("binding_id") != self.specialization.get(
            "binding_id"
        ):
            raise ValueError("DC anchor compiler/RTL binding mismatch")
        expected_parameters = {
            "BLOCK_DIM": profile.block_size,
            "MX_SCALE_WIDTH": profile.scale_bits,
        }
        for prefix, token in (
            ("WT", profile.weight_format),
            ("ACT", profile.activation_format),
            ("KV", profile.key_format),
        ):
            descriptor = format_descriptor(token)
            if descriptor.family == "mxint":
                expected_parameters[f"{prefix}_MX_INT_ENABLE"] = 1
                expected_parameters[f"{prefix}_MX_INT_WIDTH"] = (
                    descriptor.element_bits
                )
            else:
                expected_parameters[f"{prefix}_MX_INT_ENABLE"] = 0
                expected_parameters[f"{prefix}_MX_EXP_WIDTH"] = (
                    descriptor.exponent_bits
                )
                expected_parameters[f"{prefix}_MX_MANT_WIDTH"] = (
                    descriptor.mantissa_bits
                )
        vector = format_descriptor(profile.vector_format)
        for prefix in ("V_FP", "M_FP", "S_FP"):
            expected_parameters[f"{prefix}_EXP_WIDTH"] = vector.exponent_bits
            expected_parameters[f"{prefix}_MANT_WIDTH"] = vector.mantissa_bits
        if self.specialization.get("rtl_precision_parameters") != dict(
            sorted(expected_parameters.items())
        ):
            raise ValueError("DC anchor RTL precision parameters mismatch")
        runtime = self.compiler_binding["runtime_precision_contract"]
        if runtime.get("rtl_precision_parameters") != dict(
            sorted(expected_parameters.items())
        ):
            raise ValueError("DC anchor compiler RTL parameters mismatch")
        if (
            runtime["matrix_semantics"]["content_hash"]
            != self.specialization["matrix_semantics_sha256"]
        ):
            raise ValueError("DC anchor compiler matrix semantics mismatch")

    def validate_prediction(
        self,
        prediction: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply selected-point prediction-versus-DC publication gates."""

        if prediction.get("profile_id") != self.profile_id:
            raise ValueError("DC prediction profile identity mismatch")
        if prediction.get("candidate_id") != self.candidate_id:
            raise ValueError("DC prediction candidate identity mismatch")
        if prediction.get("selector_enabled") is not True:
            raise ValueError("DC prediction omits the PackedKV selector")
        if (
            int(prediction.get("MLEN", 0)) != self.candidate.mlen
            or int(prediction.get("BLEN", 0)) != self.candidate.blen
        ):
            raise ValueError("DC prediction geometry mismatch")
        context = prediction.get("synthesis_context")
        if not isinstance(context, Mapping):
            raise ValueError("DC prediction synthesis context is missing")
        for field in ("dc_tool_version", "library_id", "process_corner"):
            if context.get(field) != self.synthesis_context.get(field):
                raise ValueError(
                    f"DC prediction synthesis context mismatch: {field}"
                )
        comparisons = {
            "area": (
                float(prediction["area_mm2"]),
                self.area_mm2,
            ),
            "dynamic_power": (
                float(prediction["dynamic_power_w"]),
                self.dynamic_power_w,
            ),
            "leakage_power": (
                float(prediction["leakage_power_w"]),
                self.leakage_power_w,
            ),
        }
        errors: dict[str, float] = {}
        for name, (predicted, measured) in comparisons.items():
            if (
                not math.isfinite(predicted)
                or predicted <= 0
                or not math.isfinite(measured)
                or measured <= 0
            ):
                raise ValueError(f"DC {name} comparison is invalid")
            error = abs(predicted - measured) / measured * 100.0
            errors[name] = error
            if error > ANCHOR_ERROR_LIMITS_PCT[name]:
                raise ValueError(
                    f"DC {name} anchor error exceeds publication gate"
                )
        hbm_j = float(prediction["hbm_j_per_token"])
        hbm_calibration_id = str(prediction["hbm_calibration_id"])
        if not math.isfinite(hbm_j) or hbm_j < 0 or not hbm_calibration_id:
            raise ValueError("DC prediction HBM attribution is invalid")
        return {
            "passed": True,
            "limits_pct": dict(ANCHOR_ERROR_LIMITS_PCT),
            "errors_pct": dict(sorted(errors.items())),
            "prediction_calibration_id": str(
                prediction["prediction_calibration_id"]
            ),
            "hbm_j_per_token": hbm_j,
            "hbm_calibration_id": hbm_calibration_id,
        }

    def energy(
        self,
        *,
        duration_s: float,
        hbm_j: float,
        hbm_calibration_id: str,
    ) -> CalibratedEnergy:
        if (
            not math.isfinite(duration_s)
            or duration_s <= 0
            or not math.isfinite(hbm_j)
            or hbm_j < 0
            or not hbm_calibration_id
        ):
            raise ValueError("DC anchor energy inputs are invalid")
        calibration_id = "exact-dc-system-" + _content_hash(
            {
                "anchor_id": self.anchor_id,
                "hbm_calibration_id": hbm_calibration_id,
            }
        )
        return CalibratedEnergy(
            calibration_id=calibration_id,
            compute_j=0.0,
            vector_j=0.0,
            sram_j=0.0,
            hbm_j=hbm_j,
            leakage_j=self.leakage_power_w * duration_s,
            unattributed_dynamic_j=self.dynamic_power_w * duration_s,
            duration_s=duration_s,
        )

    def to_status(
        self,
        prediction_status: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        status = {
            "schema_version": EXACT_DC_ANCHOR_SCHEMA,
            "anchor_id": self.anchor_id,
            "profile_id": self.profile_id,
            "candidate_id": self.candidate_id,
            "metric_scope": EXACT_DC_SCOPE,
            "area_mm2": self.area_mm2,
            "dynamic_power_w": self.dynamic_power_w,
            "leakage_power_w": self.leakage_power_w,
            "clock_period_ns": self.clock_period_ns,
            "worst_slack_ns": self.worst_slack_ns,
            "record_hash": self.record_hash,
            "artifact_sha256": dict(self.artifact_sha256),
            "compiler_binding_id": self.compiler_binding["binding_id"],
            "specialization_id": self.specialization["specialization_id"],
            "prediction_validation": (
                dict(prediction_status)
                if prediction_status is not None
                else None
            ),
        }
        return status


@dataclass(frozen=True)
class ExactDCAnchorIndex:
    source_path: Path
    source_sha256: str
    model_name: str
    model_revision: str
    rtl_source_tree_sha256: str
    workload: Mapping[str, Any]
    synthesis_context: Mapping[str, Any]
    anchors: tuple[ExactDCAnchor, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        model_name: str,
        model_revision: str,
        workload: Mapping[str, Any],
        rtl_source_tree_sha256: str,
    ) -> "ExactDCAnchorIndex":
        source = Path(path).resolve()
        payload = source.read_bytes()
        raw = json.loads(payload)
        if not isinstance(raw, Mapping):
            raise TypeError("DC anchor index root must be an object")
        body = dict(raw)
        content_hash = str(body.pop("content_hash", ""))
        if _content_hash(body) != content_hash:
            raise ValueError("DC anchor index content hash mismatch")
        if set(body) != {
            "schema_version",
            "model_name",
            "model_revision",
            "rtl_source_tree_sha256",
            "metric_scope",
            "workload",
            "synthesis_context",
            "records",
        }:
            raise ValueError("DC anchor index fields differ from the schema")
        if body["schema_version"] != EXACT_DC_ANCHOR_SCHEMA:
            raise ValueError("unsupported DC anchor index schema")
        if body["model_name"] != model_name:
            raise ValueError("DC anchor model name mismatch")
        if body["model_revision"] != model_revision:
            raise ValueError("DC anchor model revision mismatch")
        if (
            body["rtl_source_tree_sha256"] != rtl_source_tree_sha256
            or len(rtl_source_tree_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in rtl_source_tree_sha256
            )
        ):
            raise ValueError("DC anchor RTL source-tree identity mismatch")
        if body["metric_scope"] != EXACT_DC_SCOPE:
            raise ValueError("DC anchor metric scope mismatch")
        if body["workload"] != dict(workload):
            raise ValueError("DC anchor workload mismatch")
        if (
            body["workload"].get("scope") != "steady_state_cached_q1"
            or body["workload"].get("query_length") != 1
            or body["workload"].get("admission_included") is not False
        ):
            raise ValueError("DC anchor must exclude decode admission")
        context = body["synthesis_context"]
        if not isinstance(context, Mapping) or set(context) != {
            "dc_tool_version",
            "library_id",
            "process_corner",
            "clock_period_ns",
            "mx_block_size",
            "power_analysis_mode",
            "area_unit",
        }:
            raise ValueError("DC anchor synthesis context differs from the schema")
        if any(
            not str(context[name]).strip()
            for name in ("dc_tool_version", "library_id", "process_corner")
        ):
            raise ValueError("DC anchor synthesis context is incomplete")
        if (
            float(context["clock_period_ns"]) != 1.0
            or int(context["mx_block_size"]) != 8
            or context["power_analysis_mode"] != "saif_annotated"
            or context["area_unit"] != "um2"
        ):
            raise ValueError("DC anchor synthesis context is unsupported")
        records = body["records"]
        if not isinstance(records, list) or not records:
            raise ValueError("DC anchor index must contain measured records")
        anchors: list[ExactDCAnchor] = []
        keys: set[tuple[str, str]] = set()
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                raise TypeError("DC anchor records must be objects")
            record_body = dict(raw_record)
            record_hash = str(record_body.pop("record_hash", ""))
            if _content_hash(record_body) != record_hash:
                raise ValueError("DC anchor record hash mismatch")
            if set(record_body) != {
                "profile_id",
                "candidate",
                "workload",
                "timing_evidence_id",
                "layout_id",
                "traffic_ledger_id",
                "activity_coverage",
                "artifacts",
            }:
                raise ValueError("DC anchor record fields differ from the schema")
            candidate_raw = record_body["candidate"]
            if not isinstance(candidate_raw, Mapping):
                raise TypeError("DC anchor candidate must be an object")
            candidate = HardwareCandidate.from_dict(
                candidate_raw,
                allow_legacy_single_chip=True,
            )
            if record_body["workload"] != body["workload"]:
                raise ValueError("DC anchor record workload mismatch")
            coverage = record_body["activity_coverage"]
            if (
                not isinstance(coverage, list)
                or frozenset(coverage) != REQUIRED_ACTIVITY
                or len(coverage) != len(REQUIRED_ACTIVITY)
            ):
                raise ValueError("DC anchor activity coverage is incomplete")
            artifacts = _load_artifacts(
                record_body["artifacts"],
                root=source.parent,
            )
            source_manifest = _load_source_manifest(
                artifacts["rtl_source_manifest"][0]
            )
            if (
                source_manifest["source_tree_sha256"]
                != rtl_source_tree_sha256
            ):
                raise ValueError(
                    "DC anchor record uses another RTL source tree"
                )
            compiler_binding = _load_compiler_binding(
                artifacts["compiler_precision_binding"][0],
                profile_id=str(record_body["profile_id"]),
                candidate=candidate,
            )
            specialization = _load_specialization(
                artifacts["rtl_specialization"][0],
                profile_id=str(record_body["profile_id"]),
                candidate=candidate,
                artifacts=artifacts,
                source_manifest=source_manifest,
                compiler_binding=compiler_binding,
            )
            _validate_synthesis_log_bindings(
                artifacts["synthesis_log"][0],
                artifacts=artifacts,
                source_tree_sha256=source_manifest[
                    "source_tree_sha256"
                ],
            )
            area = _parse_area(artifacts["area_report"][0])
            dynamic, leakage = _parse_power(
                artifacts["power_report"][0]
            )
            period, slack = _parse_timing(
                artifacts["timing_report"][0]
            )
            if abs(period - float(context["clock_period_ns"])) > 1e-12:
                raise ValueError("DC anchor report clock differs from context")
            anchor = ExactDCAnchor(
                profile_id=str(record_body["profile_id"]),
                candidate=candidate,
                workload=dict(body["workload"]),
                timing_evidence_id=str(record_body["timing_evidence_id"]),
                layout_id=str(record_body["layout_id"]),
                traffic_ledger_id=str(record_body["traffic_ledger_id"]),
                area_mm2=area,
                dynamic_power_w=dynamic,
                leakage_power_w=leakage,
                clock_period_ns=period,
                worst_slack_ns=slack,
                record_hash=record_hash,
                artifact_sha256=tuple(
                    sorted(
                        (kind, digest)
                        for kind, (_, digest) in artifacts.items()
                    )
                ),
                synthesis_context=dict(context),
                specialization=specialization,
                compiler_binding=compiler_binding,
            )
            if any(
                not value
                for value in (
                    anchor.profile_id,
                    anchor.timing_evidence_id,
                    anchor.layout_id,
                    anchor.traffic_ledger_id,
                )
            ):
                raise ValueError("DC anchor identities must be non-empty")
            key = (anchor.profile_id, anchor.candidate_id)
            if key in keys:
                raise ValueError("DC anchor index repeats a candidate profile")
            keys.add(key)
            anchors.append(anchor)
        return cls(
            source_path=source,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            model_name=model_name,
            model_revision=model_revision,
            rtl_source_tree_sha256=rtl_source_tree_sha256,
            workload=dict(workload),
            synthesis_context=dict(context),
            anchors=tuple(anchors),
        )

    def get(
        self,
        profile_id: str,
        candidate: HardwareCandidate,
    ) -> ExactDCAnchor | None:
        key = (profile_id, candidate.candidate_id)
        return next(
            (
                anchor
                for anchor in self.anchors
                if (anchor.profile_id, anchor.candidate_id) == key
            ),
            None,
        )

    def to_status(self) -> dict[str, Any]:
        return {
            "schema_version": EXACT_DC_ANCHOR_SCHEMA,
            "source_sha256": self.source_sha256,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "rtl_source_tree_sha256": self.rtl_source_tree_sha256,
            "record_count": len(self.anchors),
            "anchor_ids": sorted(anchor.anchor_id for anchor in self.anchors),
            "synthesis_context": dict(self.synthesis_context),
        }


__all__ = [
    "EXACT_DC_ANCHOR_SCHEMA",
    "EXACT_DC_SCOPE",
    "ExactDCAnchor",
    "ExactDCAnchorIndex",
    "REQUIRED_ACTIVITY",
    "REQUIRED_ARTIFACT_KINDS",
]
