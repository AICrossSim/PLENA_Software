"""Schema-checked bridge to calibrated simulator power artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from decode_dse.hardware.design_space import CalibratedEnergy
from decode_dse.profiles import DecodePrecisionProfile

SUPPORTED_POWER_MODEL_VERSIONS = ("plena-event-power-v4",)
EXPECTED_CALIBRATION_MANIFEST_HASH = (
    "baad3cf6e7648069f2121b06475eda40000cd6c9a069c6534c3137a00b0eb241"
)
EXPECTED_POWER_INTERPOLATION_DOMAIN = {
    "mlen_min": 16,
    "mlen_max": 64,
    "blen_min": 4,
    "blen_max": 16,
}
EXPECTED_HARDWARE_FP_BINDING = "FP_E6M5"
EXPECTED_FIT_SUMMARY = {
    "complete_rows": 502,
    "train_rows": 332,
    "holdout_rows": 170,
    "signature_count": 80,
    "structural_area_signature_count": 72,
    "vector_area_signature_count": 7,
    "coverage_failures": 0,
}
SELECTOR_SIGNATURE = "SELECTOR:PACKED_KV"
STRUCTURAL_AREA_SCHEMA = "plena-structural-area-evidence"
STRUCTURAL_AREA_MODEL_VERSION = "matrix_structural_census"
STRUCTURAL_FEATURES = (
    "pe_tl",
    "pe_sum",
    "pe_0",
    "reduce",
    "scale",
    "out",
    "fixed",
    "const",
)
REQUIRED_STRUCTURAL_AREA_SOURCES = frozenset(
    {
        "matrix_structural_coefficients.json",
        "asap7_sram_macro_table.csv",
        "matrix_machine_mxint.csv",
        "matrix_machine_mxfp.csv",
    }
)
MXINT_WEIGHT_FORMATS = ("MXINT4", "MXINT8")
MXINT_OPERAND_FORMATS = ("MXINT2", "MXINT4", "MXINT8")
MXFP_HARDWARE_FORMATS = ("E1M2", "E2M1", "E4M3", "E5M2")
VECTOR_HARDWARE_FORMATS = (
    "FP_E3M2",
    "FP_E2M3",
    "FP_E6M5",
    "FP_E5M6",
    "FP_E4M7",
    "FP_E8M5",
    "BF16",
)


def _simulator_token(token: str) -> str:
    if token.startswith("MXINT"):
        return token
    if token.startswith("E"):
        return f"MXFP_{token}"
    return token


def required_profile_power_signatures(
    profile: DecodePrecisionProfile,
) -> tuple[str, ...]:
    """Return the calibrated operation signatures used by one profile."""

    weight = _simulator_token(profile.weight_format)
    activation = _simulator_token(profile.activation_format)
    key = _simulator_token(profile.key_format)
    value = _simulator_token(profile.value_format)
    return tuple(
        sorted(
            {
                f"LINEAR:{weight}x{activation}",
                f"QK:{key}x{activation}",
                f"PV:{value}x{activation}",
                f"VECTOR:{profile.vector_format}",
                SELECTOR_SIGNATURE,
            }
        )
    )


def required_hardware_power_signatures() -> tuple[str, ...]:
    """Return complete matrix and vector calibration coverage."""

    linear = {
        f"LINEAR:{weight}x{activation}"
        for weight, activation in itertools.product(
            MXINT_WEIGHT_FORMATS,
            MXINT_OPERAND_FORMATS,
        )
    }
    qk = {
        f"QK:{kv}x{activation}"
        for kv, activation in itertools.product(
            MXINT_OPERAND_FORMATS,
            MXINT_OPERAND_FORMATS,
        )
    }
    pv = {
        f"PV:{kv}x{activation}"
        for kv, activation in itertools.product(
            MXINT_OPERAND_FORMATS,
            MXINT_OPERAND_FORMATS,
        )
    }
    mxfp = tuple(_simulator_token(token) for token in MXFP_HARDWARE_FORMATS)
    linear.update(
        f"LINEAR:{weight}x{activation}"
        for weight, activation in itertools.product(mxfp, mxfp)
    )
    qk.update(
        f"QK:{kv}x{activation}"
        for kv, activation in itertools.product(mxfp, mxfp)
    )
    pv.update(
        f"PV:{kv}x{activation}"
        for kv, activation in itertools.product(mxfp, mxfp)
    )
    vector = {f"VECTOR:{token}" for token in VECTOR_HARDWARE_FORMATS}
    return tuple(sorted(linear | qk | pv | vector | {SELECTOR_SIGNATURE}))


@dataclass(frozen=True)
class SimulatorPowerStatus:
    """Validated state of one simulator power-calibration artifact."""

    source_path: Path
    source_sha256: str
    model_version: str
    provenance_hash: str
    required_signatures: tuple[str, ...]
    available_signatures: tuple[str, ...]
    missing_signatures: tuple[str, ...]
    failures: tuple[str, ...]
    raw: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return not self.failures and not self.missing_signatures

    @property
    def calibration_id(self) -> str:
        content = (
            f"{self.model_version}:{self.provenance_hash}:"
            f"{self.source_sha256}:"
            f"{','.join(self.required_signatures)}"
        ).encode("utf-8")
        return f"sim-power-{hashlib.sha256(content).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "model_version": self.model_version,
            "provenance_hash": self.provenance_hash,
            "required_signatures": list(self.required_signatures),
            "available_signatures": list(self.available_signatures),
            "missing_signatures": list(self.missing_signatures),
            "failures": list(self.failures),
            "passed": self.passed,
            "calibration_id": self.calibration_id,
        }


def _finite_metric(
    validation: Mapping[str, Any],
    key: str,
    failures: list[str],
    *,
    minimum: float | None = 0.0,
    maximum: float | None = None,
) -> float | None:
    raw = validation.get(key)
    if raw is None:
        failures.append(f"validation_missing:{key}")
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        failures.append(f"validation_invalid:{key}")
        return None
    if not math.isfinite(value):
        failures.append(f"validation_nonfinite:{key}")
        return None
    if minimum is not None and value < minimum:
        failures.append(f"validation_range:{key}")
        return None
    if maximum is not None and value > maximum:
        failures.append(f"validation_range:{key}")
        return None
    return value


def _valid_area_coefficients(coefficients: Any) -> bool:
    if not isinstance(coefficients, (Mapping, list, tuple)):
        return False
    raw_values = (
        coefficients.values() if isinstance(coefficients, Mapping) else coefficients
    )
    try:
        values = tuple(float(value) for value in raw_values)
    except (TypeError, ValueError):
        return False
    return len(values) == 3 and all(
        math.isfinite(value) and value >= 0 for value in values
    ) and any(
        value > 0 for value in values
    )


def _is_sha256(value: object) -> bool:
    token = str(value)
    return len(token) == 64 and all(
        character in "0123456789abcdef" for character in token
    )


def _structural_features(
    mlen: int,
    blen: int,
    t_width: int,
    l_width: int,
    scale_width: int,
) -> dict[str, float]:
    if mlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError("structural area geometry is invalid")
    return {
        "pe_tl": float(mlen * blen * t_width * l_width),
        "pe_sum": float(mlen * blen * (t_width + l_width)),
        "pe_0": float(mlen * blen),
        "reduce": float(blen * (mlen - blen)),
        "scale": float(mlen * scale_width),
        "out": float(blen * blen),
        "fixed": float(mlen // blen),
        "const": 1.0,
    }


def _format_width(token: str) -> tuple[str, int]:
    value = token.strip().upper()
    if value.startswith("MXINT"):
        bits = int(value.removeprefix("MXINT").lstrip("_"))
        if bits not in {2, 4, 8}:
            raise ValueError(f"unsupported structural format {token!r}")
        return "mxint", bits
    if value.startswith("MXFP_E") and "M" in value:
        exp, mant = value.removeprefix("MXFP_E").split("M", 1)
        pair = int(exp), int(mant)
        if pair not in {(1, 2), (2, 1), (4, 3), (5, 2)}:
            raise ValueError(f"unsupported structural format {token!r}")
        return "mxfp", 1 + pair[0] + pair[1]
    raise ValueError(f"unsupported structural format {token!r}")


def _matrix_operands(signature: str) -> tuple[str, str, str]:
    try:
        operation, operands = signature.split(":", 1)
        left, right = operands.split("x", 1)
    except ValueError as exc:
        raise ValueError(f"invalid matrix signature {signature!r}") from exc
    if operation not in {"LINEAR", "QK", "PV"}:
        raise ValueError(f"invalid matrix operation {operation!r}")
    return operation, left, right


def _structural_matrix_mm2(
    raw: Mapping[str, Any],
    signature: str,
    *,
    mlen: int,
    blen: int,
    reference_corner: bool,
    scale_width: int = 8,
) -> float:
    _, left, right = _matrix_operands(signature)
    left_family, left_width = _format_width(left)
    right_family, right_width = _format_width(right)
    if left_family != right_family:
        raise ValueError("mixed-family structural area is unsupported")
    coefficients = raw["coefficients"][left_family]
    features = _structural_features(
        mlen,
        blen,
        left_width,
        right_width,
        scale_width,
    )
    area = sum(
        float(coefficients[name]) * features[name]
        for name in STRUCTURAL_FEATURES
    )
    if reference_corner:
        area *= float(raw["pdk_scale_reference"])
    if not math.isfinite(area) or area <= 0:
        raise ValueError("structural matrix area must be positive and finite")
    return area / 1e6


def _structural_failures(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, Mapping):
        return ("structural_area_schema",)
    failures: list[str] = []
    if raw.get("schema_version") != STRUCTURAL_AREA_SCHEMA:
        failures.append("structural_area_schema")
    if raw.get("model_version") != STRUCTURAL_AREA_MODEL_VERSION:
        failures.append("structural_area_model_version")
    sources = raw.get("source_sha256", {})
    if (
        not isinstance(sources, Mapping)
        or set(sources) != REQUIRED_STRUCTURAL_AREA_SOURCES
    ):
        failures.append("structural_area_sources")
    elif any(not name or not _is_sha256(value) for name, value in sources.items()):
        failures.append("structural_area_sources")
    coefficients = raw.get("coefficients", {})
    if not isinstance(coefficients, Mapping):
        failures.append("structural_area_coefficients")
    else:
        for family in ("mxint", "mxfp"):
            model = coefficients.get(family, {})
            if not isinstance(model, Mapping) or set(model) != set(
                STRUCTURAL_FEATURES
            ):
                failures.append(f"structural_area_coefficients:{family}")
                continue
            try:
                values = tuple(
                    float(model[name]) for name in STRUCTURAL_FEATURES
                )
            except (TypeError, ValueError):
                failures.append(f"structural_area_coefficients:{family}")
                continue
            if any(
                not math.isfinite(value) or value < 0 for value in values
            ) or not any(value > 0 for value in values):
                failures.append(f"structural_area_coefficients:{family}")
    try:
        pdk_scale = float(raw["pdk_scale_reference"])
        reference_anchor = float(raw["reference_anchor_um2"])
    except (KeyError, TypeError, ValueError):
        failures.append("structural_area_reference")
        pdk_scale, reference_anchor = 0.0, 0.0
    if (
        not math.isfinite(pdk_scale)
        or pdk_scale <= 0
        or not math.isfinite(reference_anchor)
        or reference_anchor <= 0
    ):
        failures.append("structural_area_reference")
    holdouts = raw.get("holdout_mape_pct", {})
    if not isinstance(holdouts, Mapping):
        failures.append("structural_area_holdout")
    else:
        for family in ("mxint", "mxfp"):
            try:
                value = float(holdouts[family])
            except (KeyError, TypeError, ValueError):
                failures.append(f"structural_area_holdout:{family}")
                continue
            if not math.isfinite(value) or value < 0 or value > 10.0:
                failures.append(f"structural_area_holdout:{family}")
    macros = raw.get("sram_macros", ())
    if (
        not isinstance(macros, list)
        or not macros
        or any(
            not isinstance(macro, Mapping)
            or not str(macro.get("macro", "")).strip()
            for macro in macros
        )
    ):
        failures.append("structural_area_sram_macros")
    else:
        for index, macro in enumerate(macros):
            try:
                depth = int(macro["depth"])
                width = int(macro["width"])
                area = float(macro["area_um2"])
            except (KeyError, TypeError, ValueError):
                failures.append(f"structural_area_sram_macro:{index}")
                continue
            if (
                depth <= 0
                or width <= 0
                or not math.isfinite(area)
                or area <= 0
            ):
                failures.append(f"structural_area_sram_macro:{index}")
    if not failures:
        try:
            anchor = (
                _structural_matrix_mm2(
                    raw,
                    "LINEAR:MXINT4xMXINT4",
                    mlen=1024,
                    blen=4,
                    reference_corner=True,
                )
                * 1e6
            )
            if abs(anchor - reference_anchor) / reference_anchor * 100.0 > 1.0:
                failures.append("structural_area_anchor")
            shapes = ((16, 4), (32, 4), (64, 8), (256, 8), (1024, 4))
            areas = [
                _structural_matrix_mm2(
                    raw,
                    "LINEAR:MXINT4xMXINT4",
                    mlen=mlen,
                    blen=blen,
                    reference_corner=False,
                )
                for mlen, blen in shapes
            ]
            if any(right <= left for left, right in zip(areas, areas[1:])):
                failures.append("structural_area_monotonicity")
            precision = [
                _structural_matrix_mm2(
                    raw,
                    signature,
                    mlen=1024,
                    blen=4,
                    reference_corner=False,
                )
                for signature in (
                    "LINEAR:MXINT4xMXINT2",
                    "LINEAR:MXINT4xMXINT4",
                    "LINEAR:MXINT8xMXINT8",
                )
            ]
            if not precision[0] < precision[1] < precision[2]:
                failures.append("structural_area_precision_order")
        except (KeyError, TypeError, ValueError):
            failures.append("structural_area_evaluation")
    return tuple(sorted(set(failures)))


def _structural_area_id(raw: Mapping[str, Any]) -> str:
    payload = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"structural-area-{hashlib.sha256(payload).hexdigest()}"


def _fp_width(vector_fp: str) -> int:
    token = vector_fp.strip().upper().removeprefix("FP_")
    if token == "BF16":
        return 16
    if not token.startswith("E") or "M" not in token:
        raise ValueError(f"unsupported vector format {vector_fp!r}")
    exp, mant = token[1:].split("M", 1)
    return 1 + int(exp) + int(mant)


def _tile_area_um2(
    depth: int,
    width: int,
    ports: int,
    macros: Iterable[Mapping[str, Any]],
) -> float:
    values = tuple(macros)
    if depth <= 0 or width <= 0 or ports <= 0 or not values:
        raise ValueError("invalid SRAM tiling inputs")
    return min(
        math.ceil(depth / int(macro["depth"]))
        * math.ceil(width / int(macro["width"]))
        * ports
        * float(macro["area_um2"])
        for macro in values
    )


def _structural_sram_mm2(
    raw: Mapping[str, Any],
    signatures: Iterable[str],
    *,
    vector_fp: str,
    area_config: Mapping[str, Any],
    mlen: int,
    blen: int,
) -> float:
    required = (
        "MLEN",
        "BLEN",
        "VLEN",
        "MATRIX_SRAM_DEPTH",
        "VECTOR_SRAM_DEPTH",
        "INT_SRAM_DEPTH",
        "FP_SRAM_DEPTH",
        "INT_DATA_WIDTH",
        "MX_SCALE_WIDTH",
        "BLOCK_DIM",
    )
    if any(name not in area_config for name in required):
        raise ValueError("simulator area config is incomplete")
    config = {name: int(area_config[name]) for name in required}
    if any(value <= 0 for value in config.values()):
        raise ValueError("simulator area config values must be positive")
    if config["MLEN"] != mlen or config["BLEN"] != blen:
        raise ValueError("simulator area config geometry mismatch")
    if config["VLEN"] != config["MLEN"]:
        raise ValueError("simulator area config requires VLEN == MLEN")
    if config["BLOCK_DIM"] != 8:
        raise ValueError("simulator area config requires native MX block size 8")
    roles: dict[str, str] = {}
    activations: set[str] = set()
    for signature in signatures:
        operation, left, right = _matrix_operands(signature)
        activations.add(right)
        roles[{"LINEAR": "weight", "QK": "key", "PV": "value"}[operation]] = left
    if set(roles) != {"weight", "key", "value"} or len(activations) != 1:
        raise ValueError("simulator area signatures are incomplete")
    tokens = (
        roles["weight"],
        next(iter(activations)),
        roles["key"],
        roles["value"],
    )
    formats = [_format_width(token) for token in tokens]
    if len({family for family, _ in formats}) != 1:
        raise ValueError("mixed-family structural SRAM area is unsupported")
    t_width = max(formats[index][1] for index in (0, 2, 3))
    act_width = formats[1][1]
    kv_width = max(formats[2][1], formats[3][1])
    scale = config["MX_SCALE_WIDTH"]
    matrix_width = mlen * (t_width + scale)
    vector_width = (
        config["VLEN"] * (_fp_width(vector_fp) + act_width + kv_width)
        + 2 * scale * max(1, config["VLEN"] // config["BLOCK_DIM"])
    )
    macros = raw["sram_macros"]
    return (
        _tile_area_um2(
            config["MATRIX_SRAM_DEPTH"],
            matrix_width,
            2,
            macros,
        )
        + _tile_area_um2(
            config["VECTOR_SRAM_DEPTH"],
            vector_width,
            2,
            macros,
        )
        + _tile_area_um2(
            config["INT_SRAM_DEPTH"],
            config["INT_DATA_WIDTH"],
            1,
            macros,
        )
        + _tile_area_um2(
            config["FP_SRAM_DEPTH"],
            _fp_width(vector_fp),
            1,
            macros,
        )
    ) / 1e6


def load_simulator_power_artifact(
    path: str | Path,
    *,
    required_signatures: Iterable[str] | None = None,
) -> SimulatorPowerStatus:
    """Load an artifact without making unvalidated coefficients rankable."""

    source = Path(path).resolve()
    payload = source.read_bytes()
    raw = json.loads(payload)
    if not isinstance(raw, Mapping):
        raise TypeError("power artifact root must be an object")
    failures: list[str] = []
    model_version = str(raw.get("model_version", ""))
    if model_version not in SUPPORTED_POWER_MODEL_VERSIONS:
        failures.append("model_version")
    provenance_hash = str(raw.get("provenance_hash", ""))
    if not _is_sha256(provenance_hash):
        failures.append("provenance_hash")
    if not _is_sha256(raw.get("activity_provenance_hash", "")):
        failures.append("activity_provenance_hash")
    if not _is_sha256(raw.get("artifact_catalog_sha256", "")):
        failures.append("artifact_catalog_sha256")
    manifest_hash = str(raw.get("calibration_manifest_hash", ""))
    if manifest_hash != EXPECTED_CALIBRATION_MANIFEST_HASH:
        failures.append("calibration_manifest_hash")
    if not str(raw.get("hbm_energy_source", "")).strip():
        failures.append("hbm_energy_source")
    fit_summary = raw.get("fit_summary", {})
    if not isinstance(fit_summary, Mapping):
        failures.append("fit_summary_schema")
    else:
        for name, expected in EXPECTED_FIT_SUMMARY.items():
            try:
                observed = int(fit_summary.get(name))
            except (TypeError, ValueError):
                failures.append(f"fit_summary_missing:{name}")
                continue
            if observed != expected:
                failures.append(f"fit_summary_mismatch:{name}")
    synthesis_context = raw.get("synthesis_context", {})
    if not isinstance(synthesis_context, Mapping):
        failures.append("synthesis_context_schema")
    else:
        for name in (
            "dc_tool_version",
            "library_id",
            "process_corner",
            "mx_block_size",
            "hardware_fp_binding",
            "activity_generator",
        ):
            if not str(synthesis_context.get(name, "")).strip():
                failures.append(f"synthesis_context_missing:{name}")
        if str(synthesis_context.get("mx_block_size", "")) != "8":
            failures.append("synthesis_context_gate:mx_block_size")
        if (
            str(synthesis_context.get("hardware_fp_binding", ""))
            != EXPECTED_HARDWARE_FP_BINDING
        ):
            failures.append(
                "synthesis_context_gate:hardware_fp_binding"
            )
    models_raw = raw.get("event_energy_models", {})
    if not isinstance(models_raw, Mapping):
        raise TypeError("event_energy_models must be an object")
    available: list[str] = []
    for signature, coefficients in models_raw.items():
        try:
            values = tuple(float(value) for value in coefficients)
        except (TypeError, ValueError):
            failures.append(f"coefficient_schema:{signature}")
            continue
        if len(values) != 3 or any(
            not math.isfinite(value) or value < 0 for value in values
        ) or not any(value > 0 for value in values):
            failures.append(f"coefficient_schema:{signature}")
            continue
        available.append(str(signature))
    required = tuple(
        sorted(
            {
                str(signature)
                for signature in (
                    required_signatures
                    if required_signatures is not None
                    else required_hardware_power_signatures()
                )
            }
        )
    )
    available_set = frozenset(available)
    missing = tuple(
        signature for signature in required if signature not in available_set
    )

    validation = raw.get("validation", {})
    if not isinstance(validation, Mapping):
        raise TypeError("validation must be an object")
    if validation.get("passed") is not True:
        failures.append("validation_passed")
    if tuple(validation.get("missing_fields", ())):
        failures.append("validation_missing_fields")
    limits = {
        "area_median_pct": 10.0,
        "area_max_pct": 15.0,
        "dynamic_median_pct": 15.0,
        "dynamic_max_pct": 25.0,
        "leakage_median_pct": 15.0,
        "leakage_max_pct": 25.0,
        "cycle_error_pct": 5.0,
        "latency_mape_pct": 10.0,
    }
    for name, limit in limits.items():
        value = _finite_metric(validation, name, failures)
        if value is not None and value > limit:
            failures.append(f"validation_gate:{name}")
    rank = _finite_metric(
        validation,
        "rank_correlation",
        failures,
        minimum=-1.0,
        maximum=1.0,
    )
    if rank is not None and rank < 0.90:
        failures.append("validation_gate:rank_correlation")

    scalar_coefficients: dict[str, float] = {}
    for name in (
        "hbm_energy_j_per_byte",
        "fixed_area_mm2",
    ):
        try:
            value = float(raw.get(name))
        except (TypeError, ValueError):
            failures.append(f"coefficient_missing:{name}")
            continue
        if not math.isfinite(value) or value < 0:
            failures.append(f"coefficient_invalid:{name}")
        else:
            scalar_coefficients[name] = value
    for name in (
        "hbm_energy_j_per_byte",
        "fixed_area_mm2",
    ):
        if scalar_coefficients.get(name, 0.0) <= 0:
            failures.append(f"coefficient_missing:{name}")
    if not _valid_area_coefficients(raw.get("leakage_power_model")):
        failures.append("leakage_power_model")
    if not _valid_area_coefficients(raw.get("selector_area_model")):
        failures.append("selector_area_model")
    vector_area = raw.get("vector_area_models", {})
    if not isinstance(vector_area, Mapping):
        failures.append("vector_area_models_schema")
    else:
        for vector_format in VECTOR_HARDWARE_FORMATS:
            if not _valid_area_coefficients(vector_area.get(vector_format)):
                failures.append(f"vector_area_model:{vector_format}")
    structural_area = raw.get("structural_area_model", {})
    failures.extend(_structural_failures(structural_area))
    required_matrix_signatures = tuple(
        signature
        for signature in required
        if signature.startswith(("LINEAR:", "QK:", "PV:"))
    )
    if isinstance(structural_area, Mapping) and not _structural_failures(
        structural_area
    ):
        for signature in required_matrix_signatures:
            try:
                _structural_matrix_mm2(
                    structural_area,
                    signature,
                    mlen=16,
                    blen=4,
                    reference_corner=True,
                )
            except (KeyError, TypeError, ValueError):
                failures.append(f"structural_area_signature:{signature}")

    return SimulatorPowerStatus(
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        model_version=model_version,
        provenance_hash=provenance_hash,
        required_signatures=required,
        available_signatures=tuple(sorted(available_set)),
        missing_signatures=missing,
        failures=tuple(sorted(set(failures))),
        raw=dict(raw),
    )


def _estimate_matches_status(
    status: SimulatorPowerStatus,
    estimate: Mapping[str, Any],
) -> bool:
    if estimate.get("calibration_source_sha256") != status.source_sha256:
        return False
    if estimate.get("calibration_provenance_hash") != status.provenance_hash:
        return False
    if (
        estimate.get("calibration_activity_provenance_hash")
        != status.raw.get("activity_provenance_hash")
    ):
        return False
    structural = status.raw.get("structural_area_model", {})
    if not isinstance(structural, Mapping):
        return False
    if estimate.get("structural_area_id") != _structural_area_id(structural):
        return False
    return True


def _estimate_within_calibration_domain(
    status: SimulatorPowerStatus,
    estimate: Mapping[str, Any],
) -> bool:
    if estimate.get("calibration_domain") != EXPECTED_POWER_INTERPOLATION_DOMAIN:
        return False
    if estimate.get("extrapolated") is not False:
        return False
    if tuple(estimate.get("extrapolation_reasons", ())):
        return False
    try:
        mlen = int(estimate["MLEN"])
        blen = int(estimate["BLEN"])
    except (KeyError, TypeError, ValueError):
        return False
    domain = EXPECTED_POWER_INTERPOLATION_DOMAIN
    if not (
        domain["mlen_min"] <= mlen <= domain["mlen_max"]
        and domain["blen_min"] <= blen <= domain["blen_max"]
    ):
        return False
    context = status.raw.get("synthesis_context", {})
    if not isinstance(context, Mapping):
        return False
    return (
        estimate.get("vector_fp")
        == context.get("hardware_fp_binding")
        == EXPECTED_HARDWARE_FP_BINDING
    )


def calibrated_energy_from_simulator(
    status: SimulatorPowerStatus,
    estimate: Mapping[str, Any],
    *,
    duration_s: float,
) -> CalibratedEnergy | None:
    """Convert only a rankable simulator estimate into calibrated energy."""

    if not status.passed:
        return None
    if not _estimate_matches_status(status, estimate):
        return None
    if estimate.get("calibrated") is not True or estimate.get("rankable") is not True:
        return None
    if not _estimate_within_calibration_domain(status, estimate):
        return None
    if tuple(estimate.get("missing_signatures", ())):
        return None
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    event_rows = estimate.get("events")
    if not isinstance(event_rows, list) or not event_rows:
        raise ValueError("simulator energy needs explicit event counts")
    expected_compute = 0.0
    expected_vector = 0.0
    expected_selector = 0.0
    geometries: set[tuple[int, int]] = set()
    operation_signatures = {
        operation: set() for operation in ("LINEAR", "QK", "PV")
    }
    vector_fp = str(estimate["vector_fp"])
    vector_events = 0
    selector_events = 0
    models = status.raw["event_energy_models"]
    for row in event_rows:
        if not isinstance(row, Mapping):
            raise ValueError("simulator event rows must be objects")
        signature = str(row["signature"])
        count = int(row["count"])
        mlen = int(row["MLEN"])
        blen = int(row["BLEN"])
        if count < 0 or mlen <= 0 or blen <= 0 or mlen % blen:
            raise ValueError("simulator event row is invalid")
        geometries.add((mlen, blen))
        try:
            coefficients = tuple(float(value) for value in models[signature])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"uncalibrated simulator event {signature!r}") from exc
        event_energy = (
            coefficients[0]
            + coefficients[1] * mlen * blen
            + coefficients[2] * (mlen + blen)
        ) * count
        if signature.startswith(("LINEAR:", "QK:", "PV:")):
            expected_compute += event_energy
            operation_signatures[signature.split(":", 1)[0]].add(signature)
        elif signature.startswith("VECTOR:"):
            expected_vector += event_energy
            if signature == f"VECTOR:{vector_fp}":
                vector_events += count
        elif signature == SELECTOR_SIGNATURE:
            expected_selector += event_energy
            selector_events += count
        else:
            raise ValueError(f"unsupported simulator event {signature!r}")
    if len(geometries) != 1:
        raise ValueError("simulator events must use one geometry")
    if not operation_signatures["LINEAR"]:
        raise ValueError("simulator events need at least one LINEAR signature")
    if any(
        len(operation_signatures[operation]) != 1
        for operation in ("QK", "PV")
    ):
        raise ValueError("simulator events need exactly one QK and PV signature")
    if vector_events <= 0:
        raise ValueError("simulator events are missing the selected vector format")
    selector_enabled = estimate.get("selector_enabled")
    if not isinstance(selector_enabled, bool):
        raise ValueError("selector_enabled must be boolean")
    if selector_enabled != (selector_events > 0):
        raise ValueError("selector state and selector events are inconsistent")
    expected_dynamic = expected_compute + expected_vector + expected_selector
    dynamic = float(estimate["dynamic_energy_j"])
    compute = float(estimate["compute_dynamic_energy_j"])
    vector = float(estimate["vector_dynamic_energy_j"])
    selector = float(estimate["selector_dynamic_energy_j"])
    leakage = float(estimate["leakage_energy_j"])
    leakage_power = float(estimate["leakage_power_w"])
    hbm = float(estimate["hbm_energy_j"])
    total = float(estimate["total_energy_j"])
    average_power = float(estimate["average_power_w"])
    if any(
        not math.isfinite(value) or value < 0
        for value in (
            dynamic,
            compute,
            vector,
            selector,
            leakage,
            leakage_power,
            hbm,
            total,
            average_power,
        )
    ):
        raise ValueError("simulator energy values must be finite and non-negative")
    mlen = int(estimate["MLEN"])
    blen = int(estimate["BLEN"])
    if mlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError("simulator leakage geometry is invalid")
    if geometries != {(mlen, blen)}:
        raise ValueError("simulator estimate and event geometries differ")
    for name, observed, expected in (
        ("compute dynamic", compute, expected_compute),
        ("vector dynamic", vector, expected_vector),
        ("selector dynamic", selector, expected_selector),
        ("total dynamic", dynamic, expected_dynamic),
    ):
        if abs(observed - expected) > max(1e-15, abs(expected) * 1e-9):
            raise ValueError(f"simulator {name} energy is inconsistent with calibration")
    coefficients = tuple(
        float(value) for value in status.raw["leakage_power_model"]
    )
    expected_leakage_power = (
        coefficients[0]
        + coefficients[1] * mlen * blen
        + coefficients[2] * (mlen + blen)
    )
    if abs(leakage_power - expected_leakage_power) > max(
        1e-12,
        abs(expected_leakage_power) * 1e-9,
    ):
        raise ValueError("simulator leakage power is inconsistent with calibration")
    if abs(leakage - leakage_power * duration_s) > max(
        1e-15,
        abs(leakage) * 1e-9,
    ):
        raise ValueError("simulator leakage energy is inconsistent with elapsed time")
    hbm_bytes = float(estimate["hbm_bytes"])
    if not math.isfinite(hbm_bytes) or hbm_bytes < 0:
        raise ValueError("simulator HBM bytes must be finite and non-negative")
    expected_hbm = hbm_bytes * float(status.raw["hbm_energy_j_per_byte"])
    if abs(hbm - expected_hbm) > max(1e-15, abs(expected_hbm) * 1e-9):
        raise ValueError("simulator HBM energy is inconsistent with calibration")
    expected_total = dynamic + leakage + hbm
    tolerance = max(1e-15, abs(total) * 1e-9)
    if abs(total - expected_total) > tolerance:
        raise ValueError("simulator energy components do not sum to total")
    if abs(average_power - total / duration_s) > max(
        1e-12,
        abs(average_power) * 1e-9,
    ):
        raise ValueError("simulator average power is inconsistent with elapsed time")
    return CalibratedEnergy(
        calibration_id=status.calibration_id,
        compute_j=compute,
        vector_j=vector,
        sram_j=0.0,
        hbm_j=hbm,
        leakage_j=leakage,
        duration_s=duration_s,
        unattributed_dynamic_j=selector,
    )


def calibrated_area_from_simulator(
    status: SimulatorPowerStatus,
    estimate: Mapping[str, Any],
) -> float | None:
    """Return area only when its artifact and estimate are fully rankable."""

    if not status.passed:
        return None
    if not _estimate_matches_status(status, estimate):
        return None
    if estimate.get("calibrated") is not True or estimate.get("rankable") is not True:
        return None
    if not _estimate_within_calibration_domain(status, estimate):
        return None
    if tuple(estimate.get("missing_signatures", ())):
        return None
    area = float(estimate["area_mm2"])
    if not math.isfinite(area) or area <= 0:
        raise ValueError("simulator area must be positive and finite")
    mlen = int(estimate["MLEN"])
    blen = int(estimate["BLEN"])
    if mlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError("simulator area geometry is invalid")
    signatures = tuple(sorted({str(value) for value in estimate["array_signatures"]}))
    if not signatures:
        raise ValueError("simulator area needs operation signatures")
    structural = status.raw["structural_area_model"]
    scale_width = int(estimate["area_config"]["MX_SCALE_WIDTH"])
    matrix_areas = [
        _structural_matrix_mm2(
            structural,
            signature,
            mlen=mlen,
            blen=blen,
            reference_corner=True,
            scale_width=scale_width,
        )
        for signature in signatures
    ]
    expected_matrix = max(matrix_areas)
    observed_matrix = float(estimate["matrix_area_mm2"])
    if abs(observed_matrix - expected_matrix) > max(
        1e-12,
        abs(expected_matrix) * 1e-9,
    ):
        raise ValueError("simulator matrix area is inconsistent with calibration")
    vector_fp = str(estimate["vector_fp"])
    pdk_scale = float(structural["pdk_scale_reference"])
    vector_coefficients = tuple(
        float(value)
        for value in status.raw["vector_area_models"][vector_fp]
    )
    expected_vector = (
        vector_coefficients[0]
        + vector_coefficients[1] * mlen * blen
        + vector_coefficients[2] * (mlen + blen)
    ) * pdk_scale
    selector_enabled = estimate["selector_enabled"]
    if not isinstance(selector_enabled, bool):
        raise ValueError("selector_enabled must be boolean")
    selector_coefficients = tuple(
        float(value) for value in status.raw["selector_area_model"]
    )
    expected_selector = (
        (
            selector_coefficients[0]
            + selector_coefficients[1] * mlen * blen
            + selector_coefficients[2] * (mlen + blen)
        )
        * pdk_scale
        if selector_enabled
        else 0.0
    )
    expected_fixed = float(status.raw["fixed_area_mm2"]) * pdk_scale
    expected_sram = _structural_sram_mm2(
        structural,
        signatures,
        vector_fp=vector_fp,
        area_config=estimate["area_config"],
        mlen=mlen,
        blen=blen,
    )
    for name, observed, expected in (
        ("SRAM", float(estimate["sram_area_mm2"]), expected_sram),
        ("fixed", float(estimate["fixed_area_mm2"]), expected_fixed),
        ("vector", float(estimate["vector_area_mm2"]), expected_vector),
        ("selector", float(estimate["selector_area_mm2"]), expected_selector),
    ):
        if not math.isfinite(observed) or observed < 0:
            raise ValueError(f"simulator {name} area is invalid")
        if abs(observed - expected) > max(1e-12, abs(expected) * 1e-9):
            raise ValueError(
                f"simulator {name} area is inconsistent with calibration"
            )
    expected_area = (
        expected_fixed
        + expected_matrix
        + expected_vector
        + expected_sram
        + expected_selector
    )
    if abs(area - expected_area) > max(1e-12, abs(expected_area) * 1e-9):
        raise ValueError("simulator area is inconsistent with calibration")
    return expected_area


__all__ = [
    "SUPPORTED_POWER_MODEL_VERSIONS",
    "EXPECTED_CALIBRATION_MANIFEST_HASH",
    "EXPECTED_FIT_SUMMARY",
    "SELECTOR_SIGNATURE",
    "SimulatorPowerStatus",
    "calibrated_area_from_simulator",
    "calibrated_energy_from_simulator",
    "load_simulator_power_artifact",
    "required_hardware_power_signatures",
    "required_profile_power_signatures",
]
