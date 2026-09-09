"""In-process bridge to the PLENA decode-chip analytic model.

The analytic model (`analytic_models/performance/disagg_decode.py`) uses
script-style imports and ships no installable package, so it can't be
pip-imported. This module is the only place that reaches into it: it adds the
performance directory to `sys.path` once, imports the helpers, and re-exports a
small typed surface (`Precision`, `DecodeMetrics`, `DecodeSimulator`). Everything
else in the DSE imports from here, so if the simulator becomes a real package
only this file changes.

Set ``PLENA_SIMULATOR_PATH`` to the simulator checkout; when it is unset the
sibling ``PLENA_Simulator`` checkout is used and that fallback is recorded in
evaluator provenance.  Resolution itself lives in
``decode_dse.hardware.power_bridge`` so every reader of the simulator tree
resolves the same root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

_BANDWIDTH_TRAFFIC_CLASSES = ("weights_kv", "activations", "writeback")
STEP_COMPOSITION = "max_compute_memory"
COMPILER_TRACE_POINT_SCHEMA = "plena-compiler-trace-point-v1"
FULL_MODEL_DECODE_SCOPE = "full_model_decode_step_independent_request_batch"
_CONTENT_ADDRESSED_ID = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[0-9a-f]{64}$"
)


def _require_content_addressed_id(
    name: str,
    value: str,
    prefix: str,
) -> None:
    if (
        not isinstance(value, str)
        or not _CONTENT_ADDRESSED_ID.fullmatch(value)
        or not value.startswith(prefix)
    ):
        raise ValueError(f"{name} must be a content-addressed identity")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_file(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
    elif hasattr(value, "dict"):
        result = value.dict()
    else:
        raise TypeError("simulator configuration is not serializable")
    if not isinstance(result, dict):
        raise TypeError("simulator configuration must serialize to an object")
    return result


def _sim_root() -> Path:
    # Imported lazily: the resolver lives beside the analytic-power bridge,
    # which imports the hardware design space, and this module is imported
    # from inside it.
    from decode_dse.hardware.power_bridge import resolve_simulator_root

    root = resolve_simulator_root().root
    perf = root / "analytic_models" / "performance"
    if not perf.exists():
        raise FileNotFoundError(
            f"simulator checkout at {root} is incomplete (missing {perf}). "
            f"Clone it or set PLENA_SIMULATOR_PATH to the checkout root."
        )
    return root


@lru_cache(maxsize=1)
def _disagg():
    """Import and return the disagg_decode module, injecting sys.path once."""
    root = _sim_root()
    perf = root / "analytic_models" / "performance"
    if str(root) not in sys.path:
        # Compiler-trace timing imports the simulator packages by qualified
        # name, while the historical analytic model uses script-style imports.
        sys.path.insert(0, str(root))
    if str(perf) not in sys.path:
        # Importing the module also puts memory/utilisation/roofline on sys.path.
        sys.path.insert(0, str(perf))
    import disagg_decode  # noqa: E402  (import deferred until path is set)

    return disagg_decode


def default_model_lib() -> Path:
    return _sim_root() / "compiler" / "doc" / "Model_Lib"


def resolve_model_json(model: str, model_lib: str | os.PathLike | None = None) -> str:
    """Accept either a full path to a model JSON or a bare name in the Model_Lib."""
    p = Path(model)
    if p.suffix == ".json" and p.exists():
        return str(p.resolve())
    lib = Path(model_lib) if model_lib else default_model_lib()
    candidate = lib / (model if model.endswith(".json") else f"{model}.json")
    if not candidate.exists():
        raise FileNotFoundError(f"model config {candidate} not found (model={model!r}).")
    return str(candidate.resolve())


# ---------------------------------------------------------------------------
# Typed surface
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Precision:
    """A decode precision point in the analytic model's native shape.

    `spec` is the dict `disagg_decode.evaluate` consumes; the scalar fields are
    kept alongside it for logging and for the software-DSE CSV bridge.
    """

    spec: dict[str, Any]
    w_fmt: str
    kv_fmt: str
    attn_w: Any
    ffn_w: Any
    kv: Any
    block: int
    m_bits: int
    key_fmt: str | None = None
    value_fmt: str | None = None
    key: Any = None
    value: Any = None

    @property
    def attn_eff_bits(self) -> float:
        return self.spec["attn_bits"]

    @property
    def ffn_eff_bits(self) -> float:
        return self.spec["ffn_bits"]

    @property
    def kv_eff_bits(self) -> float:
        return self.spec["kv_bits"]

    @property
    def key_eff_bits(self) -> float:
        return self.spec.get("key_bits", self.spec["kv_bits"])

    @property
    def value_eff_bits(self) -> float:
        return self.spec.get("value_bits", self.spec["kv_bits"])

    @property
    def stream_bits(self) -> int:
        return _disagg().stream_bits(self.spec)

    @property
    def tag(self) -> str:
        a = self.spec["attn_label"]
        f = self.spec["ffn_label"]
        k = self.spec.get("key_label", self.spec["kv_label"])
        v = self.spec.get("value_label", self.spec["kv_label"])
        m = f"_M{self.m_bits}" if self.m_bits else ""
        return f"attn-{a}__ffn-{f}__key-{k}__value-{v}{m}"


@dataclass(frozen=True)
class CompilerTracePointDescriptor:
    """Immutable lowering and timing inputs for one exact DSE point."""

    canonical_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise TypeError("compiler trace point must use canonical JSON")
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise TypeError("compiler trace point must be an object")
        if value.get("schema_version") != COMPILER_TRACE_POINT_SCHEMA:
            raise ValueError("unsupported compiler trace point schema")
        if value.get("artifact_scope") != FULL_MODEL_DECODE_SCOPE:
            raise ValueError("compiler trace point lacks full-model decode scope")
        if self.canonical_json != _canonical_json(value):
            raise ValueError("compiler trace point JSON is not canonical")

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "CompilerTracePointDescriptor":
        return cls(_canonical_json(value))

    @property
    def descriptor_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)


class CompilerTraceRequestBinder(Protocol):
    """Bind a fresh context-request factory to one immutable DSE point."""

    def bind(
        self,
        point: CompilerTracePointDescriptor,
    ) -> Callable[[int], Any]:
        ...


@dataclass(frozen=True)
class DecodeMetrics:
    """The subset of `disagg_decode.evaluate` output the DSE optimises on.

    ``n_chips`` is the chip count the point was evaluated at. With ``n_chips=0``
    (auto) the model adds chips until the footprint fits, so ``fits_in_hbm`` is
    always true — a search that wants the HBM-fit constraint to bind must pin
    ``n_chips`` (the co-design pins 1).
    """

    tps: float
    tpot: float
    total_time: float
    first_step: float
    frac_mem_bound: float
    frac_classical_mem_bound: float
    frac_architecture_issue_mem_bound: float
    frac_algorithmic_mem_bound: float
    frac_serialization_bound: float
    fits_in_hbm: bool
    hbm_required: float
    area_mm2: float
    # Legacy alias for bytes per batch decode step.
    avg_bytes_per_token: float
    # A batch step emits one token for each active sequence.
    avg_hbm_bytes_per_batch_step: float
    avg_hbm_bytes_per_generated_token: float
    hbm_traffic_per_batch_step: tuple[tuple[str, float], ...]
    hbm_traffic_per_generated_token: tuple[tuple[str, float], ...]
    mem_bound: bool
    bottleneck: str
    classical_roofline_bottleneck: str
    architecture_issue_bottleneck: str
    n_chips: int
    # Largest batch whose weights+KV fit the evaluated HBM capacity. KV precision
    # moves this smoothly, so it stays a throughput signal even when a compute
    # ceiling flattens per-batch TPS.
    max_batch: int = 0
    max_resident_batch: int = 0
    max_synchronous_batch: int = 0
    max_runtime_batch: int = 0
    fits_onchip_sram: bool = False
    fits_runtime: bool = False
    hbm_capacity: int = 0
    runtime_hbm_reserve_bytes: int = 0
    weight_element_plane_bytes: int = 0
    weight_scale_plane_bytes: int = 0
    weight_bf16_bytes: int = 0
    kv_element_plane_bytes: int = 0
    kv_scale_plane_bytes: int = 0
    vector_sram_capacity_bytes: int = 0
    vector_sram_required_bytes: int = 0
    matrix_sram_capacity_bytes: int = 0
    matrix_sram_required_bytes: int = 0
    timing_mode: str = "rtl_serialized"
    timing_calibrated: bool = False
    timing_reason: str = "missing_timing_evidence"
    timing_evidence_id: str | None = None
    timing_evidence_tier: str | None = None
    packed_q1_timing_validated: bool = False
    packed_q1_timing_reason: str = "missing_packed_q1_timing_contract"
    packed_q1_timing_contract_id: str | None = None
    bandwidth_calibration_id: str | None = None
    kv_layout: str = "dense_selector"
    kv_layout_id: str = ""
    output_head_location: str = "decode_bf16_unmodeled"
    avg_peak_compute_seconds: float | None = None
    avg_ideal_compute_seconds: float | None = None
    avg_realized_compute_seconds: float | None = None
    avg_memory_seconds: float | None = None
    step_composition: str = STEP_COMPOSITION
    execution_mode: str = "legacy_aggregate_bandwidth"
    compiler_trace_timing: dict[str, Any] | None = None
    moe_workload: dict[str, Any] | None = None
    local_output_head: dict[str, Any] | None = None
    body_physical_layout: dict[str, Any] | None = None
    slowest_rank_hbm_required_bytes: int | None = None
    per_chip_hbm_capacity_bytes: int | None = None


@dataclass(frozen=True)
class HBMOperatingPointStatus:
    """Calibration support for one generation, rate, and interface-unit count."""

    generation: str
    interface_units: int
    pin_rate_gbps: float
    technology_schema: str
    source_url: str
    source_label: str
    emulator_generation: str | None
    emulator_pin_rate_gbps: float | None
    hbm_width_bits: int
    hbm_capacity_bytes: int
    calibration_id: str | None
    calibrated_channel_min: int | None
    calibrated_channel_max: int | None
    rankable: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.generation:
            raise ValueError("HBM generation must be non-empty")
        if self.interface_units <= 0:
            raise ValueError("HBM interface-unit count must be positive")
        if self.pin_rate_gbps <= 0:
            raise ValueError("HBM pin rate must be positive")
        if (
            not self.technology_schema
            or not self.source_url
            or not self.source_label
        ):
            raise ValueError("HBM technology provenance must be non-empty")
        if (self.emulator_generation is None) != (
            self.emulator_pin_rate_gbps is None
        ):
            raise ValueError("HBM emulator identity and rate must be paired")
        if (
            self.emulator_pin_rate_gbps is not None
            and self.emulator_pin_rate_gbps <= 0
        ):
            raise ValueError("HBM emulator rate must be positive")
        if self.hbm_width_bits <= 0 or self.hbm_width_bits % 8:
            raise ValueError("HBM width must be a positive whole-byte value")
        if self.hbm_capacity_bytes <= 0:
            raise ValueError("HBM capacity must be positive")
        bounds = (
            self.calibrated_channel_min,
            self.calibrated_channel_max,
        )
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("HBM calibration bounds must be paired")
        if (
            bounds[0] is not None
            and (bounds[0] <= 0 or bounds[1] < bounds[0])
        ):
            raise ValueError("HBM calibration bounds are invalid")
        if self.rankable != bool(self.calibration_id):
            raise ValueError("rankable HBM points require a calibration identity")
        if self.calibration_id is not None:
            _require_content_addressed_id(
                "HBM calibration",
                self.calibration_id,
                "bandwidth-operating-point-",
            )
        if not self.reason:
            raise ValueError("HBM operating-point reason must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "interface_unit_bits": 64,
            "interface_units": self.interface_units,
            "pin_rate_gbps": self.pin_rate_gbps,
            "technology_schema": self.technology_schema,
            "source_url": self.source_url,
            "source_label": self.source_label,
            "emulator_generation": self.emulator_generation,
            "emulator_pin_rate_gbps": self.emulator_pin_rate_gbps,
            "hbm_width_bits": self.hbm_width_bits,
            "hbm_capacity_bytes": self.hbm_capacity_bytes,
            "calibration_id": self.calibration_id,
            "calibrated_channel_min": self.calibrated_channel_min,
            "calibrated_channel_max": self.calibrated_channel_max,
            "rankable": self.rankable,
            "reason": self.reason,
        }


class DecodeSimulator:
    """Evaluator for one model: loads dims / base hardware / base memory once,
    then evaluates (precision, hardware, batch) points cheaply.

    All config parsing happens in ``__init__``, so an Optuna loop calling
    :meth:`evaluate` thousands of times pays only the per-point arithmetic.
    """

    def __init__(
        self,
        model: str,
        *,
        model_lib: str | os.PathLike | None = None,
        settings_toml: str | os.PathLike | None = None,
        isa_path: str | os.PathLike | None = None,
        timing_mode: str = "rtl_serialized",
        timing_evidence: str | os.PathLike | None = None,
        packed_q1_timing_contract: str | os.PathLike | None = None,
    ):
        dd = _disagg()
        root = _sim_root()
        self._dd = dd
        self.model = model
        self.model_json = resolve_model_json(model, model_lib)
        self.settings_toml = str(
            Path(settings_toml) if settings_toml else root / "plena_settings.toml"
        )
        self.isa_path = str(
            Path(isa_path)
            if isa_path
            else root / "analytic_models" / "performance" / "customISA_lib.json"
        )
        self.dims = dd.load_model_dims(self.model_json)
        self.base_hw = dd.load_hardware_config_from_toml(self.settings_toml)
        self.base_mem = dd.load_memory_config_from_toml(self.settings_toml)
        self.timing_mode = timing_mode
        self.timing_evidence = (
            dd.TimingEvidence.load(timing_evidence)
            if timing_evidence is not None
            else None
        )
        self.packed_q1_timing_contract = (
            dd.PackedQ1TimingContract.load(packed_q1_timing_contract)
            if packed_q1_timing_contract is not None
            else None
        )
        self.execution_mode = dd.LEGACY_AGGREGATE_BANDWIDTH
        self.trace_timing_provider = None
        self.trace_request_binder = None

    # -- precision construction ---------------------------------------------
    #
    # Two compute widths:
    #   N — the HBM stream widths (attnW / ffnW / KV). Set memory time and the
    #       MLEN bandwidth cap.
    #   M — the MAC operand width. The array multiplies weight x activation (FFN)
    #       and KV x activation (attention), and activations are computed at low
    #       precision inside the array (stored bf16 on-chip), so M must cover
    #       max(weights, KV, activation), not just the widest weight.
    #   density_exp — iso-area density exponent k: an M-bit array fits (4/M)^k x
    #       the reference multipliers. Default 0.0 assumes narrow operands are
    #       upcast back to full compute, which fits a memory-bound decode chip
    #       whose gain comes from effective bandwidth. Set k=2.0 to enable the
    #       density layer once a Synopsys-DC sweep calibrates it.
    DENSITY_EXP: float = 0.0

    def make_precision(
        self,
        *,
        attn_w: Any,
        ffn_w: Any,
        kv: Any = None,
        key: Any = None,
        value: Any = None,
        w_fmt: str = "mxint",
        kv_fmt: str = "mxint",
        key_fmt: str | None = None,
        value_fmt: str | None = None,
        block: int = 8,
        act_w: Any = None,
        act_fmt: str = "mxint",
        m_bits: int = 0,
        density_exp: float | None = None,
    ) -> Precision:
        """Build a decode precision from per-component widths.

        Widths are ints for MXINT (e.g. 4) and (exp, frac) tuples or "E2M1"
        tokens for MXFP. attn and ffn share the weight format with independent
        widths; KV has its own, so MXINT weights + MXFP KV is expressible.
        ``act_w`` is the activation compute width (a MAC operand; stored bf16
        on-chip, never in HBM). ``m_bits`` overrides M (0 = max operand width,
        activations included).
        """
        dd = self._dd

        def _w(fmt: str, tok: Any):
            if fmt == "mxint":
                return int(tok)
            return dd.MXFP_FORMATS[tok] if isinstance(tok, str) else tuple(tok)

        if key is None and value is None:
            if kv is None:
                raise ValueError("precision requires KV or distinct key/value widths")
            key = value = kv
        elif key is None or value is None or kv is not None:
            raise ValueError("key and value widths must be supplied together without KV")
        key_fmt = kv_fmt if key_fmt is None else key_fmt
        value_fmt = kv_fmt if value_fmt is None else value_fmt
        aw, fw = _w(w_fmt, attn_w), _w(w_fmt, ffn_w)
        key_width = _w(key_fmt, key)
        value_width = _w(value_fmt, value)
        elems = [dd.element_bits(w_fmt, aw), dd.element_bits(w_fmt, fw),
                 dd.element_bits(key_fmt, key_width),
                 dd.element_bits(value_fmt, value_width)]
        activation_width = None
        activation_element_bits = None
        if act_w is not None:
            activation_width = _w(act_fmt, act_w)
            activation_element_bits = dd.element_bits(
                act_fmt,
                activation_width,
            )
            elems.append(activation_element_bits)
        spec = dd.precision_from_components(
            dd.effective_bits(w_fmt, aw, block),
            dd.effective_bits(w_fmt, fw, block),
            (
                dd.effective_bits(key_fmt, key_width, block)
                + dd.effective_bits(value_fmt, value_width, block)
            )
            / 2.0,
            dd.width_label(w_fmt, aw),
            dd.width_label(w_fmt, fw),
            (
                dd.width_label(key_fmt, key_width)
                if key_fmt == value_fmt and key_width == value_width
                else (
                    f"K_{dd.width_label(key_fmt, key_width)}__"
                    f"V_{dd.width_label(value_fmt, value_width)}"
                )
            ),
            attn_elem=elems[0],
            ffn_elem=elems[1],
            kv_elem=max(elems[2:4]),
            key_bits=dd.effective_bits(key_fmt, key_width, block),
            value_bits=dd.effective_bits(value_fmt, value_width, block),
            key_label=dd.width_label(key_fmt, key_width),
            value_label=dd.width_label(value_fmt, value_width),
            key_elem=elems[2],
            value_elem=elems[3],
            m_bits=m_bits or max(elems),
            density_exp=self.DENSITY_EXP if density_exp is None else density_exp,
            block_size=block,
        )
        if activation_width is not None and activation_element_bits is not None:
            spec.update(
                {
                    "head_bits": spec["attn_bits"],
                    "head_elem": spec["attn_elem"],
                    "head_label": spec["attn_label"],
                    "head_activation_bits": dd.effective_bits(
                        act_fmt,
                        activation_width,
                        block,
                    ),
                    "head_activation_elem": activation_element_bits,
                    "head_activation_label": dd.width_label(
                        act_fmt,
                        activation_width,
                    ),
                    "lm_head_quantized": False,
                    "lm_head_top_k": 20,
                }
            )
        return Precision(
            spec,
            w_fmt,
            key_fmt if key_fmt == value_fmt else "split",
            aw,
            fw,
            key_width if key_width == value_width else (key_width, value_width),
            block,
            spec["m_bits"],
            key_fmt,
            value_fmt,
            key_width,
            value_width,
        )

    def precision_from_eff_bits(
        self,
        attn_bits: float,
        ffn_bits: float,
        kv_bits: float,
        *,
        act_bits: float | None = None,
        block: int = 8,
        m_bits: int = 0,
        density_exp: float | None = None,
    ) -> Precision:
        """Rebuild a precision from effective-bit numbers (the software-DSE CSV
        columns).

        Element widths subtract the per-block scale share before rounding.
        ``act_bits`` is the activation compute width; it joins the operand max
        for M but never the HBM stream."""
        dd = self._dd
        scale_share = dd.SCALE_BITS / block

        def _elem(bits: float) -> int:
            return max(1, round(float(bits) - scale_share))

        elems = [_elem(attn_bits), _elem(ffn_bits), _elem(kv_bits)]
        if act_bits is not None:
            elems.append(_elem(act_bits))
        spec = dd.precision_from_components(
            attn_bits, ffn_bits, kv_bits,
            attn_elem=elems[0], ffn_elem=elems[1], kv_elem=elems[2],
            m_bits=m_bits or max(elems),
            density_exp=self.DENSITY_EXP if density_exp is None else density_exp,
            block_size=block,
        )
        if act_bits is not None:
            spec.update(
                {
                    "head_bits": spec["attn_bits"],
                    "head_elem": spec["attn_elem"],
                    "head_label": spec["attn_label"],
                    "head_activation_bits": float(act_bits),
                    "head_activation_elem": elems[-1],
                    "head_activation_label": f"MXINT{elems[-1]}",
                    "lm_head_quantized": False,
                    "lm_head_top_k": 20,
                }
            )
        return Precision(
            spec, "mxint", "mxint",
            spec["attn_elem"], spec["ffn_elem"], spec["kv_elem"], block, spec["m_bits"],
        )

    def precision_from_row(self, row: dict[str, Any], **kw) -> Precision:
        """Precision for one software-DSE CSV row (as parsed by
        ``attn_bits``, ``ffn_bits``, ``kv_bits``, and ``act_bits`` fields."""
        return self.precision_from_eff_bits(
            row["attn_bits"], row["ffn_bits"], row["kv_bits"],
            act_bits=row.get("act_bits"), block=int(row.get("block") or 8), **kw,
        )

    # -- hardware helpers ----------------------------------------------------
    def hbm_overrides(self, gen: str, channels: int = 0) -> dict[str, int]:
        """HBM_WIDTH + HBM_SIZE for an HBM generation x channel count.

        HBM is fixed technology: bandwidth and capacity move together, so this
        is the only sanctioned way to change either.
        """
        if channels < 0:
            raise ValueError("HBM channels must be non-negative")
        if gen not in self._dd.HBM_GENS:
            raise ValueError(f"unsupported HBM generation {gen!r}")
        ov = self._dd.hbm_overrides(gen, channels)
        return {"HBM_WIDTH": ov["HBM_WIDTH"], "HBM_SIZE": ov["HBM_SIZE"]}

    def hbm_operating_point(
        self,
        gen: str,
        channels: int,
    ) -> HBMOperatingPointStatus:
        """Return exact calibration scope without extrapolating channel counts."""

        if channels <= 0:
            raise ValueError("HBM interface-unit count must be positive")
        technology = self._dd.hbm_technology(gen)
        overrides = self.hbm_overrides(gen, channels)
        bw_model = getattr(self, "_bw_model", None)
        if bw_model is None:
            return HBMOperatingPointStatus(
                generation=technology.generation,
                interface_units=channels,
                pin_rate_gbps=technology.pin_rate_gbps,
                technology_schema=technology.schema_version,
                source_url=technology.source_url,
                source_label=technology.source_label,
                emulator_generation=technology.emulator_generation,
                emulator_pin_rate_gbps=technology.emulator_pin_rate_gbps,
                hbm_width_bits=overrides["HBM_WIDTH"],
                hbm_capacity_bytes=overrides["HBM_SIZE"],
                calibration_id=None,
                calibrated_channel_min=None,
                calibrated_channel_max=None,
                rankable=False,
                reason="peak_bandwidth_sensitivity",
            )

        calibration_id = bw_model.operating_point_calibration_id(
            technology.generation,
            technology.pin_rate_gbps,
        )
        if calibration_id is None:
            return HBMOperatingPointStatus(
                generation=technology.generation,
                interface_units=channels,
                pin_rate_gbps=technology.pin_rate_gbps,
                technology_schema=technology.schema_version,
                source_url=technology.source_url,
                source_label=technology.source_label,
                emulator_generation=technology.emulator_generation,
                emulator_pin_rate_gbps=technology.emulator_pin_rate_gbps,
                hbm_width_bits=overrides["HBM_WIDTH"],
                hbm_capacity_bytes=overrides["HBM_SIZE"],
                calibration_id=None,
                calibrated_channel_min=None,
                calibrated_channel_max=None,
                rankable=False,
                reason="generation_rate_not_calibrated",
            )

        channel_sets = [
            tuple(bw_model.channel_counts(name, technology.generation))
            for name in _BANDWIDTH_TRAFFIC_CLASSES
        ]
        size_model = getattr(bw_model, "size_model", None)
        if size_model is not None:
            channel_sets.append(
                tuple(size_model.channel_counts(technology.generation))
            )
        if not channel_sets or any(not values for values in channel_sets):
            return HBMOperatingPointStatus(
                generation=technology.generation,
                interface_units=channels,
                pin_rate_gbps=technology.pin_rate_gbps,
                technology_schema=technology.schema_version,
                source_url=technology.source_url,
                source_label=technology.source_label,
                emulator_generation=technology.emulator_generation,
                emulator_pin_rate_gbps=technology.emulator_pin_rate_gbps,
                hbm_width_bits=overrides["HBM_WIDTH"],
                hbm_capacity_bytes=overrides["HBM_SIZE"],
                calibration_id=None,
                calibrated_channel_min=None,
                calibrated_channel_max=None,
                rankable=False,
                reason="calibrated_traffic_class_missing",
            )
        lower = max(min(values) for values in channel_sets)
        upper = min(max(values) for values in channel_sets)
        supported = lower <= channels <= upper
        return HBMOperatingPointStatus(
            generation=technology.generation,
            interface_units=channels,
            pin_rate_gbps=technology.pin_rate_gbps,
            technology_schema=technology.schema_version,
            source_url=technology.source_url,
            source_label=technology.source_label,
            emulator_generation=technology.emulator_generation,
            emulator_pin_rate_gbps=technology.emulator_pin_rate_gbps,
            hbm_width_bits=overrides["HBM_WIDTH"],
            hbm_capacity_bytes=overrides["HBM_SIZE"],
            calibration_id=calibration_id if supported else None,
            calibrated_channel_min=lower,
            calibrated_channel_max=upper,
            rankable=supported,
            reason=(
                "emulator_dma_calibrated"
                if supported
                else "channel_count_outside_calibration"
            ),
        )

    def validate_calibrated_hardware_space(
        self,
        generation: str,
        channels: Iterable[int],
    ) -> tuple[HBMOperatingPointStatus, ...]:
        """Require every headline HBM point to have measured calibration support."""

        values = tuple(
            self.hbm_operating_point(generation, int(channel))
            for channel in channels
        )
        unsupported = [
            status
            for status in values
            if not status.rankable
        ]
        if unsupported:
            details = ", ".join(
                f"{status.generation}/{status.interface_units}:"
                f"{status.reason}"
                for status in unsupported
            )
            raise ValueError(
                "uncalibrated headline HBM operating points: " + details
            )
        return values

    def mlen_cap(self, precision: Precision, hw_over: dict | None = None) -> int:
        """Largest MLEN this precision's stream width lets the HBM feed per cycle."""
        hw = self.base_hw.model_copy(update=hw_over) if hw_over else self.base_hw
        return self._dd.mlen_bandwidth_cap(hw, precision.spec)

    def shipped_over(self, precision: Precision, hbm_over: dict | None = None) -> dict[str, int]:
        """Feasible overrides for the shipped (un-searched) chip at a precision.

        The shipped TOML geometry can break the bandwidth bound for wide streams
        (e.g. MLEN=1024 at FP16 needs 16384 bits/cycle), so MLEN is clamped to the
        largest power of two within the bandwidth cap, VLEN follows MLEN (compiler
        requires MLEN == VLEN), HLEN is clamped under MLEN, and the prefetch
        follows MLEN. This baseline chip obeys the same rules as the co-design
        search, so it is a legal design point too.
        """
        hbm = dict(hbm_over or {})
        cap = self.mlen_cap(precision, hw_over=hbm)
        mlen = min(self.base_hw.MLEN, 1 << max(0, cap.bit_length() - 1))
        hlen = min(self.base_hw.HLEN, mlen)
        return {"MLEN": mlen, "VLEN": mlen, "HLEN": hlen,
                "HBM_M_Prefetch_Amount": mlen, **hbm}

    # -- evaluation ----------------------------------------------------------
    def use_calibrated_bandwidth(self) -> None:
        """Price memory at the emulator-measured effective bandwidth instead
        of the aggregate peak (disagg_serve.memory calibration table)."""
        root = _sim_root()
        sys.path.insert(0, str(root))
        from analytic_models.disagg_serve.memory import CalibratedBandwidth

        self._bw_model = CalibratedBandwidth.load()

    def use_compiler_trace_timing(
        self,
        provider: Any,
        request_binder: CompilerTraceRequestBinder,
    ) -> None:
        """Select full-model trace timing with immutable point binding."""

        if getattr(provider, "artifact_scope", None) != (
            self._dd.FULL_MODEL_DECODE_SCOPE
        ):
            raise ValueError(
                "compiler trace provider lacks full-model decode scope"
            )
        if not callable(getattr(request_binder, "bind", None)):
            raise TypeError(
                "compiler trace request binder must provide bind(point)"
            )
        self.execution_mode = self._dd.COMPILER_TRACE
        self.trace_timing_provider = provider
        self.trace_request_binder = request_binder

    def compiler_trace_point(
        self,
        precision: Precision,
        *,
        hardware: Any,
        overrides: dict[str, Any],
        batch: int,
        input_seq: int,
        output_seq: int,
        stride: int,
        n_chips: int,
        hbm_gen: str,
        hbm_channels: int,
        kv_layout: str,
        runtime_hbm_reserve_bytes: int,
        output_head_location: str,
    ) -> CompilerTracePointDescriptor:
        """Seal every lowering and timing determinant except context length."""

        dd = self._dd
        if n_chips <= 0:
            raise ValueError(
                "compiler trace timing requires an explicit chip count"
            )
        topology = dd._parallel_topology(overrides, n_chips)
        if not topology["explicit_topology"]:
            raise ValueError(
                "compiler trace timing requires explicit TP/KVP topology"
            )
        resolved_timing_mode = (
            (
                dd.DRAIN_OVERLAPPED
                if bool(topology["drain_overlapped"])
                else dd.RTL_SERIALIZED
            )
            if bool(topology["architecture_knobs_explicit"])
            else self.timing_mode
        )
        memory = self.base_mem.model_copy(
            update={
                "weight_bits": precision.spec["ffn_bits"],
                "activation_bits": dd.ACT_BITS,
                "kv_cache_bits": precision.spec["kv_bits"],
                **overrides,
            }
        )
        from compiler_trace_timing import HBMOperatingPoint

        hbm = HBMOperatingPoint(
            generation=hbm_gen,
            channels=hbm_channels,
            pin_rate_gbps=dd.hbm_technology(hbm_gen).pin_rate_gbps,
        )
        return CompilerTracePointDescriptor.from_mapping(
            {
                "schema_version": COMPILER_TRACE_POINT_SCHEMA,
                "artifact_scope": FULL_MODEL_DECODE_SCOPE,
                "model": {
                    "model_json_sha256": _sha256_file(self.model_json),
                    "dimensions": dict(self.dims),
                    "layer_scope": "all_decoder_layers",
                    "output_head_location": output_head_location,
                },
                "precision": {
                    "specification": dict(precision.spec),
                    "weight_format": precision.w_fmt,
                    "kv_format": precision.kv_fmt,
                    "key_format": precision.key_fmt or precision.kv_fmt,
                    "value_format": precision.value_fmt or precision.kv_fmt,
                    "block_size": precision.block,
                    "mac_bits": precision.m_bits,
                },
                "hardware": {
                    "array_geometry": {
                        "mlen": int(hardware.MLEN),
                        "blen": int(hardware.BLEN),
                        "vlen": int(hardware.VLEN),
                        "hlen": int(hardware.HLEN),
                    },
                    "hbm_timing_geometry": hbm.to_dict(),
                    "configuration": _model_mapping(hardware),
                    "memory_configuration": _model_mapping(memory),
                    "overrides": dict(overrides),
                    "topology": dict(topology),
                },
                "serving": {
                    "batch": batch,
                    "input_tokens": input_seq,
                    "generation_tokens": output_seq,
                    "sample_stride": stride,
                    "kv_layout": kv_layout,
                    "runtime_hbm_reserve_bytes": runtime_hbm_reserve_bytes,
                },
                "compiler": {
                    "settings_sha256": _sha256_file(self.settings_toml),
                    "latency_library_sha256": _sha256_file(self.isa_path),
                    "timing_mode": resolved_timing_mode,
                    "frequency_hz": dd.FREQ_HZ,
                },
            }
        )

    def _bind_trace_request_factory(
        self,
        point: CompilerTracePointDescriptor,
    ):
        binder = self.trace_request_binder
        if binder is None:
            raise RuntimeError("compiler trace request binder is missing")
        bound = binder.bind(point)
        if not callable(bound):
            raise TypeError("compiler trace binder must return a callable")
        descriptor = point.to_dict()
        expected_batch = int(descriptor["serving"]["batch"])
        expected_geometry = descriptor["hardware"]["array_geometry"]
        expected_hbm = descriptor["hardware"]["hbm_timing_geometry"]
        expected_frequency = float(descriptor["compiler"]["frequency_hz"])

        def request_for_context(context_tokens: int):
            request = bound(context_tokens)
            if getattr(request, "compiler_inputs_sha256", None) != (
                point.descriptor_sha256
            ):
                raise ValueError(
                    "compiler timing request differs from its point descriptor"
                )
            if getattr(request, "context_tokens", None) != context_tokens:
                raise ValueError("compiler timing request context is inconsistent")
            if getattr(request, "batch", None) != expected_batch:
                raise ValueError("compiler timing request batch is inconsistent")
            geometry = getattr(request, "geometry", None)
            if not callable(getattr(geometry, "to_dict", None)) or (
                geometry.to_dict() != expected_geometry
            ):
                raise ValueError("compiler timing request geometry is inconsistent")
            hbm = getattr(request, "hbm", None)
            if not callable(getattr(hbm, "to_dict", None)) or (
                hbm.to_dict() != expected_hbm
            ):
                raise ValueError("compiler timing request HBM geometry is inconsistent")
            if float(getattr(request, "frequency_hz", 0.0)) != expected_frequency:
                raise ValueError("compiler timing request frequency is inconsistent")
            return request

        return request_for_context

    def evaluate(
        self,
        precision: Precision,
        *,
        batch: int,
        input_seq: int,
        output_seq: int,
        hw_over: dict | None = None,
        n_chips: int = 0,
        stride: int | None = None,
        hbm_gen: str = "HBM2",
        hbm_channels: int = 8,
        kv_layout: str = "dense_selector",
        runtime_hbm_reserve_bytes: int = 0,
        output_head_location: str = "decode_bf16_unmodeled",
    ) -> DecodeMetrics:
        """Evaluate one (precision, hardware, batch) point.

        `hw_over` overrides HardwareConfig fields on top of the base config
        (MLEN/BLEN/VLEN/HLEN and, via :meth:`hbm_overrides`, HBM_WIDTH/HBM_SIZE).
        It is applied to both the compute config (peak bandwidth, MLEN cap) and
        the memory config (capacity, traffic) so the two agree. `hbm_gen`/
        `hbm_channels` key the calibrated bandwidth lookup when
        :meth:`use_calibrated_bandwidth` is on; otherwise memory uses peak.
        """
        dd = self._dd
        over = dict(hw_over or {})
        # Matrix prefetch tracks MLEN so a searched MLEN incurs no wasted reads.
        if "MLEN" in over:
            over.setdefault("HBM_M_Prefetch_Amount", over["MLEN"])
        hw_cfg = self.base_hw.model_copy(update=over) if over else self.base_hw
        if runtime_hbm_reserve_bytes < 0:
            raise ValueError("runtime_hbm_reserve_bytes must be non-negative")
        bw_model = (
            getattr(self, "_bw_model", None)
            if self.execution_mode == dd.LEGACY_AGGREGATE_BANDWIDTH
            else None
        )
        if bw_model is not None:
            operating_point = self.hbm_operating_point(
                hbm_gen,
                hbm_channels,
            )
            if (
                operating_point.reason
                in {
                    "calibrated_traffic_class_missing",
                    "channel_count_outside_calibration",
                }
            ):
                raise ValueError(
                    f"unsupported calibrated HBM operating point "
                    f"{hbm_gen}/{hbm_channels}: {operating_point.reason}"
                )
            expected = self.hbm_overrides(hbm_gen, hbm_channels)
            for field in ("HBM_WIDTH", "HBM_SIZE"):
                actual = int(getattr(hw_cfg, field))
                if actual != expected[field]:
                    raise ValueError(
                        f"calibrated {hbm_gen}/{hbm_channels} expects "
                        f"{field}={expected[field]}, got {actual}"
                    )

        if stride is None:
            stride = max(1, output_seq // 24)  # subsample the growing-context loop

        trace_point = None
        trace_request_factory = None
        if self.execution_mode == dd.COMPILER_TRACE:
            trace_point = self.compiler_trace_point(
                precision,
                hardware=hw_cfg,
                overrides=over,
                batch=batch,
                input_seq=input_seq,
                output_seq=output_seq,
                stride=stride,
                n_chips=n_chips,
                hbm_gen=hbm_gen,
                hbm_channels=hbm_channels,
                kv_layout=kv_layout,
                runtime_hbm_reserve_bytes=runtime_hbm_reserve_bytes,
                output_head_location=output_head_location,
            )
            trace_request_factory = self._bind_trace_request_factory(
                trace_point
            )

        loop = dd.evaluate(
            self.model_json, self.dims, hw_cfg, self.isa_path, self.base_mem,
            precision.spec, batch, input_seq, output_seq,
            hw_over=over, stride=stride, n_chips=n_chips,
            bw_model=bw_model,
            hbm_gen=hbm_gen, hbm_channels=hbm_channels,
            kv_layout=kv_layout,
            timing_mode=self.timing_mode,
            timing_evidence=self.timing_evidence,
            packed_q1_timing_contract=self.packed_q1_timing_contract,
            runtime_hbm_reserve_bytes=runtime_hbm_reserve_bytes,
            output_head_location=output_head_location,
            execution_mode=self.execution_mode,
            trace_timing_provider=self.trace_timing_provider,
            trace_request_factory=trace_request_factory,
        )
        if trace_point is not None:
            trace_evidence = loop.get("compiler_trace_timing")
            if not isinstance(trace_evidence, dict) or (
                trace_evidence.get("compiler_input_descriptor_sha256")
                != trace_point.descriptor_sha256
            ):
                raise ValueError(
                    "compiler trace evidence differs from the bound DSE point"
                )
        ledger = loop["physical_ledger"]
        quantized_weights = (
            ledger.weights.attention
            + ledger.weights.ffn_resident
            + ledger.weights.lm_head_resident
        )
        sram = ledger.sram
        return DecodeMetrics(
            tps=float(loop["tps"]),
            tpot=float(loop["tpot"]),
            total_time=float(loop["total_time"]),
            first_step=float(loop["first_step"]),
            frac_mem_bound=float(loop["frac_mem_bound"]),
            frac_classical_mem_bound=float(
                loop["frac_classical_mem_bound"]
            ),
            frac_architecture_issue_mem_bound=float(
                loop["frac_architecture_issue_mem_bound"]
            ),
            frac_algorithmic_mem_bound=float(
                loop.get("frac_algorithmic_mem_bound", loop["frac_mem_bound"])
            ),
            frac_serialization_bound=float(loop.get("frac_serialization_bound", 0.0)),
            fits_in_hbm=bool(loop["fits_in_hbm"]),
            hbm_required=float(loop["hbm_required"]),
            area_mm2=float(dd.area_mm2(hw_cfg)),
            avg_bytes_per_token=float(loop.get("avg_bytes_per_token", 0.0)),
            avg_hbm_bytes_per_batch_step=float(loop["avg_bytes_per_batch_step"]),
            avg_hbm_bytes_per_generated_token=float(
                loop["avg_bytes_per_generated_token"]
            ),
            hbm_traffic_per_batch_step=tuple(
                sorted(loop["traffic_breakdown_per_batch_step"].items())
            ),
            hbm_traffic_per_generated_token=tuple(
                sorted(loop["traffic_breakdown_per_generated_token"].items())
            ),
            mem_bound=float(loop["frac_mem_bound"]) >= 0.5,
            bottleneck=str(dd.decode_bound_label(loop)),
            classical_roofline_bottleneck=str(
                dd.classical_roofline_bound_label(loop)
            ),
            architecture_issue_bottleneck=str(
                dd.architecture_issue_bound_label(loop)
            ),
            n_chips=int(loop["n_chips"]),
            max_batch=int(dd.max_batch_capacity(loop, batch)),
            max_resident_batch=int(loop["max_resident_batch"]),
            max_synchronous_batch=int(loop["max_synchronous_batch"]),
            max_runtime_batch=int(loop["max_runtime_batch"]),
            fits_onchip_sram=bool(loop["fits_onchip_sram"]),
            fits_runtime=bool(loop["fits_runtime"]),
            hbm_capacity=int(loop["hbm_capacity"]),
            runtime_hbm_reserve_bytes=int(loop["runtime_hbm_reserve_bytes"]),
            weight_element_plane_bytes=int(quantized_weights.element_aligned),
            weight_scale_plane_bytes=int(quantized_weights.scale_aligned),
            weight_bf16_bytes=int(ledger.weights.bf16_resident.total_aligned),
            kv_element_plane_bytes=int(ledger.kv.element_bytes),
            kv_scale_plane_bytes=int(ledger.kv.scale_bytes),
            vector_sram_capacity_bytes=int(sram.vector_capacity_bytes),
            vector_sram_required_bytes=int(sram.vector_required_bytes),
            matrix_sram_capacity_bytes=int(sram.matrix_capacity_bytes),
            matrix_sram_required_bytes=int(sram.matrix_required_bytes),
            timing_mode=str(loop["timing_mode"]),
            timing_calibrated=bool(loop["timing_calibrated"]),
            timing_reason=str(loop["timing_reason"]),
            timing_evidence_id=loop.get("timing_evidence_id"),
            timing_evidence_tier=loop.get("timing_evidence_tier"),
            packed_q1_timing_validated=bool(
                loop["packed_q1_timing_validated"]
            ),
            packed_q1_timing_reason=str(loop["packed_q1_timing_reason"]),
            packed_q1_timing_contract_id=loop.get(
                "packed_q1_timing_contract_id"
            ),
            bandwidth_calibration_id=loop.get("bandwidth_calibration_id"),
            kv_layout=str(loop["kv_layout"]),
            kv_layout_id=str(ledger.kv.layout_id),
            output_head_location=str(loop["output_head_location"]),
            avg_peak_compute_seconds=float(
                loop["avg_peak_compute_seconds"]
            ),
            avg_ideal_compute_seconds=float(
                loop["avg_ideal_compute_seconds"]
            ),
            avg_realized_compute_seconds=float(
                loop["avg_realized_compute_seconds"]
            ),
            avg_memory_seconds=float(loop["avg_memory_seconds"]),
            step_composition=str(loop["step_composition"]),
            execution_mode=str(loop["execution_mode"]),
            compiler_trace_timing=(
                dict(loop["compiler_trace_timing"])
                if loop.get("compiler_trace_timing") is not None
                else None
            ),
            moe_workload=(
                dict(loop["moe_workload"])
                if loop.get("moe_workload") is not None
                else None
            ),
            local_output_head=(
                dict(loop["local_output_head"])
                if loop.get("local_output_head") is not None
                else None
            ),
            body_physical_layout=(
                dict(loop["body_physical_layout"])
                if loop.get("body_physical_layout") is not None
                else None
            ),
            slowest_rank_hbm_required_bytes=(
                int(ledger.slowest_rank_hbm_required_bytes)
                if ledger.slowest_rank_hbm_required_bytes is not None
                else None
            ),
            per_chip_hbm_capacity_bytes=(
                int(ledger.per_chip_hbm_capacity_bytes)
                if ledger.per_chip_hbm_capacity_bytes is not None
                else None
            ),
        )
