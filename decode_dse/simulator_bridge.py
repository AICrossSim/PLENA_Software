"""In-process bridge to the PLENA decode-chip analytic model.

The analytic model (`analytic_models/performance/disagg_decode.py`) uses
script-style imports and ships no installable package, so it can't be
pip-imported. This module is the only place that reaches into it: it adds the
performance directory to `sys.path` once, imports the helpers, and re-exports a
small typed surface (`Precision`, `DecodeMetrics`, `DecodeSimulator`). Everything
else in the DSE imports from here, so if the simulator becomes a real package
only this file changes.

Set ``PLENA_SIMULATOR_PATH`` to the simulator checkout; defaults to
``/home/sr1325/PLENA_Simulator``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_SIM_ROOT = "/home/sr1325/PLENA_Simulator"


def _sim_root() -> Path:
    root = Path(os.environ.get("PLENA_SIMULATOR_PATH", _DEFAULT_SIM_ROOT)).expanduser()
    perf = root / "analytic_models" / "performance"
    if not perf.exists():
        raise FileNotFoundError(
            f"PLENA_Simulator not found at {root} (missing {perf}). Clone it or set "
            f"PLENA_SIMULATOR_PATH to the checkout root."
        )
    return root


@lru_cache(maxsize=1)
def _disagg():
    """Import and return the disagg_decode module, injecting sys.path once."""
    perf = _sim_root() / "analytic_models" / "performance"
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
    def stream_bits(self) -> int:
        return _disagg().stream_bits(self.spec)

    @property
    def tag(self) -> str:
        a, f, k = self.spec["attn_label"], self.spec["ffn_label"], self.spec["kv_label"]
        m = f"_M{self.m_bits}" if self.m_bits else ""
        return f"attn-{a}__ffn-{f}__kv-{k}{m}"


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
    fits_in_hbm: bool
    hbm_required: float
    area_mm2: float
    avg_bytes_per_token: float
    mem_bound: bool
    n_chips: int
    # Largest batch whose weights+KV fit the evaluated HBM capacity. KV precision
    # moves this smoothly, so it stays a throughput signal even when a compute
    # ceiling flattens per-batch TPS.
    max_batch: int = 0


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
        kv: Any,
        w_fmt: str = "mxint",
        kv_fmt: str = "mxint",
        block: int = 32,
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

        aw, fw, kvw = _w(w_fmt, attn_w), _w(w_fmt, ffn_w), _w(kv_fmt, kv)
        elems = [dd.element_bits(w_fmt, aw), dd.element_bits(w_fmt, fw),
                 dd.element_bits(kv_fmt, kvw)]
        if act_w is not None:
            elems.append(dd.element_bits(act_fmt, _w(act_fmt, act_w)))
        spec = dd.precision_from_components(
            dd.effective_bits(w_fmt, aw, block),
            dd.effective_bits(w_fmt, fw, block),
            dd.effective_bits(kv_fmt, kvw, block),
            dd.width_label(w_fmt, aw),
            dd.width_label(w_fmt, fw),
            dd.width_label(kv_fmt, kvw),
            attn_elem=elems[0],
            ffn_elem=elems[1],
            kv_elem=elems[2],
            m_bits=m_bits or max(elems),
            density_exp=self.DENSITY_EXP if density_exp is None else density_exp,
        )
        return Precision(spec, w_fmt, kv_fmt, aw, fw, kvw, block, spec["m_bits"])

    def precision_from_eff_bits(
        self,
        attn_bits: float,
        ffn_bits: float,
        kv_bits: float,
        *,
        act_bits: float | None = None,
        block: int = 32,
        m_bits: int = 0,
        density_exp: float | None = None,
    ) -> Precision:
        """Rebuild a precision from effective-bit numbers (the software-DSE CSV
        columns).

        Element widths subtract the per-block scale share before rounding
        (``elem = round(eff - SCALE_BITS/block)``); rounding eff bits directly
        mis-sizes odd widths (e.g. MXINT3 at block 16 = 3.5 eff bits would round
        to 4). ``act_bits`` is the activation compute width; it joins the operand
        max for M but never the HBM stream (activations stay on-chip)."""
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
        )
        return Precision(
            spec, "mxint", "mxint",
            spec["attn_elem"], spec["ffn_elem"], spec["kv_elem"], block, spec["m_bits"],
        )

    def precision_from_row(self, row: dict[str, Any], **kw) -> Precision:
        """Precision for one software-DSE CSV row (as parsed by
        :func:`decode_dse.results.load_software_rows`)."""
        return self.precision_from_eff_bits(
            row["attn_bits"], row["ffn_bits"], row["kv_bits"],
            act_bits=row.get("act_bits"), block=int(row.get("block") or 32), **kw,
        )

    # -- hardware helpers ----------------------------------------------------
    def hbm_overrides(self, gen: str, channels: int = 0) -> dict[str, int]:
        """HBM_WIDTH + HBM_SIZE for an HBM generation x channel count.

        HBM is fixed technology: bandwidth and capacity move together, so this
        is the only sanctioned way to change either.
        """
        ov = self._dd.hbm_overrides(gen, channels)
        return {"HBM_WIDTH": ov["HBM_WIDTH"], "HBM_SIZE": ov["HBM_SIZE"]}

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

        if stride is None:
            stride = max(1, output_seq // 24)  # subsample the growing-context loop

        loop = dd.evaluate(
            self.model_json, self.dims, hw_cfg, self.isa_path, self.base_mem,
            precision.spec, batch, input_seq, output_seq,
            hw_over=over, stride=stride, n_chips=n_chips,
            bw_model=getattr(self, "_bw_model", None),
            hbm_gen=hbm_gen, hbm_channels=hbm_channels,
        )
        return DecodeMetrics(
            tps=float(loop["tps"]),
            tpot=float(loop["tpot"]),
            total_time=float(loop["total_time"]),
            first_step=float(loop["first_step"]),
            frac_mem_bound=float(loop["frac_mem_bound"]),
            fits_in_hbm=bool(loop["fits_in_hbm"]),
            hbm_required=float(loop["hbm_required"]),
            area_mm2=float(dd.area_mm2(hw_cfg)),
            avg_bytes_per_token=float(loop.get("avg_bytes_per_token", 0.0)),
            mem_bound=float(loop["frac_mem_bound"]) >= 0.5,
            n_chips=int(loop["n_chips"]),
            max_batch=int(dd.max_batch_capacity(loop, batch)),
        )
