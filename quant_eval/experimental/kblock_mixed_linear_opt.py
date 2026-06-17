"""
K-block mixed precision Linear — hardware-faithful INT accumulation.

Each K-tile maps to a physical GEMM position:
    physical_pos = tile_id % physical_pattern_len

This ensures q/k/v/o/gate/up/down projections share the same hardware mask.

Unlike chop fake-quant, this models:
  - Scale stored as integer byte (floor alignment, not exact)
  - Integer-domain block accumulation with per-block scale alignment
  - Configurable accumulator / output float formats

Config dict schemas
-------------------
mxint : {"mode": "mxint", "block_size": 32, "element_bits": 8, "element_frac_bits": 4}
mxfp  : {"mode": "mxfp", "block_size": 32, "element_exp_bits": 4, "element_frac_bits": 3}
bypass: {"mode": "bypass"}
"""

import math
import re
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_eval.experimental.kblock_mixed_linear import fake_float_quant, _max_finite_for_float


# ------------------------------------------------------------------ #
# Quantizer primitives                                                 #
# ------------------------------------------------------------------ #

def _broadcast_clip(clip_p: torch.Tensor, max_abs: torch.Tensor) -> torch.Tensor:
    cp = clip_p.to(max_abs.device, max_abs.dtype)
    if cp.dim() == 0:
        return cp
    if cp.dim() == 1:
        return cp.reshape(*([1] * (max_abs.dim() - 2)), -1, 1)
    while cp.dim() < max_abs.dim():
        cp = cp.unsqueeze(0)
    return cp


def mxfp_quantizer(
    x: torch.Tensor,
    cfg: dict,
    clip_p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MXFP block fake-quant (quantize + dequantize)."""
    exp_bits = cfg["element_exp_bits"]
    mant_bits = cfg["element_frac_bits"]
    block_size = cfg["block_size"]

    if exp_bits >= 8 and mant_bits >= 23:
        return x

    orig_shape = x.shape
    pad = (-x.shape[-1]) % block_size
    if pad:
        x = F.pad(x, (0, pad))

    xb = x.reshape(*x.shape[:-1], -1, block_size).to(torch.float32)
    max_abs = xb.abs().amax(-1, keepdim=True)
    if clip_p is not None:
        max_abs = max_abs * _broadcast_clip(clip_p, max_abs)

    elem_max = _max_finite_for_float(exp_bits, mant_bits)
    safe = max_abs.clamp(1e-30)
    scale_exp = torch.ceil(torch.log2(safe / elem_max))
    scale_exp = torch.where(max_abs > 0, scale_exp, torch.zeros_like(scale_exp))
    scale = torch.pow(torch.tensor(2.0, device=x.device, dtype=torch.float32), scale_exp)

    q = fake_float_quant((xb / scale).clamp(-elem_max, elem_max), exp_bits=exp_bits, mant_bits=mant_bits)
    dq = (q * scale).reshape(*xb.shape[:-2], -1)
    return dq[..., :orig_shape[-1]].reshape(orig_shape).to(x.dtype)


def mxint_quantizer_hw(
    x: torch.Tensor,
    cfg: dict,
    clip_p: Optional[torch.Tensor] = None,
    scale_bias: int = 127,
):
    """
    INT block quantization — hardware faithful.

    Scale is chosen via ceil(log2(...)) then stored as an integer byte (floor),
    unlike chop's exact-scale fake-quant.

    Returns
    -------
    q_int : integer-valued float, shape = x.shape
    scale_byte : [..., num_blocks]
    """
    element_bits = cfg["element_bits"]
    element_frac_bits = cfg["element_frac_bits"]
    block_size = cfg["block_size"]
    max_int = (1 << (element_bits - 1)) - 1
    min_int = -(1 << (element_bits - 1))

    orig_shape = x.shape
    pad = (-x.shape[-1]) % block_size
    if pad:
        x = F.pad(x, (0, pad))

    xb = x.reshape(*x.shape[:-1], -1, block_size).to(torch.float32)
    max_abs = xb.abs().amax(-1, keepdim=True)
    if clip_p is not None:
        max_abs = max_abs * _broadcast_clip(clip_p, max_abs)

    safe = max_abs.clamp(1e-30)
    scale_exp = torch.ceil(torch.log2(safe * float(1 << element_frac_bits) / float(max_int)))
    scale_exp = torch.where(max_abs > 0, scale_exp, torch.zeros_like(scale_exp))
    scale_byte = (scale_exp + float(scale_bias)).clamp(0, 255).to(torch.float32)

    scale = torch.pow(
        torch.tensor(2.0, device=x.device, dtype=torch.float32),
        scale_byte - float(scale_bias),
    )
    q = torch.round(xb / scale * float(1 << element_frac_bits)).clamp(min_int, max_int)

    q = q.reshape(*xb.shape[:-2], -1)[..., :orig_shape[-1]].reshape(orig_shape).to(torch.float32)
    return q, scale_byte.squeeze(-1)


def _mode(cfg: dict) -> str:
    """Normalize precision mode names.

    ``int`` is kept as a compatibility alias for MXINT.
    """
    mode = cfg.get("mode", "bypass")
    return "mxint" if mode == "int" else mode


# ------------------------------------------------------------------ #
# Hadamard                                                             #
# ------------------------------------------------------------------ #

_HADAMARD_CACHE: dict = {}


def _hadamard(n: int, device, dtype=torch.float32) -> torch.Tensor:
    assert n & (n - 1) == 0, f"n must be power of 2, got {n}"
    key = (n, str(device))
    if key not in _HADAMARD_CACHE:
        H = torch.tensor([[1.0]], device=device, dtype=dtype)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HADAMARD_CACHE[key] = H / math.sqrt(n)
    return _HADAMARD_CACHE[key]


# ------------------------------------------------------------------ #
# INT partial accumulation                                             #
# ------------------------------------------------------------------ #

def _align_and_sum_int_partials(
    int_partials: list,
    scale_bias: int,
    acc_fp_exp_bits: int,
    acc_fp_mant_bits: int,
) -> torch.Tensor:
    """
    Align per-tile INT partial sums to a common exponent and accumulate.

    Each entry: (raw_acc, scale_sum, element_frac_bits).
      raw_acc   : integer dot-product [..., out_features]
      scale_sum : max per-output scale exponent [..., out_features]
      element_frac_bits : fixed-point fractional bits for this tile
    """
    pow2 = lambda e: torch.pow(torch.tensor(2.0, device=e.device, dtype=torch.float32), e)

    eff_exps, entries = [], []
    for raw_acc, scale_sum, frac in int_partials:
        eff = scale_sum.to(torch.float32) - float(2 * scale_bias) - float(2 * frac)
        eff_exps.append(eff)
        entries.append((raw_acc.to(torch.float32), eff))

    max_exp = eff_exps[0]
    for e in eff_exps[1:]:
        max_exp = torch.maximum(max_exp, e)

    total = None
    for raw_acc, eff in entries:
        aligned = raw_acc / pow2((max_exp - eff).clamp(0, 63))
        total = aligned if total is None else total + aligned

    out = total * pow2(max_exp)
    return fake_float_quant(out, exp_bits=acc_fp_exp_bits, mant_bits=acc_fp_mant_bits).to(torch.float32)


# ------------------------------------------------------------------ #
# KBlockMixedLinear                                                    #
# ------------------------------------------------------------------ #

_TARGET = re.compile(
    r"model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))$"
)


class KBlockMixedLinear(nn.Module):
    """
    K-block mixed precision Linear with hardware-faithful INT accumulation.

    Placement strategies: uniform, clustered, fixed, sensitivity.
    Per-position override: precision_map = {pos: config_dict, ...}
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        k_tile: int = 128,
        low_percent: int = 50,
        placement: str = "uniform",
        physical_pattern_len: int = 16,
        high_config: dict | None = None,
        low_config: dict | None = None,
        precision_map: dict[int, dict] | None = None,
        acc_fp_exp_bits: int = 8,
        acc_fp_mant_bits: int = 23,
        out_fp_exp_bits: int = 8,
        out_fp_mant_bits: int = 23,
        scale_bias: int = 127,
        use_erry_clip: bool = False,
        use_rotation: bool = False,
        fixed_low_positions=None,
        selected_low_tile_ids=None,
        sensitivity_pattern_len=None,
    ):
        super().__init__()

        if placement not in ("uniform", "clustered", "fixed", "sensitivity"):
            raise ValueError(f"placement must be uniform/clustered/fixed/sensitivity, got {placement!r}")
        if not 0 <= low_percent <= 100:
            raise ValueError(f"low_percent must be in [0, 100], got {low_percent}")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.k_tile = k_tile
        self.low_percent = low_percent
        self.placement = placement
        self.physical_pattern_len = int(physical_pattern_len)
        self.fixed_low_positions = set(int(x) for x in (fixed_low_positions or []))
        self.selected_low_tile_ids = set(selected_low_tile_ids or [])
        self.sensitivity_pattern_len = sensitivity_pattern_len

        self.high_config = high_config or {"mode": "bypass"}
        self.low_config = low_config or {"mode": "bypass"}
        self.precision_map = precision_map

        self.acc_fp_exp_bits = acc_fp_exp_bits
        self.acc_fp_mant_bits = acc_fp_mant_bits
        self.out_fp_exp_bits = out_fp_exp_bits
        self.out_fp_mant_bits = out_fp_mant_bits
        self.scale_bias = scale_bias

        self.use_erry_clip = use_erry_clip
        self.use_rotation_enabled = use_rotation

        self.weight = nn.Parameter(base_linear.weight.detach().clone())
        self.bias = (
            nn.Parameter(base_linear.bias.detach().clone())
            if base_linear.bias is not None else None
        )

        self.register_buffer("clip_p_table", None, persistent=False)
        self.rotate_input = False
        self.weight_is_rotated = False

        self._validate_positions()

    # ---------------------------------------------------------------- #
    # Physical mapping (same logic as kblock_mixed_chop.py)            #
    # ---------------------------------------------------------------- #

    def _pattern_len(self) -> int:
        if self.placement == "sensitivity" and self.sensitivity_pattern_len:
            return int(self.sensitivity_pattern_len)
        return self.physical_pattern_len

    def num_tiles(self) -> int:
        return (self.in_features + self.k_tile - 1) // self.k_tile

    def _low_tile_num(self) -> int:
        if self.placement == "fixed":
            return len(self.fixed_low_positions)
        if self.placement == "sensitivity" and self.selected_low_tile_ids:
            return len(self.selected_low_tile_ids)
        n = self._pattern_len()
        if self.low_percent <= 0:
            return 0
        return max(1, min((n * self.low_percent + 50) // 100, n))

    def _active_physical_positions(self) -> set:
        if self.placement == "fixed":
            return set(self.fixed_low_positions)
        if self.placement == "sensitivity":
            return set(self.selected_low_tile_ids)
        if self.placement == "clustered":
            return set(range(self._low_tile_num()))
        n, low_num = self._pattern_len(), self._low_tile_num()
        return {p for p in range(n) if ((p + 1) * low_num) // n != (p * low_num) // n}

    def physical_position_for_tile(self, tile_id: int) -> int:
        return int(tile_id) % self._pattern_len()

    def use_low_tile(self, tile_id: int) -> bool:
        if self.placement not in ("fixed", "sensitivity") and self.low_percent <= 0:
            return False
        return (tile_id % self._pattern_len()) in self._active_physical_positions()

    def _config_for_tile(self, tile_id: int) -> dict:
        pos = self.physical_position_for_tile(tile_id)
        if self.precision_map is not None and pos in self.precision_map:
            return self.precision_map[pos]
        return self.low_config if self.use_low_tile(tile_id) else self.high_config

    def _validate_positions(self):
        n = self._pattern_len()
        for pos in (*self.fixed_low_positions, *self.selected_low_tile_ids):
            if not 0 <= pos < n:
                raise ValueError(f"position {pos} outside [0, {n})")
        if self.precision_map:
            for pos in self.precision_map:
                if not 0 <= pos < n:
                    raise ValueError(f"precision_map key {pos} outside [0, {n})")

    def _tile_range(self, tile_id: int):
        k0 = tile_id * self.k_tile
        return k0, min(k0 + self.k_tile, self.in_features)

    def _low_tile_ids(self) -> list:
        return [i for i in range(self.num_tiles()) if self.use_low_tile(i)]

    def _tile_clip_p(self, tile_id: int) -> Optional[torch.Tensor]:
        if not self.use_erry_clip or self.clip_p_table is None:
            return None
        return self.clip_p_table[tile_id]

    # ---------------------------------------------------------------- #
    # Rotation                                                          #
    # ---------------------------------------------------------------- #

    def _rotate_input_tiles(self, x: torch.Tensor) -> torch.Tensor:
        low_ids = self._low_tile_ids()
        if not low_ids:
            return x
        x_fp32, chunks, prev = x.to(torch.float32), [], 0
        for tile_id in low_ids:
            k0, k1 = self._tile_range(tile_id)
            if prev < k0:
                chunks.append(x_fp32[..., prev:k0])
            chunks.append(x_fp32[..., k0:k1] @ _hadamard(k1 - k0, x.device))
            prev = k1
        if prev < self.in_features:
            chunks.append(x_fp32[..., prev:])
        return torch.cat(chunks, -1).to(x.dtype)

    def apply_rotation_to_weight(self):
        if self.weight_is_rotated:
            return
        W = self.weight.detach().to(torch.float32).clone()
        for tile_id in self._low_tile_ids():
            k0, k1 = self._tile_range(tile_id)
            W[:, k0:k1] @= _hadamard(k1 - k0, self.weight.device)
        self.weight.data.copy_(W.to(self.weight.dtype))
        self.weight_is_rotated = True

    # ---------------------------------------------------------------- #
    # Per-tile quantized partials                                       #
    # ---------------------------------------------------------------- #

    def _fp_partial(self, x_blk, w_blk, cfg, clip_p):
        x_q = mxfp_quantizer(x_blk, cfg)
        w_q = mxfp_quantizer(w_blk, cfg, clip_p=clip_p)
        partial = F.linear(x_q, w_q).to(torch.float32)
        return fake_float_quant(
            partial, exp_bits=self.acc_fp_exp_bits, mant_bits=self.acc_fp_mant_bits
        ).to(torch.float32)

    def _int_partial(self, x_blk, w_blk, cfg, clip_p):
        block_size = cfg["block_size"]
        frac = cfg["element_frac_bits"]
        num_blocks = x_blk.shape[-1] // block_size

        x_int, x_sb = mxint_quantizer_hw(x_blk, cfg, scale_bias=self.scale_bias)
        w_int, w_sb = mxint_quantizer_hw(w_blk, cfg, clip_p=clip_p, scale_bias=self.scale_bias)

        x_b = x_int.reshape(*x_int.shape[:-1], num_blocks, block_size)
        w_b = w_int.reshape(self.out_features, num_blocks, block_size)
        raw = torch.einsum("...bk,obk->...bo", x_b, w_b)  # [..., num_blocks, out_features]

        # x_sb: [..., num_blocks], w_sb: [out_features, num_blocks] → [num_blocks, out_features]
        w_sb_t = w_sb.T if w_sb.shape[0] == self.out_features else w_sb
        scale_sum = (
            x_sb.unsqueeze(-1)
            + w_sb_t.reshape(*([1] * (x_sb.dim() - 1)), *w_sb_t.shape)
        )  # [..., num_blocks, out_features]

        max_scale = scale_sum.max(-2, keepdim=True).values
        shift = (max_scale - scale_sum).clamp(0, 63)
        pow2 = lambda e: torch.pow(torch.tensor(2.0, device=raw.device, dtype=torch.float32), e)
        tile_acc = torch.floor(raw / pow2(shift)).sum(-2)
        tile_scale = max_scale.squeeze(-2)

        return tile_acc, tile_scale, frac

    # ---------------------------------------------------------------- #
    # Forward                                                           #
    # ---------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        if self.use_rotation_enabled and self.rotate_input:
            if not self.weight_is_rotated:
                self.apply_rotation_to_weight()
            x = self._rotate_input_tiles(x)

        fp_acc, int_partials = None, []

        for tile_id, k0 in enumerate(range(0, self.in_features, self.k_tile)):
            k1 = min(k0 + self.k_tile, self.in_features)
            x_blk = x[..., k0:k1]
            w_blk = self.weight[:, k0:k1]
            cfg = self._config_for_tile(tile_id)
            clip_p = self._tile_clip_p(tile_id)
            mode = _mode(cfg)

            if mode == "mxint":
                int_partials.append(self._int_partial(x_blk, w_blk, cfg, clip_p))
            else:
                partial = (
                    self._fp_partial(x_blk, w_blk, cfg, clip_p)
                    if mode == "mxfp" else
                    F.linear(x_blk, w_blk).to(torch.float32)
                )
                fp_acc = partial if fp_acc is None else fp_acc + partial
                fp_acc = fake_float_quant(
                    fp_acc, exp_bits=self.acc_fp_exp_bits, mant_bits=self.acc_fp_mant_bits
                ).to(torch.float32)

        if fp_acc is None:
            fp_acc = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=torch.float32)

        int_acc = (
            _align_and_sum_int_partials(
                int_partials, self.scale_bias, self.acc_fp_exp_bits, self.acc_fp_mant_bits
            )
            if int_partials else torch.zeros_like(fp_acc)
        )

        out = fake_float_quant(
            fp_acc + int_acc, exp_bits=self.out_fp_exp_bits, mant_bits=self.out_fp_mant_bits
        ).to(torch.float32)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.to(orig_dtype)

    @torch.no_grad()
    def forward_fp_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Full-precision forward. Used during calibration."""
        W = self.weight
        if self.weight_is_rotated:
            W = W.detach().to(torch.float32).clone()
            for tid in self._low_tile_ids():
                k0, k1 = self._tile_range(tid)
                W[:, k0:k1] @= _hadamard(k1 - k0, self.weight.device)
            W = W.to(self.weight.dtype)
        return F.linear(x, W, bias=self.bias)


# ------------------------------------------------------------------ #
# Model-level replacement                                              #
# ------------------------------------------------------------------ #

def replace_linears_with_kblock_mixed(
    model: nn.Module,
    k_tile: int = 128,
    low_percent: int = 50,
    placement: str = "uniform",
    physical_pattern_len: int = 16,
    high_config: dict | None = None,
    low_config: dict | None = None,
    precision_map: dict[int, dict] | None = None,
    acc_fp_exp_bits: int = 8,
    acc_fp_mant_bits: int = 23,
    out_fp_exp_bits: int = 8,
    out_fp_mant_bits: int = 23,
    scale_bias: int = 127,
    use_erry_clip: bool = False,
    use_rotation: bool = False,
    fixed_low_positions=None,
    sensitivity_score_csv: str | None = None,
    sensitivity_mode: str | None = None,
) -> list[str]:
    """Replace target Linear layers with KBlockMixedLinear."""
    if isinstance(fixed_low_positions, str):
        fixed_low_positions = [int(x) for x in fixed_low_positions.split(",") if x.strip()]

    selected_positions: set = set()
    sens_pattern_len: int | None = None

    if placement == "sensitivity":
        if not sensitivity_score_csv:
            raise ValueError("placement='sensitivity' requires sensitivity_score_csv")
        if not sensitivity_mode:
            raise ValueError("placement='sensitivity' requires sensitivity_mode")

        import csv as _csv

        rows = []
        with open(sensitivity_score_csv, newline="") as f:
            for row in _csv.DictReader(f):
                if row.get("mode") == sensitivity_mode:
                    rows.append(row)

        if not rows:
            raise ValueError(f"No rows for sensitivity_mode={sensitivity_mode!r} in {sensitivity_score_csv}")

        module_tile_counts: dict = {}
        for r in rows:
            tid = int(r["tile_id"])
            module_tile_counts[r["module"]] = max(module_tile_counts.get(r["module"], 0), tid + 1)
        sens_pattern_len = min(module_tile_counts.values())

        pos_scores: dict = defaultdict(list)
        for r in rows:
            pos_scores[int(r["tile_id"]) % sens_pattern_len].append(float(r["rel_mse"]))
        pos_mean = {p: sum(v) / len(v) for p, v in pos_scores.items()}

        n = sens_pattern_len
        keep = max(1, min((n * low_percent + 50) // 100, n)) if low_percent > 0 else 0
        selected_positions = {p for p, _ in sorted(pos_mean.items(), key=lambda kv: kv[1])[:keep]}
        print(f"[SENS] mode={sensitivity_mode} selected {len(selected_positions)}/{n} positions: {sorted(selected_positions)}")

    replaced = []
    for name, module in list(model.named_modules()):
        if not _TARGET.match(name) or not isinstance(module, nn.Linear):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        new_module = KBlockMixedLinear(
            module,
            k_tile=k_tile,
            low_percent=low_percent,
            placement=placement,
            physical_pattern_len=physical_pattern_len,
            high_config=high_config,
            low_config=low_config,
            precision_map=precision_map,
            acc_fp_exp_bits=acc_fp_exp_bits,
            acc_fp_mant_bits=acc_fp_mant_bits,
            out_fp_exp_bits=out_fp_exp_bits,
            out_fp_mant_bits=out_fp_mant_bits,
            scale_bias=scale_bias,
            use_erry_clip=use_erry_clip,
            use_rotation=use_rotation,
            fixed_low_positions=fixed_low_positions,
            selected_low_tile_ids=selected_positions if placement == "sensitivity" else None,
            sensitivity_pattern_len=sens_pattern_len if placement == "sensitivity" else None,
        )
        setattr(model.get_submodule(parent_name), child_name, new_module)
        replaced.append(name)

    return replaced


# ------------------------------------------------------------------ #
# Audit helper                                                         #
# ------------------------------------------------------------------ #

def summarize_physical_mapping(model: nn.Module, max_modules: int | None = None) -> list[str]:
    lines, n_seen = [], 0
    for name, module in model.named_modules():
        if not isinstance(module, KBlockMixedLinear):
            continue
        if max_modules is not None and n_seen >= max_modules:
            break
        active = sorted(module._active_physical_positions())
        entries = []
        for tile_id in range(module.num_tiles()):
            pos = module.physical_position_for_tile(tile_id)
            cfg = module._config_for_tile(tile_id)
            mode = _mode(cfg)
            if mode == "mxint":
                prec = f"MXINT{cfg['element_bits']}Q{cfg['element_frac_bits']}"
            elif mode == "mxfp":
                prec = f"E{cfg['element_exp_bits']}M{cfg['element_frac_bits']}"
            else:
                prec = "FP"
            source = "map" if (module.precision_map and pos in module.precision_map) else (
                "low" if module.use_low_tile(tile_id) else "high"
            )
            entries.append(f"{tile_id}->p{pos}:{source}:{prec}")
        lines.append(
            f"{name}: pattern_len={module._pattern_len()}, active={active}, "
            f"tiles=[{', '.join(entries)}]"
        )
        n_seen += 1
    return lines


# ------------------------------------------------------------------ #
# Calibration                                                          #
# ------------------------------------------------------------------ #

_P_CANDIDATES = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
_ROTATION_THRESHOLD = 0.005


def _qdq_w_for_clip(w_blk: torch.Tensor, cfg: dict, p: float, scale_bias: int) -> torch.Tensor:
    """Quantize-dequantize a weight block with scalar clip factor p."""
    mode = _mode(cfg)
    if mode == "bypass":
        return w_blk.to(torch.float32)
    clip = torch.tensor(p, device=w_blk.device, dtype=torch.float32)
    if mode == "mxfp":
        return mxfp_quantizer(w_blk, cfg, clip_p=clip).to(torch.float32)
    # MXINT: quant then manual dequant
    block_size = cfg["block_size"]
    frac = cfg["element_frac_bits"]
    q, sb = mxint_quantizer_hw(w_blk, cfg, clip_p=clip, scale_bias=scale_bias)
    num_blocks = q.shape[-1] // block_size
    exp = sb - float(scale_bias)  # [..., num_blocks]
    scale = torch.pow(torch.tensor(2.0, device=w_blk.device, dtype=torch.float32), exp)
    q_b = q.reshape(*q.shape[:-1], num_blocks, block_size)
    dq = q_b / float(1 << frac) * scale.unsqueeze(-1)
    return dq.reshape(q.shape).to(torch.float32)


@torch.no_grad()
def calibrate_erry_clip_for_layer(
    layer: KBlockMixedLinear,
    calib_x: torch.Tensor,
    device: str,
):
    """Per-tile per-block erry-clip calibration. Sets layer.clip_p_table."""
    # Infer block_size from configs
    block_size = None
    for cfg in [layer.high_config, layer.low_config, *(layer.precision_map or {}).values()]:
        if cfg.get("block_size"):
            block_size = cfg["block_size"]
            break
    if block_size is None:
        return

    calib_x = calib_x.to(device, torch.float32)
    weight = layer.weight.to(device, torch.float32)
    num_blocks = layer.k_tile // block_size
    clip_table = torch.ones(layer.num_tiles(), num_blocks, device=device)

    for tile_id in range(layer.num_tiles()):
        k0, k1 = layer._tile_range(tile_id)
        cfg = layer._config_for_tile(tile_id)
        if _mode(cfg) == "bypass":
            continue

        for blk_id in range(num_blocks):
            bk0, bk1 = k0 + blk_id * block_size, k0 + (blk_id + 1) * block_size
            if bk1 > k1:
                continue
            x_blk = calib_x[:, bk0:bk1]
            w_blk = weight[:, bk0:bk1]
            y_ref = x_blk @ w_blk.T

            best_p, best_err = 1.0, float("inf")
            for p in _P_CANDIDATES:
                w_q = _qdq_w_for_clip(w_blk, cfg, p, layer.scale_bias)
                err = ((y_ref - x_blk @ w_q.T) ** 2).sum().item()
                if err < best_err:
                    best_err, best_p = err, p
            clip_table[tile_id, blk_id] = best_p

    layer.clip_p_table = clip_table


@torch.no_grad()
def selective_rotation_search_for_layer(
    layer: KBlockMixedLinear,
    calib_x: torch.Tensor,
    device: str,
):
    """Decide whether local low-tile Hadamard rotation helps this layer."""
    if not layer._low_tile_ids():
        layer.rotate_input = False
        return False, 0.0

    calib_x = calib_x.to(device, torch.float32)
    if calib_x.shape[0] > 512:
        calib_x = calib_x[torch.randperm(calib_x.shape[0])[:512]]
    y_ref = calib_x @ layer.weight.to(device, torch.float32).T

    layer.rotate_input = False
    y_no_rot = layer.forward(calib_x.to(layer.weight.dtype)).to(torch.float32)
    err_no = ((y_ref - y_no_rot) ** 2).mean().item()

    layer.rotate_input = True
    w_backup = layer.weight.data.clone()
    layer.weight_is_rotated = False
    layer.apply_rotation_to_weight()
    y_rot = layer.forward(calib_x.to(layer.weight.dtype)).to(torch.float32)
    err_rot = ((y_ref - y_rot) ** 2).mean().item()

    layer.weight.data.copy_(w_backup)
    layer.weight_is_rotated = False

    improvement = (err_no - err_rot) / max(err_no, 1e-12)
    decision = improvement >= _ROTATION_THRESHOLD
    layer.rotate_input = decision
    if decision:
        layer.apply_rotation_to_weight()
    return decision, improvement


@torch.no_grad()
def run_calibration(
    model: nn.Module,
    calib_batches,
    device: str = "cuda:0",
    use_erry_clip: bool = True,
    use_rotation: bool = True,
    max_samples: int = 1024,
    verbose: bool = True,
) -> nn.Module:
    """Capture activations and run per-layer erry-clip + rotation calibration."""
    if not (use_erry_clip or use_rotation):
        return model

    captured: dict = {
        name: [] for name, m in model.named_modules()
        if isinstance(m, KBlockMixedLinear)
    }
    handles = []
    for name, module in model.named_modules():
        if name not in captured:
            continue
        def _hook(n):
            def hook(mod, inp, out):
                captured[n].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
            return hook
        handles.append(module.register_forward_hook(_hook(name)))

    model.eval()
    for i, batch in enumerate(calib_batches):
        inputs = batch.to(device) if isinstance(batch, torch.Tensor) else batch["input_ids"].to(device)
        model(inputs)
        if verbose and (i + 1) % 4 == 0:
            print(f"[Calib] batch {i+1}/{len(calib_batches)}")
    for h in handles:
        h.remove()

    n_rot = 0
    for n_done, (name, module) in enumerate(
        (n, m) for n, m in model.named_modules() if isinstance(m, KBlockMixedLinear)
    ):
        acts = captured.pop(name, [])
        if not acts:
            continue
        calib_x = torch.cat(acts, 0)
        if calib_x.shape[0] > max_samples:
            calib_x = calib_x[torch.randperm(calib_x.shape[0])[:max_samples]]

        if use_erry_clip:
            calibrate_erry_clip_for_layer(module, calib_x, device)
        if use_rotation:
            decision, improv = selective_rotation_search_for_layer(module, calib_x, device)
            if decision:
                n_rot += 1
            if verbose and n_done < 5:
                print(f"[Calib]   {name}: rotation={decision} ({improv*100:.2f}%)")

    if verbose and use_rotation:
        n_total = sum(1 for _, m in model.named_modules() if isinstance(m, KBlockMixedLinear))
        print(f"[Calib] Done. rotation on {n_rot}/{n_total} layers")
    return model
