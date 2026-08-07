"""Static legality and cross-stack capability contracts for decode profiles.

Legality answers whether a precision profile can run at all; capability adds
the PackedKV runtime geometry each implementation layer requires.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from decode_dse.profiles import (
    ACCUMULATOR_RULE,
    VECTOR_FORMATS,
    FIXED_ACCUMULATOR_FRACTION_BITS,
    FIXED_ACCUMULATOR_INTEGER_BITS,
    MATRIX_SEMANTICS,
    MIXED_MATRIX_RULE,
    MX_BLOCK_SIZE,
    MXFP_MATRIX_RULE,
    MXINT_MATRIX_RULE,
    MX_SCALE_BITS,
    MX_SCALE_FORMAT,
    OUTPUT_RULE,
    PROFILE_KIND_BF16_REFERENCE,
    DecodePrecisionProfile,
    format_descriptor,
)

HARDWARE_MXINT_WEIGHT_FORMATS = ("MXINT4", "MXINT8")
HARDWARE_MXINT_OPERAND_FORMATS = ("MXINT2", "MXINT4", "MXINT8")
HARDWARE_MXFP_FORMATS = ("E1M2", "E2M1", "E4M3", "E5M2")


@dataclass(frozen=True)
class StackValidity:
    """Measured validity is independent for each implementation layer."""

    software_valid: bool | None = None
    compiler_valid: bool | None = None
    emulator_valid: bool | None = None
    rtl_valid: bool | None = None
    dc_calibrated: bool | None = None

    def __post_init__(self) -> None:
        for value in self.to_dict().values():
            _optional_bool(value)

    def to_dict(self) -> dict[str, bool | None]:
        return {
            "software_valid": self.software_valid,
            "compiler_valid": self.compiler_valid,
            "emulator_valid": self.emulator_valid,
            "rtl_valid": self.rtl_valid,
            "dc_calibrated": self.dc_calibrated,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "StackValidity":
        data = value or {}
        return cls(
            software_valid=_optional_bool(data.get("software_valid")),
            compiler_valid=_optional_bool(data.get("compiler_valid")),
            emulator_valid=_optional_bool(data.get("emulator_valid")),
            rtl_valid=_optional_bool(data.get("rtl_valid")),
            dc_calibrated=_optional_bool(data.get("dc_calibrated")),
        )

    def updated(self, **values: bool | None) -> "StackValidity":
        unknown = set(values) - set(self.to_dict())
        if unknown:
            raise ValueError(f"unknown validity fields: {sorted(unknown)}")
        return replace(self, **values)


def _optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise TypeError(f"validity values must be bool or None, got {value!r}")


@dataclass(frozen=True)
class LegalityIssue:
    code: str
    message: str
    stages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stages": list(self.stages),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LegalityIssue":
        return cls(
            code=str(value["code"]),
            message=str(value["message"]),
            stages=tuple(str(stage) for stage in value["stages"]),
        )


@dataclass(frozen=True)
class ProfileLegality:
    """Static eligibility does not claim that a stack validation has run."""

    software_supported: bool
    hardware_candidate: bool
    issues: tuple[LegalityIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "software_supported": self.software_supported,
            "hardware_candidate": self.hardware_candidate,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileLegality":
        software_supported = value["software_supported"]
        hardware_candidate = value["hardware_candidate"]
        if not isinstance(software_supported, bool) or not isinstance(
            hardware_candidate, bool
        ):
            raise TypeError("legality flags must be booleans")
        return cls(
            software_supported=software_supported,
            hardware_candidate=hardware_candidate,
            issues=tuple(
                LegalityIssue.from_dict(issue) for issue in value.get("issues", ())
            ),
        )


@dataclass(frozen=True)
class MatrixSemanticsBinding:
    """Structural arithmetic binding for one matrix operand pair."""

    operation: str
    left_role: str
    right_role: str
    family: str
    rule: str
    structurally_supported: bool
    numerical_trace_conformant: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "left_role": self.left_role,
            "right_role": self.right_role,
            "family": self.family,
            "rule": self.rule,
            "structurally_supported": self.structurally_supported,
            "numerical_trace_conformant": self.numerical_trace_conformant,
        }


def matrix_semantics_bindings(
    profile: DecodePrecisionProfile,
) -> tuple[MatrixSemanticsBinding, ...]:
    """Resolve matrix rules without treating structural support as trace evidence."""

    operation_roles = (
        ("linear", "activation", "weight"),
        ("qk", "activation", "key"),
        ("pv", "activation", "value"),
    )
    role_formats = {
        "weight": profile.weight_format,
        "activation": profile.activation_format,
        "key": profile.key_format,
        "value": profile.value_format,
    }
    bindings: list[MatrixSemanticsBinding] = []
    for operation, left_role, right_role in operation_roles:
        left_family = format_descriptor(role_formats[left_role]).family
        right_family = format_descriptor(role_formats[right_role]).family
        if left_family == right_family == "mxint":
            family = "mxint"
            rule = MXINT_MATRIX_RULE
            structurally_supported = True
        elif left_family == right_family == "mxfp":
            family = "mxfp"
            rule = MXFP_MATRIX_RULE
            structurally_supported = True
        else:
            family = "mixed"
            rule = MIXED_MATRIX_RULE
            structurally_supported = False
        bindings.append(
            MatrixSemanticsBinding(
                operation=operation,
                left_role=left_role,
                right_role=right_role,
                family=family,
                rule=rule,
                structurally_supported=structurally_supported,
            )
        )
    return tuple(bindings)


def evaluate_profile_legality(profile: DecodePrecisionProfile) -> ProfileLegality:
    """Apply the conservative PLENA hardware-candidate policy."""

    issues: list[LegalityIssue] = []
    if profile.kind == PROFILE_KIND_BF16_REFERENCE:
        issues.append(
            LegalityIssue(
                "reference_only",
                "The BF16 point is a numerical reference, not a deployment candidate.",
                ("compiler", "emulator", "rtl", "dc"),
            )
        )
        return ProfileLegality(True, False, tuple(issues))

    if (
        profile.matrix_semantics != MATRIX_SEMANTICS
        or profile.accumulator_rule != ACCUMULATOR_RULE
        or profile.output_rule != OUTPUT_RULE
    ):
        issues.append(
            LegalityIssue(
                "unsupported_matrix_semantics",
                "The matrix arithmetic contract does not match the PLENA datapath.",
                ("compiler", "emulator", "rtl", "dc"),
            )
        )

    descriptors = tuple(
        format_descriptor(token)
        for token in (
            profile.weight_format,
            profile.activation_format,
            profile.key_format,
            profile.value_format,
        )
    )
    families = {descriptor.family for descriptor in descriptors}
    if len(families) != 1:
        issues.append(
            LegalityIssue(
                "mixed_mx_families",
                "The matrix datapath requires one MX family across W, A, K, and V.",
                ("compiler", "emulator", "rtl", "dc"),
            )
        )
    elif families == {"mxint"}:
        if profile.weight_format not in HARDWARE_MXINT_WEIGHT_FORMATS:
            issues.append(
                LegalityIssue(
                    "unsupported_mxint_weight",
                    f"{profile.weight_format} is numerical-only for decoder weights.",
                    ("compiler", "emulator", "rtl", "dc"),
                )
            )
        for role, token in (
            ("activation", profile.activation_format),
            ("key", profile.key_format),
            ("value", profile.value_format),
        ):
            if token not in HARDWARE_MXINT_OPERAND_FORMATS:
                issues.append(
                    LegalityIssue(
                        "unsupported_mxint_operand",
                        f"{token} is numerical-only for the {role} operand.",
                        ("compiler", "emulator", "rtl", "dc"),
                    )
                )
    elif families == {"mxfp"}:
        for role, token in (
            ("weight", profile.weight_format),
            ("activation", profile.activation_format),
            ("key", profile.key_format),
            ("value", profile.value_format),
        ):
            if token not in HARDWARE_MXFP_FORMATS:
                issues.append(
                    LegalityIssue(
                        "unsupported_mxfp_operand",
                        f"{token} is numerical-only for the {role} operand.",
                        ("compiler", "emulator", "rtl", "dc"),
                    )
                )

    if profile.block_size != MX_BLOCK_SIZE:
        issues.append(
            LegalityIssue(
                "unsupported_block_size",
                f"The matrix datapath requires block size {MX_BLOCK_SIZE}.",
                ("compiler", "emulator", "rtl", "dc"),
            )
        )
    if profile.scale_format != MX_SCALE_FORMAT or profile.scale_bits != MX_SCALE_BITS:
        issues.append(
            LegalityIssue(
                "unsupported_scale",
                "Quantized hardware profiles require an 8-bit E8M0 scale.",
                ("compiler", "emulator", "rtl", "dc"),
            )
        )
    return ProfileLegality(True, not issues, tuple(issues))


@dataclass(frozen=True)
class TensorPackingContract:
    """Physical constraints for one packed element and scale-plane transfer."""

    role: str
    element_count: int
    base_address_bytes: int
    scale_base_address_bytes: int
    hbm_bus_bits: int
    hbm_alignment_bytes: int
    require_full_blocks: bool = True

    def __post_init__(self) -> None:
        if self.role not in {"weight", "activation", "key", "value"}:
            raise ValueError(f"unsupported tensor role {self.role!r}")
        if self.element_count <= 0:
            raise ValueError("element_count must be positive")
        if self.base_address_bytes < 0 or self.scale_base_address_bytes < 0:
            raise ValueError("base addresses must be non-negative")
        if self.hbm_bus_bits <= 0 or self.hbm_bus_bits % 8:
            raise ValueError("hbm_bus_bits must be a positive whole-byte width")
        if self.hbm_alignment_bytes <= 0:
            raise ValueError("hbm_alignment_bytes must be positive")


def _role_format(profile: DecodePrecisionProfile, role: str) -> str:
    return {
        "weight": profile.weight_format,
        "activation": profile.activation_format,
        "key": profile.key_format,
        "value": profile.value_format,
    }[role]


def validate_tensor_packing(
    profile: DecodePrecisionProfile,
    contract: TensorPackingContract,
) -> tuple[LegalityIssue, ...]:
    """Validate block packing, plane sizes, addresses, and HBM bus width."""

    issues: list[LegalityIssue] = []
    descriptor = format_descriptor(_role_format(profile, contract.role))
    if descriptor.token == "BF16":
        block_count = 0
        scale_bits = 0
    else:
        block_count = math.ceil(contract.element_count / profile.block_size)
        scale_bits = block_count * profile.scale_bits
        if contract.require_full_blocks and contract.element_count % profile.block_size:
            issues.append(
                LegalityIssue(
                    "partial_mx_block",
                    f"{contract.role} length is not divisible by block size {profile.block_size}.",
                    ("compiler", "emulator", "rtl"),
                )
            )

    element_bits = descriptor.element_bits * contract.element_count
    if element_bits % 8 or scale_bits % 8:
        issues.append(
            LegalityIssue(
                "non_byte_aligned_plane",
                "Element and scale planes must each occupy a whole number of bytes.",
                ("compiler", "emulator", "rtl"),
            )
        )
    planes = [("element", contract.base_address_bytes)]
    if scale_bits:
        planes.append(("scale", contract.scale_base_address_bytes))
    for plane, address in planes:
        if address % contract.hbm_alignment_bytes:
            issues.append(
                LegalityIssue(
                    "misaligned_hbm_base",
                    f"The {contract.role} {plane} plane is not HBM aligned.",
                    ("compiler", "emulator", "rtl"),
                )
            )
    if contract.hbm_bus_bits < descriptor.element_bits * profile.block_size:
        issues.append(
            LegalityIssue(
                "hbm_bus_too_narrow",
                f"The HBM bus cannot transfer one {contract.role} MX block per beat.",
                ("compiler", "emulator", "rtl"),
            )
        )
    return tuple(issues)


def required_accumulator_bits(
    left_format: str,
    right_format: str,
    reduction_depth: int,
) -> int:
    """Return the required fixed-bank width for one PLENA matrix reduction."""

    if reduction_depth <= 0:
        raise ValueError("reduction_depth must be positive")
    left = format_descriptor(left_format)
    right = format_descriptor(right_format)
    if left.family != "mxint" or right.family != "mxint":
        return (
            FIXED_ACCUMULATOR_INTEGER_BITS
            + FIXED_ACCUMULATOR_FRACTION_BITS
        )
    block_depth = min(reduction_depth, MX_BLOCK_SIZE)
    widened_mac_bits = (
        left.element_bits
        + right.element_bits
        + math.ceil(math.log2(block_depth))
    )
    return max(
        widened_mac_bits,
        FIXED_ACCUMULATOR_INTEGER_BITS + FIXED_ACCUMULATOR_FRACTION_BITS,
    )


def validate_accumulator_width(
    profile: DecodePrecisionProfile,
    accumulator_bits: int,
    reduction_depth: int,
) -> tuple[LegalityIssue, ...]:
    """Check both W×A and KV×A accumulator width requirements."""

    if accumulator_bits <= 0:
        raise ValueError("accumulator_bits must be positive")
    required = max(
        required_accumulator_bits(
            profile.weight_format,
            profile.activation_format,
            reduction_depth,
        ),
        required_accumulator_bits(
            profile.key_format,
            profile.activation_format,
            reduction_depth,
        ),
        required_accumulator_bits(
            profile.value_format,
            profile.activation_format,
            reduction_depth,
        ),
    )
    if accumulator_bits >= required:
        return ()
    return (
        LegalityIssue(
            "accumulator_too_narrow",
            f"{accumulator_bits}-bit accumulation is below the required {required} bits.",
            ("compiler", "emulator", "rtl", "dc"),
        ),
    )



CAPABILITY_STAGES = ("software", "compiler", "emulator", "rtl", "dc")
RTL_MXINT_ACTIVATION_FORMATS = ("MXINT4", "MXINT8")


@dataclass(frozen=True)
class PackedKVRuntimeTarget:
    """Geometry and ISA fields required by one PackedKV realization."""

    mlen: int = 1024
    blen: int = 8
    hlen: int = 128
    batch: int = 1
    kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 8
    selector_bits: int = 4
    packed_kv: bool = True
    batched_attention: bool = True

    def __post_init__(self) -> None:
        for name in (
            "mlen",
            "blen",
            "hlen",
            "batch",
            "kv_heads",
            "head_dim",
            "block_size",
            "selector_bits",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "mlen": self.mlen,
            "blen": self.blen,
            "hlen": self.hlen,
            "batch": self.batch,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "block_size": self.block_size,
            "selector_bits": self.selector_bits,
            "packed_kv": self.packed_kv,
            "batched_attention": self.batched_attention,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PackedKVRuntimeTarget":
        return cls(
            mlen=int(value["mlen"]),
            blen=int(value.get("blen", 8)),
            hlen=int(value["hlen"]),
            batch=int(value.get("batch", 1)),
            kv_heads=int(value["kv_heads"]),
            head_dim=int(value["head_dim"]),
            block_size=int(value["block_size"]),
            selector_bits=int(value["selector_bits"]),
            packed_kv=bool(value["packed_kv"]),
            batched_attention=bool(value["batched_attention"]),
        )


DEFAULT_PACKED_KV_TARGET = PackedKVRuntimeTarget()


@dataclass(frozen=True)
class CrossStackCapability:
    """Structural support is distinct from completed runtime validation."""

    profile_id: str
    target: PackedKVRuntimeTarget
    issues: tuple[LegalityIssue, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id must be non-empty")

    def stage_supported(self, stage: str) -> bool:
        if stage not in CAPABILITY_STAGES:
            raise ValueError(f"unknown capability stage {stage!r}")
        return not any(stage in issue.stages for issue in self.issues)

    @property
    def stage_support(self) -> dict[str, bool]:
        return {
            stage: self.stage_supported(stage)
            for stage in CAPABILITY_STAGES
        }

    @property
    def validity_floor(self) -> StackValidity:
        support = self.stage_support
        return StackValidity(
            software_valid=None if support["software"] else False,
            compiler_valid=None if support["compiler"] else False,
            emulator_valid=None if support["emulator"] else False,
            rtl_valid=None if support["rtl"] else False,
            dc_calibrated=None if support["dc"] else False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "target": self.target.to_dict(),
            "stage_support": self.stage_support,
            "validity_floor": self.validity_floor.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _append_issue(
    issues: list[LegalityIssue],
    code: str,
    message: str,
    stages: tuple[str, ...],
) -> None:
    issue = LegalityIssue(code=code, message=message, stages=stages)
    if issue not in issues:
        issues.append(issue)


def _validate_packed_geometry(
    profile: DecodePrecisionProfile,
    target: PackedKVRuntimeTarget,
    issues: list[LegalityIssue],
) -> None:
    runtime_stages = ("compiler", "emulator", "rtl")
    if profile.block_size != target.block_size:
        _append_issue(
            issues,
            "runtime_block_mismatch",
            "The runtime MX block size does not match the precision profile.",
            runtime_stages + ("dc",),
        )
    if not target.packed_kv:
        return
    if target.mlen % target.block_size or target.head_dim % target.block_size:
        _append_issue(
            issues,
            "packedkv_block_alignment",
            "PackedKV rows and head groups must contain complete MX blocks.",
            runtime_stages,
        )
    if target.mlen % target.hlen:
        _append_issue(
            issues,
            "packedkv_hlen_tiling",
            "MLEN must contain an integral number of HLEN groups.",
            runtime_stages,
        )
    if target.head_dim != target.hlen:
        _append_issue(
            issues,
            "packedkv_selector_stride",
            "The selector stride is HLEN and must equal the KV head dimension.",
            runtime_stages,
        )
    if target.kv_heads * target.head_dim > target.mlen:
        _append_issue(
            issues,
            "packedkv_row_overflow",
            "The active KV heads do not fit in one packed MLEN row.",
            runtime_stages,
        )
    if target.mlen % target.head_dim:
        _append_issue(
            issues,
            "packedkv_head_slots",
            "MLEN must contain an integral number of KV head slots.",
            runtime_stages,
        )
    if target.kv_heads > 1 << target.selector_bits:
        _append_issue(
            issues,
            "packedkv_selector_encoding",
            "The instruction selector field cannot encode every KV head.",
            runtime_stages,
        )


def evaluate_stack_capability(
    profile: DecodePrecisionProfile,
    target: PackedKVRuntimeTarget = DEFAULT_PACKED_KV_TARGET,
) -> CrossStackCapability:
    """Return structural capability without claiming any validation passed."""

    issues: list[LegalityIssue] = []
    legality = evaluate_profile_legality(profile)
    if not legality.software_supported:
        _append_issue(
            issues,
            "software_profile_unsupported",
            "The numerical profile is outside the software implementation.",
            CAPABILITY_STAGES,
        )
    if not legality.hardware_candidate:
        for issue in legality.issues:
            _append_issue(issues, issue.code, issue.message, issue.stages)

    if profile.vector_format not in VECTOR_FORMATS:
        _append_issue(
            issues,
            "vector_format_unsupported",
            f"{profile.vector_format} is not a configured vector format.",
            ("compiler", "emulator", "rtl", "dc"),
        )

    _validate_packed_geometry(profile, target, issues)

    if profile.kind != PROFILE_KIND_BF16_REFERENCE and legality.hardware_candidate:
        descriptors = {
            role: format_descriptor(token)
            for role, token in (
                ("weight", profile.weight_format),
                ("activation", profile.activation_format),
                ("key", profile.key_format),
                ("value", profile.value_format),
            )
        }
        if target.packed_kv and target.batched_attention and any(
            descriptor.family != "mxint"
            for descriptor in descriptors.values()
        ):
            _append_issue(
                issues,
                "rtl_batched_mxfp_unsupported",
                "The packed batched attention selector is implemented only on the MXINT matrix path.",
                ("rtl",),
            )
        if (
            target.packed_kv
            and target.batched_attention
            and all(
                descriptor.family == "mxint"
                for descriptor in descriptors.values()
            )
            and target.block_size % target.blen != 0
        ):
            _append_issue(
                issues,
                "rtl_mxint_scale_segment_mismatch",
                "The MXINT MCU requires BLEN to divide the MX block size.",
                ("rtl",),
            )
        if (
            all(
                descriptor.family == "mxint"
                for descriptor in descriptors.values()
            )
            and profile.activation_format not in RTL_MXINT_ACTIVATION_FORMATS
        ):
            _append_issue(
                issues,
                "rtl_mxint_activation_requant_unvalidated",
                "The RTL FP-to-MXINT activation path is validated for MXINT4 and MXINT8.",
                ("rtl",),
            )

    return CrossStackCapability(
        profile_id=profile.profile_id,
        target=target,
        issues=tuple(issues),
    )


def merge_stack_validity(*values: StackValidity) -> StackValidity:
    """Merge observations so any unsupported or failed stage remains false."""

    fields = tuple(StackValidity().to_dict())
    merged: dict[str, bool | None] = {}
    for field in fields:
        observations = tuple(getattr(value, field) for value in values)
        if False in observations:
            merged[field] = False
        elif True in observations:
            merged[field] = True
        else:
            merged[field] = None
    return StackValidity(**merged)


def constrain_stack_validity(
    profile: DecodePrecisionProfile,
    observed: StackValidity,
    target: PackedKVRuntimeTarget = DEFAULT_PACKED_KV_TARGET,
) -> StackValidity:
    """Apply structural capability as a false-only floor to observations."""

    capability = evaluate_stack_capability(profile, target)
    return merge_stack_validity(capability.validity_floor, observed)


def scope_stack_validity(
    observed: StackValidity,
    *,
    evidence_target: PackedKVRuntimeTarget,
    runtime_target: PackedKVRuntimeTarget,
) -> StackValidity:
    """Retain successful hardware evidence only on its measured target."""

    runtime_fields = (
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
    )
    dc_fields = tuple(field for field in runtime_fields if field != "batch")
    exact_runtime_match = all(
        getattr(evidence_target, field) == getattr(runtime_target, field)
        for field in runtime_fields
    )
    compiler_emulator_match = exact_runtime_match
    dc_match = all(
        getattr(evidence_target, field) == getattr(runtime_target, field)
        for field in dc_fields
    )

    def scoped(value: bool | None, matches: bool) -> bool | None:
        if value is False:
            return False
        if value is True and not matches:
            return None
        return value

    return StackValidity(
        software_valid=observed.software_valid,
        compiler_valid=scoped(
            observed.compiler_valid,
            compiler_emulator_match,
        ),
        emulator_valid=scoped(
            observed.emulator_valid,
            compiler_emulator_match,
        ),
        rtl_valid=scoped(observed.rtl_valid, exact_runtime_match),
        dc_calibrated=scoped(observed.dc_calibrated, dc_match),
    )


def load_stack_validity(
    path: str | Path,
    *,
    scope_profile_ids: Sequence[str] | None = None,
    required_stages: Sequence[str] = (),
    run_plan_hash: str | None = None,
    manifest: Any | None = None,
    scope_name: str = "stack-validity",
) -> dict[str, StackValidity]:
    """Load measured per-profile validity and enforce the required stages.

    The file records one validity object per profile plus the run-plan and
    manifest hashes it was measured under; a hash mismatch means the
    measurements describe a different sweep and must not be reused. When a
    manifest is supplied it must expose ``canonical_hash``.
    """

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError(f"{scope_name} file must contain a JSON object")
    recorded_hash = document.get("run_plan_hash")
    if run_plan_hash is not None and recorded_hash != run_plan_hash:
        raise ValueError(f"{scope_name} was measured under a different run plan")
    if manifest is not None:
        recorded_manifest_hash = document.get("manifest_hash")
        if recorded_manifest_hash != manifest.canonical_hash:
            raise ValueError(
                f"{scope_name} was measured under a different sweep manifest"
            )
    records = document.get("profiles", document)
    if not isinstance(records, Mapping):
        raise TypeError(f"{scope_name} profiles must be a JSON object")

    validity = {
        str(profile_id): StackValidity.from_dict(record)
        for profile_id, record in records.items()
    }
    if scope_profile_ids is not None:
        scope = set(scope_profile_ids)
        unknown = sorted(set(validity) - scope)
        if unknown:
            raise ValueError(
                f"{scope_name} covers {len(unknown)} profiles outside its scope"
            )
        validity = {key: value for key, value in validity.items() if key in scope}

    unsupported = sorted(set(required_stages) - set(CAPABILITY_STAGES))
    if unsupported:
        raise ValueError(f"unknown validity stages: {unsupported}")
    for profile_id, measured in sorted(validity.items()):
        for stage in required_stages:
            if getattr(measured, f"{stage}_valid") is not True:
                raise ValueError(
                    f"{scope_name} profile {profile_id} has no passing {stage} stage"
                )
    return validity


# Retained call-site name for the sweep and executor entry points.
load_built_stack_validity = load_stack_validity


__all__ = [
    "CAPABILITY_STAGES",
    "DEFAULT_PACKED_KV_TARGET",
    "HARDWARE_MXFP_FORMATS",
    "HARDWARE_MXINT_OPERAND_FORMATS",
    "HARDWARE_MXINT_WEIGHT_FORMATS",
    "RTL_MXINT_ACTIVATION_FORMATS",
    "CrossStackCapability",
    "LegalityIssue",
    "MatrixSemanticsBinding",
    "PackedKVRuntimeTarget",
    "ProfileLegality",
    "StackValidity",
    "TensorPackingContract",
    "constrain_stack_validity",
    "evaluate_profile_legality",
    "evaluate_stack_capability",
    "load_built_stack_validity",
    "load_stack_validity",
    "matrix_semantics_bindings",
    "merge_stack_validity",
    "required_accumulator_bits",
    "scope_stack_validity",
    "validate_accumulator_width",
    "validate_tensor_packing",
]
