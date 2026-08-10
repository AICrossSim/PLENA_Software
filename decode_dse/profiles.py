"""Canonical precision profiles for the exhaustive decode sweep."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

PROFILE_SCHEMA = "decode-precision-profile"
MATRIX_SEMANTICS_SCHEMA = "plena-matrix-semantics"
MX_BLOCK_SIZE = 8
MX_SCALE_FORMAT = "E8M0"
MX_SCALE_BITS = 8
MXINT_MATRIX_RULE = "block8_range_safe_scale_widened_mac_max_shift16_rne_vector"
MXFP_MATRIX_RULE = "product_cast_to_m_fp_then_fixed16_16_bank"
MIXED_MATRIX_RULE = "deployment_unsupported_without_trace_evidence"
ACCUMULATOR_RULE = "plena_fixed16_16_accumulate_truncate"
OUTPUT_RULE = "truncate_to_vector_format"
MXINT_MAX_SHIFT = 16
FIXED_ACCUMULATOR_INTEGER_BITS = 16
FIXED_ACCUMULATOR_FRACTION_BITS = 16
M_FP_FORMAT_BINDING = "profile.vector_format"
MATRIX_STORAGE_FP_BINDING = "profile.vector_format"
MATRIX_INSTRUCTION_K_PARTITION = "MLEN"
QK_LOGICAL_K_PARTITION = "HLEN"
MXINT_PARTIAL_CONVERSION = "per_mm_ic_integer_reduction_to_vector_storage_fp"
MXINT_CROSS_INSTRUCTION_ACCUMULATION = "signed_fixed16_16_wraparound"

MXINT_FORMATS = ("MXINT2", "MXINT4", "MXINT8")
MXFP_FORMATS = ("E1M2", "E2M1", "E3M4", "E4M3", "E5M2")
DECODE_FORMATS = MXINT_FORMATS + MXFP_FORMATS
VECTOR_FP_FORMATS = (
    "FP_E3M2",
    "FP_E2M3",
    "FP_E6M5",
    "FP_E5M6",
    "FP_E4M7",
    "FP_E8M5",
)
VECTOR_FORMATS = VECTOR_FP_FORMATS + ("BF16",)

PROFILE_KIND_QUANTIZED = "quantized"
PROFILE_KIND_VECTOR_BF16_CONTROL = "vector_bf16_control"
PROFILE_KIND_BF16_REFERENCE = "bf16_reference"
PROFILE_KINDS = (
    PROFILE_KIND_QUANTIZED,
    PROFILE_KIND_VECTOR_BF16_CONTROL,
    PROFILE_KIND_BF16_REFERENCE,
)

DEFAULT_WEIGHT_OPERATORS = (
    "attention_linear",
    "ffn_linear",
)
DEFAULT_ACTIVATION_OPERATORS = (
    "attention_linear",
    "ffn_linear",
    "qk_matmul",
    "pv_matmul",
)
DEFAULT_KV_OPERATORS = (
    "kv_cache",
    "qk_matmul",
    "pv_matmul",
)
DEFAULT_VECTOR_OPERATORS = (
    "input_rmsnorm",
    "post_attention_rmsnorm",
    "q_norm",
    "k_norm",
    "rope",
    "softmax",
    "silu_gate",
    "residual",
    "final_rmsnorm",
)
DEFAULT_BF16_OPERATORS = (
    "embedding",
    "lm_head",
)


@dataclass(frozen=True)
class FormatDescriptor:
    """Structural properties of one canonical numeric format."""

    token: str
    family: str
    element_bits: int
    exponent_bits: int | None = None
    mantissa_bits: int | None = None
    signed: bool = True
    block_scaled: bool = False


def _make_descriptors() -> dict[str, FormatDescriptor]:
    descriptors: dict[str, FormatDescriptor] = {}
    for token in MXINT_FORMATS:
        width = int(token.removeprefix("MXINT"))
        descriptors[token] = FormatDescriptor(
            token=token,
            family="mxint",
            element_bits=width,
            signed=True,
            block_scaled=True,
        )
    for token in MXFP_FORMATS:
        exponent, mantissa = token.removeprefix("E").split("M")
        descriptors[token] = FormatDescriptor(
            token=token,
            family="mxfp",
            element_bits=1 + int(exponent) + int(mantissa),
            exponent_bits=int(exponent),
            mantissa_bits=int(mantissa),
            signed=True,
            block_scaled=True,
        )
    for token in VECTOR_FP_FORMATS:
        exponent, mantissa = token.removeprefix("FP_E").split("M")
        descriptors[token] = FormatDescriptor(
            token=token,
            family="fp",
            element_bits=1 + int(exponent) + int(mantissa),
            exponent_bits=int(exponent),
            mantissa_bits=int(mantissa),
            signed=True,
            block_scaled=False,
        )
    descriptors["BF16"] = FormatDescriptor(
        token="BF16",
        family="bf16",
        element_bits=16,
        exponent_bits=8,
        mantissa_bits=7,
        signed=True,
        block_scaled=False,
    )
    return descriptors


FORMAT_DESCRIPTORS: Mapping[str, FormatDescriptor] = MappingProxyType(
    _make_descriptors()
)


def format_descriptor(token: str) -> FormatDescriptor:
    """Return format metadata without converting formats to effective bits."""

    canonical = str(token).upper()
    if canonical in MXFP_FORMATS or canonical in MXINT_FORMATS or canonical == "BF16":
        pass
    elif canonical.startswith("FP_"):
        pass
    elif canonical in {token.removeprefix("FP_") for token in VECTOR_FP_FORMATS}:
        canonical = f"FP_{canonical}"
    try:
        return FORMAT_DESCRIPTORS[canonical]
    except KeyError as exc:
        raise ValueError(f"unsupported precision format {token!r}") from exc


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class MatrixSemanticsContract:
    """Versioned matrix arithmetic rules implemented by the PLENA datapath."""

    schema_version: str = MATRIX_SEMANTICS_SCHEMA
    block_size: int = MX_BLOCK_SIZE
    mxint_rule: str = MXINT_MATRIX_RULE
    mxint_max_shift: int = MXINT_MAX_SHIFT
    mxint_vector_rounding: str = "round_to_nearest_even"
    mxint_partial_conversion: str = MXINT_PARTIAL_CONVERSION
    mxint_cross_instruction_accumulation: str = (
        MXINT_CROSS_INSTRUCTION_ACCUMULATION
    )
    mxfp_rule: str = MXFP_MATRIX_RULE
    m_fp_format_binding: str = M_FP_FORMAT_BINDING
    matrix_storage_fp_binding: str = MATRIX_STORAGE_FP_BINDING
    matrix_instruction_k_partition: str = MATRIX_INSTRUCTION_K_PARTITION
    qk_logical_k_partition: str = QK_LOGICAL_K_PARTITION
    fixed_accumulator_integer_bits: int = FIXED_ACCUMULATOR_INTEGER_BITS
    fixed_accumulator_fraction_bits: int = FIXED_ACCUMULATOR_FRACTION_BITS
    accumulator_rule: str = ACCUMULATOR_RULE
    output_rule: str = OUTPUT_RULE
    mixed_family_rule: str = MIXED_MATRIX_RULE
    mixed_family_deployment_supported: bool = False

    def __post_init__(self) -> None:
        expected = {
            "schema_version": MATRIX_SEMANTICS_SCHEMA,
            "block_size": MX_BLOCK_SIZE,
            "mxint_rule": MXINT_MATRIX_RULE,
            "mxint_max_shift": MXINT_MAX_SHIFT,
            "mxint_vector_rounding": "round_to_nearest_even",
            "mxint_partial_conversion": MXINT_PARTIAL_CONVERSION,
            "mxint_cross_instruction_accumulation": (
                MXINT_CROSS_INSTRUCTION_ACCUMULATION
            ),
            "mxfp_rule": MXFP_MATRIX_RULE,
            "m_fp_format_binding": M_FP_FORMAT_BINDING,
            "matrix_storage_fp_binding": MATRIX_STORAGE_FP_BINDING,
            "matrix_instruction_k_partition": MATRIX_INSTRUCTION_K_PARTITION,
            "qk_logical_k_partition": QK_LOGICAL_K_PARTITION,
            "fixed_accumulator_integer_bits": FIXED_ACCUMULATOR_INTEGER_BITS,
            "fixed_accumulator_fraction_bits": FIXED_ACCUMULATOR_FRACTION_BITS,
            "accumulator_rule": ACCUMULATOR_RULE,
            "output_rule": OUTPUT_RULE,
            "mixed_family_rule": MIXED_MATRIX_RULE,
            "mixed_family_deployment_supported": False,
        }
        for field, required in expected.items():
            if getattr(self, field) != required:
                raise ValueError(
                    f"unsupported matrix semantics field {field!r}: "
                    f"{getattr(self, field)!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "block_size": self.block_size,
            "mxint_rule": self.mxint_rule,
            "mxint_max_shift": self.mxint_max_shift,
            "mxint_vector_rounding": self.mxint_vector_rounding,
            "mxint_partial_conversion": self.mxint_partial_conversion,
            "mxint_cross_instruction_accumulation": (
                self.mxint_cross_instruction_accumulation
            ),
            "mxfp_rule": self.mxfp_rule,
            "m_fp_format_binding": self.m_fp_format_binding,
            "matrix_storage_fp_binding": self.matrix_storage_fp_binding,
            "matrix_instruction_k_partition": (
                self.matrix_instruction_k_partition
            ),
            "qk_logical_k_partition": self.qk_logical_k_partition,
            "fixed_accumulator_integer_bits": self.fixed_accumulator_integer_bits,
            "fixed_accumulator_fraction_bits": self.fixed_accumulator_fraction_bits,
            "accumulator_rule": self.accumulator_rule,
            "output_rule": self.output_rule,
            "mixed_family_rule": self.mixed_family_rule,
            "mixed_family_deployment_supported": (
                self.mixed_family_deployment_supported
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatrixSemanticsContract":
        expected_fields = set(cls().to_dict())
        if set(value) != expected_fields:
            raise ValueError("matrix semantics fields differ from the schema")
        deployment_supported = value["mixed_family_deployment_supported"]
        if not isinstance(deployment_supported, bool):
            raise TypeError("mixed_family_deployment_supported must be boolean")
        return cls(
            schema_version=str(value["schema_version"]),
            block_size=int(value["block_size"]),
            mxint_rule=str(value["mxint_rule"]),
            mxint_max_shift=int(value["mxint_max_shift"]),
            mxint_vector_rounding=str(value["mxint_vector_rounding"]),
            mxint_partial_conversion=str(value["mxint_partial_conversion"]),
            mxint_cross_instruction_accumulation=str(
                value["mxint_cross_instruction_accumulation"]
            ),
            mxfp_rule=str(value["mxfp_rule"]),
            m_fp_format_binding=str(value["m_fp_format_binding"]),
            matrix_storage_fp_binding=str(
                value["matrix_storage_fp_binding"]
            ),
            matrix_instruction_k_partition=str(
                value["matrix_instruction_k_partition"]
            ),
            qk_logical_k_partition=str(value["qk_logical_k_partition"]),
            fixed_accumulator_integer_bits=int(
                value["fixed_accumulator_integer_bits"]
            ),
            fixed_accumulator_fraction_bits=int(
                value["fixed_accumulator_fraction_bits"]
            ),
            accumulator_rule=str(value["accumulator_rule"]),
            output_rule=str(value["output_rule"]),
            mixed_family_rule=str(value["mixed_family_rule"]),
            mixed_family_deployment_supported=deployment_supported,
        )


MATRIX_SEMANTICS = MatrixSemanticsContract()


@dataclass(frozen=True)
class DecodePrecisionProfile:
    """Immutable numerical contract shared by every decode subsystem."""

    kind: str
    weight_format: str
    activation_format: str
    key_format: str
    value_format: str
    vector_format: str
    block_size: int = MX_BLOCK_SIZE
    scale_format: str = MX_SCALE_FORMAT
    scale_bits: int = MX_SCALE_BITS
    accumulator_rule: str = ACCUMULATOR_RULE
    output_rule: str = OUTPUT_RULE
    matrix_semantics: MatrixSemanticsContract = MATRIX_SEMANTICS
    method: str = "rtn"
    weight_operators: tuple[str, ...] = DEFAULT_WEIGHT_OPERATORS
    activation_operators: tuple[str, ...] = DEFAULT_ACTIVATION_OPERATORS
    kv_operators: tuple[str, ...] = DEFAULT_KV_OPERATORS
    vector_operators: tuple[str, ...] = DEFAULT_VECTOR_OPERATORS
    bf16_operators: tuple[str, ...] = DEFAULT_BF16_OPERATORS
    schema_version: str = PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA:
            raise ValueError(f"unsupported profile schema {self.schema_version!r}")
        if self.kind not in PROFILE_KINDS:
            raise ValueError(f"unsupported profile kind {self.kind!r}")
        if self.block_size != MX_BLOCK_SIZE:
            raise ValueError(f"decode sweep requires block size {MX_BLOCK_SIZE}")
        if self.method != "rtn":
            raise ValueError("the exhaustive sweep foundation requires RTN")
        if self.key_format != self.value_format:
            raise ValueError("K and V must use the same storage format")
        if self.accumulator_rule != ACCUMULATOR_RULE:
            raise ValueError(f"unsupported accumulator rule {self.accumulator_rule!r}")
        if self.output_rule != OUTPUT_RULE:
            raise ValueError(f"unsupported output rule {self.output_rule!r}")
        if self.matrix_semantics != MATRIX_SEMANTICS:
            raise ValueError("unsupported matrix semantics contract")
        if self.matrix_semantics.block_size != self.block_size:
            raise ValueError("profile and matrix-semantics block sizes differ")
        if self.matrix_semantics.accumulator_rule != self.accumulator_rule:
            raise ValueError("profile and matrix-semantics accumulator rules differ")
        if self.matrix_semantics.output_rule != self.output_rule:
            raise ValueError("profile and matrix-semantics output rules differ")

        if self.kind == PROFILE_KIND_BF16_REFERENCE:
            if any(
                token != "BF16"
                for token in (
                    self.weight_format,
                    self.activation_format,
                    self.key_format,
                    self.value_format,
                    self.vector_format,
                )
            ):
                raise ValueError("the BF16 reference must use BF16 for every role")
            if self.scale_format != "NONE" or self.scale_bits != 0:
                raise ValueError("the BF16 reference has no shared MX scale")
            return

        for role, token in (
            ("weight", self.weight_format),
            ("activation", self.activation_format),
            ("key", self.key_format),
            ("value", self.value_format),
        ):
            if token not in DECODE_FORMATS:
                raise ValueError(f"{role} format {token!r} is outside the decode sweep")
        if self.scale_format != MX_SCALE_FORMAT or self.scale_bits != MX_SCALE_BITS:
            raise ValueError("quantized profiles require an 8-bit E8M0 shared scale")
        if self.kind == PROFILE_KIND_QUANTIZED and self.vector_format not in VECTOR_FP_FORMATS:
            raise ValueError("quantized profiles require one canonical vector FP setting")
        if self.kind == PROFILE_KIND_VECTOR_BF16_CONTROL and self.vector_format != "BF16":
            raise ValueError("vector controls must keep vector operations in BF16")

    @classmethod
    def quantized(
        cls,
        weight_format: str,
        activation_format: str,
        kv_format: str,
        vector_format: str,
    ) -> "DecodePrecisionProfile":
        return cls(
            kind=PROFILE_KIND_QUANTIZED,
            weight_format=weight_format,
            activation_format=activation_format,
            key_format=kv_format,
            value_format=kv_format,
            vector_format=vector_format,
        )

    @classmethod
    def vector_bf16_control(
        cls,
        weight_format: str,
        activation_format: str,
        kv_format: str,
    ) -> "DecodePrecisionProfile":
        return cls(
            kind=PROFILE_KIND_VECTOR_BF16_CONTROL,
            weight_format=weight_format,
            activation_format=activation_format,
            key_format=kv_format,
            value_format=kv_format,
            vector_format="BF16",
        )

    @classmethod
    def bf16_reference(cls) -> "DecodePrecisionProfile":
        return cls(
            kind=PROFILE_KIND_BF16_REFERENCE,
            weight_format="BF16",
            activation_format="BF16",
            key_format="BF16",
            value_format="BF16",
            vector_format="BF16",
            scale_format="NONE",
            scale_bits=0,
        )

    @property
    def kv_format(self) -> str:
        return self.key_format

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "weight_format": self.weight_format,
            "activation_format": self.activation_format,
            "key_format": self.key_format,
            "value_format": self.value_format,
            "vector_format": self.vector_format,
            "block_size": self.block_size,
            "scale_format": self.scale_format,
            "scale_bits": self.scale_bits,
            "accumulator_rule": self.accumulator_rule,
            "output_rule": self.output_rule,
            "matrix_semantics": self.matrix_semantics.to_dict(),
            "method": self.method,
            "operator_coverage": {
                "weight": list(self.weight_operators),
                "activation": list(self.activation_operators),
                "kv": list(self.kv_operators),
                "vector": list(self.vector_operators),
                "bf16": list(self.bf16_operators),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodePrecisionProfile":
        expected_fields = set(cls.quantized(
            "MXINT4", "MXINT4", "MXINT4", "FP_E3M2"
        ).to_dict())
        if set(value) != expected_fields:
            raise ValueError("profile fields differ from the schema")
        if value["schema_version"] != PROFILE_SCHEMA:
            raise ValueError(f"unsupported profile schema {value['schema_version']!r}")
        coverage = value["operator_coverage"]
        if not isinstance(coverage, Mapping) or set(coverage) != {
            "weight",
            "activation",
            "kv",
            "vector",
            "bf16",
        }:
            raise ValueError("operator_coverage fields differ from the schema")

        def operators(field: str) -> tuple[str, ...]:
            items = coverage[field]
            if (
                not isinstance(items, list)
                or not items
                or any(not isinstance(item, str) or not item for item in items)
            ):
                raise TypeError(f"operator_coverage.{field} must be a string list")
            return tuple(items)

        return cls(
            schema_version=str(value["schema_version"]),
            kind=str(value["kind"]),
            weight_format=str(value["weight_format"]),
            activation_format=str(value["activation_format"]),
            key_format=str(value["key_format"]),
            value_format=str(value["value_format"]),
            vector_format=str(value["vector_format"]),
            block_size=int(value["block_size"]),
            scale_format=str(value["scale_format"]),
            scale_bits=int(value["scale_bits"]),
            accumulator_rule=str(value["accumulator_rule"]),
            output_rule=str(value["output_rule"]),
            matrix_semantics=MatrixSemanticsContract.from_dict(
                value["matrix_semantics"]
            ),
            method=str(value["method"]),
            weight_operators=operators("weight"),
            activation_operators=operators("activation"),
            kv_operators=operators("kv"),
            vector_operators=operators("vector"),
            bf16_operators=operators("bf16"),
        )

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def profile_id(self) -> str:
        return f"dqp-{self.canonical_hash}"


@dataclass(frozen=True)
class DeclaredSearchSpace:
    """The pre-declared precision subspace a sweep enumerates exhaustively.

    Every axis is a canonical-order subsequence of the module's format
    constants, so the enumeration order of a subspace is exactly the
    canonical order with excluded formats absent. Any format missing from
    an axis must carry a disclosed rationale; the sweep remains an
    unpruned, deterministic enumeration over the space declared here.
    """

    weight_formats: tuple[str, ...]
    activation_formats: tuple[str, ...]
    kv_formats: tuple[str, ...]
    vector_formats: tuple[str, ...]
    exclusions: Mapping[str, Mapping[str, str]]

    @property
    def expected_quantized_profiles(self) -> int:
        return (
            len(self.weight_formats)
            * len(self.activation_formats)
            * len(self.kv_formats)
            * len(self.vector_formats)
        )

    @property
    def expected_vector_bf16_controls(self) -> int:
        return (
            len(self.weight_formats)
            * len(self.activation_formats)
            * len(self.kv_formats)
        )

    @property
    def expected_total_profiles(self) -> int:
        return self.expected_quantized_profiles + self.expected_vector_bf16_controls + 1

    @property
    def is_canonical(self) -> bool:
        return (
            self.weight_formats == DECODE_FORMATS
            and self.activation_formats == DECODE_FORMATS
            and self.kv_formats == DECODE_FORMATS
            and self.vector_formats == VECTOR_FP_FORMATS
        )


CANONICAL_SEARCH_SPACE = DeclaredSearchSpace(
    weight_formats=DECODE_FORMATS,
    activation_formats=DECODE_FORMATS,
    kv_formats=DECODE_FORMATS,
    vector_formats=VECTOR_FP_FORMATS,
    exclusions=MappingProxyType({}),
)

_SEARCH_AXES = (
    ("weight_w", "weight_formats", DECODE_FORMATS),
    ("act_w", "activation_formats", DECODE_FORMATS),
    ("kv", "kv_formats", DECODE_FORMATS),
    ("vector_fp", "vector_formats", VECTOR_FP_FORMATS),
)


def declared_search_space(search: Mapping[str, Any]) -> DeclaredSearchSpace:
    """Validate and freeze the search block's declared precision subspace.

    Each axis must be a non-empty, canonical-order subsequence of the
    canonical format tuple. Every excluded format needs a non-empty
    rationale under ``search.declared_exclusions[<axis>][<format>]``, and
    rationales for formats that are not actually excluded are rejected, so
    the config cannot drift from the space it claims to enumerate.
    """

    declared_exclusions = search.get("declared_exclusions", {})
    if not isinstance(declared_exclusions, Mapping):
        raise ValueError("search.declared_exclusions must be a mapping")
    axes: dict[str, tuple[str, ...]] = {}
    exclusions: dict[str, dict[str, str]] = {}
    for key, field_name, canonical in _SEARCH_AXES:
        declared = tuple(search.get(key, ()))
        if not declared:
            raise ValueError(f"search.{key} must declare at least one format")
        unknown = [value for value in declared if value not in canonical]
        if unknown:
            raise ValueError(f"search.{key} has unknown formats: {unknown}")
        canonical_subsequence = tuple(
            value for value in canonical if value in set(declared)
        )
        if declared != canonical_subsequence:
            raise ValueError(
                f"search.{key} must keep canonical format order: "
                f"expected {canonical_subsequence!r}, got {declared!r}"
            )
        excluded = tuple(value for value in canonical if value not in set(declared))
        rationales = declared_exclusions.get(key, {})
        if not isinstance(rationales, Mapping):
            raise ValueError(f"search.declared_exclusions.{key} must be a mapping")
        missing = [value for value in excluded if not str(rationales.get(value, "")).strip()]
        if missing:
            raise ValueError(
                f"search.{key} excludes {missing} without a disclosed rationale "
                "in search.declared_exclusions"
            )
        stray = [value for value in rationales if value not in excluded]
        if stray:
            raise ValueError(
                f"search.declared_exclusions.{key} names formats that are not "
                f"excluded: {stray}"
            )
        axes[field_name] = declared
        if excluded:
            exclusions[key] = {value: str(rationales[value]) for value in excluded}
    unknown_axes = [key for key in declared_exclusions if key not in dict(
        (axis, None) for axis, _, _ in _SEARCH_AXES
    )]
    if unknown_axes:
        raise ValueError(f"search.declared_exclusions has unknown axes: {unknown_axes}")
    return DeclaredSearchSpace(
        weight_formats=axes["weight_formats"],
        activation_formats=axes["activation_formats"],
        kv_formats=axes["kv_formats"],
        vector_formats=axes["vector_formats"],
        exclusions=MappingProxyType(
            {axis: MappingProxyType(dict(values)) for axis, values in exclusions.items()}
        ),
    )


def iter_quantized_profiles(
    space: DeclaredSearchSpace = CANONICAL_SEARCH_SPACE,
) -> Iterable[DecodePrecisionProfile]:
    """Yield the declared space's quantized profiles in canonical nested-loop order."""

    for weight_format in space.weight_formats:
        for activation_format in space.activation_formats:
            for kv_format in space.kv_formats:
                for vector_format in space.vector_formats:
                    yield DecodePrecisionProfile.quantized(
                        weight_format,
                        activation_format,
                        kv_format,
                        vector_format,
                    )


def iter_vector_bf16_controls(
    space: DeclaredSearchSpace = CANONICAL_SEARCH_SPACE,
) -> Iterable[DecodePrecisionProfile]:
    """Yield one BF16-vector control for every declared W/A/KV triple."""

    for weight_format in space.weight_formats:
        for activation_format in space.activation_formats:
            for kv_format in space.kv_formats:
                yield DecodePrecisionProfile.vector_bf16_control(
                    weight_format,
                    activation_format,
                    kv_format,
                )


def enumerate_decode_profiles(
    space: DeclaredSearchSpace = CANONICAL_SEARCH_SPACE,
) -> tuple[DecodePrecisionProfile, ...]:
    """Return the deterministic numerical sweep over the declared space."""

    return (
        *iter_quantized_profiles(space),
        *iter_vector_bf16_controls(space),
        DecodePrecisionProfile.bf16_reference(),
    )


__all__ = [
    "DECODE_FORMATS",
    "ACCUMULATOR_RULE",
    "FIXED_ACCUMULATOR_FRACTION_BITS",
    "FIXED_ACCUMULATOR_INTEGER_BITS",
    "FORMAT_DESCRIPTORS",
    "MATRIX_SEMANTICS",
    "MATRIX_SEMANTICS_SCHEMA",
    "MIXED_MATRIX_RULE",
    "M_FP_FORMAT_BINDING",
    "MATRIX_STORAGE_FP_BINDING",
    "MATRIX_INSTRUCTION_K_PARTITION",
    "QK_LOGICAL_K_PARTITION",
    "MXINT_PARTIAL_CONVERSION",
    "MXINT_CROSS_INSTRUCTION_ACCUMULATION",
    "MX_BLOCK_SIZE",
    "MXFP_MATRIX_RULE",
    "MXINT_MATRIX_RULE",
    "MXINT_MAX_SHIFT",
    "MX_SCALE_BITS",
    "MX_SCALE_FORMAT",
    "MXFP_FORMATS",
    "MXINT_FORMATS",
    "PROFILE_KIND_BF16_REFERENCE",
    "PROFILE_KIND_QUANTIZED",
    "PROFILE_KIND_VECTOR_BF16_CONTROL",
    "PROFILE_SCHEMA",
    "VECTOR_FORMATS",
    "VECTOR_FP_FORMATS",
    "CANONICAL_SEARCH_SPACE",
    "DeclaredSearchSpace",
    "declared_search_space",
    "DecodePrecisionProfile",
    "FormatDescriptor",
    "MatrixSemanticsContract",
    "OUTPUT_RULE",
    "enumerate_decode_profiles",
    "format_descriptor",
    "iter_quantized_profiles",
    "iter_vector_bf16_controls",
]
