"""Evaluate one decode precision profile with a cached one-token loop."""

from __future__ import annotations

import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from decode_dse.legality import (
    constrain_stack_validity,
    merge_stack_validity,
)
from decode_dse.legality import StackValidity
from decode_dse.manifest import SweepManifestEntry
from decode_dse.profiles import (
    DECODE_FORMATS,
    PROFILE_KIND_BF16_REFERENCE,
    DecodePrecisionProfile,
    format_descriptor,
)
from decode_dse.software.cached_decode import (
    ContinuationExample,
    TorchHFCachedDecodeBackend,
    _legacy_cache_layers,
    evaluate_teacher_forced_cached_batched,
)
from decode_dse.software.sweep_runner import EvaluationOutcome
from decode_dse.software.sweep_plan import (
    ExecutorContext,
    _mase_tree_hash,
    _software_tree_hash,
    profile_to_decode_quant_spec,
)
from decode_dse.software.token_samples import (
    DecodeTokenSample,
    load_sample_bundle,
)
from decode_dse.software.runtime_environment import (
    RuntimeEnvironment,
    capture_runtime_environment,
    estimate_artifact_footprint,
    initialize_numerical_runtime,
    require_runtime_environment,
    seal_runtime_environment,
)
from decode_dse.software.cache_artifacts import (
    ArtifactProvenance,
    BF16CacheConverter,
    DecodeCacheArtifact,
    FunctionalCacheConverter,
    QuantizedTensorPayload,
    TensorPayload,
    admit_prefill_cache,
    admit_prefill_cache_split,
    load_decode_cache_artifact,
    load_prefill_artifact,
    save_decode_cache_artifact,
)
from decode_dse.legality import (
    load_built_stack_validity,
)
from decode_dse.software.sweep_plan import load_immutable_json, resolve_bound_path

PACKED_CACHE_LAYOUT = "packed-gqa-mlen1024-block8-native-encoding"
_PACKED_CACHE_LAYOUT_RE = re.compile(
    r"^packed-gqa-mlen(?P<row_elements>[0-9]+)-block8-native-encoding$"
)
ADMISSION_CONVERTER_SCHEMA = "packedkv-admission"
ADMISSION_INDEX_SCHEMA = "decode-admission-index"
ADMISSION_PREPARATION_SCHEMA = "decode-admission-preparation"
ADMISSION_NUMERICAL_VALIDATION_SCHEMA = "decode-admission-numerical-validation"
E8M0_SCALE_BIAS = 127


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(
    value: str | os.PathLike[str],
    *,
    workspace_root: str | os.PathLike[str] | None = None,
) -> Path:
    return resolve_bound_path(
        value,
        repository_root=_repository_root(),
        workspace_root=workspace_root,
    )


def _document_token(document_id: str) -> str:
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()


def _prefill_path(root: Path, document_id: str) -> Path:
    return root / _document_token(document_id)


def _decode_cache_path(
    root: Path,
    key_format: str,
    document_id: str,
    value_format: str | None = None,
    contract_id: str | None = None,
) -> Path:
    value = key_format if value_format is None else value_format
    precision_path = key_format if key_format == value else f"K-{key_format}__V-{value}"
    namespace = root if contract_id is None else root / contract_id
    return namespace / precision_path / _document_token(document_id)


def _kv_precision_id(key_format: str, value_format: str) -> str:
    return (
        key_format if key_format == value_format else f"K={key_format};V={value_format}"
    )


def _artifact_bytes(artifact: DecodeCacheArtifact) -> int:
    return sum(
        len(tensor.element_plane)
        + len(tensor.scale_plane)
        + len(tensor.numerical_view.data)
        for layer in artifact.layers
        for tensor in (layer.key, layer.value)
    )


def _directory_bytes(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _available_host_bytes() -> int:
    value = Path("/proc/meminfo")
    if not value.is_file():
        raise RuntimeError("host-memory availability cannot be measured")
    for line in value.read_text(encoding="utf-8").splitlines():
        name, separator, payload = line.partition(":")
        if name == "MemAvailable" and separator:
            amount, unit = payload.strip().split()
            if unit != "kB":
                break
            return int(amount) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _packed_tensor_disk_components(
    element_count: int,
    format_id: str,
) -> dict[str, int]:
    """Return exact persisted planes for one logical cache tensor set."""

    if element_count <= 0:
        raise ValueError("element_count must be positive")
    descriptor = format_descriptor(format_id)
    if descriptor.family == "bf16":
        element_bytes = 2 * element_count
        scale_bytes = 0
    else:
        if element_count % 8:
            raise ValueError("native PackedKV tensors must be block-8 divisible")
        element_bytes = math.ceil(element_count * descriptor.element_bits / 8)
        scale_bytes = element_count // 8
    return {
        "element_plane_bytes": element_bytes,
        "scale_plane_bytes": scale_bytes,
        "numerical_view_bytes": 2 * element_count,
    }


def _packed_tensor_disk_bytes(element_count: int, format_id: str) -> int:
    """Return persisted element, scale, and BF16 numerical-view bytes."""

    return sum(_packed_tensor_disk_components(element_count, format_id).values())


def _validate_admission_resource_projection(
    value: Any,
) -> dict[str, int | float | str]:
    if not isinstance(value, Mapping):
        raise TypeError("admission resource projection must be an object")
    if value.get("persistence_contract") != "packed_planes_plus_bf16_numerical_view":
        raise ValueError("admission persistence contract differs")
    integer_fields = (
        "artifact_space_reserve_bytes",
        "projected_element_plane_bytes",
        "projected_scale_plane_bytes",
        "projected_numerical_view_bytes",
        "projected_metadata_reserve_bytes",
        "projected_cold_artifact_bytes",
        "required_cold_capacity_bytes",
        "observed_cold_available_bytes",
        "projected_peak_host_bytes",
        "required_host_bytes",
        "observed_host_available_bytes",
    )
    integers: dict[str, int] = {}
    for name in integer_fields:
        field = value.get(name)
        if isinstance(field, bool) or not isinstance(field, int) or field < 0:
            raise ValueError(f"admission resource {name} is invalid")
        integers[name] = field
    safety_factor = value.get("artifact_space_safety_factor")
    if (
        isinstance(safety_factor, bool)
        or not isinstance(safety_factor, (int, float))
        or not math.isfinite(float(safety_factor))
        or float(safety_factor) < 1.0
    ):
        raise ValueError("admission resource safety factor is invalid")
    projected = sum(
        integers[name]
        for name in (
            "projected_element_plane_bytes",
            "projected_scale_plane_bytes",
            "projected_numerical_view_bytes",
            "projected_metadata_reserve_bytes",
        )
    )
    minimum_capacity = (
        math.ceil(projected * float(safety_factor))
        + integers["artifact_space_reserve_bytes"]
    )
    if (
        projected <= 0
        or integers["projected_numerical_view_bytes"] <= 0
        or integers["projected_cold_artifact_bytes"] != projected
        or integers["required_cold_capacity_bytes"] < minimum_capacity
        or integers["observed_cold_available_bytes"]
        < integers["required_cold_capacity_bytes"]
        or integers["required_host_bytes"] < integers["projected_peak_host_bytes"]
        or integers["observed_host_available_bytes"] < integers["required_host_bytes"]
    ):
        raise ValueError("admission resource projection is inconsistent")
    return {
        "persistence_contract": str(value["persistence_contract"]),
        "artifact_space_safety_factor": float(safety_factor),
        **integers,
    }


def _pack_tensor_codes(codes: Any, width: int) -> bytes:
    """Pack unsigned fixed-width codes in flattened LSB-first order."""

    import torch

    if width <= 0 or width > 8:
        raise ValueError("element width must be in [1, 8]")
    values = codes.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if values.numel() == 0:
        return b""
    mask = (1 << width) - 1
    if torch.any((values < 0) | (values > mask)):
        raise ValueError("element code is outside its declared width")
    positions = torch.arange(values.numel(), dtype=torch.int64) * width
    byte_indices = torch.div(positions, 8, rounding_mode="floor")
    shifts = positions.remainder(8)
    byte_count = math.ceil(values.numel() * width / 8)
    packed = torch.zeros(byte_count, dtype=torch.int64)
    packed.scatter_add_(
        0,
        byte_indices,
        (values << shifts).bitwise_and(0xFF),
    )
    spills = shifts + width > 8
    if torch.any(spills):
        packed.scatter_add_(
            0,
            byte_indices[spills] + 1,
            values[spills] >> (8 - shifts[spills]),
        )
    return packed.to(torch.uint8).numpy().tobytes(order="C")


def _uint8_bytes(values: Any) -> bytes:
    import torch

    return (
        values.detach()
        .to(device="cpu", dtype=torch.int64)
        .bitwise_and(0xFF)
        .to(torch.uint8)
        .reshape(-1)
        .numpy()
        .tobytes(order="C")
    )


class MaseMXCacheConverter:
    """Create physical MX planes and the exact MASE numerical cache view."""

    def __init__(self, format_id: str, block_size: int = 8) -> None:
        descriptor = format_descriptor(format_id)
        if descriptor.family not in {"mxint", "mxfp"}:
            raise ValueError(f"{format_id!r} is not an MX cache format")
        self.format_id = descriptor.token
        self.descriptor = descriptor
        self.block_size = int(block_size)
        if self.block_size != 8:
            raise ValueError("the exhaustive sweep requires block size 8")

    def convert(
        self,
        tensor: TensorPayload,
        role: str,
        layer_index: int,
        precision_id: str,
        layout_id: str,
    ) -> QuantizedTensorPayload:
        del role, layer_index, precision_id
        layout = _PACKED_CACHE_LAYOUT_RE.fullmatch(layout_id)
        if layout is None:
            raise ValueError(f"unsupported numerical cache layout {layout_id!r}")
        import torch

        source = tensor.to_torch("cpu").contiguous()
        if source.dtype != torch.bfloat16:
            raise TypeError("decode admission requires a BF16 source tensor")
        row_elements = int(layout.group("row_elements"))
        if source.shape[1] * source.shape[3] != row_elements:
            raise ValueError("packed KV row width differs from the declared layout")
        if source.shape[-1] % self.block_size:
            raise ValueError("cache head dimension is not block divisible")
        physical = source.permute(0, 2, 1, 3).contiguous()
        blocks = physical.reshape(-1, self.block_size)

        if self.descriptor.family == "mxint":
            from chop.nn.quantizers.mxint.fake import (
                compose_mxint_tensor,
                extract_mxint_components,
                mxint_sign_magnitude_codes,
            )
            from chop.nn.quantizers.mxint.meta import MXIntMeta

            meta = MXIntMeta(
                block_size=self.block_size,
                scale_bits=8,
                element_bits=self.descriptor.element_bits,
            )
            scales, elements = extract_mxint_components(blocks, meta)
            numerical = compose_mxint_tensor(scales, elements, meta).reshape(
                physical.shape
            )
            element_codes = mxint_sign_magnitude_codes(
                elements,
                self.descriptor.element_bits,
            )
            element_plane = _pack_tensor_codes(
                element_codes,
                self.descriptor.element_bits,
            )
            scale_plane = _uint8_bytes(scales)
            element_encoding = "MXINT_SIGN_MAGNITUDE_LSB"
        else:
            from chop.nn.quantizers._minifloat_mx import (
                extract_minifloat_component,
            )
            from chop.nn.quantizers.mxfp.fake import (
                compose_mxfp_tensor,
                extract_mxfp_components,
            )
            from chop.nn.quantizers.mxfp.meta import MXFPMeta

            meta = MXFPMeta(
                block_size=self.block_size,
                scale_exp_bits=8,
                element_exp_bits=int(self.descriptor.exponent_bits),
                element_frac_bits=int(self.descriptor.mantissa_bits),
                element_is_finite=self.descriptor.exponent_bits == 1,
                round_mode="rn",
            )
            scales, elements = extract_mxfp_components(blocks, meta)
            numerical = compose_mxfp_tensor(
                scales,
                elements,
                meta,
                output_dtype=torch.float32,
            ).reshape(physical.shape)
            element_codes = extract_minifloat_component(
                elements,
                meta.element_meta,
            )
            element_plane = _pack_tensor_codes(
                element_codes,
                self.descriptor.element_bits,
            )
            biased_scales = scales.to(torch.int64) + E8M0_SCALE_BIAS
            if torch.any((biased_scales < 0) | (biased_scales > 0xFF)):
                raise ValueError("MXFP shared exponent is outside E8M0 range")
            scale_plane = _uint8_bytes(biased_scales)
            element_encoding = "MXFP_IEEE_LSB"

        numerical_payload = TensorPayload.from_torch(
            numerical.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16),
            dtype="bfloat16",
        )
        return QuantizedTensorPayload(
            format_id=self.format_id,
            block_size=self.block_size,
            element_bits=self.descriptor.element_bits,
            logical_shape=tensor.shape,
            element_plane=element_plane,
            scale_plane=scale_plane,
            numerical_view=numerical_payload,
            element_encoding=element_encoding,
            scale_encoding="E8M0_BIAS127_U8",
        )


@dataclass(frozen=True)
class AdmissionCacheHandle:
    kv_format: str
    paths: Mapping[str, Path]
    value_format: str | None = None

    @property
    def key_format(self) -> str:
        return self.kv_format

    @property
    def resolved_value_format(self) -> str:
        return self.kv_format if self.value_format is None else self.value_format

    @property
    def split_kv(self) -> bool:
        return self.key_format != self.resolved_value_format


@dataclass(frozen=True)
class DecodeWeightBank:
    model: Any
    device: Any
    weight_format: str
    binding_plan: "DecodeBindingPlan"
    identity_guard: "DecodeWeightBankIdentity"
    quantization_guard: "DecodeWeightQuantizationGuard"
    build_seconds: float
    weight_method: str = "rtn"


@dataclass(frozen=True)
class DecodeBindingTarget:
    pattern_index: int
    name: str
    module: Any


@dataclass(frozen=True)
class DecodeBindingPlan:
    """Cache the modules selected by each quantization regex."""

    patterns: tuple[str, ...]
    targets: tuple[DecodeBindingTarget, ...]

    def resolve(
        self,
        pass_args: Mapping[str, Any],
    ) -> tuple[tuple[DecodeBindingTarget, Mapping[str, Any]], ...]:
        configs = _binding_configs(pass_args)
        actual_patterns = tuple(pattern for pattern, _ in configs)
        if actual_patterns != self.patterns:
            raise RuntimeError("decode binding patterns changed within a weight bank")
        return tuple(
            (target, configs[target.pattern_index][1]) for target in self.targets
        )


@dataclass(frozen=True)
class WeightQuantizationIdentity:
    name: str
    module: Any
    object_id: int
    events: int


@dataclass(frozen=True)
class DecodeWeightQuantizationGuard:
    """Seal decode linears and detect any later weight-bank reconstruction."""

    modules: tuple[WeightQuantizationIdentity, ...]

    @classmethod
    def capture(
        cls,
        binding_plan: DecodeBindingPlan,
        *,
        expected_modules: int,
    ) -> "DecodeWeightQuantizationGuard":
        records = []
        seen: set[int] = set()
        for target in binding_plan.targets:
            module = target.module
            seal = getattr(module, "seal_decode_weight_bank", None)
            if not callable(seal):
                continue
            if id(module) in seen:
                raise RuntimeError("decode binding plan repeats a weight module")
            seen.add(id(module))
            records.append(
                WeightQuantizationIdentity(
                    name=target.name,
                    module=module,
                    object_id=id(module),
                    events=int(seal()),
                )
            )
        if len(records) != expected_modules:
            raise RuntimeError(
                "decode weight-bank seal coverage mismatch: "
                f"{len(records)} != {expected_modules}"
            )
        return cls(modules=tuple(records))

    def verify(self) -> int:
        total = 0
        for record in self.modules:
            module = record.module
            if id(module) != record.object_id:
                raise RuntimeError(f"decode weight module was replaced: {record.name}")
            if not bool(getattr(module, "_decode_bank_sealed", False)):
                raise RuntimeError(f"decode weight bank was unsealed: {record.name}")
            events = int(getattr(module, "_weight_quantization_events", -1))
            if events != record.events:
                raise RuntimeError(f"decode weight bank was requantized: {record.name}")
            total += events
        return total


@dataclass(frozen=True)
class WeightParameterIdentity:
    name: str
    object_id: int
    data_pointer: int | None
    version: int | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "object_id": self.object_id,
            "data_pointer": self.data_pointer,
            "version": self.version,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
        }


@dataclass(frozen=True)
class DecodeWeightBankIdentity:
    """Detect parameter replacement, relocation, or in-place modification."""

    parameters: tuple[WeightParameterIdentity, ...]
    fingerprint: str
    structure_fingerprint: str

    @classmethod
    def capture(cls, model: Any) -> "DecodeWeightBankIdentity":
        parameters = tuple(
            _parameter_identity(name, parameter)
            for name, parameter in model.named_parameters()
        )
        if not parameters:
            raise RuntimeError("decode weight bank contains no parameters")
        names = tuple(parameter.name for parameter in parameters)
        if len(names) != len(set(names)):
            raise RuntimeError("decode weight bank contains duplicate parameter names")
        return cls(
            parameters=parameters,
            fingerprint=_identity_fingerprint(parameters, structural=False),
            structure_fingerprint=_identity_fingerprint(
                parameters,
                structural=True,
            ),
        )

    def verify(self, model: Any) -> str:
        current = tuple(
            _parameter_identity(name, parameter)
            for name, parameter in model.named_parameters()
        )
        fingerprint = _identity_fingerprint(current, structural=False)
        if current != self.parameters or fingerprint != self.fingerprint:
            mismatch = next(
                (
                    expected.name
                    for expected, actual in zip(self.parameters, current)
                    if expected != actual
                ),
                (
                    "<parameter-count>"
                    if len(current) != len(self.parameters)
                    else "<fingerprint>"
                ),
            )
            raise RuntimeError(
                f"decode weight bank changed during runtime rebinding: {mismatch}"
            )
        return fingerprint


@dataclass(frozen=True)
class BindingMeasurement:
    performed: bool
    seconds: float
    target_count: int
    used_cached_targets: bool
    weight_requantizations: int
    sealed_weight_modules: int
    weight_quantization_events_before: int
    weight_quantization_events_after: int
    identity_before: str
    identity_after: str
    structure_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "performed": self.performed,
            "seconds": self.seconds,
            "target_count": self.target_count,
            "used_cached_targets": self.used_cached_targets,
            "weight_requantizations": self.weight_requantizations,
            "sealed_weight_modules": self.sealed_weight_modules,
            "weight_quantization_events_before": (
                self.weight_quantization_events_before
            ),
            "weight_quantization_events_after": (self.weight_quantization_events_after),
            "weight_identity_before": self.identity_before,
            "weight_identity_after": self.identity_after,
            "weight_structure_fingerprint": self.structure_fingerprint,
        }


def _parameter_identity(name: str, parameter: Any) -> WeightParameterIdentity:
    pointer: int | None
    data_ptr = getattr(parameter, "data_ptr", None)
    try:
        pointer = int(data_ptr()) if callable(data_ptr) else None
    except (RuntimeError, TypeError, ValueError):
        pointer = None
    version_value = getattr(parameter, "_version", None)
    version = int(version_value) if version_value is not None else None
    return WeightParameterIdentity(
        name=str(name),
        object_id=id(parameter),
        data_pointer=pointer,
        version=version,
        shape=tuple(int(value) for value in parameter.shape),
        dtype=str(parameter.dtype),
        device=str(parameter.device),
        requires_grad=bool(parameter.requires_grad),
    )


def _identity_fingerprint(
    parameters: Sequence[WeightParameterIdentity],
    *,
    structural: bool,
) -> str:
    if structural:
        payload = [
            {
                "name": parameter.name,
                "shape": list(parameter.shape),
                "dtype": parameter.dtype,
                "requires_grad": parameter.requires_grad,
            }
            for parameter in parameters
        ]
    else:
        payload = [parameter.to_dict() for parameter in parameters]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binding_configs(
    pass_args: Mapping[str, Any],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    configs: list[tuple[str, Mapping[str, Any]]] = []
    for pattern, payload in pass_args.items():
        if pattern in {
            "by",
            "gptq",
            "rotation_search",
            "collapse_decode_banks",
        }:
            continue
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("config"), Mapping
        ):
            raise TypeError(f"invalid decode binding payload for {pattern!r}")
        configs.append((str(pattern), payload["config"]))
    if not configs:
        raise RuntimeError("decode profile contains no runtime bindings")
    return tuple(configs)


def build_decode_binding_plan(
    model: Any,
    pass_args: Mapping[str, Any],
) -> DecodeBindingPlan:
    """Resolve quantization regexes once for a reusable weight bank."""

    configs = _binding_configs(pass_args)
    compiled = tuple(re.compile(pattern) for pattern, _ in configs)
    matched = [0] * len(compiled)
    targets: list[DecodeBindingTarget] = []
    for name, module in model.named_modules():
        pattern_index = next(
            (
                index
                for index, pattern in enumerate(compiled)
                if pattern.fullmatch(name)
            ),
            None,
        )
        if pattern_index is None:
            continue
        matched[pattern_index] += 1
        targets.append(
            DecodeBindingTarget(
                pattern_index=pattern_index,
                name=str(name),
                module=module,
            )
        )
    missing = tuple(
        configs[index][0] for index, count in enumerate(matched) if count == 0
    )
    if missing:
        raise RuntimeError(f"decode profile patterns matched no modules: {missing}")
    return DecodeBindingPlan(
        patterns=tuple(pattern for pattern, _ in configs),
        targets=tuple(targets),
    )


def _validate_bank_structure(
    model: Any,
    binding_plan: DecodeBindingPlan,
    collapsed_linears: int,
    model_architecture: Mapping[str, Any],
) -> None:
    layers = int(model_architecture["num_hidden_layers"])
    qk_norm = bool(model_architecture.get("use_qk_norm", False))
    rmsnorms = layers * (4 if qk_norm else 2) + 1
    expected_pattern_counts = (
        layers,
        4 * layers,
        3 * layers,
        layers,
        layers,
        rmsnorms,
    )
    pattern_counts = tuple(
        sum(target.pattern_index == index for target in binding_plan.targets)
        for index in range(len(binding_plan.patterns))
    )
    if pattern_counts != expected_pattern_counts:
        raise RuntimeError(
            "decode binding coverage mismatch: "
            f"{pattern_counts} != {expected_pattern_counts}"
        )
    expected_bindings = sum(expected_pattern_counts)
    expected_linears = 7 * layers
    if len(binding_plan.targets) != expected_bindings:
        raise RuntimeError(
            f"model requires exactly {expected_bindings} runtime bindings"
        )
    if collapsed_linears != expected_linears:
        raise RuntimeError(
            f"model requires exactly {expected_linears} collapsed linears"
        )
    layer_indices = {
        int(match.group(1))
        for target in binding_plan.targets
        if (match := re.fullmatch(r"model\.layers\.(\d+)", target.name))
    }
    if layer_indices != set(range(layers)):
        raise RuntimeError("decoder-layer coverage is incomplete")
    for name in ("model.embed_tokens", "lm_head"):
        module = model.get_submodule(name)
        weight = getattr(module, "weight", None)
        if weight is None or str(weight.dtype) != "torch.bfloat16":
            raise RuntimeError(f"{name} must remain a BF16 weight module")
        if hasattr(module, "phase_q_config"):
            raise RuntimeError(f"{name} must remain outside decode quantization")


class _DecodeCacheLRU:
    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("decode-cache LRU capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self.total_bytes = 0
        self.values: OrderedDict[Path, tuple[DecodeCacheArtifact, int]] = OrderedDict()

    def get(self, path: Path) -> DecodeCacheArtifact:
        cached = self.values.pop(path, None)
        if cached is not None:
            self.values[path] = cached
            return cached[0]
        artifact = load_decode_cache_artifact(path)
        size = _artifact_bytes(artifact)
        while self.values and self.total_bytes + size > self.capacity_bytes:
            _, (_, removed) = self.values.popitem(last=False)
            self.total_bytes -= removed
        if size <= self.capacity_bytes:
            self.values[path] = (artifact, size)
            self.total_bytes += size
        return artifact

    def clear(self) -> None:
        self.values.clear()
        self.total_bytes = 0


def _load_stack_validity(
    path: Path,
    context: ExecutorContext,
) -> dict[str, StackValidity]:
    required_stages = (
        ("compiler", "emulator") if context.sample_contract.compiler_required else ()
    )
    result = load_built_stack_validity(
        path,
        manifest=context.master_manifest,
        scope_profile_ids=context.run_plan.hardware_validation_profile_ids,
        required_stages=required_stages,
        scope_name="hardware-validation",
        run_plan_hash=context.run_plan.canonical_hash,
    )
    required = {entry.profile_id for entry in context.stage_manifest.entries}
    if context.sample_contract.compiler_required:
        missing = sorted(required - set(result))
        if missing:
            raise ValueError(
                f"stack-validity manifest misses {len(missing)} stage profiles"
            )
        unchecked = sorted(
            profile_id
            for profile_id in required
            if result[profile_id].compiler_valid is None
            or result[profile_id].emulator_valid is None
        )
        if unchecked:
            raise ValueError(
                f"compiler/emulator checks are missing for {len(unchecked)} profiles"
            )
    return result


class DecodeEvaluator:
    """Evaluate fixed split-prefill artifacts with one reusable bank per W."""

    def _initialize_model_contract(self) -> None:
        architecture = self.config.get("model_architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("config.model_architecture is required")
        required = {
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "use_qk_norm",
        }
        missing = sorted(required - set(architecture))
        if missing:
            raise ValueError("model architecture is incomplete: " + ", ".join(missing))
        self.model_architecture = dict(architecture)
        self.cache_row_elements = int(architecture["num_key_value_heads"]) * int(
            architecture["head_dim"]
        )
        layout = _PACKED_CACHE_LAYOUT_RE.fullmatch(
            str(self.executor_config.get("layout_id", ""))
        )
        if layout is None:
            raise ValueError("executor.layout_id is not a native PackedKV layout")
        if int(layout.group("row_elements")) != self.cache_row_elements:
            raise ValueError(
                "executor.layout_id row width differs from model_architecture"
            )
        model_name = str(self.config.get("model_name", "")).lower()
        if "qwen3" in model_name:
            self.model_family = "qwen3"
        elif "llama" in model_name:
            self.model_family = "llama"
        else:
            raise ValueError(
                "the decode executor supports dense Qwen3 and Llama models"
            )

        placement = self.config.get("model_placement")
        if not isinstance(placement, Mapping):
            raise ValueError("config.model_placement is required")
        if (
            placement.get("policy") != "single_device"
            or int(placement.get("device_count", 0)) != 1
            or placement.get("automatic_device_map") is not False
        ):
            raise ValueError(
                "the executor requires explicit single-device placement with "
                "automatic_device_map disabled"
            )
        self.model_placement = dict(placement)

    def _initialize_admission_resources(
        self,
        *,
        workspace_root: str | os.PathLike[str],
    ) -> None:
        executor_config = self.executor_config
        self.device = str(self.config.get("device", "cuda:0"))
        self.runtime_seed = int(self.config.get("seed", 0))
        initialize_numerical_runtime(self.runtime_seed)
        self.runtime_environment = capture_runtime_environment(
            self.device,
            seed=self.runtime_seed,
        )
        self.sample_bundle_path = _resolve_path(
            str(executor_config["sample_bundle"]),
            workspace_root=workspace_root,
        )
        self.prefill_root = _resolve_path(
            str(executor_config["prefill_artifact_root"]),
            workspace_root=workspace_root,
        )
        self.admission_root = _resolve_path(
            str(executor_config["admission_artifact_root"]),
            workspace_root=workspace_root,
        )
        self.layout_id = str(executor_config.get("layout_id", PACKED_CACHE_LAYOUT))
        if _PACKED_CACHE_LAYOUT_RE.fullmatch(self.layout_id) is None:
            raise ValueError(f"unsupported layout_id {self.layout_id!r}")
        self._configure_mase_path()
        repository = _repository_root()
        admission_identity = {
            "schema_version": ADMISSION_CONVERTER_SCHEMA,
            "software_tree_sha256": _software_tree_hash(repository),
            "mase_tree_sha256": _mase_tree_hash(repository, self.config),
            "runtime_environment_fingerprint": (
                self.runtime_environment.logical_fingerprint
            ),
            "layout_id": self.layout_id,
            "block_size": 8,
            "scale_format": "E8M0",
            "scale_encoding": "bias127",
            "mxint_element_encoding": "sign_magnitude_lsb",
            "mxfp_element_encoding": "ieee_lsb",
            "plane_order": "row_major_lsb_first",
        }
        self.admission_code_revision = hashlib.sha256(
            json.dumps(
                admission_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.admission_contract_id = (
            f"{ADMISSION_CONVERTER_SCHEMA.rsplit('/', 1)[-1]}-"
            f"{self.admission_code_revision[:16]}"
        )
        self.bundle = load_sample_bundle(self.sample_bundle_path)

    def __init__(self, context: ExecutorContext) -> None:
        self.context = context
        self.config = context.config
        executor_config = self.config.get("executor")
        if not isinstance(executor_config, Mapping):
            raise ValueError("config.executor is required")
        self.executor_config = executor_config
        self._initialize_model_contract()
        self._initialize_admission_resources(
            workspace_root=self.context.workspace_root,
        )
        self._validate_bundle()
        self.samples = self.bundle.samples_for_prompt_set(
            context.sample_contract.prompt_set
        )
        contract_stage = context.sample_contract.name
        self.decode_microbatch_size = int(
            context.run_plan.numerical_screen_microbatch_size
            if contract_stage == "numerical-screen"
            else context.run_plan.hardware_validation_microbatch_size
        )
        if self.decode_microbatch_size <= 0:
            raise ValueError("decode_microbatch_size must be positive")
        self.deep_append_validation = context.stage in {
            "preflight",
            "validation-pilot",
        }
        self._validate_prefill_index()
        self._validate_workspace_runtime_environment()
        self.admission_catalog = self._validate_admission_preparation()
        self._validate_host_resources()
        capacity_gib = float(executor_config.get("max_cpu_cache_gib", 24.0))
        self.cache_lru = _DecodeCacheLRU(int(capacity_gib * (1 << 30)))
        validity_path = executor_config.get("stack_validity_manifest")
        resolved_validity = (
            _resolve_path(
                str(validity_path),
                workspace_root=self.context.workspace_root,
            )
            if validity_path
            else None
        )
        self.stack_validity = (
            _load_stack_validity(resolved_validity, context)
            if resolved_validity is not None and resolved_validity.is_file()
            else {}
        )
        if context.sample_contract.compiler_required and not self.stack_validity:
            raise ValueError(
                "hardware validation requires a measured executor.stack_validity_manifest"
            )

    def _validate_host_resources(self) -> None:
        projected_bytes = self._projected_new_admission_bytes()
        if projected_bytes:
            raise RuntimeError(
                "sealed admission preparation is missing required artifacts"
            )
        safety_factor = float(
            self.executor_config.get("artifact_space_safety_factor", 1.05)
        )
        reserve_gib = float(self.executor_config.get("artifact_space_reserve_gib", 8.0))
        if not math.isfinite(safety_factor) or safety_factor < 1.0:
            raise ValueError("artifact_space_safety_factor must be at least one")
        if not math.isfinite(reserve_gib) or reserve_gib < 0:
            raise ValueError("artifact_space_reserve_gib must be non-negative")
        if (
            self.executor_config.get("artifact_policy")
            == "content_addressed_recompute_per_format"
        ):
            footprint = estimate_artifact_footprint(
                self.config,
                self.model_architecture,
            )
            required_artifact_bytes = footprint.required_workspace_bytes
        else:
            required_artifact_bytes = int(reserve_gib * (1 << 30))
        artifact_gib = required_artifact_bytes / float(1 << 30)
        probe = self.admission_root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free_gib = shutil.disk_usage(probe).free / float(1 << 30)
        if free_gib < artifact_gib:
            raise RuntimeError(
                f"artifact filesystem has {free_gib:.1f} GiB free; "
                f"{artifact_gib:.1f} GiB is required"
            )
        self.resource_projection = {
            "projected_new_admission_bytes": projected_bytes,
            "safety_factor": safety_factor,
            "reserve_bytes": int(reserve_gib * (1 << 30)),
            "required_free_bytes": required_artifact_bytes,
            "observed_free_bytes": int(free_gib * (1 << 30)),
        }

        host_gib = float(self.executor_config.get("min_available_host_gib", 0))
        if not math.isfinite(host_gib) or host_gib < 0:
            raise ValueError("min_available_host_gib must be finite and non-negative")
        if host_gib == 0:
            return
        meminfo = Path("/proc/meminfo")
        if not meminfo.is_file():
            raise RuntimeError("host-memory availability cannot be measured")
        fields = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition(":")
            if separator:
                fields[name] = value.strip()
        available = fields.get("MemAvailable", "")
        if not available.endswith(" kB"):
            raise RuntimeError("MemAvailable is missing from /proc/meminfo")
        available_gib = int(available.removesuffix(" kB")) * 1024 / float(1 << 30)
        if available_gib < host_gib:
            raise RuntimeError(
                f"host has {available_gib:.1f} GiB available; "
                f"{host_gib:.1f} GiB is required"
            )

    def _projected_new_admission_bytes(self) -> int:
        if (
            self.executor_config.get("artifact_policy")
            == "content_addressed_recompute_per_format"
        ):
            return 0
        if not all(
            hasattr(self, name)
            for name in (
                "context",
                "samples",
                "prefill_root",
                "admission_root",
                "admission_contract_id",
            )
        ):
            return 0
        formats = tuple(
            sorted(
                {
                    entry.profile.kv_format
                    for entry in self.context.stage_manifest.entries
                }
            )
        )
        projected = 0
        for sample in self.samples:
            manifest_path = (
                _prefill_path(self.prefill_root, sample.document_id) / "manifest.json"
            )
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if value.get("schema") != "plena.prefill":
                raise ValueError(f"invalid prefill manifest: {manifest_path}")
            tensor_elements = 0
            layers = value.get("layers")
            if not isinstance(layers, list) or not layers:
                raise ValueError(f"prefill manifest has no layers: {manifest_path}")
            for layer in layers:
                for role in ("key", "value"):
                    tensor = layer.get(role)
                    if not isinstance(tensor, Mapping):
                        raise ValueError(
                            f"prefill manifest lacks {role}: {manifest_path}"
                        )
                    if tensor.get("dtype") != "bfloat16":
                        raise ValueError("prefill cache storage must be BF16")
                    shape = tuple(int(size) for size in tensor.get("shape", ()))
                    if (
                        len(shape) != 4
                        or shape[0] != 1
                        or shape[1] * shape[3] != self.cache_row_elements
                        or shape[-2] != 512
                    ):
                        raise ValueError(
                            "prefill cache geometry differs from the model contract"
                        )
                    elements = math.prod(shape)
                    if elements * 2 <= 0:
                        raise ValueError("prefill tensor is empty")
                    tensor_elements += elements
            for format_id in formats:
                path = _decode_cache_path(
                    self.admission_root,
                    format_id,
                    sample.document_id,
                    contract_id=self.admission_contract_id,
                )
                if path.exists():
                    continue
                projected += _packed_tensor_disk_bytes(
                    tensor_elements,
                    format_id,
                )
                projected += 1 << 20
        return projected

    def _configure_mase_path(self) -> None:
        value = self.executor_config.get("mase_src")
        if value is None:
            raise ValueError("executor.mase_src is required")
        resolved = _resolve_path(str(value))
        if not resolved.is_dir():
            raise ValueError(f"MASE source tree does not exist: {resolved}")
        path = str(resolved)
        if path not in sys.path:
            sys.path.insert(0, path)
        spec = importlib.util.find_spec("chop")
        if spec is None or spec.origin is None:
            raise RuntimeError("the configured MASE source does not provide chop")
        origin = Path(spec.origin).resolve()
        expected = (resolved / "chop").resolve()
        if expected not in (origin, *origin.parents):
            raise RuntimeError(
                f"chop resolved outside configured MASE source: {origin}"
            )
        self.mase_source_root = resolved

    def _validate_bundle(self) -> None:
        if self.bundle.model_revision != str(self.config["model_revision"]):
            raise ValueError("sample bundle model revision mismatch")
        if self.bundle.tokenizer_revision != str(self.config["tokenizer_revision"]):
            raise ValueError("sample bundle tokenizer revision mismatch")
        if self.bundle.prompt_manifest() != self.context.prompts:
            raise ValueError("sample bundle differs from the workspace prompts")
        expected_count = self.context.sample_contract.prompt_count
        samples = self.bundle.samples_for_prompt_set(
            self.context.sample_contract.prompt_set
        )
        if len(samples) != expected_count:
            raise ValueError("sample count differs from the stage contract")
        steps = self.context.sample_contract.decode_steps
        if any(len(sample.decode_target_ids) < steps for sample in samples):
            raise ValueError("sample bundle lacks stage decode targets")

    def _validate_prefill_index(self) -> None:
        value = load_immutable_json(self.prefill_root / "index.json")
        if value.get("schema_version") != "decode-prefill-index":
            raise ValueError("unsupported prefill index schema")
        if value.get("model_revision") != self.bundle.model_revision:
            raise ValueError("prefill index model revision mismatch")
        if value.get("tokenizer_revision") != self.bundle.tokenizer_revision:
            raise ValueError("prefill index tokenizer revision mismatch")
        if value.get("sample_bundle_hash") != self.bundle.canonical_hash:
            raise ValueError("prefill index sample-bundle mismatch")
        current_code_revision = _software_tree_hash(_repository_root())
        if value.get("code_revision") != current_code_revision:
            raise ValueError("prefill index source-tree mismatch")
        recorded_runtime = RuntimeEnvironment.from_dict(
            value.get("runtime_environment", {})
        )
        if (
            recorded_runtime.logical_fingerprint
            != self.runtime_environment.logical_fingerprint
        ):
            raise ValueError("prefill index runtime-environment mismatch")
        records = {
            str(record["document_id"]): record for record in value.get("records", ())
        }
        expected = {
            sample.document_id
            for sample in self.bundle.numerical_screen + self.bundle.hardware_validation
        }
        if set(records) != expected:
            raise ValueError("prefill index document coverage mismatch")
        for sample in self.samples:
            path = _prefill_path(self.prefill_root, sample.document_id)
            if not path.is_dir():
                raise ValueError(f"missing prefill artifact for {sample.document_id}")
            record = records[sample.document_id]
            if record.get("prompt_hash") != sample.prompt_hash:
                raise ValueError("prefill index prompt hash mismatch")
            if (
                record.get("relative_path")
                != path.relative_to(self.prefill_root).as_posix()
            ):
                raise ValueError("prefill index path mismatch")
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if (
                manifest.get("artifact_id") != record.get("artifact_id")
                or manifest.get("prompt_hash") != sample.prompt_hash
                or manifest.get("provenance", {}).get("code_revision")
                != current_code_revision
                or manifest.get("provenance", {})
                .get("parameters", {})
                .get("runtime_environment_fingerprint")
                != self.runtime_environment.logical_fingerprint
                or manifest.get("metadata", {}).get("preparation_device_uuid")
                != record.get("preparation_device", {}).get("device_uuid")
            ):
                raise ValueError("prefill artifact identity differs from its index")

    def seal_workspace_runtime_environment(
        self,
        workspace_root: str | Path,
    ) -> Mapping[str, Any]:
        """Bind admission and later shards to one numerical stack."""

        return seal_runtime_environment(
            Path(workspace_root).resolve() / "runtime_environment.json",
            self.runtime_environment,
        )

    def _validate_workspace_runtime_environment(self) -> None:
        require_runtime_environment(
            self.context.workspace_root / "runtime_environment.json",
            self.runtime_environment,
        )

    def _capture_current_runtime_environment(self) -> RuntimeEnvironment:
        current = capture_runtime_environment(
            self.device,
            seed=self.runtime_seed,
        )
        if current.logical_fingerprint != self.runtime_environment.logical_fingerprint:
            raise RuntimeError("numerical runtime changed during execution")
        return current

    @classmethod
    def for_admission_preparation(
        cls,
        config: Mapping[str, Any],
        *,
        workspace_root: str | os.PathLike[str],
    ) -> "DecodeEvaluator":
        """Construct the artifact-only admission path without loading weights."""

        self = cls.__new__(cls)
        self.config = config
        executor_config = config.get("executor")
        if not isinstance(executor_config, Mapping):
            raise ValueError("config.executor is required")
        self.executor_config = executor_config
        self._initialize_model_contract()
        self._initialize_admission_resources(workspace_root=workspace_root)
        if self.bundle.model_revision != str(config["model_revision"]):
            raise ValueError("sample bundle model revision mismatch")
        if self.bundle.tokenizer_revision != str(config["tokenizer_revision"]):
            raise ValueError("sample bundle tokenizer revision mismatch")
        self.samples = self.bundle.numerical_screen + self.bundle.hardware_validation
        self._validate_prefill_index()
        return self

    @property
    def admission_index_path(self) -> Path:
        return self.admission_root / self.admission_contract_id / "index.json"

    @staticmethod
    def _admission_timing_path(path: Path) -> Path:
        return path.parent / ".timings" / f"{path.name}.json"

    def _admission_record_keys(self) -> set[tuple[str, str]]:
        formats = DECODE_FORMATS + ("BF16",)
        return {
            (format_id, sample.document_id)
            for format_id in formats
            for sample in self.bundle.numerical_screen + self.bundle.hardware_validation
        }

    def _validate_admission_catalog(
        self,
        value: Mapping[str, Any],
        *,
        deep: bool,
    ) -> tuple[
        dict[tuple[str, str], Path],
        tuple[dict[str, Any], ...],
    ]:
        if value.get("schema_version") != ADMISSION_INDEX_SCHEMA:
            raise ValueError("unsupported admission index schema")
        if value.get("admission_contract_id") != self.admission_contract_id:
            raise ValueError("admission index contract mismatch")
        if value.get("admission_code_revision") != self.admission_code_revision:
            raise ValueError("admission index source-tree mismatch")
        if (
            value.get("runtime_environment_fingerprint")
            != self.runtime_environment.logical_fingerprint
        ):
            raise ValueError("admission index runtime-environment mismatch")
        if value.get("sample_bundle_hash") != self.bundle.canonical_hash:
            raise ValueError("admission index sample-bundle mismatch")
        if tuple(value.get("quantized_formats", ())) != DECODE_FORMATS:
            raise ValueError("admission index quantized-format coverage mismatch")
        if tuple(value.get("reference_formats", ())) != ("BF16",):
            raise ValueError("admission index reference coverage mismatch")
        records = tuple(value.get("records", ()))
        expected_keys = self._admission_record_keys()
        if len(records) != len(expected_keys):
            raise ValueError("admission index record count mismatch")
        paths: dict[tuple[str, str], Path] = {}
        samples = {
            sample.document_id: sample
            for sample in self.bundle.numerical_screen + self.bundle.hardware_validation
        }
        ordered_records = sorted(
            records,
            key=lambda record: (
                str(record["document_id"]),
                str(record["format_id"]),
            ),
        )
        deep_document_id: str | None = None
        deep_prefill: Any | None = None
        numerical_records: list[dict[str, Any]] = []
        for record in ordered_records:
            key = (str(record["format_id"]), str(record["document_id"]))
            if key in paths or key not in expected_keys:
                raise ValueError("admission index contains an invalid record key")
            format_id, document_id = key
            sample = samples[document_id]
            expected_path = _decode_cache_path(
                self.admission_root,
                format_id,
                document_id,
                contract_id=self.admission_contract_id,
            )
            relative = expected_path.relative_to(self.admission_root).as_posix()
            if (
                record.get("relative_path") != relative
                or record.get("prompt_hash") != sample.prompt_hash
            ):
                raise ValueError("admission index record binding mismatch")
            manifest_path = expected_path / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"missing admitted cache manifest: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_precision = _kv_precision_id(format_id, format_id)
            if (
                manifest.get("schema") != "plena.decode_cache"
                or manifest.get("artifact_id") != record.get("artifact_id")
                or manifest.get("source_artifact_id")
                != record.get("source_artifact_id")
                or manifest.get("precision_id") != expected_precision
                or manifest.get("layout_id") != self.layout_id
            ):
                raise ValueError("admitted cache manifest differs from its index")
            if int(record.get("persisted_bytes", -1)) != _directory_bytes(
                expected_path
            ):
                raise ValueError("admitted cache persisted-byte count mismatch")
            timing_path = self._admission_timing_path(expected_path)
            timing = load_immutable_json(timing_path)
            if (
                timing.get("schema_version") != "decode-admission-timing"
                or timing.get("artifact_id") != record.get("artifact_id")
                or timing.get("format_id") != format_id
                or timing.get("document_id") != document_id
                or timing.get("admission_contract_id") != self.admission_contract_id
                or timing.get("basis") != "conversion_persistence_and_deep_validation"
                or float(timing.get("build_seconds", -1.0)) <= 0
                or float(timing.get("build_seconds", -1.0))
                != float(record.get("build_seconds", -2.0))
            ):
                raise ValueError("admission timing record mismatch")
            if deep:
                artifact = load_decode_cache_artifact(expected_path)
                if deep_document_id != document_id:
                    deep_prefill = load_prefill_artifact(
                        _prefill_path(self.prefill_root, document_id)
                    )
                    deep_document_id = document_id
                if deep_prefill is None:
                    raise AssertionError("deep prefill cache was not loaded")
                self._validate_admitted(
                    artifact,
                    prefill=deep_prefill,
                    sample=sample,
                    key_format=format_id,
                    value_format=format_id,
                    provenance=self._admission_provenance(
                        format_id,
                        format_id,
                    ),
                )
                numerical_records.append(
                    self._validate_admitted_numerical(
                        artifact,
                        prefill=deep_prefill,
                        document_id=document_id,
                    )
                )
            paths[key] = expected_path
        if set(paths) != expected_keys:
            raise ValueError("admission index coverage is incomplete")
        artifact_seconds = sum(float(record["build_seconds"]) for record in records)
        quantized_seconds = sum(
            float(record["build_seconds"])
            for record in records
            if record["format_id"] in DECODE_FORMATS
        )
        reference_seconds = sum(
            float(record["build_seconds"])
            for record in records
            if record["format_id"] == "BF16"
        )
        prefill_seconds = float(value.get("prefill_read_seconds", -1.0))
        cold_seconds = float(value.get("cold_build_seconds", -1.0))
        persisted_bytes = sum(int(record["persisted_bytes"]) for record in records)
        payload_bytes = sum(int(record["payload_bytes"]) for record in records)
        resources = _validate_admission_resource_projection(
            value.get("resource_projection")
        )
        if (
            int(value.get("document_count", -1))
            != len(self.bundle.numerical_screen + self.bundle.hardware_validation)
            or int(value.get("artifact_count", -1)) != len(records)
            or not math.isclose(
                float(value.get("artifact_build_seconds", -1.0)),
                artifact_seconds,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(value.get("quantized_build_seconds", -1.0)),
                quantized_seconds,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(value.get("reference_build_seconds", -1.0)),
                reference_seconds,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or prefill_seconds <= 0
            or not math.isclose(
                cold_seconds,
                artifact_seconds + prefill_seconds,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or int(value.get("persisted_bytes", -1)) != persisted_bytes
            or int(value.get("payload_bytes", -1)) != payload_bytes
            or int(resources.get("projected_cold_artifact_bytes", -1)) < persisted_bytes
        ):
            raise ValueError("admission index aggregate accounting mismatch")
        return paths, tuple(numerical_records)

    def _validate_admission_preparation(self) -> Mapping[str, Any]:
        catalog = load_immutable_json(self.admission_index_path)
        if (
            catalog.get("persistence_policy")
            == "content_addressed_recompute_per_format"
        ):
            records = self._validate_recomputable_admission_catalog(catalog)
            receipt = load_immutable_json(
                self.context.workspace_root / "admission_preparation.json"
            )
            validation_path = Path(str(receipt.get("numerical_validation_path", "")))
            validation = (
                load_immutable_json(validation_path)
                if validation_path.is_file()
                else {}
            )
            if (
                receipt.get("schema_version") != ADMISSION_PREPARATION_SCHEMA
                or receipt.get("persistence_policy")
                != "content_addressed_recompute_per_format"
                or receipt.get("manifest_hash")
                != self.context.master_manifest.canonical_hash
                or receipt.get("run_plan_hash") != self.context.run_plan.canonical_hash
                or receipt.get("prompt_manifest_hash")
                != self.context.prompts.canonical_hash
                or receipt.get("admission_index_hash") != catalog.get("content_hash")
                or receipt.get("admission_contract_id") != self.admission_contract_id
                or receipt.get("runtime_environment_fingerprint")
                != self.runtime_environment.logical_fingerprint
                or receipt.get("numerical_validation_hash")
                != validation.get("content_hash")
                or validation.get("schema_version")
                != ADMISSION_NUMERICAL_VALIDATION_SCHEMA
                or validation.get("passed") is not True
                or validation.get("admission_index_hash") != catalog.get("content_hash")
                or int(validation.get("artifact_count", -1)) != len(records)
                or int(receipt.get("persisted_bytes", -1)) != 0
            ):
                raise ValueError(
                    "recomputable admission preparation is not workspace-valid"
                )
            self.admission_records = records
            self.admission_paths = {}
            return catalog
        paths, _ = self._validate_admission_catalog(catalog, deep=False)
        receipt_path = self.context.workspace_root / "admission_preparation.json"
        receipt = load_immutable_json(receipt_path)
        validation_path = Path(str(receipt.get("numerical_validation_path", "")))
        validation = (
            load_immutable_json(validation_path) if validation_path.is_file() else {}
        )
        if (
            receipt.get("schema_version") != ADMISSION_PREPARATION_SCHEMA
            or receipt.get("manifest_hash")
            != self.context.master_manifest.canonical_hash
            or receipt.get("run_plan_hash") != self.context.run_plan.canonical_hash
            or receipt.get("prompt_manifest_hash")
            != self.context.prompts.canonical_hash
            or receipt.get("admission_index_hash") != catalog.get("content_hash")
            or receipt.get("admission_contract_id") != self.admission_contract_id
            or receipt.get("runtime_environment_fingerprint")
            != self.runtime_environment.logical_fingerprint
            or receipt.get("numerical_validation_hash")
            != validation.get("content_hash")
            or validation.get("schema_version") != ADMISSION_NUMERICAL_VALIDATION_SCHEMA
            or validation.get("passed") is not True
            or validation.get("admission_index_hash") != catalog.get("content_hash")
            or validation.get("admission_contract_id") != self.admission_contract_id
            or validation.get("admission_code_revision") != self.admission_code_revision
            or validation.get("runtime_environment_fingerprint")
            != self.runtime_environment.logical_fingerprint
            or validation.get("sample_bundle_hash") != self.bundle.canonical_hash
            or validation.get("layout_id") != self.layout_id
            or int(validation.get("block_size", -1)) != 8
            or tuple(validation.get("quantized_formats", ())) != DECODE_FORMATS
            or tuple(validation.get("reference_formats", ())) != ("BF16",)
            or int(validation.get("artifact_count", -1))
            != int(catalog.get("artifact_count", -2))
            or int(validation.get("document_count", -1))
            != int(catalog.get("document_count", -2))
            or int(validation.get("tensor_count", -1))
            != sum(
                int(record.get("tensor_count", -2))
                for record in validation.get("records", ())
            )
        ):
            raise ValueError("admission preparation receipt is not workspace-valid")
        self.admission_paths = paths
        return catalog

    def _admission_cold_resource_projection(self) -> Mapping[str, Any]:
        self.admission_root.mkdir(parents=True, exist_ok=True)
        projected_bytes = 0
        projected_element_bytes = 0
        projected_scale_bytes = 0
        projected_numerical_view_bytes = 0
        projected_metadata_reserve_bytes = 0
        existing_bytes = 0
        max_prefill_bytes = 0
        max_artifact_bytes = 0
        formats = DECODE_FORMATS + ("BF16",)
        for sample in self.samples:
            prefill_path = _prefill_path(
                self.prefill_root,
                sample.document_id,
            )
            max_prefill_bytes = max(
                max_prefill_bytes,
                _directory_bytes(prefill_path),
            )
            manifest = json.loads(
                (prefill_path / "manifest.json").read_text(encoding="utf-8")
            )
            tensor_elements = sum(
                math.prod(tuple(int(size) for size in layer[role]["shape"]))
                for layer in manifest["layers"]
                for role in ("key", "value")
            )
            for format_id in formats:
                components = _packed_tensor_disk_components(
                    tensor_elements,
                    format_id,
                )
                metadata_reserve = 1 << 20
                estimate = sum(components.values()) + metadata_reserve
                projected_bytes += estimate
                projected_element_bytes += components["element_plane_bytes"]
                projected_scale_bytes += components["scale_plane_bytes"]
                projected_numerical_view_bytes += components["numerical_view_bytes"]
                projected_metadata_reserve_bytes += metadata_reserve
                max_artifact_bytes = max(max_artifact_bytes, estimate)
                path = _decode_cache_path(
                    self.admission_root,
                    format_id,
                    sample.document_id,
                    contract_id=self.admission_contract_id,
                )
                if path.exists():
                    existing_bytes += _directory_bytes(path)
        safety_factor = float(
            self.executor_config.get(
                "artifact_space_safety_factor",
                1.05,
            )
        )
        reserve_bytes = int(
            float(
                self.executor_config.get(
                    "artifact_space_reserve_gib",
                    8.0,
                )
            )
            * (1 << 30)
        )
        configured_artifact_gib = float(
            self.executor_config.get("min_free_artifact_gib", 0.0)
        )
        if (
            not math.isfinite(safety_factor)
            or safety_factor < 1.0
            or reserve_bytes < 0
            or not math.isfinite(configured_artifact_gib)
            or configured_artifact_gib < 0
        ):
            raise ValueError("invalid admission resource configuration")
        required_capacity = max(
            math.ceil(projected_bytes * safety_factor) + reserve_bytes,
            int(configured_artifact_gib * (1 << 30)),
        )
        disk = shutil.disk_usage(self.admission_root)
        observed_cold_available = min(
            disk.total,
            disk.free + existing_bytes,
        )
        configured_host_gib = float(
            self.executor_config.get(
                "min_available_host_gib",
                0.0,
            )
        )
        if not math.isfinite(configured_host_gib) or configured_host_gib < 0:
            raise ValueError("min_available_host_gib must be non-negative")
        configured_host_bytes = int(configured_host_gib * (1 << 30))
        projected_peak_host = max_prefill_bytes + 2 * max_artifact_bytes
        required_host = max(projected_peak_host, configured_host_bytes)
        observed_host = _available_host_bytes()
        if observed_cold_available < required_capacity:
            raise RuntimeError("artifact filesystem cannot hold the cold admission set")
        if observed_host < required_host:
            raise RuntimeError(
                "host memory is below the admission preparation requirement"
            )
        if projected_bytes != sum(
            (
                projected_element_bytes,
                projected_scale_bytes,
                projected_numerical_view_bytes,
                projected_metadata_reserve_bytes,
            )
        ):
            raise AssertionError("admission storage projection is inconsistent")
        return {
            "persistence_contract": ("packed_planes_plus_bf16_numerical_view"),
            "artifact_space_safety_factor": safety_factor,
            "artifact_space_reserve_bytes": reserve_bytes,
            "projected_element_plane_bytes": projected_element_bytes,
            "projected_scale_plane_bytes": projected_scale_bytes,
            "projected_numerical_view_bytes": (projected_numerical_view_bytes),
            "projected_metadata_reserve_bytes": (projected_metadata_reserve_bytes),
            "projected_cold_artifact_bytes": projected_bytes,
            "required_cold_capacity_bytes": required_capacity,
            "observed_cold_available_bytes": observed_cold_available,
            "projected_peak_host_bytes": projected_peak_host,
            "required_host_bytes": required_host,
            "observed_host_available_bytes": observed_host,
        }

    def _load_model(self) -> Any:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "decode execution requires torch and transformers"
            ) from exc
        if str(self.config.get("dtype", "bfloat16")).lower() != "bfloat16":
            raise ValueError("decode execution requires bfloat16 storage")
        model = AutoModelForCausalLM.from_pretrained(
            str(self.config["model_name"]),
            revision=str(self.config["model_revision"]),
            torch_dtype=torch.bfloat16,
            cache_dir=self.config.get("hf_cache_dir"),
            local_files_only=bool(self.config.get("local_files_only", True)),
            trust_remote_code=bool(self.config.get("trust_remote_code", False)),
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        return model.eval()

    def _validate_device_label(self) -> None:
        import torch

        if not self.device.startswith("cuda"):
            if self.context.device_label.lower() != "cpu":
                raise ValueError("CPU execution requires device label CPU")
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested but CUDA is unavailable")
        index = torch.device(self.device).index
        name = torch.cuda.get_device_name(index or 0).lower()
        label = self.context.device_label.lower()
        if label not in name:
            raise ValueError(
                f"planned device {self.context.device_label!r} does not match {name!r}"
            )
        minimum_mb = int(self.config.get("gpu_min_free_mb", 0))
        if minimum_mb < 0:
            raise ValueError("gpu_min_free_mb must be non-negative")
        free_bytes, _ = torch.cuda.mem_get_info(index or 0)
        free_mb = free_bytes / float(1 << 20)
        if free_mb < minimum_mb:
            raise RuntimeError(
                f"GPU has {free_mb:.0f} MiB free; {minimum_mb} MiB is required"
            )

    @contextmanager
    def open_weight_bank(
        self,
        weight_format: str,
        entries: tuple[SweepManifestEntry, ...],
    ) -> Iterator[DecodeWeightBank]:
        import torch

        if not entries or any(
            entry.profile.weight_format != weight_format for entry in entries
        ):
            raise ValueError("weight-bank group is empty or inconsistent")
        self.cache_lru.clear()
        gc.collect()
        build_started = time.perf_counter()
        initialize_numerical_runtime(self.runtime_seed)
        self._capture_current_runtime_environment()
        self._validate_device_label()
        model = None
        try:
            with self._weight_bank_build_lock():
                model = self._load_model()
                if weight_format != "BF16":
                    from chop.passes.module.transforms.quantize.quantize import (
                        install_phase_context_pre_hooks,
                        quantize_module_transform_pass,
                    )
                    from decode_dse.software.precision_bindings import (
                        build_decode_pass_args,
                    )

                    representative = profile_to_decode_quant_spec(entries[0].profile)
                    if representative is None:
                        raise ValueError("quantized weight group has a BF16 profile")
                    pass_args = build_decode_pass_args(
                        str(self.config["model_name"]),
                        self.device,
                        representative,
                    )
                    pass_args["collapse_decode_banks"] = True
                    bank_device_variable = "MASE_PHASE_BANK_DEVICE"
                    previous_bank_device = os.environ.get(bank_device_variable)
                    try:
                        if self.device.startswith("cuda"):
                            os.environ[bank_device_variable] = self.device
                        else:
                            os.environ.pop(bank_device_variable, None)
                        model, _ = quantize_module_transform_pass(
                            model,
                            pass_args,
                        )
                    finally:
                        if previous_bank_device is None:
                            os.environ.pop(bank_device_variable, None)
                        else:
                            os.environ[bank_device_variable] = previous_bank_device
                    model = model.to(self.device).eval()
                    install_phase_context_pre_hooks(model)
                    binding_plan = build_decode_binding_plan(model, pass_args)
                    quantization_guard = DecodeWeightQuantizationGuard.capture(
                        binding_plan,
                        expected_modules=(
                            7 * int(self.model_architecture["num_hidden_layers"])
                        ),
                    )
                    _validate_bank_structure(
                        model,
                        binding_plan,
                        len(quantization_guard.modules),
                        self.model_architecture,
                    )
                else:
                    if any(
                        entry.profile.kind != PROFILE_KIND_BF16_REFERENCE
                        for entry in entries
                    ):
                        raise ValueError(
                            "BF16 weight bank contains a quantized profile"
                        )
                    model = model.to(self.device).eval()
                    binding_plan = DecodeBindingPlan(patterns=(), targets=())
                    quantization_guard = DecodeWeightQuantizationGuard.capture(
                        binding_plan,
                        expected_modules=0,
                    )
            identity_guard = DecodeWeightBankIdentity.capture(model)
            yield DecodeWeightBank(
                model=model,
                device=torch.device(self.device),
                weight_format=weight_format,
                binding_plan=binding_plan,
                identity_guard=identity_guard,
                quantization_guard=quantization_guard,
                build_seconds=time.perf_counter() - build_started,
            )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @contextmanager
    def _weight_bank_build_lock(self) -> Iterator[None]:
        if not bool(self.executor_config.get("serialize_weight_bank_builds", True)):
            yield
            return
        scratch = _resolve_path(
            str(self.config.get("scratch_dir", "results/decode_dse_scratch")),
            workspace_root=self.context.workspace_root,
        )
        lock_root = scratch / ".locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        model_token = hashlib.sha256(
            str(self.config["model_name"]).encode("utf-8")
        ).hexdigest()[:16]
        lock_path = lock_root / f"{model_token}-weight-bank-build.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _admission_provenance(
        self,
        key_format: str,
        value_format: str,
    ) -> ArtifactProvenance:
        created_at = getattr(self, "admission_provenance_created_at", None)
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self.admission_provenance_created_at = created_at
        return ArtifactProvenance(
            producer="packedkv-decode-admission",
            code_revision=self.admission_code_revision,
            created_at_utc=str(created_at),
            parameters=(
                ("key_format", key_format),
                ("value_format", value_format),
                ("layout_id", self.layout_id),
                ("block_size", "8"),
                ("converter_schema", ADMISSION_CONVERTER_SCHEMA),
                (
                    "element_encoding",
                    "MXINT_SIGN_MAGNITUDE_OR_MXFP_IEEE_LSB",
                ),
                ("scale_encoding", "E8M0_bias127"),
                ("sample_bundle_hash", self.bundle.canonical_hash),
                (
                    "runtime_environment_fingerprint",
                    self.runtime_environment.logical_fingerprint,
                ),
            ),
        )

    @contextmanager
    def _artifact_lock(self, path: Path) -> Iterator[None]:
        lock_root = self.admission_root / ".locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f"{hashlib.sha256(str(path).encode()).hexdigest()}.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _validate_admitted(
        self,
        artifact: DecodeCacheArtifact,
        *,
        prefill: Any,
        sample: DecodeTokenSample,
        key_format: str,
        value_format: str,
        provenance: ArtifactProvenance,
    ) -> None:
        if artifact.source_artifact_id != prefill.artifact_id:
            raise ValueError("admitted cache source mismatch")
        if artifact.precision_id != _kv_precision_id(
            key_format,
            value_format,
        ):
            raise ValueError("admitted cache precision mismatch")
        if artifact.key_format != key_format:
            raise ValueError("admitted cache key-format mismatch")
        if artifact.value_format != value_format:
            raise ValueError("admitted cache value-format mismatch")
        if artifact.layout_id != self.layout_id:
            raise ValueError("admitted cache layout mismatch")
        if artifact.provenance.producer != provenance.producer:
            raise ValueError("admitted cache producer mismatch")
        if artifact.provenance.code_revision != provenance.code_revision:
            raise ValueError("admitted cache source-tree mismatch")
        if artifact.provenance.parameters != provenance.parameters:
            raise ValueError("admitted cache conversion-contract mismatch")
        metadata = dict(artifact.metadata)
        if metadata.get("document_id") != sample.document_id:
            raise ValueError("admitted cache document mismatch")
        if metadata.get("sample_bundle_hash") != self.bundle.canonical_hash:
            raise ValueError("admitted cache sample-bundle mismatch")
        if metadata.get("admission_contract_id") != self.admission_contract_id:
            raise ValueError("admitted cache contract identity mismatch")
        if len(artifact.layers) != len(prefill.layers):
            raise ValueError("admitted cache layer count mismatch")
        for layer_index, (admitted_layer, prefill_layer) in enumerate(
            zip(artifact.layers, prefill.layers)
        ):
            for role, payload, source, format_id in (
                ("key", admitted_layer.key, prefill_layer.key, key_format),
                ("value", admitted_layer.value, prefill_layer.value, value_format),
            ):
                descriptor = format_descriptor(format_id)
                expected_block = 1 if descriptor.family == "bf16" else 8
                elements = math.prod(source.shape)
                expected_element_bytes = math.ceil(
                    elements * descriptor.element_bits / 8
                )
                expected_scale_bytes = (
                    0 if descriptor.family == "bf16" else elements // 8
                )
                expected_element_encoding = {
                    "bf16": "BF16_LE",
                    "mxint": "MXINT_SIGN_MAGNITUDE_LSB",
                    "mxfp": "MXFP_IEEE_LSB",
                }[descriptor.family]
                expected_scale_encoding = (
                    "NONE" if descriptor.family == "bf16" else "E8M0_BIAS127_U8"
                )
                if (
                    payload.logical_shape != source.shape
                    or payload.format_id != descriptor.token
                    or payload.block_size != expected_block
                    or payload.element_bits != descriptor.element_bits
                    or payload.element_encoding != expected_element_encoding
                    or payload.scale_encoding != expected_scale_encoding
                    or len(payload.element_plane) != expected_element_bytes
                    or len(payload.scale_plane) != expected_scale_bytes
                    or payload.numerical_view.dtype != "bfloat16"
                ):
                    raise ValueError(
                        f"admitted layer {layer_index} {role} plane contract mismatch"
                    )

    def _validate_admitted_numerical(
        self,
        artifact: DecodeCacheArtifact,
        *,
        prefill: Any,
        document_id: str,
    ) -> dict[str, Any]:
        """Rebuild every tensor and require exact persisted admission bytes."""

        checks: list[dict[str, Any]] = []
        for layer_index, (admitted_layer, prefill_layer) in enumerate(
            zip(artifact.layers, prefill.layers)
        ):
            for role, observed, source in (
                ("key", admitted_layer.key, prefill_layer.key),
                ("value", admitted_layer.value, prefill_layer.value),
            ):
                converter: Any = (
                    BF16CacheConverter()
                    if observed.format_id == "BF16"
                    else MaseMXCacheConverter(
                        observed.format_id,
                        block_size=observed.block_size,
                    )
                )
                rebuilt = converter.convert(
                    source,
                    role=role,
                    layer_index=layer_index,
                    precision_id=artifact.precision_id,
                    layout_id=artifact.layout_id,
                )
                exact = (
                    rebuilt.descriptor() == observed.descriptor()
                    and rebuilt.element_plane == observed.element_plane
                    and rebuilt.scale_plane == observed.scale_plane
                    and rebuilt.numerical_view.data == observed.numerical_view.data
                )
                if not exact:
                    raise ValueError(
                        f"admitted layer {layer_index} {role} differs "
                        "from source conversion"
                    )
                check = {
                    "layer_index": layer_index,
                    "role": role,
                    "format_id": observed.format_id,
                    "source_sha256": source.sha256,
                    "element_sha256": observed.element_sha256,
                    "scale_sha256": observed.scale_sha256,
                    "numerical_view_sha256": (observed.numerical_view.sha256),
                    "descriptor_sha256": _mapping_sha256(observed.descriptor()),
                    "exact_match": True,
                }
                checks.append({**check, "check_hash": _mapping_sha256(check)})
        return {
            "document_id": document_id,
            "format_id": artifact.key_format,
            "source_artifact_id": artifact.source_artifact_id,
            "artifact_id": artifact.artifact_id,
            "layer_count": len(artifact.layers),
            "tensor_count": len(checks),
            "tensor_checks": checks,
        }

    def _validate_native_append(
        self,
        cache: Any,
        start: int,
        end: int,
        artifact: DecodeCacheArtifact,
    ) -> None:
        import torch

        started = time.perf_counter()
        tensor_checks = 0
        quantized_tensor_checks = 0
        try:
            if end != start + 1:
                raise AssertionError("decode append validation requires one token")
            layers = _legacy_cache_layers(cache)
            if len(layers) != len(artifact.layers):
                raise AssertionError("runtime cache layer count changed")
            for layer_index, ((key, value), payload) in enumerate(
                zip(layers, artifact.layers)
            ):
                for role, runtime, expected_payload in (
                    ("key", key, payload.key),
                    ("value", value, payload.value),
                ):
                    appended = runtime[..., start:end, :]
                    if appended.shape[-2] != 1:
                        raise AssertionError("runtime cache append is not q_len=1")
                    tensor_checks += 1
                    if expected_payload.format_id == "BF16":
                        if appended.dtype != torch.bfloat16:
                            raise TypeError("BF16 cache append changed storage dtype")
                        continue
                    source = TensorPayload.from_torch(
                        appended.detach().to("cpu", dtype=torch.bfloat16),
                        dtype="bfloat16",
                    )
                    converted = MaseMXCacheConverter(
                        expected_payload.format_id,
                        block_size=expected_payload.block_size,
                    ).convert(
                        source,
                        role,
                        layer_index,
                        artifact.precision_id,
                        artifact.layout_id,
                    )
                    expected = converted.numerical_view.to_torch(appended.device)
                    if appended.dtype != torch.bfloat16 or not torch.equal(
                        appended,
                        expected,
                    ):
                        raise AssertionError(
                            f"layer {layer_index} {role} append is outside "
                            f"{expected_payload.format_id}"
                        )
                    quantized_tensor_checks += 1
            self._native_append_validation_calls += 1
            self._native_append_tensor_checks += tensor_checks
            self._native_append_quantized_tensor_checks += quantized_tensor_checks
        finally:
            self._native_append_validation_seconds += time.perf_counter() - started

    def _load_or_create_admitted(
        self,
        sample: DecodeTokenSample,
        key_format: str,
        value_format: str,
        provenance: ArtifactProvenance,
        prefill: Any | None = None,
        artifact_root: Path | None = None,
    ) -> tuple[Path, DecodeCacheArtifact, Any]:
        root = self.admission_root if artifact_root is None else artifact_root
        path = _decode_cache_path(
            root,
            key_format,
            sample.document_id,
            value_format,
            contract_id=self.admission_contract_id,
        )
        with self._artifact_lock(path):
            if prefill is None:
                prefill = load_prefill_artifact(
                    _prefill_path(self.prefill_root, sample.document_id)
                )
            if prefill.prompt_hash != sample.prompt_hash:
                raise ValueError("prefill artifact prompt hash mismatch")
            if path.exists():
                artifact = load_decode_cache_artifact(path)
                self._validate_admitted(
                    artifact,
                    prefill=prefill,
                    sample=sample,
                    key_format=key_format,
                    value_format=value_format,
                    provenance=provenance,
                )
                return path, artifact, prefill
            if key_format == value_format == "BF16":
                converter: Any = BF16CacheConverter()
                artifact = admit_prefill_cache(
                    prefill,
                    precision_id="BF16",
                    layout_id=self.layout_id,
                    converter=converter,
                    provenance=provenance,
                    metadata={
                        "document_id": sample.document_id,
                        "sample_bundle_hash": self.bundle.canonical_hash,
                        "admission_contract_id": self.admission_contract_id,
                    },
                )
            else:
                if "BF16" in {key_format, value_format}:
                    raise ValueError("split refinement cannot mix BF16 and MX cache")
                key_converter = MaseMXCacheConverter(key_format, block_size=8)
                value_converter = MaseMXCacheConverter(value_format, block_size=8)
                artifact = admit_prefill_cache_split(
                    prefill,
                    precision_id=_kv_precision_id(key_format, value_format),
                    layout_id=self.layout_id,
                    key_converter=FunctionalCacheConverter(key_converter.convert),
                    value_converter=FunctionalCacheConverter(value_converter.convert),
                    key_format=key_format,
                    value_format=value_format,
                    provenance=provenance,
                    metadata={
                        "document_id": sample.document_id,
                        "sample_bundle_hash": self.bundle.canonical_hash,
                        "admission_contract_id": self.admission_contract_id,
                    },
                )
            self._validate_admitted(
                artifact,
                prefill=prefill,
                sample=sample,
                key_format=key_format,
                value_format=value_format,
                provenance=provenance,
            )
            save_decode_cache_artifact(artifact, path)
            del artifact
            persisted = load_decode_cache_artifact(path)
            self._validate_admitted(
                persisted,
                prefill=prefill,
                sample=sample,
                key_format=key_format,
                value_format=value_format,
                provenance=provenance,
            )
            return path, persisted, prefill

    def _ensure_admitted(
        self,
        sample: DecodeTokenSample,
        key_format: str,
        value_format: str,
        provenance: ArtifactProvenance,
    ) -> Path:
        return self._load_or_create_admitted(
            sample,
            key_format,
            value_format,
            provenance,
        )[0]

    def _validate_recomputable_admission_catalog(
        self,
        value: Mapping[str, Any],
    ) -> dict[tuple[str, str], Mapping[str, Any]]:
        """Validate content identities without requiring retained tensors."""

        if (
            value.get("schema_version") != ADMISSION_INDEX_SCHEMA
            or value.get("persistence_policy")
            != "content_addressed_recompute_per_format"
            or value.get("admission_contract_id") != self.admission_contract_id
            or value.get("admission_code_revision") != self.admission_code_revision
            or value.get("runtime_environment_fingerprint")
            != self.runtime_environment.logical_fingerprint
            or value.get("sample_bundle_hash") != self.bundle.canonical_hash
            or value.get("layout_id") != self.layout_id
            or tuple(value.get("quantized_formats", ())) != DECODE_FORMATS
            or tuple(value.get("reference_formats", ())) != ("BF16",)
        ):
            raise ValueError("recomputable admission catalog identity mismatch")
        created_at = value.get("admission_provenance_created_at")
        if not isinstance(created_at, str) or not created_at.endswith("Z"):
            raise ValueError("admission catalog lacks a provenance timestamp")
        self.admission_provenance_created_at = created_at
        expected = self._admission_record_keys()
        samples = {
            sample.document_id: sample
            for sample in self.bundle.numerical_screen + self.bundle.hardware_validation
        }
        records: dict[tuple[str, str], Mapping[str, Any]] = {}
        sha256 = re.compile(r"^[0-9a-f]{64}$")
        for record in value.get("records", ()):
            if not isinstance(record, Mapping):
                raise TypeError("admission catalog records must be objects")
            key = (str(record.get("format_id")), str(record.get("document_id")))
            if key in records or key not in expected:
                raise ValueError("admission catalog contains an invalid record key")
            format_id, document_id = key
            expected_relative = _decode_cache_path(
                Path("."),
                format_id,
                document_id,
                contract_id=self.admission_contract_id,
            ).as_posix()
            if (
                record.get("prompt_hash") != samples[document_id].prompt_hash
                or record.get("relative_path") != expected_relative
                or not sha256.fullmatch(str(record.get("artifact_id", "")))
                or not sha256.fullmatch(str(record.get("source_artifact_id", "")))
                or int(record.get("persisted_bytes", 0)) <= 0
                or int(record.get("payload_bytes", 0)) <= 0
                or float(record.get("build_seconds", 0.0)) <= 0.0
            ):
                raise ValueError("admission catalog record is incomplete")
            records[key] = dict(record)
        if set(records) != expected:
            raise ValueError("admission catalog coverage is incomplete")
        logical_bytes = sum(
            int(record["persisted_bytes"]) for record in records.values()
        )
        by_format = {
            format_id: sum(
                int(record["persisted_bytes"])
                for (candidate, _), record in records.items()
                if candidate == format_id
            )
            for format_id in DECODE_FORMATS + ("BF16",)
        }
        resources = value.get("resource_projection")
        if (
            not isinstance(resources, Mapping)
            or resources.get("policy") != "content_addressed_recompute_per_format"
            or int(resources.get("logical_total_bytes", -1)) != logical_bytes
            or int(resources.get("runtime_peak_format_bytes", -1))
            != max(by_format.values())
            or int(resources.get("persisted_after_preparation_bytes", -1)) != 0
            or int(resources.get("required_cold_capacity_bytes", 0)) <= 0
            or int(resources.get("observed_cold_available_bytes", -1))
            < int(resources.get("required_cold_capacity_bytes", 0))
            or int(resources.get("required_host_bytes", 0)) <= 0
            or int(resources.get("observed_host_available_bytes", -1))
            < int(resources.get("required_host_bytes", 0))
            or int(value.get("artifact_count", -1)) != len(records)
            or int(value.get("document_count", -1)) != len(samples)
            or int(value.get("logical_artifact_bytes", -1)) != logical_bytes
            or int(value.get("persisted_bytes", -1)) != 0
        ):
            raise ValueError("admission catalog resource accounting mismatch")
        return records

    def _prepare_recomputable_admission_catalog(
        self,
        *,
        workspace_root: Path,
        workspace_identity: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """Validate every conversion once and retain only its content identity."""

        required_identity = {
            "manifest_hash",
            "run_plan_hash",
            "prompt_manifest_hash",
        }
        if set(workspace_identity) != required_identity or any(
            not str(workspace_identity[key]) for key in required_identity
        ):
            raise ValueError("workspace identity is incomplete")
        self.admission_root.mkdir(parents=True, exist_ok=True)
        validation_path = self.admission_index_path.with_name(
            "numerical_validation.json"
        )
        if self.admission_index_path.exists():
            catalog = load_immutable_json(self.admission_index_path)
            records = self._validate_recomputable_admission_catalog(catalog)
            validation = load_immutable_json(validation_path)
        else:
            self.admission_provenance_created_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )
            formats = DECODE_FORMATS + ("BF16",)
            provenances = {
                format_id: self._admission_provenance(format_id, format_id)
                for format_id in formats
            }
            records_list: list[dict[str, Any]] = []
            numerical_records: list[dict[str, Any]] = []
            format_bytes = {format_id: 0 for format_id in formats}
            max_prefill_bytes = 0
            max_artifact_bytes = 0
            temporary_parent = self.admission_root / ".preparation"
            temporary_parent.mkdir(parents=True, exist_ok=True)
            temporary_root = Path(
                tempfile.mkdtemp(
                    prefix="admission-",
                    dir=temporary_parent,
                )
            )
            try:
                for sample in self.samples:
                    prefill = load_prefill_artifact(
                        _prefill_path(self.prefill_root, sample.document_id)
                    )
                    max_prefill_bytes = max(
                        max_prefill_bytes,
                        _directory_bytes(
                            _prefill_path(
                                self.prefill_root,
                                sample.document_id,
                            )
                        ),
                    )
                    for format_id in formats:
                        started = time.perf_counter()
                        path, artifact, _ = self._load_or_create_admitted(
                            sample,
                            format_id,
                            format_id,
                            provenances[format_id],
                            prefill=prefill,
                            artifact_root=temporary_root,
                        )
                        elapsed = time.perf_counter() - started
                        persisted_bytes = _directory_bytes(path)
                        max_artifact_bytes = max(
                            max_artifact_bytes,
                            persisted_bytes,
                        )
                        format_bytes[format_id] += persisted_bytes
                        numerical_records.append(
                            self._validate_admitted_numerical(
                                artifact,
                                prefill=prefill,
                                document_id=sample.document_id,
                            )
                        )
                        records_list.append(
                            {
                                "format_id": format_id,
                                "document_id": sample.document_id,
                                "prompt_hash": sample.prompt_hash,
                                "source_artifact_id": prefill.artifact_id,
                                "artifact_id": artifact.artifact_id,
                                "relative_path": _decode_cache_path(
                                    Path("."),
                                    format_id,
                                    sample.document_id,
                                    contract_id=self.admission_contract_id,
                                ).as_posix(),
                                "persisted_bytes": persisted_bytes,
                                "payload_bytes": _artifact_bytes(artifact),
                                "build_seconds": elapsed,
                            }
                        )
                        shutil.rmtree(path)
            finally:
                shutil.rmtree(temporary_root, ignore_errors=True)
            records_list.sort(
                key=lambda record: (
                    str(record["document_id"]),
                    str(record["format_id"]),
                )
            )
            logical_bytes = sum(
                int(record["persisted_bytes"]) for record in records_list
            )
            footprint = estimate_artifact_footprint(
                self.config,
                self.model_architecture,
            )
            observed_cold = shutil.disk_usage(self.admission_root).free
            required_host = max(
                1,
                max_prefill_bytes + 2 * max_artifact_bytes,
            )
            observed_host = _available_host_bytes()
            catalog_body = {
                "schema_version": ADMISSION_INDEX_SCHEMA,
                "persistence_policy": "content_addressed_recompute_per_format",
                "admission_contract_id": self.admission_contract_id,
                "admission_code_revision": self.admission_code_revision,
                "admission_provenance_created_at": (
                    self.admission_provenance_created_at
                ),
                "runtime_environment_fingerprint": (
                    self.runtime_environment.logical_fingerprint
                ),
                "sample_bundle_hash": self.bundle.canonical_hash,
                "layout_id": self.layout_id,
                "block_size": 8,
                "quantized_formats": list(DECODE_FORMATS),
                "reference_formats": ["BF16"],
                "document_count": len(self.samples),
                "artifact_count": len(records_list),
                "records": records_list,
                "logical_artifact_bytes": logical_bytes,
                "persisted_bytes": 0,
                "resource_projection": {
                    "policy": "content_addressed_recompute_per_format",
                    "logical_total_bytes": logical_bytes,
                    "runtime_peak_format_bytes": max(format_bytes.values()),
                    "persisted_after_preparation_bytes": 0,
                    "required_cold_capacity_bytes": (
                        footprint.required_workspace_bytes
                    ),
                    "observed_cold_available_bytes": observed_cold,
                    "required_host_bytes": required_host,
                    "observed_host_available_bytes": observed_host,
                },
            }
            write_immutable_json(self.admission_index_path, catalog_body)
            catalog = load_immutable_json(self.admission_index_path)
            records = self._validate_recomputable_admission_catalog(catalog)
            tensor_count = sum(
                int(record["tensor_count"]) for record in numerical_records
            )
            write_immutable_json(
                validation_path,
                {
                    "schema_version": ADMISSION_NUMERICAL_VALIDATION_SCHEMA,
                    "passed": True,
                    "basis": "independent_source_recompute_exact_planes",
                    "admission_index_hash": str(catalog["content_hash"]),
                    "admission_contract_id": self.admission_contract_id,
                    "admission_code_revision": self.admission_code_revision,
                    "runtime_environment_fingerprint": (
                        self.runtime_environment.logical_fingerprint
                    ),
                    "sample_bundle_hash": self.bundle.canonical_hash,
                    "layout_id": self.layout_id,
                    "source_dtype": "BF16",
                    "block_size": 8,
                    "quantized_formats": list(DECODE_FORMATS),
                    "reference_formats": ["BF16"],
                    "document_count": len(self.samples),
                    "artifact_count": len(numerical_records),
                    "tensor_count": tensor_count,
                    "records": numerical_records,
                    "steady_state_tpot_included": False,
                },
            )
            validation = load_immutable_json(validation_path)
        if (
            validation.get("schema_version") != ADMISSION_NUMERICAL_VALIDATION_SCHEMA
            or validation.get("passed") is not True
            or validation.get("admission_index_hash") != catalog.get("content_hash")
            or int(validation.get("artifact_count", -1)) != len(records)
        ):
            raise ValueError("admission numerical validation is not catalog-bound")
        receipt = {
            "schema_version": ADMISSION_PREPARATION_SCHEMA,
            **{key: str(workspace_identity[key]) for key in required_identity},
            "persistence_policy": "content_addressed_recompute_per_format",
            "admission_contract_id": self.admission_contract_id,
            "runtime_environment_fingerprint": (
                self.runtime_environment.logical_fingerprint
            ),
            "admission_index_path": str(self.admission_index_path),
            "admission_index_hash": str(catalog["content_hash"]),
            "numerical_validation_path": str(validation_path),
            "numerical_validation_hash": str(validation["content_hash"]),
            "quantized_format_count": len(DECODE_FORMATS),
            "document_count": int(catalog["document_count"]),
            "artifact_count": int(catalog["artifact_count"]),
            "cold_build_seconds": sum(
                float(record["build_seconds"]) for record in catalog["records"]
            ),
            "logical_artifact_bytes": int(catalog["logical_artifact_bytes"]),
            "persisted_bytes": 0,
            "resource_projection": dict(catalog["resource_projection"]),
        }
        write_immutable_json(
            workspace_root / "admission_preparation.json",
            receipt,
        )
        return load_immutable_json(workspace_root / "admission_preparation.json")

    def prepare_admission_catalog(
        self,
        *,
        workspace_root: Path,
        workspace_identity: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """Build all prompt admissions before any profile evaluation starts."""

        if (
            self.executor_config.get("artifact_policy")
            == "content_addressed_recompute_per_format"
        ):
            return self._prepare_recomputable_admission_catalog(
                workspace_root=workspace_root,
                workspace_identity=workspace_identity,
            )

        required_identity = {
            "manifest_hash",
            "run_plan_hash",
            "prompt_manifest_hash",
        }
        if set(workspace_identity) != required_identity or any(
            not str(workspace_identity[key]) for key in required_identity
        ):
            raise ValueError("workspace identity is incomplete")
        workspace_root = workspace_root.resolve()
        if self.admission_index_path.exists():
            catalog = load_immutable_json(self.admission_index_path)
        else:
            resource_projection = self._admission_cold_resource_projection()
            records = []
            prefill_read_seconds = 0.0
            formats = DECODE_FORMATS + ("BF16",)
            provenances = {
                format_id: self._admission_provenance(
                    format_id,
                    format_id,
                )
                for format_id in formats
            }
            for sample in self.samples:
                prefill_started = time.perf_counter()
                source_prefill = load_prefill_artifact(
                    _prefill_path(self.prefill_root, sample.document_id)
                )
                prefill_read_seconds += time.perf_counter() - prefill_started
                for format_id in formats:
                    provenance = provenances[format_id]
                    path = _decode_cache_path(
                        self.admission_root,
                        format_id,
                        sample.document_id,
                        contract_id=self.admission_contract_id,
                    )
                    timing_path = self._admission_timing_path(path)
                    existed = path.exists()
                    started = time.perf_counter()
                    admitted_path, artifact, prefill = self._load_or_create_admitted(
                        sample,
                        format_id,
                        format_id,
                        provenance,
                        prefill=source_prefill,
                    )
                    elapsed = time.perf_counter() - started
                    if not timing_path.exists():
                        if existed:
                            raise RuntimeError(
                                "admitted cache has no measured build-time record"
                            )
                        write_immutable_json(
                            timing_path,
                            {
                                "schema_version": "decode-admission-timing",
                                "admission_contract_id": (self.admission_contract_id),
                                "format_id": format_id,
                                "document_id": sample.document_id,
                                "artifact_id": artifact.artifact_id,
                                "build_seconds": elapsed,
                                "basis": ("conversion_persistence_and_deep_validation"),
                            },
                        )
                    timing = load_immutable_json(timing_path)
                    if (
                        timing.get("artifact_id") != artifact.artifact_id
                        or timing.get("basis")
                        != "conversion_persistence_and_deep_validation"
                    ):
                        raise ValueError("invalid admission build-time record")
                    records.append(
                        {
                            "format_id": format_id,
                            "document_id": sample.document_id,
                            "prompt_hash": sample.prompt_hash,
                            "source_artifact_id": prefill.artifact_id,
                            "artifact_id": artifact.artifact_id,
                            "relative_path": admitted_path.relative_to(
                                self.admission_root
                            ).as_posix(),
                            "persisted_bytes": _directory_bytes(admitted_path),
                            "payload_bytes": _artifact_bytes(artifact),
                            "build_seconds": float(timing["build_seconds"]),
                        }
                    )
            persisted_bytes = sum(int(record["persisted_bytes"]) for record in records)
            payload_bytes = sum(int(record["payload_bytes"]) for record in records)
            artifact_build_seconds = sum(
                float(record["build_seconds"]) for record in records
            )
            build_seconds = artifact_build_seconds + prefill_read_seconds
            if persisted_bytes > int(
                resource_projection["projected_cold_artifact_bytes"]
            ):
                raise RuntimeError(
                    "admission storage exceeds its cold resource projection"
                )
            catalog_body = {
                "schema_version": ADMISSION_INDEX_SCHEMA,
                "admission_contract_id": self.admission_contract_id,
                "admission_code_revision": self.admission_code_revision,
                "runtime_environment_fingerprint": (
                    self.runtime_environment.logical_fingerprint
                ),
                "sample_bundle_hash": self.bundle.canonical_hash,
                "layout_id": self.layout_id,
                "block_size": 8,
                "quantized_formats": list(DECODE_FORMATS),
                "reference_formats": ["BF16"],
                "document_count": len(self.samples),
                "artifact_count": len(records),
                "records": records,
                "cold_build_seconds": build_seconds,
                "prefill_read_seconds": prefill_read_seconds,
                "artifact_build_seconds": artifact_build_seconds,
                "quantized_build_seconds": sum(
                    float(record["build_seconds"])
                    for record in records
                    if record["format_id"] in DECODE_FORMATS
                ),
                "reference_build_seconds": sum(
                    float(record["build_seconds"])
                    for record in records
                    if record["format_id"] == "BF16"
                ),
                "persisted_bytes": persisted_bytes,
                "payload_bytes": payload_bytes,
                "resource_projection": dict(resource_projection),
            }
            write_immutable_json(self.admission_index_path, catalog_body)
            catalog = load_immutable_json(self.admission_index_path)

        _, numerical_records = self._validate_admission_catalog(
            catalog,
            deep=True,
        )
        tensor_count = sum(int(record["tensor_count"]) for record in numerical_records)
        validation_path = self.admission_index_path.with_name(
            "numerical_validation.json"
        )
        write_immutable_json(
            validation_path,
            {
                "schema_version": (ADMISSION_NUMERICAL_VALIDATION_SCHEMA),
                "passed": True,
                "basis": ("independent_source_conversion_exact_persisted_planes"),
                "admission_index_hash": str(catalog["content_hash"]),
                "admission_contract_id": self.admission_contract_id,
                "admission_code_revision": self.admission_code_revision,
                "runtime_environment_fingerprint": (
                    self.runtime_environment.logical_fingerprint
                ),
                "sample_bundle_hash": self.bundle.canonical_hash,
                "layout_id": self.layout_id,
                "source_dtype": "BF16",
                "block_size": 8,
                "quantized_formats": list(DECODE_FORMATS),
                "reference_formats": ["BF16"],
                "document_count": int(catalog["document_count"]),
                "artifact_count": len(numerical_records),
                "tensor_count": tensor_count,
                "records": list(numerical_records),
                "steady_state_tpot_included": False,
            },
        )
        validation = load_immutable_json(validation_path)

        receipt = {
            "schema_version": ADMISSION_PREPARATION_SCHEMA,
            **{key: str(workspace_identity[key]) for key in required_identity},
            "admission_contract_id": self.admission_contract_id,
            "runtime_environment_fingerprint": (
                self.runtime_environment.logical_fingerprint
            ),
            "admission_index_path": str(self.admission_index_path),
            "admission_index_hash": str(catalog["content_hash"]),
            "numerical_validation_path": str(validation_path),
            "numerical_validation_hash": str(validation["content_hash"]),
            "quantized_format_count": len(DECODE_FORMATS),
            "document_count": int(catalog["document_count"]),
            "artifact_count": int(catalog["artifact_count"]),
            "cold_build_seconds": float(catalog["cold_build_seconds"]),
            "persisted_bytes": int(catalog["persisted_bytes"]),
            "resource_projection": dict(catalog["resource_projection"]),
        }
        write_immutable_json(
            workspace_root / "admission_preparation.json",
            receipt,
        )
        return load_immutable_json(workspace_root / "admission_preparation.json")

    @contextmanager
    def open_kv_admission_cache(
        self,
        kv_format: str,
    ) -> Iterator[AdmissionCacheHandle]:
        descriptor = format_descriptor(kv_format)
        if descriptor.family not in {"mxint", "mxfp", "bf16"}:
            raise ValueError(f"unsupported KV format {kv_format!r}")
        if kv_format == "BF16" and any(
            entry.profile.kind != PROFILE_KIND_BF16_REFERENCE
            for entry in self.context.stage_manifest.entries
            if entry.profile.kv_format == kv_format
        ):
            raise ValueError("only the BF16 reference may use BF16 KV")
        if (
            self.executor_config.get("artifact_policy")
            == "content_addressed_recompute_per_format"
        ):
            runtime_parent = self.admission_root / ".runtime"
            runtime_parent.mkdir(parents=True, exist_ok=True)
            runtime_root = Path(
                tempfile.mkdtemp(
                    prefix=f"{kv_format.lower()}-",
                    dir=runtime_parent,
                )
            )
            provenance = self._admission_provenance(kv_format, kv_format)
            paths: dict[str, Path] = {}
            try:
                for sample in self.samples:
                    path, artifact, _ = self._load_or_create_admitted(
                        sample,
                        kv_format,
                        kv_format,
                        provenance,
                        artifact_root=runtime_root,
                    )
                    record = self.admission_records[(kv_format, sample.document_id)]
                    if (
                        artifact.artifact_id != record.get("artifact_id")
                        or artifact.source_artifact_id
                        != record.get("source_artifact_id")
                        or _artifact_bytes(artifact)
                        != int(record.get("payload_bytes", -1))
                        or _directory_bytes(path)
                        != int(record.get("persisted_bytes", -1))
                    ):
                        raise ValueError(
                            "recomputed admission differs from its content identity"
                        )
                    paths[sample.document_id] = path
                yield AdmissionCacheHandle(
                    kv_format=kv_format,
                    paths=paths,
                )
            finally:
                self.cache_lru.clear()
                shutil.rmtree(runtime_root, ignore_errors=True)
            return
        paths = {
            sample.document_id: self.admission_paths[(kv_format, sample.document_id)]
            for sample in self.samples
        }
        yield AdmissionCacheHandle(kv_format=kv_format, paths=paths)

    @contextmanager
    def open_split_kv_admission_cache(
        self,
        key_format: str,
        value_format: str,
    ) -> Iterator[AdmissionCacheHandle]:
        """Open cached prompt admissions for one refinement K/V pair."""

        for role, token in (("key", key_format), ("value", value_format)):
            if format_descriptor(token).family not in {"mxint", "mxfp"}:
                raise ValueError(f"unsupported {role} cache format {token!r}")
        provenance = self._admission_provenance(key_format, value_format)
        if (
            self.executor_config.get("artifact_policy")
            == "content_addressed_recompute_per_format"
        ):
            runtime_parent = self.admission_root / ".runtime"
            runtime_parent.mkdir(parents=True, exist_ok=True)
            runtime_root = Path(
                tempfile.mkdtemp(
                    prefix=f"k-{key_format.lower()}-v-{value_format.lower()}-",
                    dir=runtime_parent,
                )
            )
            try:
                paths = {
                    sample.document_id: self._load_or_create_admitted(
                        sample,
                        key_format,
                        value_format,
                        provenance,
                        artifact_root=runtime_root,
                    )[0]
                    for sample in self.samples
                }
                yield AdmissionCacheHandle(
                    kv_format=key_format,
                    value_format=value_format,
                    paths=paths,
                )
            finally:
                self.cache_lru.clear()
                shutil.rmtree(runtime_root, ignore_errors=True)
            return
        paths = {
            sample.document_id: self._ensure_admitted(
                sample,
                key_format,
                value_format,
                provenance,
            )
            for sample in self.samples
        }
        yield AdmissionCacheHandle(
            kv_format=key_format,
            value_format=value_format,
            paths=paths,
        )

    def _bind_profile(
        self,
        weight_bank: DecodeWeightBank,
        profile: DecodePrecisionProfile,
    ) -> BindingMeasurement:
        return self._bind_precision_spec(
            weight_bank,
            profile_to_decode_quant_spec(profile),
        )

    def bind_refinement_profile(
        self,
        weight_bank: DecodeWeightBank,
        profile: Any,
    ) -> BindingMeasurement:
        """Rebind one refinement profile without requantizing weights."""

        from decode_dse.software.refinement_schedule import (
            DecodeRefinementProfile,
            refinement_profile_to_decode_quant_spec,
        )

        if not isinstance(profile, DecodeRefinementProfile):
            raise TypeError("profile must be a DecodeRefinementProfile")
        if profile.weight_format != weight_bank.weight_format:
            raise ValueError("refinement profile does not match the weight bank")
        if profile.weight_method != weight_bank.weight_method:
            raise ValueError(
                "refinement weight method does not match the open weight bank"
            )
        return self._bind_precision_spec(
            weight_bank,
            refinement_profile_to_decode_quant_spec(profile),
        )

    def _bind_precision_spec(
        self,
        weight_bank: DecodeWeightBank,
        spec: Any | None,
    ) -> BindingMeasurement:
        identity_before = weight_bank.identity_guard.verify(weight_bank.model)
        quantization_before = weight_bank.quantization_guard.verify()
        started = time.perf_counter()
        if spec is None:
            identity_after = weight_bank.identity_guard.verify(weight_bank.model)
            quantization_after = weight_bank.quantization_guard.verify()
            return BindingMeasurement(
                performed=False,
                seconds=time.perf_counter() - started,
                target_count=0,
                used_cached_targets=True,
                weight_requantizations=(quantization_after - quantization_before),
                sealed_weight_modules=len(weight_bank.quantization_guard.modules),
                weight_quantization_events_before=quantization_before,
                weight_quantization_events_after=quantization_after,
                identity_before=identity_before,
                identity_after=identity_after,
                structure_fingerprint=(
                    weight_bank.identity_guard.structure_fingerprint
                ),
            )
        from chop.nn.quantized.functional.vector import VectorRoundingPolicy
        from chop.nn.quantized.modules.llama.rms_norm import (
            build_vector_phase_policies,
        )
        from chop.nn.quantized.modules.phase_config import (
            normalize_phase_q_config,
            resolve_module_phase_config,
        )
        from chop.nn.quantized.modules.llama import (
            LlamaAttentionMXFP,
            LlamaAttentionMXInt,
        )
        from chop.nn.quantized.modules.qwen3 import (
            Qwen3AttentionMXFP,
            Qwen3AttentionMXInt,
        )
        from decode_dse.software.precision_bindings import build_decode_pass_args

        pass_args = build_decode_pass_args(
            str(self.config["model_name"]),
            self.device,
            spec,
        )
        resolved = weight_bank.binding_plan.resolve(pass_args)
        for target, config in resolved:
            module = target.module
            phase = normalize_phase_q_config(config)
            decode = resolve_module_phase_config(phase, "decode")

            if hasattr(module, "_init_phase_state"):
                module._init_phase_state(config)
            elif hasattr(module, "_init_phase_attention_config"):
                attention_classes = {
                    ("llama", "mxint"): LlamaAttentionMXInt,
                    ("llama", "mxfp"): LlamaAttentionMXFP,
                    ("qwen3", "mxint"): Qwen3AttentionMXInt,
                    ("qwen3", "mxfp"): Qwen3AttentionMXFP,
                }
                attention_class = attention_classes[(self.model_family, spec.act_fmt)]
                if module.__class__ is not attention_class:
                    module.__class__ = attention_class
                module._init_phase_attention_config(config)
            elif hasattr(module, "_init_phase_mlp_config"):
                module._init_phase_mlp_config(
                    getattr(module, "layer_idx", None),
                    config,
                )
            elif hasattr(module, "_phase_policies") and hasattr(
                module, "variance_epsilon"
            ):
                module.q_config = config
                module.phase_q_config = phase
                module.decode_policy = phase["decode_policy"]
                module._phase_policies = {
                    key: build_vector_phase_policies(
                        resolve_module_phase_config(phase, key)
                    )
                    for key in ("prefill", "decode")
                }
                module.bypass = decode.get("bypass", False)
            elif hasattr(module, "_phase_policies"):
                module.q_config = config
                module.phase_q_config = phase
                module.decode_policy = phase["decode_policy"]
                module._phase_policies = {
                    key: VectorRoundingPolicy.from_config(
                        resolve_module_phase_config(phase, key)
                    )
                    for key in ("prefill", "decode")
                }
                module.bypass = not module._phase_policies["decode"].enabled
            else:
                raise RuntimeError(
                    f"cannot bind decode precision to module {target.name}"
                )
        identity_after = weight_bank.identity_guard.verify(weight_bank.model)
        quantization_after = weight_bank.quantization_guard.verify()
        return BindingMeasurement(
            performed=True,
            seconds=time.perf_counter() - started,
            target_count=len(resolved),
            used_cached_targets=True,
            weight_requantizations=(quantization_after - quantization_before),
            sealed_weight_modules=len(weight_bank.quantization_guard.modules),
            weight_quantization_events_before=quantization_before,
            weight_quantization_events_after=quantization_after,
            identity_before=identity_before,
            identity_after=identity_after,
            structure_fingerprint=(weight_bank.identity_guard.structure_fingerprint),
        )

    def _validity_for(self, entry: SweepManifestEntry) -> StackValidity:
        measured = self.stack_validity.get(entry.profile_id, entry.validity)
        return constrain_stack_validity(
            entry.profile,
            merge_stack_validity(
                measured,
                StackValidity(software_valid=True),
            ),
        )

    def evaluate(
        self,
        entry: SweepManifestEntry,
        *,
        weight_bank: DecodeWeightBank,
        kv_admission_cache: AdmissionCacheHandle,
    ) -> EvaluationOutcome:
        import torch

        if entry.profile.weight_format != weight_bank.weight_format:
            raise ValueError("profile does not match the open weight bank")
        if entry.profile.kv_format != kv_admission_cache.kv_format:
            raise ValueError("profile does not match the admitted KV cache")
        self._native_append_validation_calls = 0
        self._native_append_tensor_checks = 0
        self._native_append_quantized_tensor_checks = 0
        self._native_append_validation_seconds = 0.0
        binding = self._bind_profile(weight_bank, entry.profile)
        cuda_device = weight_bank.device if weight_bank.device.type == "cuda" else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)

        from chop.nn.quantized.modules.phase_context import force_runtime_phase

        backend = TorchHFCachedDecodeBackend(
            device=weight_bank.device,
            append_validator=(
                self._validate_native_append if self.deep_append_validation else None
            ),
            native_append_format=True,
        )
        documents = []
        steps = self.context.sample_contract.decode_steps
        with force_runtime_phase("decode"):
            for offset in range(0, len(self.samples), self.decode_microbatch_size):
                batch_samples = self.samples[
                    offset : offset + self.decode_microbatch_size
                ]
                examples = []
                for sample in batch_samples:
                    prefill = load_prefill_artifact(
                        _prefill_path(self.prefill_root, sample.document_id)
                    )
                    admitted = self.cache_lru.get(
                        kv_admission_cache.paths[sample.document_id]
                    )
                    first_token = prefill.first_token.token_ids[0]
                    examples.append(
                        ContinuationExample(
                            document_id=sample.document_id,
                            prefill=prefill,
                            decode_cache=admitted,
                            continuation_ids=(
                                first_token,
                                *sample.decode_target_ids[:steps],
                            ),
                        )
                    )
                documents.extend(
                    evaluate_teacher_forced_cached_batched(
                        weight_bank.model,
                        examples,
                        backend,
                    )
                )
                del examples

        expected_append_calls = (
            math.ceil(len(self.samples) / self.decode_microbatch_size) * steps
            if self.deep_append_validation
            else 0
        )
        expected_tensor_checks = (
            expected_append_calls
            * int(self.model_architecture["num_hidden_layers"])
            * 2
        )
        expected_quantized_tensor_checks = (
            0
            if entry.profile.kind == PROFILE_KIND_BF16_REFERENCE
            else expected_tensor_checks
        )
        observed_append_validation = (
            self._native_append_validation_calls,
            self._native_append_tensor_checks,
            self._native_append_quantized_tensor_checks,
        )
        expected_append_validation = (
            expected_append_calls,
            expected_tensor_checks,
            expected_quantized_tensor_checks,
        )
        if observed_append_validation != expected_append_validation:
            raise AssertionError(
                "native append oracle coverage differs from the stage contract"
            )

        nll_sum = sum(document.nll_sum for document in documents)
        token_count = sum(document.token_count for document in documents)
        if token_count != len(self.samples) * steps:
            raise AssertionError("decode token count differs from the stage contract")
        cluster_by_document = {
            sample.document_id: sample.source_cluster_id for sample in self.samples
        }
        weight_bank.identity_guard.verify(weight_bank.model)
        weight_bank.quantization_guard.verify()
        current_runtime = self._capture_current_runtime_environment()
        mean_nll = nll_sum / token_count
        perplexity = (
            math.exp(mean_nll) if mean_nll <= math.log(sys.float_info.max) else None
        )
        gpu_memory = None
        runtime_environment = {
            **dict(current_runtime.logical),
            **dict(current_runtime.observation),
            "logical_fingerprint": (current_runtime.logical_fingerprint),
            "mase_tree_sha256": _mase_tree_hash(
                _repository_root(),
                self.config,
            ),
        }
        if cuda_device is not None:
            properties = torch.cuda.get_device_properties(cuda_device)
            total_bytes = int(properties.total_memory)
            peak_allocated = int(torch.cuda.max_memory_allocated(cuda_device))
            peak_reserved = int(torch.cuda.max_memory_reserved(cuda_device))
            gpu_memory = {
                "microbatch_size": self.decode_microbatch_size,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "total_device_bytes": total_bytes,
                "peak_reserved_fraction": peak_reserved / total_bytes,
            }
            runtime_environment.update(
                {
                    "device_name": str(properties.name),
                    "device_uuid": str(getattr(properties, "uuid", "unavailable")),
                }
            )
        metrics = {
            "nll_sum": nll_sum,
            "token_count": token_count,
            "mean_nll": mean_nll,
            "mean_token_nll": mean_nll,
            "post_handoff_greedy_conditioned_nll": mean_nll,
            "post_handoff_greedy_conditioned_exp_nll": perplexity,
            "metric_id": "post_handoff_greedy_conditioned_nll/v1",
            "metric_definition": (
                "prefill-greedy handoff token is unscored; subsequent "
                "dataset tokens use cached q_len=1 teacher forcing"
            ),
            "runtime_rebinding": binding.to_dict(),
            "native_append_validation": {
                "mode": (
                    "deep_oracle" if self.deep_append_validation else "preflight_gated"
                ),
                "deep_oracle_enabled": self.deep_append_validation,
                "calls": self._native_append_validation_calls,
                "expected_calls": expected_append_calls,
                "tensor_checks": self._native_append_tensor_checks,
                "expected_tensor_checks": expected_tensor_checks,
                "quantized_tensor_checks": (
                    self._native_append_quantized_tensor_checks
                ),
                "expected_quantized_tensor_checks": (expected_quantized_tensor_checks),
                "oracle_seconds": self._native_append_validation_seconds,
                "q_len": 1,
                "layout_id": self.layout_id,
            },
            "admission_accounting": {
                "prepared_before_profile_execution": True,
                "included_in_steady_state_decode_runtime": False,
                "preparation_receipt": str(
                    self.context.workspace_root / "admission_preparation.json"
                ),
            },
            "decode_microbatch": {
                "configured_size": self.decode_microbatch_size,
                "independent_cache_count": len(self.samples),
                "equal_length_required": True,
            },
            "gpu_memory": gpu_memory,
            "runtime_environment": runtime_environment,
            "weight_bank": {
                "weight_format": weight_bank.weight_format,
                "weight_method": weight_bank.weight_method,
                "build_seconds": weight_bank.build_seconds,
                "parameter_count": len(weight_bank.identity_guard.parameters),
                "identity_fingerprint": weight_bank.identity_guard.fingerprint,
                "structure_fingerprint": (
                    weight_bank.identity_guard.structure_fingerprint
                ),
            },
            "sample_contract": self.context.sample_contract.to_dict(),
            "source_cluster_count": len(set(cluster_by_document.values())),
            "documents": [
                {
                    "document_id": document.document_id,
                    "source_cluster_id": cluster_by_document[document.document_id],
                    "nll_sum": document.nll_sum,
                    "token_count": document.token_count,
                    "mean_token_nll": document.mean_nll,
                    "first_token_id": document.first_token_id,
                    "initial_cache_length": document.initial_cache_length,
                    "final_cache_length": document.final_cache_length,
                }
                for document in documents
            ],
        }
        return EvaluationOutcome(
            metrics=metrics,
            validity=self._validity_for(entry),
            artifacts=(
                str(self.sample_bundle_path),
                str(self.prefill_root / "index.json"),
                str(self.admission_index_path),
                str(self.context.workspace_root / "admission_preparation.json"),
            ),
        )


def create_executor(context: ExecutorContext) -> DecodeEvaluator:
    """Build the configured dense-decoder exhaustive executor."""

    return DecodeEvaluator(context)


__all__ = [
    "AdmissionCacheHandle",
    "ADMISSION_INDEX_SCHEMA",
    "ADMISSION_PREPARATION_SCHEMA",
    "BindingMeasurement",
    "DecodeBindingPlan",
    "DecodeBindingTarget",
    "DecodeWeightBank",
    "DecodeWeightBankIdentity",
    "MaseMXCacheConverter",
    "PACKED_CACHE_LAYOUT",
    "DecodeEvaluator",
    "WeightParameterIdentity",
    "build_decode_binding_plan",
    "create_executor",
]
