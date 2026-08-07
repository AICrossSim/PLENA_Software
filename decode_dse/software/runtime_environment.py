"""Numerical-runtime identity for sweep artifacts.

GPU kernel selection, TF32 policy, and library versions all change decode
numerics, so results are only comparable within one recorded stack.
"""

from __future__ import annotations

import hashlib
import fcntl
import importlib.machinery
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from packaging.version import InvalidVersion, Version

from decode_dse.manifest import validate_sweep_config
from decode_dse.software.sweep_plan import (
    HARDWARE_VALIDATION_SAMPLE_CONTRACT,
    NUMERICAL_SCREEN_SAMPLE_CONTRACT,
    build_quantizer_provenance,
    load_immutable_json,
    write_immutable_json,
)

#: Stages that persist an admitted decode cache, in schedule order.
STAGE_SAMPLE_CONTRACTS = (
    NUMERICAL_SCREEN_SAMPLE_CONTRACT,
    HARDWARE_VALIDATION_SAMPLE_CONTRACT,
)

RUNTIME_ENVIRONMENT_SCHEMA = "decode-runtime-environment"
LAUNCH_PREFLIGHT_SCHEMA = "decode-launch-preflight"
#: The decode cache layout every config must declare, parameterised by MLEN.
_PACKED_CACHE_LAYOUT = re.compile(
    r"^packed-gqa-mlen(?P<mlen>[0-9]+)-block8-native-encoding$"
)
#: Architecture field a config declares for the parameter arithmetic. It is not
#: matched against the snapshot because the released configs imply it from the
#: model type rather than stating it.
_QK_NORM_FIELD = "use_qk_norm"
#: Architecture fields a config must pin and the cached snapshot must match.
_ARCHITECTURE_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "tie_word_embeddings",
    "attention_bias",
)
_MIB = 1 << 20
_GIB = 1 << 30
_ACTIVATION_LIVE_TENSOR_FACTOR = 4
_FRAMEWORK_WORKSPACE_HEADROOM_BYTES = 4 * _GIB
_ADMISSION_PERSISTED_BYTES_PER_ELEMENT = {
    "MXINT2": 2.0 + 2 / 8 + 1 / 8,
    "MXINT4": 2.0 + 4 / 8 + 1 / 8,
    "MXINT8": 2.0 + 8 / 8 + 1 / 8,
    "E1M2": 2.0 + 4 / 8 + 1 / 8,
    "E2M1": 2.0 + 4 / 8 + 1 / 8,
    "E3M4": 2.0 + 8 / 8 + 1 / 8,
    "E4M3": 2.0 + 8 / 8 + 1 / 8,
    "E5M2": 2.0 + 8 / 8 + 1 / 8,
    "BF16": 2.0 + 2.0,
}
#: Per-prompt receipt, index, and numerical-view overhead beside the planes.
_ADMISSION_METADATA_BYTES_PER_PROMPT = 9 * _MIB
_PREFILL_METADATA_BYTES_PER_PROMPT = 1 * _MIB


@dataclass(frozen=True)
class ArtifactFootprintEstimate:
    """Logical writes and peak retained bytes for recomputable cache artifacts."""

    prefill_persisted_bytes: int
    admission_total_logical_bytes: int
    admission_peak_persisted_bytes: int
    metadata_bytes: int
    reserve_bytes: int
    required_workspace_bytes: int
    concurrent_kv_formats: int
    policy: str = "content_addressed_recompute_per_format"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "prefill_persisted_bytes": self.prefill_persisted_bytes,
            "admission_total_logical_bytes": self.admission_total_logical_bytes,
            "total_logical_workspace_bytes": (
                self.prefill_persisted_bytes + self.admission_total_logical_bytes
            ),
            "admission_peak_persisted_bytes": self.admission_peak_persisted_bytes,
            "metadata_bytes": self.metadata_bytes,
            "reserve_bytes": self.reserve_bytes,
            "required_workspace_bytes": self.required_workspace_bytes,
            "concurrent_kv_formats": self.concurrent_kv_formats,
        }


def _parse_compute_capability(token: str) -> tuple[int, int] | None:
    """Split an `sm_90`/`compute_100` token into (major, minor)."""

    text = str(token).strip().lower()
    for prefix in ("sm_", "compute_"):
        if text.startswith(prefix):
            digits = text[len(prefix) :]
            if digits.isdigit() and len(digits) >= 2:
                return int(digits[:-1]), int(digits[-1])
    return None


def _architecture_supported(
    capability: str,
    arch_list: Sequence[str],
) -> bool:
    """Return whether a build covers a device capability.

    A cubin built for `sm_XY` also runs on `sm_XZ` for Z >= Y, so a device is
    covered by any listed architecture with the same major version and a minor
    version no greater than the device's.
    """

    device = _parse_compute_capability(capability)
    if device is None:
        return False
    for token in arch_list:
        built = _parse_compute_capability(token)
        if built is None:
            continue
        if built[0] == device[0] and built[1] <= device[1]:
            return True
    return False


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _backend_value(owner: Any, name: str) -> Any:
    try:
        value = getattr(owner, name)
        return value() if callable(value) else value
    except (AttributeError, RuntimeError):
        return None


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Logical numerical stack plus non-identity device observations."""

    logical: Mapping[str, Any]
    observation: Mapping[str, Any]
    schema_version: str = RUNTIME_ENVIRONMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_ENVIRONMENT_SCHEMA:
            raise ValueError("unsupported runtime-environment schema")
        if not self.logical or not self.observation:
            raise ValueError("runtime environment requires logical and observed data")

    @property
    def logical_fingerprint(self) -> str:
        return _canonical_hash(dict(self.logical))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_fingerprint": self.logical_fingerprint,
            "logical": dict(self.logical),
            "observation": dict(self.observation),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeEnvironment":
        environment = cls(
            schema_version=str(value["schema_version"]),
            logical=dict(value["logical"]),
            observation=dict(value["observation"]),
        )
        if value.get("logical_fingerprint") != environment.logical_fingerprint:
            raise ValueError("runtime-environment fingerprint mismatch")
        return environment


def initialize_numerical_runtime(seed: int) -> None:
    """Set the deterministic policy before model construction or calibration."""

    if seed < 0:
        raise ValueError("runtime seed must be non-negative")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import random

    random.seed(seed)
    try:
        import numpy
    except ImportError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(seed)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("runtime initialization requires torch") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for flag in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if hasattr(torch.backends.cuda.matmul, flag):
            setattr(torch.backends.cuda.matmul, flag, False)


def capture_runtime_environment(device: str, *, seed: int) -> RuntimeEnvironment:
    """Capture fields that can change decode numerical results."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("runtime capture requires torch") from exc

    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("the publication sweep requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA execution requested but CUDA is unavailable")
    index = selected.index or 0
    properties = torch.cuda.get_device_properties(index)
    logical = {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "numpy": importlib.metadata.version("numpy"),
        "seed": int(seed),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "device_type": "cuda",
        "device_name": str(properties.name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": _backend_value(torch.backends.cudnn, "allow_tf32"),
        "cuda_matmul_allow_tf32": _backend_value(
            torch.backends.cuda.matmul,
            "allow_tf32",
        ),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": _backend_value(
            torch.backends.cuda.matmul,
            "allow_fp16_reduced_precision_reduction",
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": _backend_value(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction",
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    observation = {
        "device_index": index,
        "device_uuid": str(getattr(properties, "uuid", "unavailable")),
        "total_memory_bytes": int(properties.total_memory),
    }
    return RuntimeEnvironment(logical=logical, observation=observation)


def seal_runtime_environment(
    path: str | Path,
    environment: RuntimeEnvironment,
) -> Mapping[str, Any]:
    """Create or verify one immutable workspace runtime contract."""

    destination = Path(path)
    write_immutable_json(destination, environment.to_dict())
    return load_immutable_json(destination)


def require_runtime_environment(
    path: str | Path,
    current: RuntimeEnvironment,
) -> RuntimeEnvironment:
    """Reject execution under a different logical numerical stack."""

    recorded = RuntimeEnvironment.from_dict(load_immutable_json(Path(path)))
    if recorded.logical_fingerprint != current.logical_fingerprint:
        raise ValueError("runtime environment differs from the immutable workspace")
    return recorded


@dataclass(frozen=True)
class GPUObservation:
    """Capacity and identity observed for one visible CUDA device."""

    index: int
    name: str
    total_bytes: int
    free_bytes: int
    compute_capability: str | None = None
    bf16_supported: bool | None = None


@dataclass(frozen=True)
class MutablePathObservation:
    """Plan-time write, lock, and capacity observation for one mutable root."""

    label: str
    path: str
    writable: bool
    lockable: bool
    free_bytes: int
    error: str | None = None


@dataclass(frozen=True)
class LaunchEnvironmentObservation:
    """Fast, model-load-free observations consumed by the aggregate gate."""

    package_versions: Mapping[str, str | None]
    cuda_devices: tuple[GPUObservation, ...]
    host_available_bytes: int | None
    model_snapshot: str | None
    model_config: Mapping[str, Any]
    model_weight_bytes: int | None
    model_assets_complete: bool
    model_asset_error: str | None
    dataset_assets: Mapping[str, str | None]
    mase_origin: str | None
    mutable_paths: tuple[MutablePathObservation, ...]
    collection_errors: tuple[str, ...] = ()
    #: CUDA architectures the installed torch wheel was built for.
    torch_arch_list: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelMemoryEstimate:
    """Single-device BF16 memory arithmetic for one execution mode."""

    mode: str
    parameter_count: int
    weight_bytes: int
    kv_cache_bytes: int
    activation_headroom_bytes: int
    framework_workspace_bytes: int
    computed_required_bytes: int
    configured_floor_bytes: int
    required_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "parameter_count": self.parameter_count,
            "dtype": "bfloat16",
            "dtype_bytes": 2,
            "weight_bytes": self.weight_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "activation_headroom_bytes": self.activation_headroom_bytes,
            "framework_workspace_bytes": self.framework_workspace_bytes,
            "computed_required_bytes": self.computed_required_bytes,
            "configured_floor_bytes": self.configured_floor_bytes,
            "required_bytes": self.required_bytes,
            "required_mib": self.required_bytes / _MIB,
        }


@dataclass(frozen=True)
class LaunchPreflightCheck:
    """One independently evaluated launch prerequisite."""

    code: str
    passed: bool
    requirement: str
    observed: str
    resolution: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "passed": self.passed,
            "requirement": self.requirement,
            "observed": self.observed,
            "resolution": self.resolution,
        }


class LaunchPreflightError(RuntimeError):
    """Raised after every launch prerequisite has been evaluated."""


@dataclass(frozen=True)
class LaunchPreflightReport:
    """Aggregate report that never hides later failures behind an earlier one."""

    checks: tuple[LaunchPreflightCheck, ...]
    memory_estimates: tuple[ModelMemoryEstimate, ...]
    artifact_footprint: ArtifactFootprintEstimate | None = None
    schema_version: str = LAUNCH_PREFLIGHT_SCHEMA

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "memory_estimates": [
                estimate.to_dict() for estimate in self.memory_estimates
            ],
            "artifact_footprint": (
                self.artifact_footprint.to_dict()
                if self.artifact_footprint is not None
                else None
            ),
        }

    def require_passed(self) -> None:
        failures = tuple(check for check in self.checks if not check.passed)
        if failures:
            lines = [f"launch preflight found {len(failures)} unmet prerequisites:"]
            for index, failure in enumerate(failures, start=1):
                lines.extend(
                    (
                        f"{index}. [{failure.code}] {failure.requirement}",
                        f"   observed: {failure.observed}",
                        f"   action: {failure.resolution}",
                    )
                )
            raise LaunchPreflightError("\n".join(lines))


def dense_decoder_parameter_count(model_config: Mapping[str, Any]) -> int:
    """Compute parameters for a dense GQA decoder from its pinned architecture.

    Covers the Llama and Qwen3 dense families. Qwen3 adds RMSNorm over the
    per-head query and key projections, which Llama does not have; the term is
    included only when the snapshot declares it.
    """

    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    )
    missing = [name for name in required if name not in model_config]
    if missing:
        raise ValueError(
            "pinned model config lacks architecture dimensions: " + ", ".join(missing)
        )
    hidden = int(model_config["hidden_size"])
    intermediate = int(model_config["intermediate_size"])
    layers = int(model_config["num_hidden_layers"])
    attention_heads = int(model_config["num_attention_heads"])
    kv_heads = int(model_config["num_key_value_heads"])
    vocab = int(model_config["vocab_size"])
    head_dim = int(model_config.get("head_dim") or hidden // attention_heads)
    tied = bool(model_config.get("tie_word_embeddings", False))
    biased = bool(model_config.get("attention_bias", False))

    query_dim = attention_heads * head_dim
    kv_dim = kv_heads * head_dim
    attention = hidden * query_dim + 2 * hidden * kv_dim + query_dim * hidden
    if biased:
        attention += query_dim + 2 * kv_dim + hidden
    feed_forward = 3 * hidden * intermediate
    # Input and post-attention RMSNorm; Qwen3 adds q_norm and k_norm.
    layer_norms = 2 * hidden
    if _has_qk_norm(model_config):
        layer_norms += 2 * head_dim

    embedding = vocab * hidden
    output_head = 0 if tied else embedding
    final_norm = hidden
    return (
        embedding
        + output_head
        + layers * (attention + feed_forward + layer_norms)
        + final_norm
    )


def _has_qk_norm(model_config: Mapping[str, Any]) -> bool:
    """Return whether the snapshot declares per-head query/key normalization."""

    if "use_qk_norm" in model_config:
        return bool(model_config["use_qk_norm"])
    architectures = model_config.get("architectures") or ()
    model_type = str(model_config.get("model_type", ""))
    return model_type.startswith("qwen3") or any(
        str(name).lower().startswith("qwen3") for name in architectures
    )


def _architecture_view(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit architecture values, deriving standard head width."""

    result = {field: model_config.get(field) for field in _ARCHITECTURE_FIELDS}
    if result["head_dim"] is None:
        hidden = result["hidden_size"]
        heads = result["num_attention_heads"]
        if (
            isinstance(hidden, int)
            and not isinstance(hidden, bool)
            and isinstance(heads, int)
            and not isinstance(heads, bool)
            and heads > 0
            and hidden % heads == 0
        ):
            result["head_dim"] = hidden // heads
    return result


def _memory_estimate(
    *,
    mode: str,
    model_config: Mapping[str, Any],
    parameter_count: int,
    weight_bytes: int,
    batch_size: int,
    sequence_length: int,
    configured_floor_mib: int,
) -> ModelMemoryEstimate:
    layers = int(model_config["num_hidden_layers"])
    kv_heads = int(model_config["num_key_value_heads"])
    head_dim = int(model_config["head_dim"])
    hidden = int(model_config["hidden_size"])
    kv_cache_bytes = batch_size * sequence_length * layers * 2 * kv_heads * head_dim * 2
    # Hidden state, residual, normalization, and attention staging live set.
    activation_headroom_bytes = (
        batch_size * sequence_length * hidden * 2 * _ACTIVATION_LIVE_TENSOR_FACTOR
    )
    framework_workspace_bytes = _FRAMEWORK_WORKSPACE_HEADROOM_BYTES
    computed = (
        weight_bytes
        + kv_cache_bytes
        + activation_headroom_bytes
        + framework_workspace_bytes
    )
    configured_floor = configured_floor_mib * _MIB
    return ModelMemoryEstimate(
        mode=mode,
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        activation_headroom_bytes=activation_headroom_bytes,
        framework_workspace_bytes=framework_workspace_bytes,
        computed_required_bytes=computed,
        configured_floor_bytes=configured_floor,
        required_bytes=max(computed, configured_floor),
    )


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir():
        raise OSError(f"no existing parent directory for {path}")
    return candidate


def _stable_probe_error(error: BaseException) -> str:
    """Describe a path-probe failure without a random temporary filename."""

    if isinstance(error, OSError) and error.errno is not None:
        reason = error.strerror or os.strerror(error.errno)
        return f"{type(error).__name__}: errno={error.errno} ({reason})"
    return type(error).__name__


def _observe_mutable_path(label: str, path: Path) -> MutablePathObservation:
    temporary_name: str | None = None
    descriptor: int | None = None
    try:
        parent = _nearest_existing_directory(path)
        free_bytes = int(shutil.disk_usage(parent).free)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=".plena-launch-probe-",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        descriptor = os.open(temporary_name, os.O_RDWR)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        descriptor = None
        os.unlink(temporary_name)
        temporary_name = None
        return MutablePathObservation(
            label=label,
            path=str(path),
            writable=True,
            lockable=True,
            free_bytes=free_bytes,
        )
    except Exception as error:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
        try:
            free_bytes = int(shutil.disk_usage(_nearest_existing_directory(path)).free)
        except OSError:
            free_bytes = 0
        return MutablePathObservation(
            label=label,
            path=str(path),
            writable=False,
            lockable=False,
            free_bytes=free_bytes,
            error=_stable_probe_error(error),
        )


def _resolve_workspace_path(value: str, repository: Path, workspace: Path) -> Path:
    if value.startswith("workspace://"):
        suffix = value.removeprefix("workspace://")
        relative = Path(suffix)
        if not suffix or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid workspace-bound path: {value!r}")
        result = (workspace / relative).resolve()
        result.relative_to(workspace.resolve())
        return result
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _model_snapshot_path(config: Mapping[str, Any]) -> Path:
    cache = Path(str(config["hf_cache_dir"])).resolve()
    repo_name = "models--" + str(config["model_name"]).replace("/", "--")
    return cache / repo_name / "snapshots" / str(config["model_revision"])


def _observe_model_assets(
    config: Mapping[str, Any],
) -> tuple[str | None, Mapping[str, Any], int | None, bool, str | None]:
    snapshot: Path | None = None
    model_config: Mapping[str, Any] = {}
    weight_bytes: int | None = None
    try:
        snapshot = _model_snapshot_path(config)
        tokenizer_snapshot = snapshot.parent / str(config["tokenizer_revision"])
        model_config_path = snapshot / "config.json"
        index_path = snapshot / "model.safetensors.index.json"
        tokenizer_paths = (
            tokenizer_snapshot / "tokenizer.json",
            tokenizer_snapshot / "tokenizer_config.json",
        )
        if not snapshot.is_dir() or not model_config_path.is_file():
            raise FileNotFoundError(f"pinned model snapshot is absent: {snapshot}")
        loaded_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_config, Mapping):
            raise ValueError("pinned model config is not a JSON object")
        model_config = loaded_config
        if not index_path.is_file():
            raise FileNotFoundError(f"model shard index is absent: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = tuple(sorted(set(index.get("weight_map", {}).values())))
        missing_shards = tuple(
            name for name in shard_names if not (snapshot / name).is_file()
        )
        if not shard_names or missing_shards:
            raise FileNotFoundError(
                "pinned model shards are absent: "
                + ", ".join(str(snapshot / name) for name in missing_shards)
            )
        weight_bytes = int(index.get("metadata", {}).get("total_size", 0))
        if weight_bytes <= 0:
            raise ValueError("model shard index lacks total_size")
        missing_tokenizer = tuple(
            path for path in tokenizer_paths if not path.is_file()
        )
        if missing_tokenizer:
            raise FileNotFoundError(
                "pinned tokenizer assets are incomplete: "
                + ", ".join(str(path) for path in missing_tokenizer)
            )
        return str(snapshot), model_config, weight_bytes, True, None
    except Exception as error:
        return (
            str(snapshot) if snapshot is not None and snapshot.is_dir() else None,
            model_config,
            weight_bytes,
            False,
            f"{type(error).__name__}: {error}",
        )


def _dataset_asset(
    data: Mapping[str, Any],
) -> str | None:
    try:
        cache = Path(str(data["cache_dir"])).resolve()
        dataset_dir = cache / str(data["dataset_name"]).replace("/", "___")
        config_dir = dataset_dir / str(data["dataset_config"])
        revision = str(data["dataset_revision"])
        matches = tuple(
            path
            for path in config_dir.rglob(revision)
            if path.is_dir() and (path / "dataset_info.json").is_file()
        )
        split = str(data["split"])
        matches = tuple(path for path in matches if any(path.glob(f"*-{split}.arrow")))
        return str(matches[0]) if len(matches) == 1 else None
    except (KeyError, OSError):
        return None


def collect_launch_environment(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    workspace_root: Path,
) -> LaunchEnvironmentObservation:
    """Collect all inexpensive launch observations without loading the model."""

    repository = repository_root.resolve()
    workspace = workspace_root.resolve()
    errors: list[str] = []
    packages: dict[str, str | None] = {}
    for package in (
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "nvidia-ml-py",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    devices: list[GPUObservation] = []
    arch_list: tuple[str, ...] = ()
    try:
        import torch

        arch_list = tuple(str(arch) for arch in torch.cuda.get_arch_list())
        if not arch_list:
            compiled_arches = _backend_value(torch._C, "_cuda_getArchFlags")
            if isinstance(compiled_arches, str):
                arch_list = tuple(compiled_arches.split())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                devices.append(
                    GPUObservation(
                        index=index,
                        name=str(properties.name),
                        total_bytes=int(total),
                        free_bytes=int(free),
                        compute_capability=(f"sm_{properties.major}{properties.minor}"),
                        bf16_supported=bool(properties.major >= 8),
                    )
                )
    except Exception as error:
        errors.append(f"CUDA observation: {type(error).__name__}: {error}")

    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        available_line = next(
            line for line in meminfo.splitlines() if line.startswith("MemAvailable:")
        )
        host_available = int(available_line.split()[1]) * 1024
    except Exception as error:
        host_available = None
        errors.append(f"host memory observation: {type(error).__name__}: {error}")

    snapshot, model_config, weight_bytes, complete, model_error = _observe_model_assets(
        config
    )
    datasets: dict[str, str | None] = {}
    evaluation = config.get("evaluation_data")
    if isinstance(evaluation, Mapping):
        datasets["evaluation"] = _dataset_asset(evaluation)
    else:
        datasets["evaluation"] = None
    refinement = config.get("refinement")
    calibration = (
        refinement.get("calibration_data") if isinstance(refinement, Mapping) else None
    )
    if isinstance(calibration, Mapping):
        datasets["refinement_calibration"] = _dataset_asset(calibration)
    else:
        datasets["refinement_calibration"] = None

    mase_origin: str | None = None
    executor = config.get("executor")
    if isinstance(executor, Mapping) and executor.get("mase_src"):
        mase_root = Path(str(executor["mase_src"]))
        if not mase_root.is_absolute():
            mase_root = repository / mase_root
        mase_root = mase_root.resolve()
        try:
            loaded = sys.modules.get("chop")
            spec = (
                getattr(loaded, "__spec__", None)
                if loaded is not None
                else importlib.machinery.PathFinder.find_spec("chop", [str(mase_root)])
            )
            if spec is not None and spec.origin is not None:
                mase_origin = str(Path(spec.origin).resolve())
        except (ImportError, OSError) as error:
            errors.append(f"MASE observation: {type(error).__name__}: {error}")

    path_values: list[tuple[str, Path]] = [("workspace", workspace)]
    if isinstance(executor, Mapping):
        for key in (
            "prefill_artifact_root",
            "admission_artifact_root",
        ):
            value = executor.get(key)
            if isinstance(value, str):
                try:
                    path_values.append(
                        (key, _resolve_workspace_path(value, repository, workspace))
                    )
                except (OSError, ValueError) as error:
                    errors.append(f"{key}: {type(error).__name__}: {error}")
    scratch = config.get("scratch_dir")
    if isinstance(scratch, str):
        try:
            path_values.append(
                ("scratch_dir", _resolve_workspace_path(scratch, repository, workspace))
            )
        except (OSError, ValueError) as error:
            errors.append(f"scratch_dir: {type(error).__name__}: {error}")
    if isinstance(refinement, Mapping):
        for key in (
            "prefill_artifact_root",
            "admission_artifact_root",
            "checkpoint_root",
        ):
            value = refinement.get(key)
            if isinstance(value, str):
                try:
                    path_values.append(
                        (
                            f"refinement.{key}",
                            _resolve_workspace_path(value, repository, workspace),
                        )
                    )
                except (OSError, ValueError) as error:
                    errors.append(f"refinement.{key}: {type(error).__name__}: {error}")
    observations = tuple(
        _observe_mutable_path(label, path) for label, path in path_values
    )
    return LaunchEnvironmentObservation(
        package_versions=packages,
        cuda_devices=tuple(devices),
        host_available_bytes=host_available,
        model_snapshot=snapshot,
        model_config=model_config,
        model_weight_bytes=weight_bytes,
        model_assets_complete=complete,
        model_asset_error=model_error,
        dataset_assets=datasets,
        mase_origin=mase_origin,
        mutable_paths=observations,
        collection_errors=tuple(errors),
        torch_arch_list=arch_list,
    )


def _append_check(
    checks: list[LaunchPreflightCheck],
    *,
    code: str,
    passed: bool,
    requirement: str,
    observed: str,
    resolution: str,
) -> None:
    checks.append(
        LaunchPreflightCheck(
            code=code,
            passed=bool(passed),
            requirement=requirement,
            observed=observed,
            resolution=resolution,
        )
    )


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _integer(value: Any, default: int = -1) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _version_at_least(observed: str | None, required: str) -> bool:
    """Compare package versions while accepting local build suffixes."""

    if observed is None:
        return False
    try:
        return Version(observed) >= Version(required)
    except InvalidVersion:
        return False


def _projected_admission_bytes(model_config: Mapping[str, Any]) -> int:
    """Return total logical admission bytes written across all cache formats."""

    layers = int(model_config["num_hidden_layers"])
    kv_heads = int(model_config["num_key_value_heads"])
    hidden = int(model_config["hidden_size"])
    attention_heads = int(model_config["num_attention_heads"])
    head_dim = int(model_config.get("head_dim") or hidden // attention_heads)
    bytes_per_element = sum(_ADMISSION_PERSISTED_BYTES_PER_ELEMENT.values())
    total = 0
    for contract in STAGE_SAMPLE_CONTRACTS:
        elements = layers * 2 * kv_heads * contract.prefill_tokens * head_dim
        total += math.ceil(contract.prompt_count * elements * bytes_per_element)
        total += contract.prompt_count * _ADMISSION_METADATA_BYTES_PER_PROMPT
    return total


def estimate_artifact_footprint(
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    *,
    concurrent_workers: int | None = None,
) -> ArtifactFootprintEstimate:
    """Derive workspace demand from model geometry and artifact policy."""

    executor = config.get("executor")
    if not isinstance(executor, Mapping):
        raise ValueError("config.executor is required")
    policy = str(
        executor.get(
            "artifact_policy",
            "content_addressed_recompute_per_format",
        )
    )
    if policy != "content_addressed_recompute_per_format":
        raise ValueError("unsupported executor.artifact_policy")
    workers = (
        _integer(config.get("max_parallel_points", 1), 1)
        if concurrent_workers is None
        else int(concurrent_workers)
    )
    if workers <= 0:
        raise ValueError("concurrent workers must be positive")

    layers = int(model_config["num_hidden_layers"])
    kv_heads = int(model_config["num_key_value_heads"])
    hidden = int(model_config["hidden_size"])
    attention_heads = int(model_config["num_attention_heads"])
    head_dim = int(model_config.get("head_dim") or hidden // attention_heads)
    total_elements = 0
    total_prompts = 0
    for contract in STAGE_SAMPLE_CONTRACTS:
        total_elements += (
            contract.prompt_count
            * layers
            * 2
            * kv_heads
            * contract.prefill_tokens
            * head_dim
        )
        total_prompts += contract.prompt_count
    prefill_metadata = total_prompts * _PREFILL_METADATA_BYTES_PER_PROMPT
    prefill_bytes = 2 * total_elements + prefill_metadata
    format_bytes = sorted(
        (
            math.ceil(total_elements * bytes_per_element)
            + total_prompts * _ADMISSION_METADATA_BYTES_PER_PROMPT
        )
        for bytes_per_element in _ADMISSION_PERSISTED_BYTES_PER_ELEMENT.values()
    )
    concurrent_formats = min(workers, len(format_bytes))
    peak_admission = sum(format_bytes[-concurrent_formats:])
    total_admission = sum(format_bytes)
    reserve = int(
        max(
            0.0,
            _finite_float(executor.get("artifact_space_reserve_gib", 8.0), 8.0),
        )
        * _GIB
    )
    safety = _finite_float(executor.get("artifact_space_safety_factor", 1.05), 1.05)
    if safety < 1.0:
        raise ValueError("artifact_space_safety_factor must be at least one")
    configured_floor = int(
        max(
            0.0,
            _finite_float(executor.get("min_free_artifact_gib", 0.0), 0.0),
        )
        * _GIB
    )
    required = max(
        math.ceil((prefill_bytes + peak_admission) * safety) + reserve,
        configured_floor,
    )
    return ArtifactFootprintEstimate(
        prefill_persisted_bytes=prefill_bytes,
        admission_total_logical_bytes=total_admission,
        admission_peak_persisted_bytes=peak_admission,
        metadata_bytes=(
            prefill_metadata
            + total_prompts
            * len(_ADMISSION_PERSISTED_BYTES_PER_ELEMENT)
            * _ADMISSION_METADATA_BYTES_PER_PROMPT
        ),
        reserve_bytes=reserve,
        required_workspace_bytes=required,
        concurrent_kv_formats=concurrent_formats,
    )


def evaluate_launch_preflight(
    config: Mapping[str, Any],
    observation: LaunchEnvironmentObservation,
    *,
    repository_root: Path,
    device_labels: Sequence[str],
) -> LaunchPreflightReport:
    """Evaluate all prerequisites and retain every failure in one report."""

    checks: list[LaunchPreflightCheck] = []
    try:
        validate_sweep_config(config)
        config_error = None
    except Exception as error:
        config_error = str(error)
    executor = config.get("executor")
    refinement = config.get("refinement")
    placement = config.get("model_placement")
    runtime_requirements = config.get("runtime_requirements")
    executor_map = executor if isinstance(executor, Mapping) else {}
    refinement_map = refinement if isinstance(refinement, Mapping) else {}
    placement_map = placement if isinstance(placement, Mapping) else {}
    requirements_map = (
        runtime_requirements if isinstance(runtime_requirements, Mapping) else {}
    )
    placement_valid = (
        placement_map.get("policy") == "single_device"
        and _integer(placement_map.get("device_count", -1)) == 1
        and placement_map.get("automatic_device_map") is False
    )
    policy_requirements = (
        bool(str(config.get("model_name", "")).strip()),
        str(config.get("dtype", "")).lower() == "bfloat16",
        str(config.get("device", "")).startswith("cuda"),
        config.get("local_files_only") is True,
        config.get("trust_remote_code") is False,
        bool(_PACKED_CACHE_LAYOUT.fullmatch(str(executor_map.get("layout_id", "")))),
        executor_map.get("serialize_weight_bank_builds") is True,
        executor_map.get("artifact_policy") == "content_addressed_recompute_per_format",
        str(config.get("scratch_dir", "")).startswith("workspace://"),
        _finite_float(executor_map.get("min_free_artifact_gib", 0)) >= 0.0,
        _finite_float(executor_map.get("artifact_space_safety_factor", -1)) >= 1.0,
        _finite_float(executor_map.get("artifact_space_reserve_gib", -1)) >= 0.0,
        _finite_float(executor_map.get("min_available_host_gib", 0)) >= 0.0,
        _finite_float(executor_map.get("max_cpu_cache_gib", 0)) > 0.0,
        _integer(config.get("gpu_min_free_mb", -1)) >= 0,
        _integer(refinement_map.get("gpu_min_free_mb", -1)) >= 0,
        _finite_float(refinement_map.get("max_cpu_cache_gib", 0)) > 0.0,
        placement_valid,
        bool(requirements_map),
    )
    _append_check(
        checks,
        code="configuration",
        passed=config_error is None and all(policy_requirements),
        requirement=(
            "a named BF16, local-only, native PackedKV decode contract; "
            "workspace scratch; serialized bank builds; positive CPU caches; "
            "non-negative artifact/host/GPU floors; safety factor >=1; and an "
            "explicit single-device placement with automatic mapping disabled"
        ),
        observed=config_error or "configuration fields inspected",
        resolution="restore the publication configuration",
    )

    missing_packages = tuple(
        name
        for name, version in observation.package_versions.items()
        if version is None
    )
    _append_check(
        checks,
        code="packages",
        passed=not missing_packages,
        requirement=(
            "torch, transformers, datasets, numpy, and nvidia-ml-py installed"
        ),
        observed=json.dumps(dict(observation.package_versions), sort_keys=True),
        resolution="enter the PLENA_Software environment containing every runtime package",
    )
    minimum_versions = requirements_map.get("minimum_package_versions")
    version_requirements = (
        dict(minimum_versions) if isinstance(minimum_versions, Mapping) else {}
    )
    version_failures = {
        str(name): {
            "required": str(required),
            "observed": observation.package_versions.get(str(name)),
        }
        for name, required in sorted(version_requirements.items())
        if not _version_at_least(
            observation.package_versions.get(str(name)),
            str(required),
        )
    }
    _append_check(
        checks,
        code="package_versions",
        passed=bool(version_requirements) and not version_failures,
        requirement=("runtime packages meet the model's declared minimum versions"),
        observed=json.dumps(
            {
                "minimum": version_requirements,
                "installed": dict(observation.package_versions),
            },
            sort_keys=True,
        ),
        resolution=(
            "install package versions satisfying runtime_requirements; "
            f"failures={json.dumps(version_failures, sort_keys=True)}"
        ),
    )

    _append_check(
        checks,
        code="model_assets",
        passed=observation.model_assets_complete,
        requirement=(
            "the exact pinned model, tokenizer, shard index, and every shard "
            "cached locally"
        ),
        observed=(
            str(observation.model_snapshot)
            if observation.model_assets_complete
            else str(observation.model_asset_error)
        ),
        resolution="populate the configured Hugging Face cache at the pinned revision",
    )

    declared_architecture = config.get("model_architecture")
    architecture_declared = isinstance(declared_architecture, Mapping) and all(
        field in declared_architecture for field in _ARCHITECTURE_FIELDS
    )
    expected_architecture: Mapping[str, Any] = (
        {field: declared_architecture[field] for field in _ARCHITECTURE_FIELDS}
        if architecture_declared
        else {}
    )
    observed_architecture = _architecture_view(observation.model_config)
    architecture_ok = (
        architecture_declared
        and all(
            observed_architecture.get(key) == value
            for key, value in expected_architecture.items()
        )
        and str(observation.model_config.get("torch_dtype", "")).lower()
        in {"bfloat16", "torch.bfloat16"}
    )
    model_name = str(config.get("model_name", "the configured model"))
    _append_check(
        checks,
        code="model_architecture",
        passed=architecture_ok,
        requirement=(
            f"the cached revision must match the architecture {model_name} pins "
            f"in model_architecture ({', '.join(_ARCHITECTURE_FIELDS)})"
        ),
        observed=(
            json.dumps(
                {key: observed_architecture.get(key) for key in _ARCHITECTURE_FIELDS},
                sort_keys=True,
            )
            if architecture_declared
            else "config does not pin model_architecture"
        ),
        resolution=(
            "pin model_architecture in the config and use the matching snapshot; "
            "do not substitute another size"
        ),
    )
    # Resource arithmetic is declarative: the snapshot is checked for identity,
    # while every required dimension comes from the pinned configuration.
    arithmetic_config: dict[str, Any] = dict(observation.model_config)
    arithmetic_config.update(expected_architecture)
    arithmetic_config.setdefault("torch_dtype", "bfloat16")
    if isinstance(declared_architecture, Mapping) and (
        _QK_NORM_FIELD in declared_architecture
    ):
        arithmetic_config[_QK_NORM_FIELD] = bool(declared_architecture[_QK_NORM_FIELD])

    for label in ("evaluation", "refinement_calibration"):
        asset = observation.dataset_assets.get(label)
        _append_check(
            checks,
            code=f"dataset_{label}",
            passed=asset is not None,
            requirement=(
                f"the exact pinned {label.replace('_', ' ')} dataset revision "
                "and split cached locally"
            ),
            observed=asset or "no matching revision/split cache found",
            resolution="materialize the configured dataset revision in its configured cache",
        )

    configured_mase = Path(str(executor_map.get("mase_src", "")))
    if not configured_mase.is_absolute():
        configured_mase = repository_root.resolve() / configured_mase
    expected_mase = (configured_mase.resolve() / "chop" / "__init__.py").resolve()
    _append_check(
        checks,
        code="mase_origin",
        passed=(
            observation.mase_origin is not None
            and Path(observation.mase_origin).resolve() == expected_mase
        ),
        requirement="chop must resolve from the configured sibling mase/src tree",
        observed=observation.mase_origin or "chop did not resolve",
        resolution="correct executor.mase_src and remove conflicting imports",
    )
    try:
        provenance = build_quantizer_provenance(repository_root, config)
        provenance_observed = (
            f"{len(provenance.sources)} files, {provenance.canonical_hash}"
        )
        provenance_ok = True
    except Exception as error:
        provenance_observed = f"{type(error).__name__}: {error}"
        provenance_ok = False
    _append_check(
        checks,
        code="quantizer_sources",
        passed=provenance_ok,
        requirement=(
            "all MASE, Simulator, RTL, and Software arithmetic sources "
            "resolvable and hashable"
        ),
        observed=provenance_observed,
        resolution="restore every configured arithmetic source tree and decode import origin",
    )

    visible = observation.cuda_devices
    _append_check(
        checks,
        code="cuda",
        passed=bool(visible),
        requirement="CUDA available with visible devices",
        observed=(
            ", ".join(
                f"{gpu.index}:{gpu.name} free={gpu.free_bytes / _MIB:.0f} MiB "
                f"total={gpu.total_bytes / _MIB:.0f} MiB"
                for gpu in visible
            )
            or "no CUDA device is visible"
        ),
        resolution="run on the intended CUDA host before creating the immutable plan",
    )
    required_arches = tuple(
        sorted({gpu.compute_capability for gpu in visible if gpu.compute_capability})
    )
    unsupported = tuple(
        arch
        for arch in required_arches
        if not _architecture_supported(arch, observation.torch_arch_list)
    )
    _append_check(
        checks,
        code="cuda_architecture",
        passed=(
            bool(observation.torch_arch_list)
            and bool(required_arches)
            and not unsupported
        ),
        requirement=(
            "the installed torch wheel must contain kernels for every device "
            "compute capability"
        ),
        observed=(
            f"devices={required_arches or '(none reported)'}; "
            f"torch_arch_list={observation.torch_arch_list or '(unavailable)'}"
        ),
        resolution=(
            "install a torch build whose CUDA architecture list covers the "
            "execution host, for example a cu128 wheel for sm_100"
        ),
    )
    required_capability = str(requirements_map.get("compute_capability", ""))
    capability_matches = bool(required_capability) and all(
        gpu.compute_capability == required_capability for gpu in visible
    )
    _append_check(
        checks,
        code="target_compute_capability",
        passed=capability_matches,
        requirement=(
            f"every execution GPU must be {required_capability or 'declared'}"
        ),
        observed=str(tuple(gpu.compute_capability for gpu in visible)),
        resolution="run preflight on hardware matching the declared capability",
    )
    bf16_ready = bool(visible) and all(gpu.bf16_supported is True for gpu in visible)
    _append_check(
        checks,
        code="bf16",
        passed=bf16_ready,
        requirement="native BF16 support on every execution GPU",
        observed=str(tuple(gpu.bf16_supported for gpu in visible)),
        resolution="run on GPUs with native BF16 support",
    )
    names = tuple(gpu.name.lower() for gpu in visible)
    bad_labels = tuple(
        label
        for label in device_labels
        if label.lower() == "cpu" or not any(label.lower() in name for name in names)
    )
    _append_check(
        checks,
        code="device_labels",
        passed=bool(device_labels) and not bad_labels,
        requirement="each plan device label is a GPU-name substring, never a CUDA ordinal",
        observed=(
            f"labels={tuple(device_labels)!r}; visible_names={tuple(gpu.name for gpu in visible)!r}"
        ),
        resolution="use a stable part-name substring such as 'a6000', not 'cuda:0'",
    )

    try:
        parameter_count = dense_decoder_parameter_count(arithmetic_config)
        parameter_count_error: str | None = None
    except Exception as error:
        parameter_count = 0
        parameter_count_error = f"{type(error).__name__}: {error}"
    expected_weight_bytes = parameter_count * 2
    observed_weight_bytes = observation.model_weight_bytes
    weight_bytes = expected_weight_bytes
    weight_identity_ok = (
        parameter_count > 0
        and observed_weight_bytes is not None
        and observed_weight_bytes == expected_weight_bytes
    )
    _append_check(
        checks,
        code="model_weight_size",
        passed=weight_identity_ok,
        requirement=f"parameter_count x BF16 bytes = {expected_weight_bytes:,} bytes",
        observed=(
            parameter_count_error
            or f"parameters={parameter_count:,}; "
            f"shard_index_total={observed_weight_bytes!r} bytes"
        ),
        resolution="use the exact pinned BF16 shard set",
    )
    microbatches = executor_map.get("decode_microbatch_size", {})
    numerical_batch = max(
        1,
        _integer(
            (
                microbatches.get("numerical_screen", 16)
                if isinstance(microbatches, Mapping)
                else 16
            ),
            16,
        ),
    )
    hardware_batch = max(
        1,
        _integer(
            (
                microbatches.get("hardware_validation", 8)
                if isinstance(microbatches, Mapping)
                else 8
            ),
            8,
        ),
    )
    try:
        estimates = (
            _memory_estimate(
                mode="numerical-screen",
                model_config=arithmetic_config,
                parameter_count=parameter_count,
                weight_bytes=weight_bytes,
                batch_size=numerical_batch,
                sequence_length=512 + 16,
                configured_floor_mib=_integer(config.get("gpu_min_free_mb", 0), 0),
            ),
            _memory_estimate(
                mode="hardware-validation",
                model_config=arithmetic_config,
                parameter_count=parameter_count,
                weight_bytes=weight_bytes,
                batch_size=hardware_batch,
                sequence_length=512 + 32,
                configured_floor_mib=_integer(config.get("gpu_min_free_mb", 0), 0),
            ),
            _memory_estimate(
                mode="refinement",
                model_config=arithmetic_config,
                parameter_count=parameter_count,
                weight_bytes=weight_bytes,
                batch_size=max(
                    1,
                    _integer(refinement_map.get("calibration_batch_size", 8), 8),
                ),
                sequence_length=max(
                    1,
                    _integer(
                        refinement_map.get("calibration_sequence_length", 2048),
                        2048,
                    ),
                ),
                configured_floor_mib=_integer(
                    refinement_map.get("gpu_min_free_mb", 0), 0
                ),
            ),
        )
    except Exception:
        estimates = ()
    capacity_options = (
        "use a larger single GPU with at least the reported total/free capacity to "
        "preserve semantics, or explicitly redesign _load_model for multi-device "
        "sharding; multi-device placement changes timing and runtime identity and "
        "requires a new hashed protocol. Automatic device_map remains disabled"
    )
    for estimate in estimates:
        matching = tuple(
            gpu
            for gpu in visible
            if any(label.lower() in gpu.name.lower() for label in device_labels)
        )
        qualifying = tuple(
            gpu
            for gpu in matching
            if gpu.total_bytes >= estimate.required_bytes
            and gpu.free_bytes >= estimate.required_bytes
        )
        needed_count = (
            max(1, _integer(config.get("max_parallel_points", 1), 1))
            if estimate.mode == "refinement"
            else 1
        )
        observed_capacity = (
            ", ".join(
                f"{gpu.name}: free {gpu.free_bytes / _MIB:.0f} MiB, "
                f"total {gpu.total_bytes / _MIB:.0f} MiB"
                for gpu in matching
            )
            or "no matching visible GPU"
        )
        # A shortfall in device *count* with sufficient per-device capacity is a
        # scheduling choice, not a hardware limit, and has a different remedy.
        count_limited = (
            len(qualifying) < needed_count
            and len(qualifying) == len(matching)
            and bool(matching)
        )
        resolution = (
            (
                f"{needed_count} concurrent workers were requested but only "
                f"{len(qualifying)} qualifying device(s) are available; set "
                f"max_parallel_points to at most {len(qualifying)} or run on a "
                "host with more devices of this size"
            )
            if count_limited
            else capacity_options
        )
        _append_check(
            checks,
            code=f"gpu_memory_{estimate.mode.replace('-', '_')}",
            passed=len(qualifying) >= needed_count,
            requirement=(
                f"{needed_count} matching single-device GPU(s), each with at least "
                f"{estimate.required_bytes / _MIB:.0f} MiB total and free; arithmetic: "
                f"weights {estimate.weight_bytes / _MIB:.2f} + KV "
                f"{estimate.kv_cache_bytes / _MIB:.2f} + activation headroom "
                f"{estimate.activation_headroom_bytes / _MIB:.2f} + framework/workspace "
                f"{estimate.framework_workspace_bytes / _MIB:.2f} MiB, with configured "
                f"floor {estimate.configured_floor_bytes / _MIB:.0f} MiB"
            ),
            observed=observed_capacity,
            resolution=resolution,
        )

    host_gib = _finite_float(executor_map.get("min_available_host_gib", 0), 0.0)
    host_floor = int(max(0.0, host_gib) * _GIB)
    # Loading materializes one BF16 copy of the weights in host memory before
    # the transfer to device, alongside the configured CPU cache per worker.
    host_cache = int(
        max(0.0, _finite_float(executor_map.get("max_cpu_cache_gib", 0), 0.0)) * _GIB
    )
    host_workers = max(1, _integer(config.get("max_parallel_points", 1), 1))
    host_computed = weight_bytes + host_workers * host_cache
    host_required = max(host_computed, host_floor)
    host_observed = observation.host_available_bytes
    _append_check(
        checks,
        code="host_memory",
        passed=host_observed is not None and host_observed >= host_required,
        requirement=(
            f"at least {host_required / _GIB:.1f} GiB MemAvailable; arithmetic: "
            f"BF16 weights {weight_bytes / _GIB:.2f} + {host_workers} x CPU cache "
            f"{host_cache / _GIB:.2f} GiB, with configured floor "
            f"{host_floor / _GIB:.1f} GiB"
        ),
        observed=(
            f"{host_observed / _GIB:.2f} GiB"
            if host_observed is not None
            else "unavailable"
        ),
        resolution="run on a host meeting the computed available-memory requirement",
    )

    try:
        footprint = estimate_artifact_footprint(config, arithmetic_config)
        disk_required = footprint.required_workspace_bytes
    except Exception:
        footprint = None
        disk_required = 0
    for path_observation in observation.mutable_paths:
        _append_check(
            checks,
            code=f"path_{path_observation.label.replace('.', '_')}",
            passed=(
                path_observation.writable
                and path_observation.lockable
                and path_observation.free_bytes >= disk_required
            ),
            requirement=(
                f"{path_observation.label} writable and flock-capable with at least "
                f"{disk_required / _GIB:.2f} GiB free for retained BF16 prefill, "
                "concurrency-bounded admitted caches, metadata, and reserve"
            ),
            observed=(
                f"path={path_observation.path}; writable={path_observation.writable}; "
                f"lockable={path_observation.lockable}; "
                f"free={path_observation.free_bytes / _GIB:.2f} GiB; "
                f"error={path_observation.error}"
            ),
            resolution="select a workspace filesystem satisfying the immutable artifact policy",
        )
    for index, error in enumerate(observation.collection_errors):
        _append_check(
            checks,
            code=f"observation_{index + 1}",
            passed=False,
            requirement="every environment observation must complete",
            observed=error,
            resolution="repair the reported host or path observation failure",
        )
    return LaunchPreflightReport(
        checks=tuple(checks),
        memory_estimates=estimates,
        artifact_footprint=footprint,
    )


def run_launch_preflight(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
    workspace_root: Path,
    device_labels: Sequence[str],
    observation: LaunchEnvironmentObservation | None = None,
) -> LaunchPreflightReport:
    """Collect and evaluate the complete plan-to-run launch contract."""

    current = observation or collect_launch_environment(
        config,
        repository_root=repository_root,
        workspace_root=workspace_root,
    )
    return evaluate_launch_preflight(
        config,
        current,
        repository_root=repository_root,
        device_labels=device_labels,
    )


__all__ = [
    "GPUObservation",
    "LAUNCH_PREFLIGHT_SCHEMA",
    "LaunchEnvironmentObservation",
    "LaunchPreflightCheck",
    "LaunchPreflightError",
    "LaunchPreflightReport",
    "ModelMemoryEstimate",
    "MutablePathObservation",
    "RUNTIME_ENVIRONMENT_SCHEMA",
    "RuntimeEnvironment",
    "capture_runtime_environment",
    "initialize_numerical_runtime",
    "collect_launch_environment",
    "evaluate_launch_preflight",
    "estimate_artifact_footprint",
    "require_runtime_environment",
    "seal_runtime_environment",
    "run_launch_preflight",
]
