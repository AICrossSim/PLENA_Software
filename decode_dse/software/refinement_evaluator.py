"""Build sealed GPTQ, clipping, and rotation banks for refinement."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

from decode_dse.software.cached_decode import (
    ContinuationExample,
    TorchHFCachedDecodeBackend,
    evaluate_teacher_forced_cached_batched,
)
from decode_dse.software.refinement_runner import (
    GPTQ_SELECTION_POLICY,
    REFINEMENT_PREFILL_INDEX_SCHEMA,
)
from decode_dse.software.decode_evaluator import (
    ADMISSION_CONVERTER_SCHEMA,
    PACKED_CACHE_LAYOUT,
    AdmissionCacheHandle,
    DecodeWeightBank,
    DecodeWeightBankIdentity,
    DecodeWeightQuantizationGuard,
    DecodeEvaluator,
    _DecodeCacheLRU,
    _decode_cache_path,
    _document_token,
    _mase_tree_hash,
    _repository_root,
    _software_tree_hash,
    _validate_bank_structure,
    build_decode_binding_plan,
)
from decode_dse.software.refinement_schedule import (
    RefinementScheduleEntry,
    refinement_profile_to_decode_quant_spec,
)
from decode_dse.software.refinement_runner import (
    RefinementBankHandle,
    RefinementBankSpec,
    RefinementDocumentMetric,
    RefinementEvaluation,
    RefinementExecutionEvidence,
    rotation_decision_contract,
    rotation_decision_contract_hash,
    refinement_rng_policy,
    refinement_rng_policy_hash,
    rotation_policy,
    seal_checkpoint_identity,
)
from decode_dse.software.token_samples import RefinementSampleBundle
from decode_dse.software.runtime_environment import (
    RuntimeEnvironment,
    capture_runtime_environment,
    initialize_numerical_runtime,
)
from decode_dse.software.cache_artifacts import (
    load_decode_cache_artifact,
    load_prefill_artifact,
)
from decode_dse.software.sweep_plan import (
    load_immutable_json,
    write_immutable_json,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _seed_refinement_runtime(
    bank: RefinementBankSpec,
    torch: Any,
) -> dict[str, Any]:
    if bank.rng_policy_hash != refinement_rng_policy_hash():
        raise ValueError("refinement bank RNG policy differs from the runtime")
    policy = refinement_rng_policy()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(policy["cublas_workspace_config"])
    initialize_numerical_runtime(bank.rng_seed)
    random.seed(bank.rng_seed)
    python_state = hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest()
    numpy_state = None
    try:
        import numpy

        numpy.random.seed(bank.rng_seed % (2**32))
        numpy_state = hashlib.sha256(
            repr(numpy.random.get_state()).encode("utf-8")
        ).hexdigest()
    except ImportError:
        pass
    torch.manual_seed(bank.rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(bank.rng_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch_state = hashlib.sha256(
        torch.get_rng_state().cpu().numpy().tobytes()
    ).hexdigest()
    cuda_state = []
    if torch.cuda.is_available():
        cuda_state = [
            hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ]
    return {
        "schema_version": "decode-refinement-rng",
        "bank_id": bank.bank_id,
        "seed": bank.rng_seed,
        "policy": policy,
        "policy_hash": bank.rng_policy_hash,
        "python_state_sha256": python_state,
        "numpy_state_sha256": numpy_state,
        "torch_cpu_state_sha256": torch_state,
        "torch_cuda_state_sha256": cuda_state,
    }


class RefinementEvaluator:
    """Run refinement without altering the screen/validation sample bundle."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        sample_bundle_path: str | Path,
        prefill_root: str | Path,
        admission_root: str | Path,
        workspace_root: str | Path,
        device_label: str,
        decode_microbatch_size: int = 8,
        max_cpu_cache_gib: float = 24.0,
    ) -> None:
        self.config = dict(config)
        refinement = config.get("refinement")
        if not isinstance(refinement, Mapping):
            raise ValueError("config.refinement is required")
        self.refinement_config = dict(refinement)
        self.gpu_min_free_mb = int(self.refinement_config.get("gpu_min_free_mb", 0))
        if self.gpu_min_free_mb <= 0:
            raise ValueError("refinement requires a positive declared GPU-memory floor")
        from decode_dse.software.token_samples import (
            load_refinement_sample_bundle,
        )

        self.sample_bundle_path = Path(sample_bundle_path).resolve()
        self.bundle = load_refinement_sample_bundle(self.sample_bundle_path)
        if self.bundle.model_revision != str(config["model_revision"]):
            raise ValueError("refinement bundle model revision mismatch")
        if self.bundle.tokenizer_revision != str(config["tokenizer_revision"]):
            raise ValueError("refinement bundle tokenizer revision mismatch")
        if decode_microbatch_size <= 0:
            raise ValueError("refinement decode microbatch must be positive")
        self.decode_microbatch_size = int(decode_microbatch_size)
        self.prefill_root = Path(prefill_root).resolve()
        self.admission_root = Path(admission_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.device = str(config.get("device", "cuda:0"))
        self.device_label = str(device_label)
        self.engine = DecodeEvaluator.__new__(DecodeEvaluator)
        self._initialize_engine(max_cpu_cache_gib)
        self._validate_prefill_index()
        self._open_banks: dict[str, DecodeWeightBank] = {}
        self._bank_receipts: dict[str, Path] = {}
        self._validated_pairs: dict[tuple[str, str], Path] = {}

    def _initialize_engine(self, max_cpu_cache_gib: float) -> None:
        executor_config = self.config.get("executor")
        if not isinstance(executor_config, Mapping):
            raise ValueError("config.executor is required")
        if not str(self.config.get("scratch_dir", "")).startswith("workspace://"):
            raise ValueError("refinement scratch_dir must be workspace-bound")
        if executor_config.get("serialize_weight_bank_builds") is not True:
            raise ValueError("refinement weight-bank build locking must be enabled")
        engine = self.engine
        engine.config = self.config
        engine.executor_config = dict(executor_config)
        engine._initialize_model_contract()
        engine.device = self.device
        engine.runtime_seed = int(self.config.get("seed", 0))
        initialize_numerical_runtime(engine.runtime_seed)
        engine.runtime_environment = capture_runtime_environment(
            engine.device,
            seed=engine.runtime_seed,
        )
        engine._configure_mase_path()
        engine.bundle = self.bundle
        engine.samples = self.bundle.samples
        engine.prefill_root = self.prefill_root
        engine.admission_root = self.admission_root
        engine.layout_id = str(executor_config.get("layout_id", PACKED_CACHE_LAYOUT))
        if engine.layout_id != str(executor_config["layout_id"]):
            raise ValueError("refinement requires the native PackedKV layout")
        identity = {
            "schema_version": ADMISSION_CONVERTER_SCHEMA,
            "sample_schema_version": self.bundle.schema_version,
            "software_tree_sha256": _software_tree_hash(_repository_root()),
            "mase_tree_sha256": _mase_tree_hash(_repository_root(), self.config),
            "runtime_environment_fingerprint": (
                engine.runtime_environment.logical_fingerprint
            ),
            "layout_id": engine.layout_id,
            "block_size": 8,
            "scale_format": "E8M0",
            "scale_encoding": "bias127",
            "mxint_element_encoding": "sign_magnitude_lsb",
            "mxfp_element_encoding": "ieee_lsb",
            "plane_order": "row_major_lsb_first",
        }
        engine.admission_code_revision = hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        engine.admission_contract_id = (
            "refinement-"
            f"{ADMISSION_CONVERTER_SCHEMA.rsplit('/', 1)[-1]}-"
            f"{engine.admission_code_revision[:16]}"
        )
        admission_provenance_path = (
            engine.admission_root / engine.admission_contract_id / "provenance.json"
        )
        with engine._artifact_lock(admission_provenance_path):
            if not admission_provenance_path.exists():
                write_immutable_json(
                    admission_provenance_path,
                    {
                        "schema_version": "decode-refinement-admission-provenance",
                        "admission_contract_id": engine.admission_contract_id,
                        "admission_code_revision": engine.admission_code_revision,
                        "sample_bundle_hash": self.bundle.canonical_hash,
                        "runtime_environment_fingerprint": (
                            engine.runtime_environment.logical_fingerprint
                        ),
                        "created_at_utc": datetime.now(timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    },
                )
            admission_provenance = load_immutable_json(admission_provenance_path)
        if (
            admission_provenance.get("schema_version")
            != "decode-refinement-admission-provenance"
            or admission_provenance.get("admission_contract_id")
            != engine.admission_contract_id
            or admission_provenance.get("admission_code_revision")
            != engine.admission_code_revision
            or admission_provenance.get("sample_bundle_hash")
            != self.bundle.canonical_hash
            or admission_provenance.get("runtime_environment_fingerprint")
            != engine.runtime_environment.logical_fingerprint
            or not str(admission_provenance.get("created_at_utc", "")).endswith("Z")
        ):
            raise ValueError("refinement admission provenance is invalid")
        engine.admission_provenance_created_at = str(
            admission_provenance["created_at_utc"]
        )
        engine.cache_lru = _DecodeCacheLRU(int(float(max_cpu_cache_gib) * (1 << 30)))
        engine.context = SimpleNamespace(
            device_label=self.device_label,
            workspace_root=self.workspace_root,
        )
        engine.deep_append_validation = False

    @property
    def admission_contract_id(self) -> str:
        return str(self.engine.admission_contract_id)

    @property
    def runtime_environment(self) -> dict[str, Any]:
        return self.engine.runtime_environment.to_dict()

    def _validate_prefill_index(self) -> None:
        index = load_immutable_json(self.prefill_root / "index.json")
        if index.get("schema_version") != REFINEMENT_PREFILL_INDEX_SCHEMA:
            raise ValueError("unsupported refinement prefill index")
        if index.get("model_revision") != self.bundle.model_revision:
            raise ValueError("refinement prefill model revision mismatch")
        if index.get("tokenizer_revision") != self.bundle.tokenizer_revision:
            raise ValueError("refinement prefill tokenizer revision mismatch")
        if index.get("sample_bundle_hash") != self.bundle.canonical_hash:
            raise ValueError("refinement prefill sample identity mismatch")
        recorded_runtime = RuntimeEnvironment.from_dict(
            index.get("runtime_environment", {})
        )
        if (
            recorded_runtime.logical_fingerprint
            != self.engine.runtime_environment.logical_fingerprint
        ):
            raise ValueError("refinement prefill runtime environment mismatch")
        code_revision = _software_tree_hash(_repository_root())
        if index.get("code_revision") != code_revision:
            raise ValueError("refinement prefill source-tree identity mismatch")
        records = {
            str(record["document_id"]): record for record in index.get("records", ())
        }
        if set(records) != {sample.document_id for sample in self.bundle.samples}:
            raise ValueError("refinement prefill index coverage mismatch")
        for sample in self.bundle.samples:
            path = self.prefill_root / _document_token(sample.document_id)
            if not path.is_dir():
                raise ValueError(
                    f"missing refinement prefill artifact: {sample.document_id}"
                )
            artifact = load_prefill_artifact(path)
            record = records[sample.document_id]
            if (
                artifact.artifact_id != record.get("artifact_id")
                or artifact.prompt_hash != sample.prompt_hash
                or artifact.model_revision != self.bundle.model_revision
                or artifact.tokenizer_revision != self.bundle.tokenizer_revision
                or artifact.first_token.selection != "greedy"
                or len(artifact.layers)
                != int(self.engine.model_architecture["num_hidden_layers"])
                or any(
                    layer.key.shape
                    != (
                        1,
                        int(self.engine.model_architecture["num_key_value_heads"]),
                        512,
                        int(self.engine.model_architecture["head_dim"]),
                    )
                    or layer.value.shape != layer.key.shape
                    for layer in artifact.layers
                )
            ):
                raise ValueError("refinement prefill artifact contract mismatch")
        self.prefill_index_path = self.prefill_root / "index.json"

    def _validate_bank_spec(
        self,
        bank: RefinementBankSpec,
        entries: tuple[RefinementScheduleEntry, ...],
    ) -> None:
        if bank.model_name != str(self.config["model_name"]):
            raise ValueError("refinement bank model differs from execution config")
        if bank.model_revision != str(self.config["model_revision"]):
            raise ValueError(
                "refinement bank model revision differs from execution config"
            )
        if not entries or any(
            (
                entry.profile.weight_format,
                entry.profile.weight_method,
            )
            != (bank.weight_format, bank.weight_method)
            for entry in entries
        ):
            raise ValueError("refinement bank group is empty or inconsistent")
        if bank.weight_method == "rotation":
            if len(entries) != 1:
                raise ValueError("rotation decisions must be profile-local")
            profile = entries[0].profile
            if bank.rotation_profile_id != profile.profile_id:
                raise ValueError("rotation bank profile identity mismatch")
            if bank.rotation_config_hash != rotation_decision_contract_hash(profile):
                raise ValueError("refinement rotation decision contract mismatch")
        calibration = Path(bank.calibration_path)
        if not calibration.is_file():
            raise ValueError("refinement GPTQ calibration artifact is missing")
        if _file_sha256(calibration) != bank.calibration_bundle_hash:
            raise ValueError("refinement GPTQ calibration bytes changed")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "refinement calibration validation requires torch"
            ) from exc
        payload = torch.load(calibration, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("loader"), list
        ):
            raise ValueError("refinement GPTQ calibration payload is malformed")
        expected_metadata = {
            "dataset_name": bank.calibration_dataset,
            "dataset_revision": bank.calibration_revision,
            "model_revision": bank.model_revision,
            "tokenizer_revision": str(self.config["tokenizer_revision"]),
            "nsamples": bank.calibration_samples,
            "seqlen": bank.calibration_sequence_length,
        }
        if any(payload.get(key) != value for key, value in expected_metadata.items()):
            raise ValueError(
                "refinement GPTQ calibration metadata differs from the bank"
            )
        if len(payload["loader"]) != bank.calibration_samples:
            raise ValueError("refinement GPTQ calibration sample coverage mismatch")
        selection = payload.get("selection")
        if (
            payload.get("selection_policy") != GPTQ_SELECTION_POLICY
            or isinstance(payload.get("selection_seed"), bool)
            or not isinstance(payload.get("selection_seed"), int)
            or not isinstance(selection, list)
            or len(selection) != bank.calibration_samples
        ):
            raise ValueError(
                "refinement GPTQ calibration selection evidence is invalid"
            )
        intervals: dict[int, list[tuple[int, int]]] = {}
        for ordinal, (record, loader_item) in enumerate(
            zip(selection, payload["loader"])
        ):
            if (
                not isinstance(record, Mapping)
                or record.get("ordinal") != ordinal
                or int(record.get("token_count", -1))
                != bank.calibration_sequence_length
            ):
                raise ValueError(
                    "refinement GPTQ calibration selection order is invalid"
                )
            document_index = int(record["document_index"])
            start = int(record["token_offset"])
            end = start + bank.calibration_sequence_length
            if any(
                start < prior_end and prior_start < end
                for prior_start, prior_end in intervals.setdefault(document_index, [])
            ):
                raise ValueError("refinement GPTQ calibration windows overlap")
            intervals[document_index].append((start, end))
            token_ids = tuple(int(token) for token in loader_item[0][0].tolist())
            digest = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if digest != record.get("window_hash"):
                raise ValueError("refinement GPTQ calibration window hash mismatch")
        if bank.weight_method != "rotation" and bank.rotation_config_hash is not None:
            raise ValueError("GPTQ+Erry bank carries a rotation policy")

    def _gptq_config(self, bank: RefinementBankSpec) -> dict[str, Any]:
        calibration = Path(bank.calibration_path).resolve()
        value: dict[str, Any] = {
            "dataset": f"file:{calibration}",
            "nsamples": bank.calibration_samples,
            "seqlen": bank.calibration_sequence_length,
            "cali_batch_size": bank.calibration_batch_size,
            "quantile_search": True,
            "clip_search_y": True,
            "checkpoint_dir": bank.checkpoint_dir,
        }
        if bank.weight_method == "rotation":
            policy = rotation_policy()
            value["rotation"] = {
                "calib_data": f"file:{calibration}",
                "calib_nsamples": policy["calibration_samples"],
                "calib_seqlen": policy["calibration_sequence_length"],
                "improvement_eps": policy["improvement_epsilon"],
                "cache_winners": policy["cache_winners"],
                "score_phase": policy["score_phase"],
                "cache_path": str(
                    Path(bank.checkpoint_dir) / "rotation_decisions.json"
                ),
            }
        return value

    def _validate_checkpoint_coverage(self, bank: RefinementBankSpec) -> None:
        root = Path(bank.checkpoint_dir)
        layer_files = {
            int(path.stem.rsplit("_", 1)[1])
            for path in root.glob("quantized_model_layer_*.safetensors")
        }
        metadata_files = {
            int(path.name.split("_layer_", 1)[1].split("_metadata", 1)[0])
            for path in root.glob("quantized_model_layer_*_metadata.json")
        }
        layers = int(self.engine.model_architecture["num_hidden_layers"])
        expected = set(range(layers))
        if layer_files != expected or metadata_files != expected:
            raise ValueError(
                f"refinement checkpoint does not cover all {layers} decoder layers"
            )
        if bank.weight_method == "rotation":
            decision_path = root / "rotation_decisions.json"
            if not decision_path.is_file():
                raise ValueError("rotation decisions are missing")
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            if not isinstance(decision.get("winners"), list):
                raise ValueError("rotation decisions contain no winner list")
            for key in ("baseline_ppl", "final_ppl"):
                if not math.isfinite(float(decision[key])):
                    raise ValueError("rotation decision score is non-finite")

    @contextmanager
    def open_weight_bank(
        self,
        bank: RefinementBankSpec,
        entries: tuple[RefinementScheduleEntry, ...],
    ) -> Iterator[RefinementBankHandle]:
        self._validate_bank_spec(bank, entries)
        if bank.bank_id in self._open_banks:
            raise RuntimeError("refinement bank is already open")
        try:
            import torch
            from chop.passes.module.transforms.quantize.quantize import (
                install_phase_context_pre_hooks,
                quantize_module_transform_pass,
            )
            from decode_dse.software.precision_bindings import build_decode_pass_args
        except ImportError as exc:
            raise RuntimeError(
                "decode refinement requires torch, transformers, and MASE"
            ) from exc
        rng_receipt = _seed_refinement_runtime(bank, torch)
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("refinement bank construction requires CUDA")
        device_index = torch.device(self.device).index or 0
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        if free_bytes / float(1 << 20) < self.gpu_min_free_mb:
            raise RuntimeError(
                "insufficient free GPU memory for the calibrated refinement bank"
            )
        checkpoint = Path(bank.checkpoint_dir)
        checkpoint.mkdir(parents=True, exist_ok=True)
        write_immutable_json(
            checkpoint / "rng_receipt.json",
            rng_receipt,
        )
        write_immutable_json(
            checkpoint / "bank_contract.json",
            {
                "schema_version": "decode-refinement-bank-contract",
                "bank": bank.to_dict(),
                "gptq": self._gptq_config(bank),
                "rotation_policy": (
                    rotation_decision_contract(entries[0].profile)
                    if bank.weight_method == "rotation"
                    else None
                ),
                "minimum_free_gpu_mib": self.gpu_min_free_mb,
                "rng": rng_receipt,
            },
        )
        self.engine.cache_lru.clear()
        gc.collect()
        self.engine._validate_device_label()
        build_started = time.perf_counter()
        properties = torch.cuda.get_device_properties(device_index)
        device_observation = {
            "device_label": self.device_label,
            "device_name": str(properties.name),
            "device_uuid": str(getattr(properties, "uuid", "unavailable")),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "free_mib_before_build": free_bytes / float(1 << 20),
            "total_mib": properties.total_memory / float(1 << 20),
            "cuda_runtime": str(torch.version.cuda),
        }
        model = None
        internal = None
        try:
            with self.engine._weight_bank_build_lock():
                model = self.engine._load_model()
                representative = refinement_profile_to_decode_quant_spec(
                    entries[0].profile
                )
                pass_args = build_decode_pass_args(
                    str(self.config["model_name"]),
                    self.device,
                    representative,
                    gptq_cfg=self._gptq_config(bank),
                )
                if bank.weight_method == "gptq_erry":
                    pass_args["collapse_decode_banks"] = True
                bank_device_variable = "MASE_PHASE_BANK_DEVICE"
                previous_bank_device = os.environ.get(bank_device_variable)
                try:
                    if self.device.startswith("cuda"):
                        os.environ[bank_device_variable] = self.device
                    else:
                        os.environ.pop(bank_device_variable, None)
                    model, _ = quantize_module_transform_pass(model, pass_args)
                finally:
                    if previous_bank_device is None:
                        os.environ.pop(bank_device_variable, None)
                    else:
                        os.environ[bank_device_variable] = previous_bank_device
                if bank.weight_method == "rotation":
                    collapsed = 0
                    for module in model.modules():
                        collapse = getattr(module, "collapse_to_decode_bank", None)
                        if callable(collapse) and collapse():
                            collapsed += 1
                    expected_linears = 7 * int(
                        self.engine.model_architecture["num_hidden_layers"]
                    )
                    if collapsed != expected_linears:
                        raise RuntimeError(
                            "rotation bank did not collapse all decode linears"
                        )
                model = model.to(self.device).eval()
                install_phase_context_pre_hooks(model)
                binding_plan = build_decode_binding_plan(model, pass_args)
                quantization_guard = DecodeWeightQuantizationGuard.capture(
                    binding_plan,
                    expected_modules=(
                        7 * int(self.engine.model_architecture["num_hidden_layers"])
                    ),
                )
                _validate_bank_structure(
                    model,
                    binding_plan,
                    len(quantization_guard.modules),
                    self.engine.model_architecture,
                )
            self._validate_checkpoint_coverage(bank)
            identity_guard = DecodeWeightBankIdentity.capture(model)
            build_seconds = time.perf_counter() - build_started
            checkpoint_tree_before_receipt = seal_checkpoint_identity(
                bank,
                checkpoint,
                identity_guard.fingerprint,
            ).checkpoint_tree_sha256
            receipt_body = {
                "schema_version": "decode-refinement-bank-measurement",
                "bank_id": bank.bank_id,
                "checkpoint_tree_before_receipt_sha256": (
                    checkpoint_tree_before_receipt
                ),
                "weight_identity": identity_guard.fingerprint,
                "build_seconds": build_seconds,
                "device": device_observation,
                "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            receipt_id = hashlib.sha256(
                json.dumps(
                    receipt_body,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            receipt_path = checkpoint / "measurements" / f"bank-{receipt_id}.json"
            write_immutable_json(receipt_path, receipt_body)
            internal = DecodeWeightBank(
                model=model,
                device=torch.device(self.device),
                weight_format=bank.weight_format,
                weight_method=bank.weight_method,
                binding_plan=binding_plan,
                identity_guard=identity_guard,
                quantization_guard=quantization_guard,
                build_seconds=build_seconds,
            )
            handle = seal_checkpoint_identity(
                bank,
                checkpoint,
                identity_guard.fingerprint,
            )
            self._open_banks[bank.bank_id] = internal
            self._bank_receipts[bank.bank_id] = receipt_path
            yield handle
        finally:
            self._open_banks.pop(bank.bank_id, None)
            self._bank_receipts.pop(bank.bank_id, None)
            del internal
            del model
            gc.collect()
            if "torch" in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _pair_index(
        self,
        key_format: str,
        value_format: str,
        paths: Mapping[str, Path],
    ) -> Path:
        root = (
            self.admission_root
            / self.admission_contract_id
            / f"K-{key_format}__V-{value_format}"
        )
        records = []
        for sample in self.bundle.samples:
            path = paths[sample.document_id]
            artifact = load_decode_cache_artifact(path)
            logical_path = _decode_cache_path(
                self.admission_root,
                key_format,
                sample.document_id,
                value_format,
                contract_id=self.admission_contract_id,
            )
            records.append(
                {
                    "document_id": sample.document_id,
                    "relative_path": logical_path.relative_to(
                        self.admission_root
                    ).as_posix(),
                    "artifact_id": artifact.artifact_id,
                    "manifest_sha256": _file_sha256(path / "manifest.json"),
                }
            )
        index = root / "index.json"
        write_immutable_json(
            index,
            {
                "schema_version": "decode-refinement-admission-pair",
                "sample_bundle_hash": self.bundle.canonical_hash,
                "admission_contract_id": self.admission_contract_id,
                "key_format": key_format,
                "value_format": value_format,
                "records": records,
            },
        )
        return index

    def _append_evidence_path(
        self,
        key_format: str,
        value_format: str,
    ) -> Path:
        return (
            self.admission_root
            / self.admission_contract_id
            / ".append-evidence"
            / f"K-{key_format}__V-{value_format}.json"
        )

    def _load_append_evidence(
        self,
        key_format: str,
        value_format: str,
        pair_index: Path,
    ) -> Path | None:
        path = self._append_evidence_path(key_format, value_format)
        if not path.exists():
            return None
        value = load_immutable_json(path)
        pair = load_immutable_json(pair_index)
        layers = int(self.engine.model_architecture["num_hidden_layers"])
        expected = {
            "schema_version": "decode-refinement-split-append-evidence",
            "sample_bundle_hash": self.bundle.canonical_hash,
            "admission_contract_id": self.admission_contract_id,
            "key_format": key_format,
            "value_format": value_format,
            "pair_index_hash": pair["content_hash"],
            "q_len": 1,
            "layers_checked": layers,
            "key_value_tensors_checked": layers * 2,
            "native_append_passed": True,
            "software_tree_sha256": _software_tree_hash(_repository_root()),
            "mase_tree_sha256": _mase_tree_hash(_repository_root(), self.config),
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError("refinement split-append evidence identity mismatch")
        return path

    @contextmanager
    def open_split_kv_admission_cache(
        self,
        key_format: str,
        value_format: str,
        samples: RefinementSampleBundle,
    ) -> Iterator[tuple[AdmissionCacheHandle, Path]]:
        if samples.canonical_hash != self.bundle.canonical_hash:
            raise ValueError("refinement admission received a different sample bundle")
        with self.engine.open_split_kv_admission_cache(
            key_format,
            value_format,
        ) as handle:
            pair_index = self._pair_index(
                key_format,
                value_format,
                handle.paths,
            )
            evidence = self._load_append_evidence(
                key_format,
                value_format,
                pair_index,
            )
            if evidence is not None:
                self._validated_pairs[(key_format, value_format)] = evidence
            yield handle, pair_index

    def evaluate(
        self,
        entry: RefinementScheduleEntry,
        *,
        samples: RefinementSampleBundle,
        weight_bank: RefinementBankHandle,
        kv_admission_cache: Any,
    ) -> RefinementEvaluation:
        if samples.canonical_hash != self.bundle.canonical_hash:
            raise ValueError("refinement evaluation received a different sample bundle")
        internal = self._open_banks.get(weight_bank.bank_id)
        if internal is None:
            raise RuntimeError("refinement weight bank is not open")
        if (
            not isinstance(kv_admission_cache, tuple)
            or len(kv_admission_cache) != 2
            or not isinstance(kv_admission_cache[0], AdmissionCacheHandle)
        ):
            raise TypeError("refinement admission handle is malformed")
        admission, pair_index = kv_admission_cache
        if (
            admission.key_format,
            admission.resolved_value_format,
        ) != (
            entry.profile.key_format,
            entry.profile.value_format,
        ):
            raise ValueError("refinement profile differs from its admitted K/V pair")
        binding = self.engine.bind_refinement_profile(
            internal,
            entry.profile,
        )
        if binding.weight_requantizations != 0:
            raise RuntimeError("refinement runtime binding requantized a weight")
        from chop.nn.quantized.modules.phase_context import force_runtime_phase

        pair = (entry.profile.key_format, entry.profile.value_format)
        needs_append_oracle = pair not in self._validated_pairs
        self.engine._native_append_validation_calls = 0
        self.engine._native_append_tensor_checks = 0
        self.engine._native_append_quantized_tensor_checks = 0
        self.engine._native_append_validation_seconds = 0.0

        def validate_first_append(cache, start, end, artifact):
            if self.engine._native_append_validation_calls == 0:
                self.engine._validate_native_append(
                    cache,
                    start,
                    end,
                    artifact,
                )

        backend = TorchHFCachedDecodeBackend(
            device=internal.device,
            append_validator=validate_first_append if needs_append_oracle else None,
            native_append_format=True,
        )
        documents = []
        with force_runtime_phase("decode"):
            for offset in range(
                0,
                len(self.bundle.samples),
                self.decode_microbatch_size,
            ):
                batch = self.bundle.samples[
                    offset : offset + self.decode_microbatch_size
                ]
                examples = []
                for sample in batch:
                    prefill = load_prefill_artifact(
                        self.prefill_root / _document_token(sample.document_id)
                    )
                    admitted = self.engine.cache_lru.get(
                        admission.paths[sample.document_id]
                    )
                    examples.append(
                        ContinuationExample(
                            document_id=sample.document_id,
                            prefill=prefill,
                            decode_cache=admitted,
                            continuation_ids=(
                                prefill.first_token.token_ids[0],
                                *sample.decode_target_ids[:128],
                            ),
                        )
                    )
                documents.extend(
                    evaluate_teacher_forced_cached_batched(
                        internal.model,
                        examples,
                        backend,
                    )
                )
        if needs_append_oracle:
            layers = int(self.engine.model_architecture["num_hidden_layers"])
            expected_tensor_checks = layers * 2
            if (
                self.engine._native_append_validation_calls != 1
                or self.engine._native_append_tensor_checks != expected_tensor_checks
                or self.engine._native_append_quantized_tensor_checks
                != expected_tensor_checks
            ):
                raise AssertionError(
                    "refinement split-append oracle did not cover every layer and role"
                )
            pair_index_value = load_immutable_json(pair_index)
            append_evidence = self._append_evidence_path(*pair)
            write_immutable_json(
                append_evidence,
                {
                    "schema_version": "decode-refinement-split-append-evidence",
                    "sample_bundle_hash": self.bundle.canonical_hash,
                    "admission_contract_id": self.admission_contract_id,
                    "key_format": pair[0],
                    "value_format": pair[1],
                    "pair_index_hash": pair_index_value["content_hash"],
                    "q_len": 1,
                    "layers_checked": layers,
                    "key_value_tensors_checked": expected_tensor_checks,
                    "native_append_passed": True,
                    "software_tree_sha256": _software_tree_hash(_repository_root()),
                    "mase_tree_sha256": _mase_tree_hash(
                        _repository_root(), self.config
                    ),
                },
            )
            self._validated_pairs[pair] = append_evidence
        append_evidence = self._validated_pairs.get(pair)
        if append_evidence is None:
            raise AssertionError("refinement split-append evidence is missing")
        identity_after = internal.identity_guard.verify(internal.model)
        internal.quantization_guard.verify()
        source_clusters = {
            sample.document_id: sample.source_cluster_id
            for sample in self.bundle.samples
        }
        metrics = tuple(
            RefinementDocumentMetric(
                document_id=document.document_id,
                source_cluster_id=source_clusters[document.document_id],
                nll_sum=document.nll_sum,
                token_count=document.token_count,
                initial_cache_length=document.initial_cache_length,
                final_cache_length=document.final_cache_length,
            )
            for document in documents
        )
        evidence = RefinementExecutionEvidence(
            prefill_precision="BF16",
            prefill_kv_precision="BF16",
            first_token_owner="prefill",
            q_len_values=(1,),
            exact_cache_positions=True,
            independent_batch_caches=True,
            admission_count_per_prompt=1,
            direct_native_kv_append=append_evidence.is_file(),
            runtime_rebinding=binding.performed,
            weight_requantizations=binding.weight_requantizations,
            weight_identity_before=binding.identity_before,
            weight_identity_after=identity_after,
            checkpoint_tree_sha256=weight_bank.checkpoint_tree_sha256,
        )
        return RefinementEvaluation(
            documents=metrics,
            evidence=evidence,
            artifacts=(
                str(self.sample_bundle_path),
                str(self.prefill_index_path),
                str(pair_index),
                str(append_evidence),
                str(self._bank_receipts[weight_bank.bank_id]),
            ),
        )


__all__ = ["RefinementEvaluator"]
