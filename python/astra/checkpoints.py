"""Immutable, checksummed local checkpoints without executable serialization.

Model tensors use safetensors. Optimizer/RNG/lane state uses a bounded JSON tree
plus safetensors leaves. A complete directory is published by an exclusive
atomic rename; no reader observes an incomplete checkpoint or overwritten ID.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
from typing import Any
import uuid

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np

from .model.actions import ActionVocabulary
from .model.config import ModelConfig, MAXIMUM_PARAMETERS
from .model.policy import AgentPolicy
from .versions import CANONICALIZER_VERSION

MAXIMUM_ARTIFACT_BYTES = 8 * 1024**3
MAXIMUM_JSON_BYTES = 8 * 1024**2
MAXIMUM_STATE_LEAVES = 20_000


class CheckpointError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as error:
        raise CheckpointError("Checkpoint metadata must be finite bounded JSON") from error
    if len(data) > MAXIMUM_JSON_BYTES:
        raise CheckpointError("Checkpoint metadata exceeds its size limit")
    return data


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAXIMUM_JSON_BYTES:
        raise CheckpointError("Checkpoint metadata is missing, oversized, or linked outside its package")
    try:
        def no_constant(value):
            raise CheckpointError("Nonfinite checkpoint metadata")
        return json.loads(path.read_bytes(), parse_constant=no_constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise CheckpointError("Checkpoint metadata is malformed") from error


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(source: Path, destination: Path) -> None:
    # macOS renamex_np(RENAME_EXCL), declared by the public SDK sys/stdio.h.
    # os.replace would overwrite an existing empty directory during a race.
    library = ctypes.CDLL(None, use_errno=True)
    function = library.renamex_np
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), 0x00000004) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
    _sync(destination.parent)


def _validate_tensor_file(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    size = path.stat().st_size
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or not 8 <= size <= MAXIMUM_ARTIFACT_BYTES:
        raise CheckpointError("Invalid tensor artifact size or file type")
    with path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= header_length <= min(MAXIMUM_JSON_BYTES, size - 8):
            raise CheckpointError("Invalid safetensors header length")
        try:
            header = json.loads(stream.read(header_length))
        except (ValueError, UnicodeDecodeError, RecursionError) as error:
            raise CheckpointError("Invalid safetensors header") from error
    if not isinstance(header, dict) or len(header) > MAXIMUM_STATE_LEAVES + 1:
        raise CheckpointError("Too many tensors in a checkpoint")
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
              "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
    ranges = []
    result = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or len(name) > 512 or not isinstance(tensor, dict):
            raise CheckpointError("Invalid tensor description")
        shape, dtype, offsets = tensor.get("shape"), tensor.get("dtype"), tensor.get("data_offsets")
        if not isinstance(shape, list) or len(shape) > 16 or any(type(value) is not int or not 0 <= value <= 2**24 for value in shape):
            raise CheckpointError("Tensor dimensions exceed resource bounds")
        if dtype not in widths or not isinstance(offsets, list) or len(offsets) != 2 or any(type(value) is not int for value in offsets):
            raise CheckpointError("Unsupported tensor dtype or storage range")
        start, end = offsets
        if not 0 <= start <= end <= size - 8 - header_length or end - start != math.prod(shape) * widths[dtype]:
            raise CheckpointError("Tensor byte range does not match its shape")
        result[name] = (dtype, tuple(shape))
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise CheckpointError("Tensor artifact has an overlap or unclaimed byte range")
        cursor = end
    if cursor != size - 8 - header_length:
        raise CheckpointError("Tensor artifact contains trailing data")
    return result


def _pack_state(value: Any) -> tuple[dict, dict[str, mx.array]]:
    tensors = {}
    nodes = 0
    def visit(item, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 32 or nodes > 100_000:
            raise CheckpointError("Training state tree exceeds its structural limit")
        if isinstance(item, (mx.array, np.ndarray)):
            if len(tensors) >= MAXIMUM_STATE_LEAVES:
                raise CheckpointError("Too many training state tensors")
            name = f"t{len(tensors):05d}"
            tensors[name] = mx.array(item)
            return {"tensor": name}
        if item is None or type(item) in (bool, str, int, float):
            if type(item) is float and not math.isfinite(item):
                raise CheckpointError("Training state contains a nonfinite scalar")
            return {"scalar": item}
        if isinstance(item, dict):
            if any(type(key) is not str or len(key) > 512 for key in item):
                raise CheckpointError("State dictionaries require bounded string keys")
            return {"dict": {key: visit(child, depth + 1) for key, child in item.items()}}
        if isinstance(item, (tuple, list)):
            return {"tuple" if isinstance(item, tuple) else "list": [visit(child, depth + 1) for child in item]}
        raise CheckpointError(f"Training state cannot serialize {type(item).__name__}")
    return visit(value), tensors


def _unpack_state(tree: dict, tensors: dict[str, mx.array]) -> Any:
    consumed = set()
    nodes = 0
    def visit(node, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 32 or nodes > 100_000 or not isinstance(node, dict) or len(node) != 1:
            raise CheckpointError("Invalid training state tree")
        kind, value = next(iter(node.items()))
        if kind == "tensor":
            if not isinstance(value, str) or value not in tensors or value in consumed:
                raise CheckpointError("Invalid or reused training tensor reference")
            consumed.add(value)
            return tensors[value]
        if kind == "scalar":
            if value is not None and type(value) not in (bool, str, int, float):
                raise CheckpointError("Invalid training scalar")
            return value
        if kind == "dict" and isinstance(value, dict):
            return {key: visit(child, depth + 1) for key, child in value.items()}
        if kind in ("list", "tuple") and isinstance(value, list):
            children = [visit(child, depth + 1) for child in value]
            return tuple(children) if kind == "tuple" else children
        raise CheckpointError("Unknown training state node")
    result = visit(tree)
    if consumed != tensors.keys():
        raise CheckpointError("Training artifact contains unreferenced tensors")
    return result


def _require_finite(tensors: dict[str, mx.array], message: str) -> None:
    if tensors and not bool(mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in tensors.values()])).item()):
        raise CheckpointError(message)


def restore_mlx_random_state(state) -> None:
    """Restore MLX's current-thread PRNG from its saved one-key state.

    MLX 0.32.2 exposes a read-only thread-local state sentinel. A Threefry key
    has two uint32 words and every such key is exactly representable by the
    public uint64 seed API. Replacing ``mx.random.state`` would only shadow the
    sentinel; it would not restore the generator used by random operations.
    """
    if (not isinstance(state, (tuple, list)) or len(state) != 1 or not isinstance(state[0], mx.array)
        or state[0].shape != (2,) or state[0].dtype != mx.uint32):
        raise CheckpointError("Invalid MLX random state")
    high, low = np.asarray(state[0]).tolist()
    mx.random.seed((int(high) << 32) | int(low))


def _restore_frozen_parameters(policy: AgentPolicy, paths: list[str]) -> None:
    # Full leaf paths distinguish, for example, a frozen backbone norm bias
    # from a trainable actor bias. Module.freeze(keys='bias') would freeze both.
    for path in paths:
        owner = policy
        components = path.split(".")
        for component in components[:-1]:
            owner = owner[int(component)] if isinstance(owner, (tuple, list)) else owner[component]
        if not isinstance(owner, nn.Module):
            raise CheckpointError("A frozen parameter has no owning model module")
        owner.freeze(recurse=False, keys=components[-1], strict=True)


@dataclass(frozen=True)
class LoadedCheckpoint:
    manifest: dict
    policy: AgentPolicy
    training_state: Any


def save_checkpoint(destination: Path, policy: AgentPolicy, *, kind: str, step: int,
                    training_state: Any = None, parent_id: str | None = None,
                    dataset_id: str | None = None, metrics: dict | None = None,
                    training_config: dict | None = None) -> dict:
    destination = Path(destination)
    if kind not in ("initial", "behavioral", "reinforcement") or type(step) is not int or step < 0:
        raise CheckpointError("Invalid checkpoint purpose or training step")
    identifier = str(uuid.UUID(destination.name))
    if any(value is not None and str(uuid.UUID(value)) != value.lower() for value in (parent_id, dataset_id)):
        raise CheckpointError("Invalid checkpoint ancestry")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Checkpoints are immutable; allocate a new checkpoint ID")
    policy.config.validate()
    parameters = dict(tree_flatten(policy.parameters()))
    if sum(value.size for value in parameters.values()) > MAXIMUM_PARAMETERS:
        raise CheckpointError("Policy exceeds the supported parameter budget")
    mx.eval(parameters)
    if any(value.dtype != mx.float32 for value in parameters.values()):
        raise CheckpointError("Checkpoint master parameters must be FP32")
    _require_finite(parameters, "Nonfinite policy parameters cannot be published")
    trainable = dict(tree_flatten(policy.trainable_parameters()))
    config = policy.config.to_dict()
    vocabulary = policy.actions.vocabulary.to_dict()
    identity = {"model": policy.config.semantic_dict(), "actions": vocabulary, "canonicalizerVersion": CANONICALIZER_VERSION}
    manifest = {"schemaVersion": 1, "id": identifier, "createdAt": datetime.now(timezone.utc).isoformat(),
                "kind": kind, "step": step, "parentID": parent_id, "datasetID": dataset_id,
                "model": config, "actions": vocabulary, "canonicalizerVersion": CANONICALIZER_VERSION,
                "frozenParameters": sorted(parameters.keys() - trainable.keys()),
                "policySignature": hashlib.sha256(_json_bytes(identity)).hexdigest(),
                "metrics": metrics or {}, "trainingConfig": training_config or {}, "artifacts": {}}
    _json_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        mx.save_safetensors(str(staging / "policy.safetensors"), parameters)
        if training_state is not None:
            tree, tensors = _pack_state(training_state)
            _require_finite(tensors, "Nonfinite training state tensors cannot be published")
            (staging / "training.json").write_bytes(_json_bytes(tree))
            if tensors:
                mx.save_safetensors(str(staging / "training.safetensors"), tensors)
        for file in sorted(staging.iterdir()):
            _sync(file)
            if file.suffix == ".safetensors":
                _validate_tensor_file(file)
            manifest["artifacts"][file.name] = {"sha256": _hash(file), "bytes": file.stat().st_size}
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        _sync(staging / "manifest.json")
        _sync(staging)
        _publish(staging, destination)
        return manifest
    finally:
        # Only this invocation's randomly allocated staging directory is owned.
        # A successfully published checkpoint is never removed after an error.
        if staging.exists():
            shutil.rmtree(staging)


def load_checkpoint(directory: Path, *, include_training: bool = False) -> LoadedCheckpoint:
    directory = Path(directory)
    if directory.is_symlink():
        raise CheckpointError("A checkpoint package cannot be a symbolic link")
    manifest = _read_json(directory / "manifest.json")
    required = {"schemaVersion", "id", "createdAt", "kind", "step", "parentID", "datasetID", "model", "actions",
                "canonicalizerVersion", "policySignature", "metrics", "trainingConfig", "artifacts", "frozenParameters"}
    if (not isinstance(manifest, dict) or set(manifest) != required
        or type(manifest["schemaVersion"]) is not int or manifest["schemaVersion"] != 1
        or type(manifest["canonicalizerVersion"]) is not int or manifest["canonicalizerVersion"] != CANONICALIZER_VERSION):
        raise CheckpointError("Unsupported checkpoint schema")
    if str(uuid.UUID(directory.name)) != manifest["id"] or manifest["kind"] not in ("initial", "behavioral", "reinforcement"):
        raise CheckpointError("Checkpoint identity or purpose is invalid")
    if type(manifest["step"]) is not int or manifest["step"] < 0:
        raise CheckpointError("Invalid checkpoint step")
    try:
        config = ModelConfig.from_dict(manifest["model"])
        vocabulary = ActionVocabulary.from_dict(manifest["actions"])
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"Invalid checkpoint configuration: {error}") from error
    identity = {"model": config.semantic_dict(), "actions": vocabulary.to_dict(), "canonicalizerVersion": CANONICALIZER_VERSION}
    if hashlib.sha256(_json_bytes(identity)).hexdigest() != manifest["policySignature"]:
        raise CheckpointError("Checkpoint configuration signature does not match")
    artifacts = manifest["artifacts"]
    allowed = {"policy.safetensors", "training.safetensors", "training.json"}
    if not isinstance(artifacts, dict) or "policy.safetensors" not in artifacts or set(artifacts) - allowed:
        raise CheckpointError("Checkpoint artifact set is invalid")
    if "training.safetensors" in artifacts and "training.json" not in artifacts:
        raise CheckpointError("Training tensors have no state description")
    policy_spec = None
    for name, expected in artifacts.items():
        path = directory / name
        if not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}:
            raise CheckpointError("Invalid checkpoint artifact manifest")
        if type(expected["bytes"]) is not int or not 0 <= expected["bytes"] <= MAXIMUM_ARTIFACT_BYTES:
            raise CheckpointError("Checkpoint artifact exceeds its resource budget")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or _hash(path) != expected["sha256"]:
            raise CheckpointError("Checkpoint artifact integrity verification failed")
        if name.endswith(".safetensors"):
            spec = _validate_tensor_file(path)
            if name == "policy.safetensors":
                policy_spec = spec
    if any(dtype != "F32" for dtype, _ in policy_spec.values()):
        raise CheckpointError("Checkpoint master parameters must be FP32")
    if sum(math.prod(shape) for _, shape in policy_spec.values()) != config.parameter_count:
        raise CheckpointError("Checkpoint tensor count does not match its model configuration")
    frozen = manifest["frozenParameters"]
    if (not isinstance(frozen, list) or any(type(path) is not str or len(path) > 512 for path in frozen)
        or frozen != sorted(set(frozen)) or not set(frozen) <= policy_spec.keys()):
        raise CheckpointError("Invalid frozen parameter paths")
    # Config.validate has already enforced the exact architecture-wide budget
    # using integers only. Construction cannot advance a caller's rollout RNG;
    # every initialized parameter will be replaced by the strict weight load.
    random_state = list(mx.random.state)
    try:
        policy = AgentPolicy(config, vocabulary)
    finally:
        restore_mlx_random_state(random_state)
    policy.load_weights(str(directory / "policy.safetensors"), strict=True)
    mx.eval(policy.parameters())
    _require_finite(dict(tree_flatten(policy.parameters())), "Checkpoint has nonfinite model parameters")
    _restore_frozen_parameters(policy, frozen)
    state = None
    if include_training and "training.json" in artifacts:
        tensors = mx.load(str(directory / "training.safetensors")) if "training.safetensors" in artifacts else {}
        _require_finite(tensors, "Checkpoint has nonfinite training state tensors")
        state = _unpack_state(_read_json(directory / "training.json"), tensors)
    return LoadedCheckpoint(manifest, policy, state)


def load_checkpoint_actor_progress(directory: Path, manifest: dict) -> dict:
    """Read an integrity-checked cursor without allocating optimizer tensors.

    The caller has already loaded this policy with load_checkpoint. The cursor
    comes from its hashed training artifact, not an arbitrary prepare payload or
    the unverified display metrics alone. GRU state is never restored here.
    """
    from .actor_progress import validate_actor_progress
    directory = Path(directory)
    expected = manifest.get('artifacts', {}).get('training.json')
    if manifest.get('kind') != 'reinforcement' or expected is None or str(uuid.UUID(directory.name)) != manifest.get('id'):
        raise CheckpointError('This checkpoint has no saved external actor progress')
    descriptor = os.open(directory / 'training.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAXIMUM_JSON_BYTES:
            raise CheckpointError('Invalid actor progress artifact extent')
        data = source.read(MAXIMUM_JSON_BYTES + 1)
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise CheckpointError('Actor progress artifact integrity verification failed')
    try:
        def reject(_):raise ValueError('Nonfinite actor progress')
        tree = json.loads(data, parse_constant=reject)
        if type(tree) is not dict or set(tree) != {'dict'} or type(tree['dict']) is not dict:
            raise ValueError('Invalid external training state tree')
        fields = tree['dict']
        required = {'kind', 'schemaVersion', 'learner', 'actorProgress', 'consumedRolloutIDs', 'requiresEnvironmentReset'}
        if set(fields) != required:
            raise ValueError('Unsupported external training state')
        kind = _unpack_state(fields['kind'], {})
        version = _unpack_state(fields['schemaVersion'], {})
        reset = _unpack_state(fields['requiresEnvironmentReset'], {})
        if kind != 'reinforcement_external' or type(version) is not int or version != 1 or reset is not True:
            raise ValueError('Unsupported external training state')
        progress = validate_actor_progress(_unpack_state(fields['actorProgress'], {}))
        mirror = validate_actor_progress(manifest.get('metrics', {}).get('actorProgress'))
        if progress != mirror:
            raise ValueError('Checkpoint actor progress differs from its recorded metrics')
        return progress
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise CheckpointError(f'Invalid saved actor progress: {error}') from error
