from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import is_tensor_like, print_section, safe_dtype, safe_shape


@dataclass
class CheckpointLoadResult:
    checkpoint: Any
    warnings: list[str] = field(default_factory=list)


@dataclass
class StateDictResult:
    state_dict: Mapping[Any, Any] | None
    source: str | None = None
    reason: str | None = None


def load_checkpoint_safe(path: str | Path) -> CheckpointLoadResult:
    import torch

    checkpoint_path = Path(path)
    warnings: list[str] = []
    if _torch_load_supports_weights_only(torch.load):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        warnings.append(
            "installed PyTorch does not support weights_only=True; fell back to "
            "torch.load(..., map_location='cpu'). Only inspect trusted files."
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return CheckpointLoadResult(checkpoint=checkpoint, warnings=warnings)


def load_checkpoint_unsafe(path: str | Path) -> CheckpointLoadResult:
    import torch

    checkpoint_path = Path(path)
    warnings = [
        "unsafe checkpoint loading uses Python pickle behavior and should only "
        "be used for trusted local files."
    ]
    if _torch_load_supports_weights_only(torch.load):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        warnings.append(
            "installed PyTorch does not support weights_only=False; used "
            "torch.load(..., map_location='cpu')."
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return CheckpointLoadResult(checkpoint=checkpoint, warnings=warnings)


def extract_state_dict(
    checkpoint: Any, state_dict_key: str | None = None
) -> StateDictResult:
    if state_dict_key:
        if not isinstance(checkpoint, Mapping):
            return StateDictResult(
                state_dict=None,
                reason=(
                    f"explicit state_dict_key '{state_dict_key}' was provided, "
                    "but the checkpoint is not a dictionary"
                ),
            )
        if state_dict_key not in checkpoint:
            return StateDictResult(
                state_dict=None,
                reason=f"explicit state_dict_key '{state_dict_key}' was not found",
            )
        candidate = checkpoint[state_dict_key]
        if is_state_dict_like(candidate):
            return StateDictResult(state_dict=candidate, source=state_dict_key)
        return StateDictResult(
            state_dict=None,
            reason=f"checkpoint['{state_dict_key}'] is not state_dict-like",
        )

    if is_state_dict_like(checkpoint):
        return StateDictResult(state_dict=checkpoint, source="top-level object")

    if not isinstance(checkpoint, Mapping):
        return StateDictResult(
            state_dict=None,
            reason="checkpoint is not a dictionary and is not state_dict-like",
        )

    for key in ("state_dict", "model_state_dict", "model"):
        if key in checkpoint:
            candidate = checkpoint[key]
            if is_state_dict_like(candidate):
                return StateDictResult(state_dict=candidate, source=key)

    return StateDictResult(
        state_dict=None,
        reason=(
            "no state_dict-like object found at top level, state_dict, "
            "model_state_dict, or model"
        ),
    )


def summarize_loaded_checkpoint(
    checkpoint: Any,
    state_dict_key: str | None = None,
    max_items: int = 40,
) -> StateDictResult:
    print_section("Loaded checkpoint")
    print(f"Top-level Python object type: {_type_name(checkpoint)}")
    if isinstance(checkpoint, Mapping):
        keys = [str(key) for key in checkpoint.keys()]
        print(f"Top-level keys: {', '.join(keys[:40]) if keys else '(empty dict)'}")
        if len(keys) > 40:
            print(f"Top-level keys shown: 40 of {len(keys)}")

    state_result = extract_state_dict(checkpoint, state_dict_key=state_dict_key)
    print(f"State dict found: {'yes' if state_result.state_dict is not None else 'no'}")
    if state_result.state_dict is None:
        print(f"Reason: {state_result.reason}")
        return state_result

    state_dict = state_result.state_dict
    print(f"State dict source: {state_result.source}")
    print(f"State dict entries: {len(state_dict)}")
    print(f"Showing first {min(max_items, len(state_dict))} entries")
    for index, (key, value) in enumerate(state_dict.items()):
        if index >= max_items:
            break
        print(
            f"- {key}: tensor_like={is_tensor_like(value)}, "
            f"shape={safe_shape(value)}, dtype={safe_dtype(value)}"
        )
    return state_result


def is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False

    string_keys = sum(1 for key in obj if isinstance(key, str))
    tensor_values = sum(1 for value in obj.values() if is_tensor_like(value))
    return string_keys == len(obj) and tensor_values >= max(1, int(len(obj) * 0.6))


def _torch_load_supports_weights_only(torch_load: Any) -> bool:
    try:
        return "weights_only" in inspect.signature(torch_load).parameters
    except (TypeError, ValueError):
        return False


def _type_name(obj: Any) -> str:
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"
