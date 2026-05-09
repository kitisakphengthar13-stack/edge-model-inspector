from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def can_instantiate_from_spec(spec: dict[str, Any]) -> tuple[bool, str]:
    checkpoint = spec.get("checkpoint", {})
    if isinstance(checkpoint, Mapping) and checkpoint.get("load_mode") in {
        "custom_loader",
        "external_loader",
    }:
        return False, "custom_loader and external_loader execution is not implemented"

    model = spec.get("model", {})
    if not isinstance(model, Mapping):
        return False, "model section must be a dictionary"
    if not model.get("module") or not model.get("class_name"):
        message = (
            "Cannot instantiate model from spec because model.module and/or "
            "model.class_name are missing. No ONNX export was performed."
        )
        conversion = spec.get("conversion", {})
        if isinstance(conversion, Mapping) and conversion.get("strategy") == "feature_extractor_only":
            message += (
                " feature_extractor_only requires module/class information or a "
                "framework-specific loader/wrapper."
            )
        framework = spec.get("framework")
        architecture = model.get("architecture")
        if framework == "anomalib" or architecture == "patchcore":
            message += (
                " For PatchCore/Anomalib, implement a framework-specific loader "
                "or export ONNX using Anomalib first."
            )
        return False, message
    return True, "model.module and model.class_name are present"


def instantiate_model_from_spec(
    spec: dict[str, Any], allow_imports: bool = False
) -> Any:
    if not allow_imports:
        raise RuntimeError(
            "Refusing to import user model code. Re-run with --allow-imports only "
            "for trusted local modules."
        )

    ok, reason = can_instantiate_from_spec(spec)
    if not ok:
        raise RuntimeError(reason)

    model_spec = spec["model"]
    module_name = model_spec["module"]
    class_name = model_spec["class_name"]
    kwargs = model_spec.get("kwargs") or {}
    if not isinstance(kwargs, Mapping):
        raise RuntimeError("model.kwargs must be a dictionary when provided")

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class(**dict(kwargs))


def load_state_dict_into_model(
    model: Any,
    state_dict: Mapping[Any, Any],
    strict: bool = True,
    prefix_to_strip: str | None = None,
) -> Any:
    if prefix_to_strip:
        state_dict = strip_state_dict_prefix(state_dict, prefix_to_strip)
    return model.load_state_dict(state_dict, strict=strict)


def strip_state_dict_prefix(
    state_dict: Mapping[Any, Any], prefix: str
) -> dict[Any, Any]:
    stripped = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped


def create_dummy_input_from_spec(spec: dict[str, Any]) -> Any:
    import torch

    input_spec = spec["input"]
    shape = input_spec["shape"]
    example_shape = input_spec.get("example_shape")
    resolved_shape = _resolve_shape(shape, example_shape)
    dtype_name = input_spec["dtype"]
    dtype = _torch_dtype(dtype_name)

    if dtype in {torch.float32, torch.float16}:
        return torch.randn(*resolved_shape, dtype=dtype)
    return torch.zeros(*resolved_shape, dtype=dtype)


def _resolve_shape(shape: list[Any], example_shape: Any) -> list[int]:
    if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
        return shape

    if example_shape is None:
        dynamic_dims = [dim for dim in shape if isinstance(dim, str)]
        raise RuntimeError(
            "input.shape contains dynamic dimensions "
            f"{dynamic_dims}; provide input.example_shape with concrete integers"
        )
    if not isinstance(example_shape, list) or len(example_shape) != len(shape):
        raise RuntimeError("input.example_shape must be a list matching input.shape length")

    resolved: list[int] = []
    for index, dim in enumerate(example_shape):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise RuntimeError(
                f"input.example_shape[{index}] must be an integer, got {dim!r}"
            )
        resolved.append(dim)
    return resolved


def _torch_dtype(dtype_name: str) -> Any:
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int64": torch.int64,
        "int32": torch.int32,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    if dtype_name not in mapping:
        raise RuntimeError(f"unsupported dummy input dtype: {dtype_name}")
    return mapping[dtype_name]
