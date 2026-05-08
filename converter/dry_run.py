from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import (
    extract_state_dict,
    load_checkpoint_safe,
    load_checkpoint_unsafe,
)
from .model_loader import (
    can_instantiate_from_spec,
    create_dummy_input_from_spec,
    instantiate_model_from_spec,
    load_state_dict_into_model,
)
from .utils import is_tensor_like, print_section, safe_dtype, safe_shape, truncate_repr


def run_model_dry_run(
    spec: dict[str, Any],
    *,
    allow_imports: bool,
    max_items: int = 40,
    strict_override: bool | None = None,
    prefix_to_strip: str | None = None,
    device: str = "cpu",
) -> int:
    if device != "cpu":
        print("Error: Phase 4 supports CPU dry-run only.")
        return 2

    checkpoint_spec = spec["checkpoint"]
    model_spec = spec["model"]
    load_mode = checkpoint_spec["load_mode"]
    checkpoint_path = spec.get("checkpoint_path")
    strict = (
        strict_override
        if strict_override is not None
        else checkpoint_spec.get("strict", True)
    )
    prefix = (
        prefix_to_strip
        if prefix_to_strip is not None
        else checkpoint_spec.get("prefix_to_strip")
    )

    print_section("Dry run")
    print(f"Spec name: {spec.get('name')}")
    print(f"Task: {spec.get('task')}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Model module: {model_spec.get('module')}")
    print(f"Model class: {model_spec.get('class_name')}")
    print(f"Checkpoint load mode: {load_mode}")
    print(f"Strict load mode: {strict}")

    if not allow_imports:
        print(
            "Error: refusing to import or instantiate user model code without "
            "--allow-imports. Only use --allow-imports for trusted local modules."
        )
        return 2

    if load_mode in {"custom_loader", "external_loader"}:
        print(
            f"Error: checkpoint load mode '{load_mode}' requires custom code. "
            "Custom and external loader execution is not implemented in Phase 4."
        )
        return 2

    can_instantiate, reason = can_instantiate_from_spec(spec)
    if not can_instantiate:
        print(f"Error: {reason}")
        return 2

    state_dict = None
    state_source = "none"
    if checkpoint_spec["kind"] == "none" and load_mode == "none":
        print("Checkpoint: none declared; running with randomly initialized weights.")
    else:
        if not checkpoint_path:
            print("Error: checkpoint_path is required unless checkpoint.kind is 'none'.")
            return 2
        path = Path(checkpoint_path)
        if not path.is_file():
            print(f"Error: checkpoint file not found: {path}")
            return 2

        try:
            if load_mode == "safe_weights_only":
                load_result = load_checkpoint_safe(path)
            elif load_mode == "unsafe_trusted_local":
                load_result = load_checkpoint_unsafe(path)
            else:
                print(
                    f"Error: checkpoint load mode '{load_mode}' is not executable "
                    "by dry-run-model in Phase 4."
                )
                return 2
        except Exception as exc:
            print(f"Error: failed to load checkpoint: {exc}")
            return 2

        for warning in load_result.warnings:
            print(f"Warning: {warning}")

        state_result = extract_state_dict(
            load_result.checkpoint,
            state_dict_key=checkpoint_spec.get("state_dict_key"),
        )
        if state_result.state_dict is None:
            print(f"Error: state_dict extraction failed: {state_result.reason}")
            return 2
        state_dict = state_result.state_dict
        state_source = state_result.source or "unknown"

    print(f"State dict source: {state_source}")

    try:
        model = instantiate_model_from_spec(spec, allow_imports=allow_imports)
    except Exception as exc:
        print(f"Error: failed to instantiate model: {exc}")
        return 2

    if state_dict is not None:
        try:
            load_info = load_state_dict_into_model(
                model, state_dict, strict=bool(strict), prefix_to_strip=prefix
            )
        except Exception as exc:
            print(f"Error: failed to load state_dict into model: {exc}")
            print(
                "Diagnostic options: try --no-strict or --prefix-to-strip if "
                "the checkpoint naming differs from the model."
            )
            return 2
        _print_load_info(load_info, max_items=max_items)

    model.eval()
    try:
        dummy_input = create_dummy_input_from_spec(spec)
    except Exception as exc:
        print(f"Error: failed to create dummy input: {exc}")
        return 2

    print(f"Dummy input shape: {safe_shape(dummy_input)}")
    print(f"Dummy input dtype: {safe_dtype(dummy_input)}")

    try:
        import torch

        with torch.no_grad():
            output = model(dummy_input)
    except Exception as exc:
        print(f"Forward status: failed")
        print(f"Error: model forward failed: {exc}")
        return 2

    print("Forward status: success")
    print_output_summary(output, max_items=max_items)
    return 0


def print_output_summary(output: Any, max_items: int = 40) -> None:
    print_section("Output summary")
    if is_tensor_like(output):
        print(f"Tensor: shape={safe_shape(output)}, dtype={safe_dtype(output)}")
    elif isinstance(output, (tuple, list)):
        print(f"{type(output).__name__}: length={len(output)}")
        for index, item in enumerate(output[:max_items]):
            print(f"- [{index}] {_describe_value(item)}")
    elif isinstance(output, dict):
        print(f"dict: keys={list(output.keys())[:max_items]}")
        for index, (key, value) in enumerate(output.items()):
            if index >= max_items:
                break
            print(f"- {key}: {_describe_value(value)}")
    else:
        print(f"{type(output).__module__}.{type(output).__qualname__}: {truncate_repr(output)}")


def _describe_value(value: Any) -> str:
    if is_tensor_like(value):
        return f"tensor_like=True, shape={safe_shape(value)}, dtype={safe_dtype(value)}"
    return f"type={type(value).__module__}.{type(value).__qualname__}, repr={truncate_repr(value)}"


def _print_load_info(load_info: Any, max_items: int) -> None:
    missing = list(getattr(load_info, "missing_keys", []) or [])
    unexpected = list(getattr(load_info, "unexpected_keys", []) or [])
    print_section("State dict load")
    print(f"Missing keys: {len(missing)}")
    for key in missing[:max_items]:
        print(f"- missing: {key}")
    print(f"Unexpected keys: {len(unexpected)}")
    for key in unexpected[:max_items]:
        print(f"- unexpected: {key}")
