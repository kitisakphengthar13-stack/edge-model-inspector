from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from .spec import validate_spec_file
from .utils import print_section

SUPPORTED_PRECISIONS = {"fp32", "fp16", "int8"}


def create_tensorrt_plan(
    onnx_path: str,
    spec_path: str | None = None,
    target: str = "generic",
    precision: str = "fp16",
    engine_output: str | None = None,
    workspace_mb: int | None = None,
    min_shape: str | None = None,
    opt_shape: str | None = None,
    max_shape: str | None = None,
    input_name: str | None = None,
    timing_cache: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    path = Path(onnx_path)
    if not path.is_file():
        raise RuntimeError(f"ONNX file not found: {path}")

    precision = precision.lower()
    if precision not in SUPPORTED_PRECISIONS:
        raise RuntimeError(
            f"unsupported precision '{precision}'. Supported values: fp32, fp16, int8"
        )

    spec = _load_valid_spec(spec_path) if spec_path else None
    spec_name = spec.get("name") if spec else None
    spec_input = spec.get("input") if spec else None
    resolved_input_name = input_name or (
        spec_input.get("name") if isinstance(spec_input, dict) else None
    )
    fixed_shape = _fixed_shape_from_spec(spec_input)
    dynamic_shapes = _resolve_dynamic_shapes(
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        input_name=resolved_input_name,
    )
    resolved_engine_output = _default_engine_output(
        path, spec_name=spec_name, target=target, precision=precision, override=engine_output
    )

    warnings: list[str] = []
    if precision == "int8":
        warnings.append(
            "INT8 usually requires calibration or explicit quantization; Phase 7 only plans the flag."
        )
    if not dynamic_shapes and fixed_shape is None:
        warnings.append(
            "No concrete fixed shape was found. Dynamic ONNX inputs may require "
            "--min-shape, --opt-shape, and --max-shape."
        )

    command = _build_command(
        onnx_path=str(path),
        engine_output=resolved_engine_output,
        precision=precision,
        workspace_mb=workspace_mb,
        timing_cache=timing_cache,
        verbose=verbose,
        dynamic_shapes=dynamic_shapes,
    )
    command_string = " ".join(shlex.quote(part) for part in command)

    _print_plan(
        onnx_path=str(path),
        spec_path=spec_path,
        spec_name=spec_name,
        target=target,
        precision=precision,
        engine_output=resolved_engine_output,
        workspace_mb=workspace_mb,
        input_name=resolved_input_name,
        fixed_shape=fixed_shape,
        dynamic_shapes=dynamic_shapes,
        timing_cache=timing_cache,
        verbose=verbose,
        warnings=warnings,
        command=command,
        command_string=command_string,
    )

    return {
        "onnx_path": str(path),
        "spec_path": spec_path,
        "spec_name": spec_name,
        "target": target,
        "precision": precision,
        "engine_output": resolved_engine_output,
        "workspace_mb": workspace_mb,
        "input_name": resolved_input_name,
        "fixed_shape": fixed_shape,
        "dynamic_shapes": dynamic_shapes,
        "timing_cache": timing_cache,
        "verbose": verbose,
        "warnings": warnings,
        "command": command,
        "command_string": command_string,
        "success": True,
    }


def _load_valid_spec(spec_path: str | None) -> dict[str, Any]:
    result = validate_spec_file(spec_path)
    if not result.valid or result.spec is None:
        errors = "; ".join(result.errors) or "spec did not load"
        raise RuntimeError(f"spec validation failed: {errors}")
    return result.spec


def _fixed_shape_from_spec(input_spec: Any) -> str | None:
    if not isinstance(input_spec, dict):
        return None
    shape = input_spec.get("shape")
    if not isinstance(shape, list):
        return None
    if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
        return "x".join(str(dim) for dim in shape)
    return None


def _resolve_dynamic_shapes(
    *,
    min_shape: str | None,
    opt_shape: str | None,
    max_shape: str | None,
    input_name: str | None,
) -> dict[str, str] | None:
    provided = [value is not None for value in (min_shape, opt_shape, max_shape)]
    if not any(provided):
        return None
    if not all(provided):
        raise RuntimeError(
            "dynamic TensorRT shapes require all three: --min-shape, --opt-shape, --max-shape"
        )
    if not input_name:
        raise RuntimeError("dynamic TensorRT shapes require --input-name or spec.input.name")
    return {
        "min": _normalize_shape(min_shape),
        "opt": _normalize_shape(opt_shape),
        "max": _normalize_shape(max_shape),
        "input_name": input_name,
    }


def _normalize_shape(value: str | None) -> str:
    if not value:
        raise RuntimeError("shape value must not be empty")
    normalized = value.replace(",", "x")
    parts = normalized.split("x")
    if not parts or any(not part.isdigit() for part in parts):
        raise RuntimeError(f"invalid TensorRT shape '{value}'. Use forms like 1x3x224x224.")
    return "x".join(parts)


def _default_engine_output(
    onnx_path: Path,
    *,
    spec_name: str | None,
    target: str,
    precision: str,
    override: str | None,
) -> str:
    if override:
        return override
    if spec_name:
        return str(
            Path("artifacts")
            / spec_name
            / "targets"
            / target
            / f"model_{precision}.engine"
        )
    return str(onnx_path.with_name(f"{onnx_path.stem}_{target}_{precision}.engine"))


def _build_command(
    *,
    onnx_path: str,
    engine_output: str,
    precision: str,
    workspace_mb: int | None,
    timing_cache: str | None,
    verbose: bool,
    dynamic_shapes: dict[str, str] | None,
) -> list[str]:
    command = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_output}"]
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")
    if workspace_mb is not None:
        command.append(f"--workspace={workspace_mb}")
    if timing_cache:
        command.append(f"--timingCacheFile={timing_cache}")
    if dynamic_shapes:
        name = dynamic_shapes["input_name"]
        command.extend(
            [
                f"--minShapes={name}:{dynamic_shapes['min']}",
                f"--optShapes={name}:{dynamic_shapes['opt']}",
                f"--maxShapes={name}:{dynamic_shapes['max']}",
            ]
        )
    if verbose:
        command.append("--verbose")
    return command


def _print_plan(**plan: Any) -> None:
    print_section("TensorRT build plan")
    print(f"ONNX path: {plan['onnx_path']}")
    print(f"Spec path: {plan['spec_path'] or '-'}")
    print(f"Spec name: {plan['spec_name'] or '-'}")
    print(f"Target: {plan['target']}")
    print(f"Precision: {plan['precision']}")
    print(f"Engine output: {plan['engine_output']}")
    print(f"Workspace MB: {plan['workspace_mb'] if plan['workspace_mb'] is not None else '-'}")
    print(f"Input name: {plan['input_name'] or '-'}")
    print(f"Fixed shape from spec: {plan['fixed_shape'] or '-'}")
    if plan["dynamic_shapes"]:
        shapes = plan["dynamic_shapes"]
        print(f"Dynamic min shape: {shapes['min']}")
        print(f"Dynamic opt shape: {shapes['opt']}")
        print(f"Dynamic max shape: {shapes['max']}")
    print(f"Timing cache: {plan['timing_cache'] or '-'}")
    print(f"Verbose trtexec: {'yes' if plan['verbose'] else 'no'}")

    print_section("Warnings")
    if plan["warnings"]:
        for warning in plan["warnings"]:
            print(f"- {warning}")
    else:
        print("None")

    print_section("Generated command")
    print(plan["command_string"])
    print()
    print("This is a plan only. trtexec was not executed.")
