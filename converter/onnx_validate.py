from __future__ import annotations

from pathlib import Path
from typing import Any

from .spec import validate_spec_file
from .utils import print_section

DTYPE_TO_NUMPY = {
    "float32": "float32",
    "float16": "float16",
    "int64": "int64",
    "int32": "int32",
    "uint8": "uint8",
    "bool": "bool",
}

ONNX_TYPE_TO_DTYPE = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


def validate_onnx_file(
    onnx_path: str,
    spec_path: str | None = None,
    input_shape: list[int] | None = None,
    input_dtype: str | None = None,
    input_name: str | None = None,
    max_items: int = 20,
) -> dict[str, Any]:
    path = Path(onnx_path)
    if not path.is_file():
        raise RuntimeError(f"ONNX file not found: {path}")

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        missing = exc.name or "required ONNX validation package"
        raise RuntimeError(
            f"Missing ONNX validation dependency '{missing}'. Install requirements.txt."
        ) from exc

    print_section("ONNX validation")
    print(f"Path: {path}")

    model = onnx.load(path)
    _print_model_info(model)

    onnx.checker.check_model(model)
    print("ONNX checker: passed")

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("ONNX Runtime session: created")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Session providers: {session.get_providers()}")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    _print_io_metadata("Model inputs", inputs, max_items=max_items)
    _print_io_metadata("Model outputs", outputs, max_items=max_items)

    if len(inputs) != 1:
        raise RuntimeError(
            "Phase 6 supports single-input ONNX models first. "
            f"Model has {len(inputs)} inputs; provide a future multi-input mapping."
        )

    resolved_input_name, resolved_shape, resolved_dtype = _resolve_input_request(
        inputs[0],
        spec_path=spec_path,
        input_shape=input_shape,
        input_dtype=input_dtype,
        input_name=input_name,
    )
    dummy_input = _create_numpy_dummy_input(
        np, resolved_shape, resolved_dtype
    )

    print_section("Inference input")
    print(f"Input name: {resolved_input_name}")
    print(f"Input shape: {list(dummy_input.shape)}")
    print(f"Input dtype: {dummy_input.dtype}")

    ort_outputs = session.run(None, {resolved_input_name: dummy_input})
    print("ONNX Runtime inference: passed")

    output_summaries = _summarize_outputs(outputs, ort_outputs, max_items=max_items)
    return {
        "onnx_path": str(path),
        "checker_passed": True,
        "ort_session_created": True,
        "inference_passed": True,
        "input_names": [item.name for item in inputs],
        "output_names": [item.name for item in outputs],
        "output_summaries": output_summaries,
        "success": True,
    }


def parse_cli_shape(value: str) -> list[int]:
    try:
        shape = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise RuntimeError("--input-shape must contain comma-separated integers") from exc
    if not shape:
        raise RuntimeError("--input-shape must not be empty")
    return shape


def _resolve_input_request(
    onnx_input: Any,
    *,
    spec_path: str | None,
    input_shape: list[int] | None,
    input_dtype: str | None,
    input_name: str | None,
) -> tuple[str, list[int], str]:
    spec_input = _load_spec_input(spec_path) if spec_path else None

    resolved_name = input_name
    if resolved_name is None and spec_input is not None:
        resolved_name = spec_input.get("name")
    if resolved_name is None:
        resolved_name = onnx_input.name
    if resolved_name != onnx_input.name:
        raise RuntimeError(
            f"input name '{resolved_name}' does not match ONNX model input '{onnx_input.name}'"
        )

    if input_shape is not None:
        resolved_shape = input_shape
    elif spec_input is not None:
        resolved_shape = _resolve_spec_shape(spec_input)
    else:
        resolved_shape = _resolve_onnx_shape(onnx_input.shape)

    if input_dtype is not None:
        resolved_dtype = input_dtype
    elif spec_input is not None:
        resolved_dtype = spec_input.get("dtype")
    else:
        resolved_dtype = ONNX_TYPE_TO_DTYPE.get(str(onnx_input.type))
        if resolved_dtype is None:
            print(
                f"Warning: could not infer dtype from ONNX type '{onnx_input.type}'; "
                "using float32"
            )
            resolved_dtype = "float32"

    if resolved_dtype not in DTYPE_TO_NUMPY:
        raise RuntimeError(f"unsupported input dtype for ONNX validation: {resolved_dtype}")
    return resolved_name, resolved_shape, resolved_dtype


def _load_spec_input(spec_path: str | None) -> dict[str, Any]:
    result = validate_spec_file(spec_path)
    if not result.valid or result.spec is None:
        errors = "; ".join(result.errors) or "spec did not load"
        raise RuntimeError(f"spec validation failed: {errors}")
    return result.spec["input"]


def _resolve_spec_shape(input_spec: dict[str, Any]) -> list[int]:
    shape = input_spec["shape"]
    if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
        return list(shape)

    example_shape = input_spec.get("example_shape")
    if example_shape is None:
        dynamic_dims = [dim for dim in shape if isinstance(dim, str)]
        raise RuntimeError(
            "spec.input.shape contains dynamic dimensions "
            f"{dynamic_dims}; provide spec.input.example_shape or --input-shape"
        )
    if not isinstance(example_shape, list) or len(example_shape) != len(shape):
        raise RuntimeError("spec.input.example_shape must match spec.input.shape length")
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in example_shape):
        raise RuntimeError("spec.input.example_shape entries must be integers")
    return list(example_shape)


def _resolve_onnx_shape(shape: list[Any]) -> list[int]:
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            raise RuntimeError(
                "ONNX input shape has dynamic or unknown dimensions; provide --input-shape"
            )
    return resolved


def _create_numpy_dummy_input(np: Any, shape: list[int], dtype_name: str) -> Any:
    dtype = np.dtype(DTYPE_TO_NUMPY[dtype_name])
    if np.issubdtype(dtype, np.floating):
        return np.random.randn(*shape).astype(dtype)
    return np.zeros(shape, dtype=dtype)


def _print_model_info(model: Any) -> None:
    print_section("ONNX model")
    print(f"IR version: {model.ir_version}")
    print(f"Producer name: {model.producer_name or '-'}")
    print(f"Producer version: {model.producer_version or '-'}")
    print(f"Graph name: {model.graph.name or '-'}")
    opsets = ", ".join(
        f"{item.domain or 'ai.onnx'}:{item.version}" for item in model.opset_import
    )
    print(f"Opset imports: {opsets}")


def _print_io_metadata(title: str, values: list[Any], max_items: int) -> None:
    print_section(title)
    for value in values[:max_items]:
        print(f"- name={value.name}, shape={value.shape}, type={value.type}")
    if len(values) > max_items:
        print(f"... {len(values) - max_items} more")


def _summarize_outputs(
    metadata: list[Any], values: list[Any], max_items: int
) -> list[dict[str, Any]]:
    print_section("Output summary")
    summaries: list[dict[str, Any]] = []
    for index, value in enumerate(values[:max_items]):
        name = metadata[index].name if index < len(metadata) else f"output_{index}"
        summary: dict[str, Any] = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        print(f"- {name}: shape={summary['shape']}, dtype={summary['dtype']}")
        if value.size and value.dtype.kind in {"f", "i", "u", "b"}:
            summary["min"] = float(value.min())
            summary["max"] = float(value.max())
            summary["mean"] = float(value.mean())
            print(
                f"  min={summary['min']:.6g}, max={summary['max']:.6g}, "
                f"mean={summary['mean']:.6g}"
            )
        summaries.append(summary)
    if len(values) > max_items:
        print(f"... {len(values) - max_items} more")
    return summaries
