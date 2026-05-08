from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from .checkpoint import extract_state_dict, load_checkpoint_safe, load_checkpoint_unsafe
from .dry_run import print_output_summary
from .model_loader import (
    can_instantiate_from_spec,
    create_dummy_input_from_spec,
    instantiate_model_from_spec,
    load_state_dict_into_model,
)
from .spec import validate_spec_file
from .utils import print_section, safe_dtype, safe_shape


def export_onnx_from_spec(
    spec_path: str,
    output_path: str | None = None,
    allow_imports: bool = False,
    opset: int | None = None,
    strict: bool | None = None,
    prefix_to_strip: str | None = None,
    device: str = "cpu",
    dynamo: bool = True,
    fallback_legacy: bool = True,
) -> dict[str, Any]:
    if device != "cpu":
        raise RuntimeError("Phase 5 supports CPU export only.")
    if not allow_imports:
        raise RuntimeError(
            "Refusing to import user model code. Re-run with --allow-imports only "
            "for trusted local modules."
        )

    validation = validate_spec_file(spec_path)
    if not validation.valid or validation.spec is None:
        errors = "; ".join(validation.errors) or "spec did not load"
        raise RuntimeError(f"spec validation failed: {errors}")

    spec = validation.spec
    checkpoint_spec = spec["checkpoint"]
    conversion = spec["conversion"]
    strategy = conversion["strategy"]
    load_mode = checkpoint_spec["load_mode"]

    if load_mode in {"custom_loader", "external_loader"}:
        raise RuntimeError(
            f"checkpoint load mode '{load_mode}' requires custom code. "
            "Custom and external loader execution is not implemented in Phase 5."
        )
    if strategy == "custom_wrapper":
        raise RuntimeError("conversion.strategy custom_wrapper is not implemented yet.")
    if strategy == "torchscript_existing":
        raise RuntimeError("conversion.strategy torchscript_existing is not implemented yet.")
    if strategy == "module_subgraph":
        raise RuntimeError("conversion.strategy module_subgraph is not implemented yet.")
    if strategy not in {"full_model", "feature_extractor_only"}:
        raise RuntimeError(
            f"conversion.strategy '{strategy}' is not executable by export-onnx in Phase 5."
        )

    can_instantiate, reason = can_instantiate_from_spec(spec)
    if not can_instantiate:
        raise RuntimeError(reason)

    export_path = _resolve_output_path(spec, output_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_opset = int(opset if opset is not None else conversion.get("opset", 17))
    input_name = spec["input"]["name"]
    output_names = spec["output"]["names"]
    dynamic_axes = conversion.get("dynamic_axes")
    strict_mode = (
        strict if strict is not None else checkpoint_spec.get("strict", True)
    )
    prefix = (
        prefix_to_strip
        if prefix_to_strip is not None
        else checkpoint_spec.get("prefix_to_strip")
    )

    print_section("ONNX export")
    print(f"Spec name: {spec.get('name')}")
    print(f"Task: {spec.get('task')}")
    print(f"Output path: {export_path}")
    print(f"Opset: {resolved_opset}")
    print(f"Input name: {input_name}")
    print(f"Output names: {output_names}")

    state_dict = None
    state_source = "none"
    if checkpoint_spec["kind"] == "none" and load_mode == "none":
        print("Checkpoint: none declared; exporting randomly initialized weights.")
    else:
        checkpoint_path = spec.get("checkpoint_path")
        if not checkpoint_path:
            raise RuntimeError("checkpoint_path is required unless checkpoint.kind is 'none'.")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise RuntimeError(f"checkpoint file not found: {path}")

        if load_mode == "safe_weights_only":
            load_result = load_checkpoint_safe(path)
        elif load_mode == "unsafe_trusted_local":
            load_result = load_checkpoint_unsafe(path)
        else:
            raise RuntimeError(
                f"checkpoint load mode '{load_mode}' is not executable by export-onnx."
            )
        for warning in load_result.warnings:
            print(f"Warning: {warning}")

        state_result = extract_state_dict(
            load_result.checkpoint,
            state_dict_key=checkpoint_spec.get("state_dict_key"),
        )
        if state_result.state_dict is None:
            raise RuntimeError(f"state_dict extraction failed: {state_result.reason}")
        state_dict = state_result.state_dict
        state_source = state_result.source or "unknown"

    print(f"State dict source: {state_source}")

    try:
        model = instantiate_model_from_spec(spec, allow_imports=True)
    except Exception as exc:
        raise RuntimeError(f"failed to instantiate model: {exc}") from exc

    if state_dict is not None:
        try:
            load_state_dict_into_model(
                model,
                state_dict,
                strict=bool(strict_mode),
                prefix_to_strip=prefix,
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to load state_dict into model: "
                f"{exc}. Diagnostic options: try --no-strict or --prefix-to-strip."
            ) from exc

    model.eval()
    dummy_input = create_dummy_input_from_spec(spec)
    print(f"Dummy input shape: {safe_shape(dummy_input)}")
    print(f"Dummy input dtype: {safe_dtype(dummy_input)}")

    import torch

    with torch.no_grad():
        output = model(dummy_input)
    print("Dry-run forward: success")
    print_output_summary(output)

    exporter_used = _export_with_selected_path(
        torch,
        model,
        dummy_input,
        export_path,
        input_name=input_name,
        output_names=output_names,
        opset=resolved_opset,
        dynamic_axes=dynamic_axes,
        dynamo=dynamo,
        fallback_legacy=fallback_legacy,
    )

    print(f"Exporter used: {exporter_used}")
    print(f"Export success: yes")
    return {
        "spec_name": spec.get("name"),
        "output_path": str(export_path),
        "opset": resolved_opset,
        "input_name": input_name,
        "output_names": output_names,
        "dynamo_used": exporter_used == "dynamo",
        "exporter_used": exporter_used,
        "success": True,
    }


def _export_with_selected_path(
    torch_module: Any,
    model: Any,
    dummy_input: Any,
    export_path: Path,
    *,
    input_name: str,
    output_names: list[str],
    opset: int,
    dynamic_axes: Any,
    dynamo: bool,
    fallback_legacy: bool,
) -> str:
    supports_dynamo = _torch_export_supports_dynamo(torch_module.onnx.export)
    common_kwargs = {
        "input_names": [input_name],
        "output_names": output_names,
        "opset_version": opset,
    }
    if dynamic_axes is not None:
        common_kwargs["dynamic_axes"] = dynamic_axes

    if dynamo and supports_dynamo:
        try:
            torch_module.onnx.export(
                model,
                (dummy_input,),
                str(export_path),
                dynamo=True,
                **common_kwargs,
            )
            return "dynamo"
        except Exception as exc:
            dependency_message = _missing_export_dependency_message(exc)
            if not fallback_legacy:
                raise RuntimeError(
                    dependency_message or f"dynamo ONNX export failed: {exc}"
                ) from exc
            print(
                "Warning: dynamo ONNX export failed; retrying legacy exporter: "
                f"{dependency_message or exc}"
            )
            try:
                _export_legacy(torch_module, model, dummy_input, export_path, common_kwargs)
            except Exception as legacy_exc:
                raise RuntimeError(
                    _missing_export_dependency_message(legacy_exc)
                    or f"legacy ONNX export failed: {legacy_exc}"
                ) from legacy_exc
            return "legacy_fallback"

    if dynamo and not supports_dynamo:
        print("Warning: installed PyTorch does not support torch.onnx.export dynamo argument.")

    try:
        _export_legacy(torch_module, model, dummy_input, export_path, common_kwargs)
    except Exception as exc:
        raise RuntimeError(
            _missing_export_dependency_message(exc)
            or f"legacy ONNX export failed: {exc}"
        ) from exc
    return "legacy"


def _export_legacy(
    torch_module: Any,
    model: Any,
    dummy_input: Any,
    export_path: Path,
    common_kwargs: dict[str, Any],
) -> None:
    kwargs = dict(common_kwargs)
    if _torch_export_supports_dynamo(torch_module.onnx.export):
        kwargs["dynamo"] = False
    torch_module.onnx.export(model, dummy_input, str(export_path), **kwargs)


def _torch_export_supports_dynamo(export_func: Any) -> bool:
    try:
        return "dynamo" in inspect.signature(export_func).parameters
    except (TypeError, ValueError):
        return False


def _resolve_output_path(spec: dict[str, Any], output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    conversion_output = spec.get("conversion", {}).get("output_path")
    if conversion_output:
        return Path(conversion_output)
    return Path("artifacts") / str(spec["name"]) / "model.onnx"


def _missing_export_dependency_message(exc: Exception) -> str | None:
    text = str(exc)
    if "onnxscript" in text:
        return "Missing ONNX export dependency 'onnxscript'. Install requirements.txt."
    if "Module onnx is not installed" in text or "No module named 'onnx'" in text:
        return "Missing ONNX export dependency 'onnx'. Install requirements.txt."
    return None
