from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import print_section, truncate_repr

REQUIRED_TOP_LEVEL_FIELDS = (
    "name",
    "task",
    "checkpoint_path",
    "checkpoint",
    "model",
    "conversion",
    "input",
    "output",
)

OPTIONAL_TOP_LEVEL_FIELDS = {
    "description",
    "framework",
    "preprocessing",
    "postprocessing",
    "runtime_assets",
    "deployment",
    "metadata",
    "notes",
    "class_names",
    "labels",
    "custom",
}

KNOWN_CHECKPOINT_KINDS = {
    "none",
    "raw_state_dict",
    "checkpoint_dict",
    "lightning_checkpoint",
    "full_model",
    "torchscript_archive",
    "generic_dict",
    "unknown",
}

KNOWN_LOAD_MODES = {
    "none",
    "safe_weights_only",
    "unsafe_trusted_local",
    "external_loader",
    "custom_loader",
}

KNOWN_CONVERSION_STRATEGIES = {
    "full_model",
    "feature_extractor_only",
    "module_subgraph",
    "custom_wrapper",
    "torchscript_existing",
    "external_exporter",
}

KNOWN_DTYPES = {"float32", "float16", "int64", "int32", "uint8", "bool"}


@dataclass
class SpecValidationResult:
    path: Path
    spec: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extra_sections: list[str] = field(default_factory=list)
    required_checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def load_spec(path: str | Path) -> dict[str, Any]:
    import yaml

    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("spec root must be a YAML mapping")
    return data


def validate_spec_file(path: str | Path) -> SpecValidationResult:
    spec_path = Path(path)
    result = SpecValidationResult(path=spec_path)

    try:
        spec = load_spec(spec_path)
    except Exception as exc:
        result.errors.append(f"failed to load YAML spec: {exc}")
        return result

    result.spec = spec
    validate_spec(spec, result)
    return result


def validate_spec(spec: dict[str, Any], result: SpecValidationResult) -> None:
    known_top_level = set(REQUIRED_TOP_LEVEL_FIELDS) | OPTIONAL_TOP_LEVEL_FIELDS
    result.extra_sections = sorted(key for key in spec if key not in known_top_level)

    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        present = field_name in spec
        result.required_checks.append(
            (field_name, present, "present" if present else "missing")
        )
        if not present:
            result.errors.append(f"missing required top-level field: {field_name}")

    if not isinstance(spec.get("name"), str):
        result.errors.append("name must be a string")
    if not isinstance(spec.get("task"), str):
        result.errors.append("task must be a string")
    checkpoint_path = spec.get("checkpoint_path")
    if checkpoint_path is not None and not isinstance(checkpoint_path, str):
        result.errors.append("checkpoint_path must be a string, null, or empty string")

    _validate_checkpoint(spec.get("checkpoint"), result)
    _validate_model(spec.get("model"), result)
    _validate_conversion(spec.get("conversion"), result)
    _validate_input(spec.get("input"), result)
    _validate_output(spec.get("output"), result)

    task = spec.get("task")
    if isinstance(task, str) and task == "custom":
        result.warnings.append("task is custom; future phases will need custom handling")
    elif isinstance(task, str) and task:
        result.warnings.append(
            "task names are descriptive only; arbitrary custom task strings are allowed"
        )

    framework = spec.get("framework")
    if isinstance(framework, str) and framework:
        result.warnings.append(
            f"framework '{framework}' is not imported or verified during validation"
        )


def print_validation_result(result: SpecValidationResult) -> None:
    print_section("Spec validation")
    print(f"Path: {result.path}")
    print(f"Valid: {result.valid}")

    print_section("Required fields")
    for field_name, ok, message in result.required_checks:
        status = "ok" if ok else "missing"
        print(f"- {field_name}: {status} ({message})")

    if result.spec:
        print_spec_summary(result.spec)

    print_section("Warnings")
    if result.warnings:
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("None")

    print_section("Extra sections")
    if result.extra_sections:
        for section in result.extra_sections:
            print(f"- {section}")
    else:
        print("None")

    print_section("Errors")
    if result.errors:
        for error in result.errors:
            print(f"- {error}")
    else:
        print("None")


def print_spec_summary(spec: dict[str, Any]) -> None:
    print_section("Summary")
    print(f"Name: {spec.get('name')}")
    print(f"Task: {spec.get('task')}")
    if spec.get("framework") is not None:
        print(f"Framework: {spec.get('framework')}")
    print(f"Checkpoint path: {spec.get('checkpoint_path')}")

    checkpoint = spec.get("checkpoint")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint kind: {checkpoint.get('kind')}")
        print(f"Checkpoint load mode: {checkpoint.get('load_mode')}")

    model = spec.get("model")
    if isinstance(model, dict):
        print(f"Model architecture: {model.get('architecture')}")
        if model.get("module") is not None:
            print(f"Model module: {model.get('module')}")
        if model.get("class_name") is not None:
            print(f"Model class: {model.get('class_name')}")

    conversion = spec.get("conversion")
    if isinstance(conversion, dict):
        print(f"Conversion strategy: {conversion.get('strategy')}")
        if conversion.get("target_format") is not None:
            print(f"Target format: {conversion.get('target_format')}")

    input_spec = spec.get("input")
    if isinstance(input_spec, dict):
        print(
            "Input: "
            f"name={input_spec.get('name')}, "
            f"shape={input_spec.get('shape')}, "
            f"dtype={input_spec.get('dtype')}"
        )

    output = spec.get("output")
    if isinstance(output, dict):
        print(f"Output names: {output.get('names')}")


def _validate_checkpoint(value: Any, result: SpecValidationResult) -> None:
    if not _require_mapping(value, "checkpoint", result):
        return

    kind = _require_string_field(value, "checkpoint", "kind", result)
    load_mode = _require_string_field(value, "checkpoint", "load_mode", result)

    if kind and kind not in KNOWN_CHECKPOINT_KINDS:
        result.warnings.append(
            f"checkpoint.kind '{kind}' is custom/unknown; validation will allow it"
        )
    if load_mode and load_mode not in KNOWN_LOAD_MODES:
        result.warnings.append(
            f"checkpoint.load_mode '{load_mode}' is custom/unknown; validation will allow it"
        )

    if (kind == "none" or load_mode == "none") and kind != load_mode:
        result.errors.append(
            "checkpoint.kind and checkpoint.load_mode must both be 'none' for no-checkpoint specs"
        )


def _validate_model(value: Any, result: SpecValidationResult) -> None:
    if not _require_mapping(value, "model", result):
        return

    module = value.get("module")
    class_name = value.get("class_name")
    if module:
        result.warnings.append(
            f"model.module '{module}' is recorded only; validate-spec does not import it"
        )
    if class_name:
        result.warnings.append(
            f"model.class_name '{class_name}' is recorded only; validate-spec does not instantiate it"
        )


def _validate_conversion(value: Any, result: SpecValidationResult) -> None:
    if not _require_mapping(value, "conversion", result):
        return

    strategy = _require_string_field(value, "conversion", "strategy", result)
    if strategy and strategy not in KNOWN_CONVERSION_STRATEGIES:
        result.warnings.append(
            f"conversion.strategy '{strategy}' is custom/unknown; validation will allow it"
        )


def _validate_input(value: Any, result: SpecValidationResult) -> None:
    if not _require_mapping(value, "input", result):
        return

    _require_string_field(value, "input", "name", result)
    dtype = _require_string_field(value, "input", "dtype", result)

    shape = value.get("shape")
    if "shape" not in value:
        result.errors.append("input.shape is required")
    elif not isinstance(shape, list) or not shape:
        result.errors.append("input.shape must be a non-empty list")
    else:
        for index, dim in enumerate(shape):
            if isinstance(dim, bool) or not isinstance(dim, (int, str)):
                result.errors.append(
                    "input.shape entries must be integers or strings; "
                    f"invalid entry at index {index}: {truncate_repr(dim)}"
                )

    if dtype and dtype not in KNOWN_DTYPES:
        result.warnings.append(
            f"input.dtype '{dtype}' is not in the known dtype list; validation will allow it"
        )


def _validate_output(value: Any, result: SpecValidationResult) -> None:
    if not _require_mapping(value, "output", result):
        return

    names = value.get("names")
    if "names" not in value:
        result.errors.append("output.names is required")
    elif not isinstance(names, list) or not names:
        result.errors.append("output.names must be a non-empty list of strings")
    else:
        for index, name in enumerate(names):
            if not isinstance(name, str):
                result.errors.append(
                    "output.names entries must be strings; "
                    f"invalid entry at index {index}: {truncate_repr(name)}"
                )


def _require_mapping(
    value: Any, section_name: str, result: SpecValidationResult
) -> bool:
    if not isinstance(value, dict):
        result.errors.append(f"{section_name} must be a dictionary")
        return False
    return True


def _require_string_field(
    section: dict[str, Any],
    section_name: str,
    field_name: str,
    result: SpecValidationResult,
) -> str | None:
    if field_name not in section:
        result.errors.append(f"{section_name}.{field_name} is required")
        return None
    value = section[field_name]
    if not isinstance(value, str):
        result.errors.append(f"{section_name}.{field_name} must be a string")
        return None
    return value
