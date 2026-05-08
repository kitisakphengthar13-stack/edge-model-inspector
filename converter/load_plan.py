from __future__ import annotations

from pathlib import Path
from typing import Any

from .spec import (
    KNOWN_CHECKPOINT_KINDS,
    KNOWN_CONVERSION_STRATEGIES,
    KNOWN_LOAD_MODES,
)
from .utils import print_section


def print_loading_plan(spec: dict[str, Any]) -> None:
    checkpoint = spec.get("checkpoint") if isinstance(spec.get("checkpoint"), dict) else {}
    model = spec.get("model") if isinstance(spec.get("model"), dict) else {}
    conversion = (
        spec.get("conversion") if isinstance(spec.get("conversion"), dict) else {}
    )

    checkpoint_path = spec.get("checkpoint_path")
    checkpoint_kind = checkpoint.get("kind")
    load_mode = checkpoint.get("load_mode")
    strategy = conversion.get("strategy")
    loader = checkpoint.get("loader")
    module = model.get("module")
    class_name = model.get("class_name")
    wrapper = conversion.get("wrapper") or model.get("wrapper")

    custom_loader_required = load_mode in {"custom_loader", "external_loader"} or bool(
        loader
    )
    unsafe_required = load_mode == "unsafe_trusted_local"
    can_load_checkpoint = load_mode in {"safe_weights_only", "unsafe_trusted_local"}
    can_instantiate_model = bool(module and class_name) and not custom_loader_required
    wrapper_likely_required = strategy in {
        "feature_extractor_only",
        "module_subgraph",
        "custom_wrapper",
    } or bool(wrapper)

    print_section("Loading plan")
    print(f"Spec name: {spec.get('name')}")
    print(f"Task: {spec.get('task')}")
    if spec.get("framework") is not None:
        print(f"Framework: {spec.get('framework')}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint exists now: {_yes_no(Path(checkpoint_path).is_file()) if isinstance(checkpoint_path, str) else 'no'}")
    print(f"Checkpoint kind: {checkpoint_kind}")
    print(f"Load mode: {load_mode}")
    print(f"Conversion strategy: {strategy}")

    print_section("Model hints")
    print(f"Architecture: {model.get('architecture')}")
    print(f"Module: {module}")
    print(f"Class name: {class_name}")
    if model.get("backbone") is not None:
        print(f"Backbone: {model.get('backbone')}")
    if model.get("layers") is not None:
        print(f"Layers: {model.get('layers')}")

    print_section("Execution requirements")
    print(f"Custom loader declared: {_yes_no(bool(loader))}")
    if loader:
        print(f"Custom loader reference: {loader}")
    print(f"Custom loader required: {_yes_no(custom_loader_required)}")
    print(
        "Checkpoint loading can be attempted safely: "
        f"{_yes_no(load_mode == 'safe_weights_only')}"
    )
    print(f"Unsafe checkpoint loading required: {_yes_no(unsafe_required)}")
    print(f"Export wrapper likely required: {_yes_no(wrapper_likely_required)}")
    if wrapper:
        print(f"Wrapper reference: {wrapper}")

    print_section("Status")
    print("can_plan: yes")
    print(f"can_load_checkpoint: {_yes_no(can_load_checkpoint)}")
    print(f"can_instantiate_model: {_yes_no(can_instantiate_model)}")
    print("can_export_now: no")

    print_section("Plan notes")
    print("- This command does not load checkpoints.")
    print("- This command does not import user modules or instantiate models.")
    print("- This command does not execute custom loaders.")
    print("- ONNX export is not implemented in Phase 4.")
    if checkpoint_kind not in KNOWN_CHECKPOINT_KINDS:
        print(f"- checkpoint.kind '{checkpoint_kind}' is custom/unknown and allowed.")
    if load_mode not in KNOWN_LOAD_MODES:
        print(f"- checkpoint.load_mode '{load_mode}' is custom/unknown and allowed.")
    if strategy not in KNOWN_CONVERSION_STRATEGIES:
        print(f"- conversion.strategy '{strategy}' is custom/unknown and allowed.")
    if can_instantiate_model:
        print(
            "- The spec contains module/class hints, but automatic model "
            "instantiation is not performed by plan-load."
        )
    else:
        print(
            "- Model instantiation is not currently possible from this plan "
            "without future loader/wrapper support."
        )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
