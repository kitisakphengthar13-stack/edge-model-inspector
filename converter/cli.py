from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

from .checkpoint import (
    load_checkpoint_safe,
    load_checkpoint_unsafe,
    summarize_loaded_checkpoint,
)
from .dry_run import run_model_dry_run
from .inspect_pt import inspect_checkpoint
from .load_plan import print_loading_plan
from .spec import SpecValidationResult, print_validation_result, validate_spec_file
from .utils import print_section


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m converter.cli",
        description="Local tools for inspecting PyTorch checkpoint files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a .pt, .pth, or .ckpt file without exporting it.",
    )
    inspect_parser.add_argument("path", help="Path to the PyTorch checkpoint file.")
    inspect_parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of state_dict entries to print. Default: 40.",
    )
    inspect_parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow full pickle loading for trusted local files only.",
    )
    inspect_parser.set_defaults(func=inspect_command)

    validate_spec_parser = subparsers.add_parser(
        "validate-spec",
        help="Validate a generic model conversion spec.yaml file.",
    )
    validate_spec_parser.add_argument("path", help="Path to the YAML spec file.")
    validate_spec_parser.set_defaults(func=validate_spec_command)

    plan_load_parser = subparsers.add_parser(
        "plan-load",
        help="Plan future model loading from a spec without executing it.",
    )
    plan_load_parser.add_argument("path", help="Path to the YAML spec file.")
    plan_load_parser.set_defaults(func=plan_load_command)

    check_checkpoint_parser = subparsers.add_parser(
        "check-checkpoint",
        help="Load only the checkpoint declared by a valid spec and summarize it.",
    )
    check_checkpoint_parser.add_argument("path", help="Path to the YAML spec file.")
    check_checkpoint_parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of state_dict entries to print. Default: 40.",
    )
    check_checkpoint_parser.set_defaults(func=check_checkpoint_command)

    dry_run_parser = subparsers.add_parser(
        "dry-run-model",
        help="Instantiate a trusted local model and run a CPU dummy forward pass.",
    )
    dry_run_parser.add_argument("path", help="Path to the YAML spec file.")
    dry_run_parser.add_argument(
        "--allow-imports",
        action="store_true",
        help="Allow importing and instantiating the model module declared in the spec.",
    )
    dry_run_parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of output or load diagnostic entries to print. Default: 40.",
    )
    strict_group = dry_run_parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=None,
        help="Force strict state_dict loading.",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow missing or unexpected state_dict keys.",
    )
    dry_run_parser.add_argument(
        "--prefix-to-strip",
        help="Override checkpoint.prefix_to_strip before loading state_dict.",
    )
    dry_run_parser.add_argument(
        "--device",
        default="cpu",
        help="Dry-run device. Phase 4 supports only cpu.",
    )
    dry_run_parser.set_defaults(func=dry_run_model_command)

    return parser


def inspect_command(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    try:
        checkpoint = load_checkpoint(path, unsafe_load=args.unsafe_load)
    except Exception as exc:
        if args.unsafe_load:
            print(f"Error: failed to load checkpoint: {exc}", file=sys.stderr)
        else:
            print(f"Error: safe checkpoint loading failed: {exc}", file=sys.stderr)
            print(
                "This file may require full pickle loading. Re-run with "
                "--unsafe-load only if the file is trusted and local.",
                file=sys.stderr,
            )
        return 2

    inspect_checkpoint(path, checkpoint, max_items=args.max_items)
    return 0


def validate_spec_command(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"Error: spec file not found: {path}", file=sys.stderr)
        return 1

    result = validate_spec_file(path)
    print_validation_result(result)
    return 0 if result.valid else 2


def plan_load_command(args: argparse.Namespace) -> int:
    result = _validate_spec_or_report(Path(args.path))
    if result is None:
        return 2

    print_loading_plan(result.spec)
    _print_spec_warnings(result)
    return 0


def check_checkpoint_command(args: argparse.Namespace) -> int:
    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    result = _validate_spec_or_report(Path(args.path))
    if result is None:
        return 2
    spec = result.spec

    checkpoint_spec = spec["checkpoint"]
    load_mode = checkpoint_spec["load_mode"]
    state_dict_key = checkpoint_spec.get("state_dict_key")

    if checkpoint_spec.get("kind") == "none" and load_mode == "none":
        print("Checkpoint: none declared; no checkpoint was loaded.")
        return 0

    checkpoint_path_value = spec.get("checkpoint_path")
    if not checkpoint_path_value:
        print("Error: checkpoint_path is required for check-checkpoint", file=sys.stderr)
        return 1

    checkpoint_path = Path(checkpoint_path_value)
    if not checkpoint_path.is_file():
        print(f"Error: checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        return 1

    if load_mode in {"custom_loader", "external_loader"}:
        print(
            f"Checkpoint load mode is '{load_mode}'. Custom or external loader "
            "execution is not implemented in Phase 3, so no checkpoint was loaded."
        )
        return 0

    try:
        if load_mode == "safe_weights_only":
            load_result = load_checkpoint_safe(checkpoint_path)
        elif load_mode == "unsafe_trusted_local":
            load_result = load_checkpoint_unsafe(checkpoint_path)
        else:
            print(
                f"Checkpoint load mode '{load_mode}' is custom/unknown. "
                "Checkpoint loading is not implemented for this mode in Phase 3."
            )
            return 0
    except Exception as exc:
        print(f"Error: failed to load checkpoint: {exc}", file=sys.stderr)
        return 2

    for warning in load_result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    summarize_loaded_checkpoint(
        load_result.checkpoint,
        state_dict_key=state_dict_key,
        max_items=args.max_items,
    )
    return 0


def dry_run_model_command(args: argparse.Namespace) -> int:
    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    result = _validate_spec_or_report(Path(args.path))
    if result is None:
        return 2

    return run_model_dry_run(
        result.spec,
        allow_imports=args.allow_imports,
        max_items=args.max_items,
        strict_override=args.strict,
        prefix_to_strip=args.prefix_to_strip,
        device=args.device,
    )


def _validate_spec_or_report(path: Path) -> SpecValidationResult | None:
    if not path.is_file():
        print(f"Error: spec file not found: {path}", file=sys.stderr)
        return None

    result = validate_spec_file(path)
    if not result.valid:
        print_validation_result(result)
        return None
    if result.spec is None:
        print("Error: spec did not load into a dictionary", file=sys.stderr)
        return None
    return result


def _print_spec_warnings(result: SpecValidationResult) -> None:
    if not result.warnings:
        return
    print_section("Spec warnings")
    for warning in result.warnings:
        print(f"- {warning}")


def load_checkpoint(path: Path, unsafe_load: bool = False) -> Any:
    import torch

    supports_weights_only = _torch_load_supports_weights_only(torch.load)

    if unsafe_load:
        print(
            "Warning: unsafe loading uses Python pickle and should only be used "
            "for trusted local files.",
            file=sys.stderr,
        )
        if supports_weights_only:
            return torch.load(path, map_location="cpu", weights_only=False)
        return torch.load(path, map_location="cpu")

    if supports_weights_only:
        return torch.load(path, map_location="cpu", weights_only=True)

    print(
        "Warning: installed PyTorch does not support weights_only=True; falling "
        "back to torch.load(..., map_location='cpu'). Only inspect trusted files.",
        file=sys.stderr,
    )
    return torch.load(path, map_location="cpu")


def _torch_load_supports_weights_only(torch_load: Any) -> bool:
    try:
        return "weights_only" in inspect.signature(torch_load).parameters
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
