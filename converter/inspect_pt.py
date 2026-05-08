from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .utils import (
    format_file_size,
    is_tensor_like,
    print_section,
    safe_dtype,
    safe_shape,
    safe_tensor_nbytes,
    truncate_repr,
)

STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "module_state_dict",
    "ema_state_dict",
)

METADATA_KEYS = (
    "hyper_parameters",
    "epoch",
    "global_step",
    "pytorch-lightning_version",
    "callbacks",
    "optimizer_states",
    "lr_schedulers",
    "config",
    "args",
    "model_args",
    "class_names",
    "names",
    "nc",
    "num_classes",
)

TASK_KEYWORDS = {
    "classification": (
        "classifier",
        "classification",
        "class_names",
        "num_classes",
        "fc.weight",
        "logits",
        "softmax",
    ),
    "object_detection": (
        "anchor",
        "anchors",
        "bbox",
        "box",
        "boxes",
        "detect",
        "detection",
        "rpn",
        "roi_heads",
        "yolo",
        "nms",
    ),
    "segmentation": (
        "seg",
        "segmentation",
        "mask",
        "masks",
        "decode_head",
        "mask_head",
        "seg_head",
    ),
    "pose_estimation": (
        "pose",
        "keypoint",
        "keypoints",
        "kpt",
        "heatmap",
        "joints",
    ),
    "anomaly_detection": (
        "anomaly",
        "anomalib",
        "stfpm",
        "padim",
        "patchcore",
        "fastflow",
        "cflow",
        "memory_bank",
        "anomaly_map",
        "image_threshold",
        "pixel_threshold",
        "post_processor",
        "image_min",
        "image_max",
        "pixel_min",
        "pixel_max",
        "feature_extractor",
    ),
    "autoencoder": (
        "autoencoder",
        "encoder",
        "decoder",
        "bottleneck",
        "latent",
        "vae",
    ),
    "ocr": (
        "ocr",
        "ctc",
        "charset",
        "text_recogn",
        "crnn",
        "recognition_head",
    ),
    "depth_estimation": (
        "depth",
        "disparity",
        "inverse_depth",
        "depth_head",
    ),
    "super_resolution": (
        "super_resolution",
        "superres",
        "sr.",
        "upsample",
        "upscale",
        "pixel_shuffle",
        "esrgan",
    ),
    "embedding_or_metric_learning": (
        "embedding",
        "embedder",
        "metric",
        "projection",
        "proj_head",
        "triplet",
        "contrastive",
    ),
}

ANOMALY_STRONG_KEYS = {
    "memory_bank",
    "anomaly_map",
    "image_threshold",
    "pixel_threshold",
    "post_processor",
    "image_min",
    "image_max",
    "pixel_min",
    "pixel_max",
}

THRESHOLD_KEYWORDS = (
    "threshold",
    "image_threshold",
    "pixel_threshold",
    "image_min",
    "image_max",
    "pixel_min",
    "pixel_max",
)

BACKBONE_LAYER_KEYWORDS = (
    "backbone",
    "feature_extractor",
    "layers",
    "layer",
    "blocks",
    "encoder",
)

LARGE_TENSOR_BYTES = 1024 * 1024


def inspect_checkpoint(path: str | Path, checkpoint: Any, max_items: int = 40) -> None:
    checkpoint_path = Path(path)
    print_section("File")
    print(f"Path: {checkpoint_path}")
    print(f"Size: {format_file_size(checkpoint_path.stat().st_size)}")
    print(f"Top-level Python object type: {_type_name(checkpoint)}")
    print(f"Detected checkpoint kind: {detect_checkpoint_kind(checkpoint, checkpoint_path)}")

    print_metadata(checkpoint)

    state_dict_name, state_dict = find_state_dict(checkpoint)
    if state_dict is not None:
        print_state_dict(state_dict_name, state_dict, max_items=max_items)

    print_deployment_signals(checkpoint, state_dict)

    key_names = [item.path for item in collect_mapping_entries(checkpoint)]
    if state_dict is not None:
        key_names.extend(str(key) for key in state_dict.keys())
    print_task_hints(key_names)

    print_section("Conversion note")
    print(
        "Task hints are heuristic only. Exact model conversion still requires "
        "the model architecture, checkpoint loading rule, input shape, output "
        "spec, and preprocessing/postprocessing rules."
    )


def detect_checkpoint_kind(obj: Any, path: Path | None = None) -> str:
    if path and is_torchscript_archive(path):
        return "TorchScript archive"

    if _is_nn_module(obj):
        return "full PyTorch model object"

    if is_state_dict_like(obj):
        return "raw state_dict"

    if isinstance(obj, Mapping):
        if is_lightning_checkpoint(obj):
            return "PyTorch Lightning checkpoint"
        if any(key in obj for key in STATE_DICT_KEYS) or "optimizer" in obj:
            return "checkpoint dict"
        return "generic dict"

    return "unknown"


def is_torchscript_archive(path: Path) -> bool:
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except Exception:
        return False

    normalized = [name.split("/", 1)[-1] if "/" in name else name for name in names]
    has_code = any("/code/" in f"/{name}" or name.startswith("code/") for name in names)
    has_constants = any(name.endswith("constants.pkl") for name in normalized)
    has_data = any(name.endswith("data.pkl") for name in normalized)
    return has_code and has_constants and has_data


def is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False

    string_keys = sum(1 for key in obj if isinstance(key, str))
    tensor_values = sum(1 for value in obj.values() if is_tensor_like(value))
    return string_keys == len(obj) and tensor_values >= max(1, int(len(obj) * 0.6))


def is_lightning_checkpoint(obj: Mapping[Any, Any]) -> bool:
    lightning_keys = {
        "pytorch-lightning_version",
        "hyper_parameters",
        "callbacks",
        "optimizer_states",
        "lr_schedulers",
    }
    return "state_dict" in obj and any(key in obj for key in lightning_keys)


def find_state_dict(obj: Any) -> tuple[str, Mapping[Any, Any] | None]:
    if is_state_dict_like(obj):
        return "top-level object", obj

    if not isinstance(obj, Mapping):
        return "", None

    for key in STATE_DICT_KEYS:
        value = obj.get(key)
        if is_state_dict_like(value):
            return key, value

    for key in ("model", "module", "net", "network"):
        value = obj.get(key)
        if is_state_dict_like(value):
            return key, value

    return "", None


def print_state_dict(name: str, state_dict: Mapping[Any, Any], max_items: int) -> None:
    print_section("State dict")
    print(f"Source: {name}")
    print(f"Entries: {len(state_dict)}")
    print(f"Showing first {min(max_items, len(state_dict))} entries")

    for index, (key, value) in enumerate(state_dict.items()):
        if index >= max_items:
            break
        tensor_like = is_tensor_like(value)
        print(
            f"- {key}: tensor_like={tensor_like}, "
            f"shape={safe_shape(value)}, dtype={safe_dtype(value)}"
        )


def print_metadata(obj: Any) -> None:
    if not isinstance(obj, Mapping):
        return

    present = [key for key in METADATA_KEYS if key in obj]
    if not present:
        return

    print_section("Common metadata")
    for key in present:
        print(f"{key}: {truncate_repr(obj[key])}")


def collect_mapping_entries(
    obj: Any, max_depth: int = 4, max_keys: int = 5000
) -> list["MappingEntry"]:
    entries: list[MappingEntry] = []
    seen: set[int] = set()

    def visit(value: Any, depth: int, prefix: str = "") -> None:
        if len(entries) >= max_keys or depth > max_depth:
            return
        if id(value) in seen:
            return
        seen.add(id(value))

        if isinstance(value, Mapping):
            for key, child in value.items():
                if len(entries) >= max_keys:
                    return
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                entries.append(MappingEntry(path=path, key=key_text, value=child))
                if isinstance(child, Mapping):
                    visit(child, depth + 1, path)
                elif isinstance(child, (list, tuple)) and depth + 1 <= max_depth:
                    for index, item in enumerate(child[:25]):
                        visit(item, depth + 1, f"{path}[{index}]")

    visit(obj, 0)
    return entries


class MappingEntry:
    def __init__(self, path: str, key: str, value: Any) -> None:
        self.path = path
        self.key = key
        self.value = value


def print_deployment_signals(
    checkpoint: Any, state_dict: Mapping[Any, Any] | None
) -> None:
    print_section("Notable deployment signals")

    entries = collect_mapping_entries(checkpoint)
    if state_dict is not None:
        entries.extend(
            MappingEntry(path=str(key), key=str(key), value=value)
            for key, value in state_dict.items()
        )

    memory_bank_entries = [
        item for item in entries if "memory_bank" in item.key.lower()
    ]
    threshold_entries = [
        item for item in entries if _matches_any(item.key.lower(), THRESHOLD_KEYWORDS)
    ]
    large_tensor_entries = []
    for item in entries:
        nbytes = safe_tensor_nbytes(item.value)
        if nbytes is not None and nbytes >= LARGE_TENSOR_BYTES:
            large_tensor_entries.append((item, nbytes))

    hyper_parameters = checkpoint.get("hyper_parameters") if isinstance(checkpoint, Mapping) else None
    backbone_hints = find_backbone_layer_hints(hyper_parameters)

    printed = False
    for item in memory_bank_entries:
        print(
            f"- memory_bank: {item.path}, shape={safe_shape(item.value)}, "
            f"dtype={safe_dtype(item.value)}"
        )
        printed = True

    for item, nbytes in sorted(large_tensor_entries, key=lambda pair: pair[1], reverse=True)[:10]:
        print(
            f"- large tensor: {item.path}, shape={safe_shape(item.value)}, "
            f"dtype={safe_dtype(item.value)}, estimated_size={format_file_size(nbytes)}"
        )
        printed = True

    if threshold_entries:
        examples = ", ".join(item.path for item in threshold_entries[:20])
        print(f"- threshold-related keys: {examples}")
        printed = True

    if backbone_hints:
        for key, value in backbone_hints[:12]:
            print(f"- backbone/layer hint: {key}={truncate_repr(value, max_length=160)}")
        printed = True

    if not printed:
        print("No memory banks, large tensor entries, thresholds, or backbone hints detected.")


def find_backbone_layer_hints(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    hints: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _matches_any(key_text.lower(), BACKBONE_LAYER_KEYWORDS):
                hints.append((path, child))
            if isinstance(child, Mapping):
                hints.extend(find_backbone_layer_hints(child, path))
    return hints


def print_task_hints(key_names: list[str]) -> None:
    print_section("Possible task hints")
    hints = detect_task_hints(key_names)
    if not hints:
        print("No task-specific key name hints detected.")
        return

    for task_name, confidence, matches in hints:
        examples = ", ".join(matches[:8])
        print(f"- {task_name}: {confidence} confidence; matched {examples}")


def detect_task_hints(key_names: list[str]) -> list[tuple[str, str, list[str]]]:
    lowered = [(key, key.lower()) for key in key_names]
    hints: list[tuple[str, str, list[str]]] = []

    for task_name, keywords in TASK_KEYWORDS.items():
        matches: list[str] = []
        matched_keywords: set[str] = set()
        for original, lower in lowered:
            for keyword in keywords:
                if keyword in lower:
                    matched_keywords.add(keyword)
                    if original not in matches:
                        matches.append(original)
                    break

        if not matches:
            continue
        if task_name == "anomaly_detection" and matched_keywords == {"feature_extractor"}:
            continue

        confidence = _confidence_level(task_name, matches, matched_keywords)
        hints.append((task_name, confidence, matches))

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    hints.sort(key=lambda item: (confidence_rank[item[1]], item[0]))
    return hints


def _confidence_level(
    task_name: str, matches: list[str], matched_keywords: set[str]
) -> str:
    match_count = len(matches)
    keyword_count = len(matched_keywords)

    if task_name == "anomaly_detection":
        matched_lower = {match.lower() for match in matches}
        has_memory_bank = any("memory_bank" in key for key in matched_lower)
        has_feature_extractor = any("feature_extractor" in key for key in matched_lower)
        if has_memory_bank and has_feature_extractor:
            return "high"
        if ANOMALY_STRONG_KEYS.intersection(matched_keywords):
            return "medium" if match_count < 5 and keyword_count < 3 else "high"

    if match_count >= 5 or keyword_count >= 3:
        return "high"
    if match_count >= 2 or keyword_count >= 2:
        return "medium"
    return "low"


def _matches_any(value: str, keywords: tuple[str, ...] | set[str]) -> bool:
    return any(keyword in value for keyword in keywords)


def _is_nn_module(obj: Any) -> bool:
    try:
        import torch.nn as nn
    except Exception:
        return False
    return isinstance(obj, nn.Module)


def _type_name(obj: Any) -> str:
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"
