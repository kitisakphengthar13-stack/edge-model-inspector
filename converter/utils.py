from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def is_tensor_like(obj: Any) -> bool:
    """Return True for torch tensors and tensor-like checkpoint values."""
    return all(hasattr(obj, attr) for attr in ("shape", "dtype"))


def safe_shape(obj: Any) -> str:
    if not hasattr(obj, "shape"):
        return "-"

    try:
        shape = getattr(obj, "shape")
        if isinstance(shape, Sequence):
            return "(" + ", ".join(str(dim) for dim in shape) + ")"
        return str(shape)
    except Exception:
        return "<unavailable>"


def safe_dtype(obj: Any) -> str:
    if not hasattr(obj, "dtype"):
        return "-"

    try:
        return str(getattr(obj, "dtype"))
    except Exception:
        return "<unavailable>"


def safe_tensor_nbytes(obj: Any) -> int | None:
    if not is_tensor_like(obj):
        return None

    try:
        nbytes = getattr(obj, "nbytes")
        if isinstance(nbytes, int):
            return nbytes
    except Exception:
        pass

    try:
        numel = obj.numel() if callable(getattr(obj, "numel", None)) else None
        element_size = (
            obj.element_size() if callable(getattr(obj, "element_size", None)) else None
        )
        if isinstance(numel, int) and isinstance(element_size, int):
            return numel * element_size
    except Exception:
        return None

    return None


def format_file_size(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def truncate_repr(value: Any, max_length: int = 240) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"

    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
