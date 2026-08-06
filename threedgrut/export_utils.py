"""Portable path and deterministic view-selection helpers for exports."""

import os


def scene_relative_path(path: str, scene_root: str) -> str:
    """Return a path inside the declared scene root."""
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(scene_root)
    if os.path.commonpath((resolved_path, resolved_root)) != resolved_root:
        message = f"Path escapes --scene-root: {path}"
        raise ValueError(message)
    return os.path.relpath(resolved_path, resolved_root)


def sample_indices(count: int, maximum: int) -> set[int]:
    """Select all or an evenly spaced deterministic subset."""
    if maximum <= 0 or maximum >= count:
        return set(range(count))
    if maximum == 1:
        return {count // 2}
    return {
        round(index * (count - 1) / (maximum - 1)) for index in range(maximum)
    }
