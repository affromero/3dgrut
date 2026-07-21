"""Resolve B2G combined-frame provenance for per-source diagnostics."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any


def _candidate_provenance_paths(image_path: str) -> list[str]:
    """Return likely frame-provenance sidecars for a COLMAP image path."""
    candidates: list[str] = []
    current = os.path.abspath(os.path.dirname(image_path))
    while True:
        candidates.append(os.path.join(current, "frame_provenance.json"))
        candidates.append(
            os.path.join(current, "generated_files", "frame_provenance.json")
        )
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return candidates


def find_frame_provenance_path(image_path: str) -> str | None:
    """Find the nearest B2G combined `frame_provenance.json` sidecar."""
    for candidate in _candidate_provenance_paths(image_path):
        if os.path.exists(candidate):
            return candidate
    return None


def _frame_index_from_image_name(image_path: str) -> int | None:
    """Parse `front_0123.png` / `left_0123.png` style image names."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return int(parts[1])


@lru_cache(maxsize=16)
def _load_scan_ids_by_camera(
    provenance_path: str,
) -> dict[int, tuple[str, ...]]:
    """Load scan IDs indexed by one-based provenance camera ID."""
    with open(provenance_path, "r", encoding="utf-8") as file:
        payload: Any = json.load(file)
    cameras = payload.get("cameras", []) if isinstance(payload, dict) else []
    scan_ids_by_camera: dict[int, tuple[str, ...]] = {}
    for camera in cameras:
        if not isinstance(camera, dict):
            continue
        camera_id = camera.get("camera")
        entries = camera.get("entries", [])
        if not isinstance(camera_id, int) or not isinstance(entries, list):
            continue
        scan_ids: list[str] = []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(
                entry.get("scan_id"), str
            ):
                scan_ids.append(entry["scan_id"])
            else:
                scan_ids.append("")
        scan_ids_by_camera[camera_id] = tuple(scan_ids)
    return scan_ids_by_camera


def source_scan_id_from_image_path(
    *,
    image_path: str,
    camera_idx: int | None,
) -> str | None:
    """Return the source scan ID for a combined B2G COLMAP image."""
    frame_index = _frame_index_from_image_name(image_path)
    if frame_index is None:
        return None
    provenance_path = find_frame_provenance_path(image_path)
    if provenance_path is None:
        return None
    scan_ids_by_camera = _load_scan_ids_by_camera(provenance_path)
    preferred_camera_id = camera_idx + 1 if camera_idx is not None else None
    if preferred_camera_id in scan_ids_by_camera:
        scan_ids = scan_ids_by_camera[preferred_camera_id]
    elif scan_ids_by_camera:
        scan_ids = next(iter(scan_ids_by_camera.values()))
    else:
        return None
    if frame_index < 0 or frame_index >= len(scan_ids):
        return None
    return scan_ids[frame_index] or None
