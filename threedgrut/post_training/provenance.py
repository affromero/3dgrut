"""Complete input and runtime provenance for resumable post-training studies."""

# ruff: noqa: EM101, PLR0913

import hashlib
import importlib.util
import platform
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import torch
from doctor_splat.post_training.ghost_repair import GhostRepairPlan

from threedgrut.ghost_repair_study import file_sha256


def scene_fingerprint(
    *,
    plan: GhostRepairPlan,
    scene: object,
    plan_path: Path,
    paths: dict[str, Path],
    doctor_root: Path,
    bundle: Path,
    image_names: list[str],
) -> dict[str, object]:
    """Bind every resumable development artifact to inputs and code."""
    source_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "study_id": plan.study_id,
        "scene": scene.scene,
        "plan_sha256": file_sha256(plan_path),
        "registered_inputs": {
            key: file_sha256(path) for key, path in sorted(paths.items())
        },
        "source_code": {
            name: file_sha256(source_root / name)
            for name in (
                "run_ghost_repair.py",
                "threedgrut/ghost_repair.py",
                "threedgrut/ghost_repair_study.py",
                "threedgrut/post_training/provenance.py",
            )
        },
        "repositories": {
            "3dgrut": repository_state(source_root),
            "doctor_splat": repository_state(doctor_root),
        },
        "scene_data": scene_data_manifest(bundle, image_names),
        "external_metric_weights": metric_weight_manifest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "active_gpu": active_gpu_manifest(source_root),
        },
        "renderer_settings": {
            "conditional_distance": True,
            "require_all_sigma_points_valid": True,
        },
        "ghost_constants": {
            "margin": plan.ghost_margin,
            "neighbors": plan.ghost_neighbors,
            "samples_per_view": plan.ghost_samples_per_view,
            "seed": plan.ghost_seed,
        },
        "metric_protocols": {
            metric.value: protocol
            for metric, protocol in plan.secondary_metric_protocols.items()
        },
    }


def scene_data_manifest(
    bundle: Path, image_names: list[str]
) -> dict[str, object]:
    """Hash the photographs and COLMAP model consumed by all metrics."""
    files = sorted((bundle / "sparse" / "0").rglob("*"))
    if not files:
        raise FileNotFoundError(bundle / "sparse" / "0")
    for directory_name in ("images", "images_2"):
        directory = bundle / directory_name
        if not directory.is_dir():
            continue
        for name in sorted(set(image_names)):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(path)
    return {
        str(path.relative_to(bundle)): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(files))
        if path.is_file()
    }


@cache
def metric_weight_manifest() -> dict[str, dict[str, int | str]]:
    """Hash the frozen SONATA and VGG-LPIPS weight files."""
    torchmetrics_spec = importlib.util.find_spec("torchmetrics")
    if torchmetrics_spec is None or torchmetrics_spec.origin is None:
        raise ModuleNotFoundError("torchmetrics")
    torchmetrics_root = Path(torchmetrics_spec.origin).parent
    paths = {
        "sonata": Path.home() / ".cache/sonata/ckpt/sonata.pth",
        "lpips_calibration": (
            torchmetrics_root / "functional/image/lpips_models/vgg.pth"
        ),
        "vgg16": (
            Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
        ),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }


def active_gpu_manifest(source_root: Path) -> dict[str, object]:
    """Fingerprint the active device, driver, and compiled extensions."""
    device = torch.cuda.current_device()
    driver = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=name,uuid,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    extensions = {
        str(path.relative_to(source_root)): file_sha256(path)
        for path in sorted((source_root / ".torchext").rglob("*.so"))
    }
    return {
        "device_index": device,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "nvidia_smi": driver,
        "extensions": extensions,
    }


def repository_state(root: Path) -> dict[str, object]:
    """Fingerprint the commit, tracked diff, and untracked source files."""

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, stderr=subprocess.STDOUT
        )

    head = git("rev-parse", "HEAD").decode().strip()
    diff = git("diff", "HEAD", "--binary")
    untracked = [
        line
        for line in git("ls-files", "--others", "--exclude-standard")
        .decode()
        .splitlines()
        if line
    ]
    return {
        "root": str(root.resolve()),
        "head": head,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": {
            relative: file_sha256(root / relative)
            for relative in sorted(untracked)
        },
    }


def artifact_fingerprint(
    base: dict[str, object], role: str, **details: object
) -> dict[str, object]:
    """Name one artifact role under a scene study fingerprint."""
    return {"scene_study": base, "role": role, **details}
