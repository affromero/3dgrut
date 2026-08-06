"""Materialize a train-only COLMAP initialization in fixed public poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import NoReturn

import numpy as np
import pycolmap


class TrainOnlyColmapError(ValueError):
    """The requested fold cannot produce an authenticated COLMAP source."""


class TrainOnlyOutputExistsError(FileExistsError):
    """The immutable output path is already occupied."""


def _fail(message: str) -> NoReturn:
    raise TrainOnlyColmapError(message)


def _occupied(message: str) -> NoReturn:
    raise TrainOnlyOutputExistsError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_manifest(path: Path) -> dict[str, object]:
    entries = []
    total_size = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        size = child.stat().st_size
        total_size += size
        entries.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": _sha256(child),
                "size_bytes": size,
            }
        )
    if not entries:
        _fail(f"artifact tree is empty: {path}")
    return {
        "sha256": _sha256_json(entries),
        "size_bytes": total_size,
        "file_count": len(entries),
    }


def _read_names(path: Path) -> list[str]:
    names = path.read_text(encoding="utf-8").splitlines()
    if not names or len(names) != len(set(names)):
        _fail(f"view list must contain unique nonempty names: {path}")
    if any(not name or name != name.strip() for name in names):
        _fail(f"view list contains a blank or padded name: {path}")
    return names


def validate_registration(*, source_names: list[str], all_names: list[str], training_names: list[str]) -> None:
    """Reject missing, reordered, or out-of-fold camera registrations."""
    if source_names != sorted(source_names):
        _fail("source COLMAP names must be lexicographically sorted")
    if all_names != source_names:
        _fail("all-view list does not exactly match source COLMAP")
    if training_names != sorted(training_names):
        _fail("training-view list must be lexicographically sorted")
    if not set(training_names) < set(source_names):
        _fail("training views must be a strict subset of all views")


def _camera_pose_records(reconstruction: pycolmap.Reconstruction, names: list[str]) -> list[dict[str, object]]:
    """Serialize the public calibration and registered pose for each image."""
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    records = []
    for name in names:
        image = images_by_name.get(name)
        if image is None:
            _fail(f"camera metadata is missing registered image: {name}")
        camera = reconstruction.cameras[image.camera_id]
        transform = image.cam_from_world()
        records.append(
            {
                "image_id": image.image_id,
                "name": name,
                "camera_id": image.camera_id,
                "camera_model": camera.model_name,
                "width": camera.width,
                "height": camera.height,
                "camera_params": camera.params.tolist(),
                "qvec_xyzw": transform.rotation.quat.tolist(),
                "tvec": transform.translation.tolist(),
            }
        )
    return records


def validate_fixed_camera_metadata(
    *,
    source: pycolmap.Reconstruction,
    output: pycolmap.Reconstruction,
    training_names: list[str],
    atol: float = 1e-12,
) -> tuple[str, str]:
    """Require the output to retain each public ID, intrinsic, and pose."""
    source_records = _camera_pose_records(source, training_names)
    output_records = _camera_pose_records(output, training_names)
    for source_record, output_record in zip(source_records, output_records, strict=True):
        for key in (
            "image_id",
            "name",
            "camera_id",
            "camera_model",
            "width",
            "height",
        ):
            if source_record[key] != output_record[key]:
                _fail(f"fixed camera metadata changed field {key}")
        for key in ("camera_params", "qvec_xyzw", "tvec"):
            if not np.allclose(
                source_record[key],
                output_record[key],
                rtol=0.0,
                atol=atol,
            ):
                _fail(f"fixed camera metadata changed field {key}")
    return _sha256_json(source_records), _sha256_json(output_records)


def _write_input_model(*, source_sparse: Path, training_names: list[str], output_path: Path) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction(str(source_sparse))
    source_names = sorted(image.name for image in reconstruction.images.values())
    if not set(training_names) <= set(source_names):
        _fail("training view is missing from the source reconstruction")
    for image in list(reconstruction.images.values()):
        if image.name not in set(training_names):
            reconstruction.deregister_frame(image.frame_id)
    reconstruction.delete_all_points2D_and_points3D()
    reconstruction.write(str(output_path))
    filtered = pycolmap.Reconstruction(str(output_path))
    filtered_names = sorted(image.name for image in filtered.images.values())
    if filtered_names != training_names:
        _fail("filtered pose model does not exactly match training views")
    return filtered


def _database_names(path: Path) -> list[str]:
    database = pycolmap.Database.open(str(path))
    try:
        return sorted(image.name for image in database.read_all_images())
    finally:
        database.close()


def _seed_database(path: Path, reconstruction: pycolmap.Reconstruction) -> None:
    database = pycolmap.Database.open(str(path))
    try:
        for camera in reconstruction.cameras.values():
            database.write_camera(camera, use_camera_id=True)
        for rig in reconstruction.rigs.values():
            database.write_rig(rig, use_rig_id=True)
        for frame_id in reconstruction.reg_frame_ids():
            database.write_frame(reconstruction.frames[frame_id], use_frame_id=True)
        for image_id in reconstruction.reg_image_ids():
            database.write_image(reconstruction.images[image_id], use_image_id=True)
    finally:
        database.close()


def materialize_train_only_bundle(
    *,
    source_scene: Path,
    all_views_path: Path,
    training_views_path: Path,
    output_root: Path,
    fold_contract_sha256: str,
) -> Path:
    """Re-match and re-triangulate using training RGB and fixed public poses.

    The public all-view model supplies camera calibration and poses only. SIFT
    extraction, matching, triangulation, and color extraction receive the
    registered reconstruction image names exclusively.
    """
    if output_root.exists():
        _occupied(f"output already exists: {output_root}")
    if len(fold_contract_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in fold_contract_sha256
    ):
        _fail("fold contract must be a lowercase SHA-256")
    source_sparse = source_scene / "sparse" / "0"
    source_images = source_scene / "images"
    source_downsampled = source_scene / "images_8"
    source = pycolmap.Reconstruction(str(source_sparse))
    source_names = sorted(image.name for image in source.images.values())
    all_names = _read_names(all_views_path)
    training_names = _read_names(training_views_path)
    validate_registration(
        source_names=source_names,
        all_names=all_names,
        training_names=training_names,
    )
    source_file_names = sorted(path.name for path in source_images.iterdir() if path.is_file())
    if source_file_names != source_names:
        _fail("source image files do not exactly match COLMAP names")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.", dir=output_root.parent) as temporary_directory:
        temporary = Path(temporary_directory)
        input_model = temporary / "input_model"
        input_model.mkdir()
        filtered = _write_input_model(
            source_sparse=source_sparse,
            training_names=training_names,
            output_path=input_model,
        )
        database_path = temporary / "database.db"
        _seed_database(database_path, filtered)
        pycolmap.extract_features(
            database_path=database_path,
            image_path=source_images,
            image_names=training_names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            device=pycolmap.Device.cuda,
        )
        if _database_names(database_path) != training_names:
            _fail("feature database contains non-training images")
        pycolmap.match_exhaustive(database_path=database_path, device=pycolmap.Device.cuda)
        sparse_output = temporary / "sparse" / "0"
        sparse_output.mkdir(parents=True)
        triangulated = pycolmap.triangulate_points(
            reconstruction=filtered,
            database_path=database_path,
            image_path=source_images,
            output_path=sparse_output,
            clear_points=True,
            refine_intrinsics=False,
        )
        triangulated.extract_colors_for_all_images(str(source_images))
        triangulated.write(str(sparse_output))
        output_names = sorted(image.name for image in triangulated.images.values())
        if output_names != training_names:
            _fail("triangulated model contains non-training images")
        source_metadata_sha256, output_metadata_sha256 = validate_fixed_camera_metadata(
            source=source,
            output=triangulated,
            training_names=training_names,
        )
        if triangulated.num_points3D() <= 0:
            _fail("train-only triangulation produced no points")
        Path(temporary / "images").symlink_to(source_images, target_is_directory=True)
        Path(temporary / "images_8").symlink_to(source_downsampled, target_is_directory=True)
        source_sparse_artifact = _tree_manifest(source_sparse)
        output_sparse_artifact = _tree_manifest(sparse_output)
        manifest = {
            "schema_version": 1,
            "kind": "train_only_colmap_fixed_public_poses",
            "fold_contract_sha256": fold_contract_sha256,
            "source_sparse": {
                "path": str(source_sparse),
                **source_sparse_artifact,
            },
            "source_images": {
                "path": str(source_images),
                **_tree_manifest(source_images),
            },
            "source_downsampled_images": {
                "path": str(source_downsampled),
                **_tree_manifest(source_downsampled),
            },
            "all_views": {
                "path": str(all_views_path),
                "sha256": _sha256(all_views_path),
                "image_count": len(all_names),
            },
            "training_views": {
                "path": str(training_views_path),
                "sha256": _sha256(training_views_path),
                "names_sha256": _sha256_json(training_names),
                "image_count": len(training_names),
            },
            "pose_protocol": ("public_all_view_camera_calibration_and_poses_only"),
            "transductive_pose_metadata_used": True,
            "heldout_rgb_used": False,
            "feature_extraction_views": "training_views_only",
            "feature_matching_views": "training_views_only",
            "triangulation_views": "training_views_only",
            "color_extraction_views": "training_views_only",
            "fixed_camera_metadata": {
                "fields": [
                    "image_id",
                    "name",
                    "camera_id",
                    "camera_model",
                    "width",
                    "height",
                    "camera_params",
                    "qvec_xyzw",
                    "tvec",
                ],
                "absolute_tolerance": 1e-12,
                "record_count": len(training_names),
                "source_sha256": source_metadata_sha256,
                "output_sha256": output_metadata_sha256,
                "preservation_verified": True,
            },
            "pycolmap_version": pycolmap.__version__,
            "point_count": triangulated.num_points3D(),
            "output_sparse": {
                "path": "sparse/0",
                **output_sparse_artifact,
            },
            "database": {
                "path": "database.db",
                "sha256": _sha256(database_path),
                "size_bytes": database_path.stat().st_size,
                "image_names_sha256": _sha256_json(_database_names(database_path)),
            },
            "materializer_sha256": _sha256(Path(__file__)),
        }
        manifest_path = temporary / "train_only_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(output_root)
    return output_root / "train_only_manifest.json"


def main(argv: list[str] | None = None) -> None:
    """CLI for the authenticated M3 base-training bundles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene", type=Path, required=True)
    parser.add_argument("--all-views", type=Path, required=True)
    parser.add_argument("--training-views", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold-contract-sha256", required=True)
    args = parser.parse_args(argv)
    manifest = materialize_train_only_bundle(
        source_scene=args.source_scene,
        all_views_path=args.all_views,
        training_views_path=args.training_views,
        output_root=args.output_root,
        fold_contract_sha256=args.fold_contract_sha256,
    )
    print(manifest)


if __name__ == "__main__":
    main()
