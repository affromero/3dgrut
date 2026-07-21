"""Behavioral tests for the sealed common-eval signal contract."""

import hashlib
import json
import os
import sys
from types import SimpleNamespace

import render_common_eval
import pytest
import torch
from omegaconf import OmegaConf
from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.post_processing import LuminanceAffine
from threedgrut.post_processing.predictive_multiscale_ppisp import (
    view_context_inference_contract,
)
from threedgrut.render import (
    POST_PROCESSING_EVAL_MODE_INFERENCE,
    POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
    POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC,
    POST_PROCESSING_EVAL_MODE_RAW,
    POST_PROCESSING_SOURCE_CHECKPOINT,
    POST_PROCESSING_SOURCE_NONE,
    Renderer,
    _ppisp_controller_restore_manifest,
    load_checkpoint_post_processing,
)


class _RendererFactory:
    calls: list[tuple[object, dict[str, object]]] = []

    @staticmethod
    def from_preloaded_model(model: object, **kwargs: object) -> object:
        _RendererFactory.calls.append((model, kwargs))
        return object()


class _RecordingPostProcessing(torch.nn.Module):
    use_temporal_affine = True

    def __init__(self) -> None:
        super().__init__()
        self.frame_idx: int | None = None
        self.camera_idx: int | None = None
        self.sequence_idx: int | None = None
        self.exposure_prior: torch.Tensor | None = torch.ones(1)

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        *,
        resolution: tuple[int, int],
        camera_idx: int,
        frame_idx: int,
        sequence_idx: int,
        exposure_prior: torch.Tensor | None,
    ) -> torch.Tensor:
        assert pixel_coords.shape == (2, 2)
        assert resolution == (2, 1)
        self.frame_idx = frame_idx
        self.camera_idx = camera_idx
        self.sequence_idx = sequence_idx
        self.exposure_prior = exposure_prior
        return rgb


class _RecordingTemporalLuminanceAffine(LuminanceAffine):
    def __init__(self) -> None:
        super().__init__(
            num_cameras=1,
            num_frames=2,
            use_temporal_affine=True,
        )
        self.received_frame_idx: int | None = None
        self.received_camera_idx: int | None = None
        self.received_sequence_idx: int | None = None
        self.received_exposure_prior: torch.Tensor | None = torch.ones(1)

    def forward(
        self,
        pred_rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        *,
        resolution: tuple[int, int],
        camera_idx: int,
        frame_idx: int,
        sequence_idx: int = -1,
        exposure_prior: torch.Tensor | None = None,
        residual_grid_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert pixel_coords.shape == (2, 2)
        assert resolution == (2, 1)
        assert residual_grid_gate is None
        self.received_frame_idx = frame_idx
        self.received_camera_idx = camera_idx
        self.received_sequence_idx = sequence_idx
        self.received_exposure_prior = exposure_prior
        return pred_rgb


class _CameraModule(torch.nn.Module):
    def __init__(self, camera_count: int) -> None:
        super().__init__()
        self.camera_log_gain = torch.nn.Parameter(torch.zeros(camera_count, 3))


class _CameraDataset:
    def __init__(
        self,
        keys: list[str],
        counts: list[int],
        *,
        holdout_path: str | None = None,
    ) -> None:
        self.keys = keys
        self.counts = counts
        self.holdout_image_list_path = holdout_path

    def __len__(self) -> int:
        return sum(self.counts)

    def get_post_processing_camera_names(self) -> list[str]:
        return self.keys

    def get_post_processing_frames_per_camera(self) -> list[int]:
        return self.counts


def _camera_contract_renderer(
    checkpoint_keys: list[str] | None,
    eval_keys: list[str],
    *,
    strict: bool = True,
) -> Renderer:
    module = _CameraModule(len(checkpoint_keys) if checkpoint_keys is not None else len(eval_keys))
    setattr(module, "_hax_checkpoint_camera_keys", checkpoint_keys)
    setattr(
        module,
        "_hax_checkpoint_camera_frame_counts",
        [10] * len(checkpoint_keys) if checkpoint_keys is not None else None,
    )
    setattr(module, "_hax_checkpoint_camera_index_mode", "dataset")
    renderer = object.__new__(Renderer)
    renderer.post_processing = module
    renderer.dataset = _CameraDataset(eval_keys, [2] * len(eval_keys))
    renderer._post_processing_camera_index_mode = "dataset"
    renderer.strict_post_processing_contract = strict
    renderer._configure_post_processing_camera_contract()
    return renderer


def _luminance_checkpoint(
    state: dict[str, torch.Tensor],
) -> dict:
    return {
        "global_step": 100,
        "config": OmegaConf.create(
            {
                "n_iterations": 100,
                "post_processing": {
                    "method": "luminance_affine",
                    "use_frame_residual": False,
                    "use_color_matrix": False,
                    "use_radial_affine": False,
                    "use_residual_grid": False,
                    "use_temporal_affine": False,
                },
            }
        ),
        "post_processing": {
            "module": state,
            "camera_keys": ["front"],
            "camera_frame_counts": [2],
            "camera_index_mode": "dataset",
            "schedulers": [],
        },
    }


def _restored_luminance_module(
    *,
    use_temporal_affine: bool,
    durable_camera_mapping: bool = True,
) -> LuminanceAffine:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=2,
        use_temporal_affine=use_temporal_affine,
    )
    setattr(
        module,
        "_hax_restoration_manifest",
        {
            "method": "luminance_affine",
            "restore_policy": "strict_exact",
            "exact": True,
            "transformed_keys": [],
            "dropped_keys": [],
        },
    )
    setattr(
        module,
        "_hax_checkpoint_camera_keys",
        ["front"] if durable_camera_mapping else None,
    )
    setattr(
        module,
        "_hax_checkpoint_camera_frame_counts",
        [2] if durable_camera_mapping else None,
    )
    setattr(
        module,
        "_hax_checkpoint_camera_index_mode",
        "dataset" if durable_camera_mapping else None,
    )
    module.eval()
    return module


def test_common_eval_cli_defaults_to_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparable field-only signal remains the default."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_common_eval.py",
            "--checkpoint",
            "checkpoint.pt",
            "--eval-bundle",
            "eval_bundle",
            "--holdout-list",
            "holdout.txt",
            "--out-dir",
            "out",
        ],
    )

    args = render_common_eval._parse_args()

    assert args.post_processing_mode == POST_PROCESSING_EVAL_MODE_RAW
    assert args.split == "val"


def test_common_eval_cli_accepts_train_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train and validation use the same sealed common evaluator."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_common_eval.py",
            "--checkpoint",
            "checkpoint.pt",
            "--eval-bundle",
            "eval_bundle",
            "--holdout-list",
            "holdout.txt",
            "--out-dir",
            "out",
            "--split",
            "train",
            "--train-exclude-list",
            "exclude.txt",
        ],
    )

    args = render_common_eval._parse_args()

    assert args.split == "train"
    assert args.train_exclude_list == "exclude.txt"


def test_common_eval_cli_rejects_train_without_exclusion_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train metrics must never mean the unqualified holdout complement."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_common_eval.py",
            "--checkpoint",
            "checkpoint.pt",
            "--eval-bundle",
            "eval_bundle",
            "--holdout-list",
            "holdout.txt",
            "--out-dir",
            "out",
            "--split",
            "train",
        ],
    )

    with pytest.raises(SystemExit):
        render_common_eval._parse_args()


def test_common_eval_train_split_contract_rejects_unsafe_lists(
    tmp_path: object,
) -> None:
    """Only a disjoint, registered exclusion partition is accepted."""
    root = str(tmp_path)
    sparse_path = os.path.join(root, "eval", "sparse", "0")
    os.makedirs(sparse_path)
    with open(
        os.path.join(sparse_path, "images.txt"),
        "w",
        encoding="utf-8",
    ) as handle:
        for image_id, name in enumerate(
            ("train.png", "holdout.png", "exclude.png"),
            start=1,
        ):
            handle.write(
                f"{image_id} 1 0 0 0 0 0 0 1 {name}\n\n"
            )
    holdout_path = os.path.join(root, "holdout.txt")
    with open(holdout_path, "w", encoding="utf-8") as handle:
        handle.write("holdout.png\n")
    exclude_path = os.path.join(root, "exclude.txt")

    with open(exclude_path, "w", encoding="utf-8") as handle:
        handle.write("exclude.png\n")
    render_common_eval._validate_train_split_contract(
        eval_bundle=os.path.join(root, "eval"),
        holdout_list=holdout_path,
        train_exclude_list=exclude_path,
    )

    for invalid_names in (
        "",
        "exclude.png\nexclude.png\n",
        "missing.png\n",
        "holdout.png\n",
    ):
        with open(exclude_path, "w", encoding="utf-8") as handle:
            handle.write(invalid_names)
        with pytest.raises((ValueError, FileNotFoundError)):
            render_common_eval._validate_train_split_contract(
                eval_bundle=os.path.join(root, "eval"),
                holdout_list=holdout_path,
                train_exclude_list=exclude_path,
            )


def test_common_eval_cli_accepts_camera_only_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Camera-only PPISP is an explicit, visibly diagnostic mode."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_common_eval.py",
            "--checkpoint",
            "checkpoint.pt",
            "--eval-bundle",
            "eval_bundle",
            "--holdout-list",
            "holdout.txt",
            "--out-dir",
            "out",
            "--post-processing-mode",
            POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC,
        ],
    )

    args = render_common_eval._parse_args()

    assert args.post_processing_mode == (POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC)


def test_common_eval_cli_accepts_sequence_metadata_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temporal sequence conditioning has an unambiguous explicit mode."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_common_eval.py",
            "--checkpoint",
            "checkpoint.pt",
            "--eval-bundle",
            "eval_bundle",
            "--holdout-list",
            "holdout.txt",
            "--out-dir",
            "out",
            "--post-processing-mode",
            POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
        ],
    )

    args = render_common_eval._parse_args()

    assert args.post_processing_mode == (POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA)


def test_common_eval_raw_mode_passes_no_post_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw evaluation never restores appearance state."""

    def fail_if_loaded(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module | None:
        raise AssertionError("raw mode must not restore post-processing")

    _RendererFactory.calls = []
    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        fail_if_loaded,
    )
    monkeypatch.setattr(render_common_eval, "Renderer", _RendererFactory)

    render_common_eval._build_renderer(
        {"global_step": 42},
        object(),
        eval_bundle="eval_bundle",
        out_dir="out",
        post_processing_mode=POST_PROCESSING_EVAL_MODE_RAW,
    )

    _, kwargs = _RendererFactory.calls[0]
    assert kwargs["post_processing"] is None
    assert kwargs["post_processing_mode"] == POST_PROCESSING_EVAL_MODE_RAW
    assert kwargs["post_processing_source"] == POST_PROCESSING_SOURCE_NONE
    assert kwargs["strict_post_processing_contract"] is False


def test_common_eval_inference_requires_proven_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full PPISP inference asks the loader for readiness proof."""
    restored = torch.nn.Identity()
    readiness_requirements: list[bool] = []

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module:
        readiness_requirements.append(require_controller_ready)
        return restored

    _RendererFactory.calls = []
    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )
    monkeypatch.setattr(render_common_eval, "Renderer", _RendererFactory)

    render_common_eval._build_renderer(
        {"global_step": 42},
        object(),
        eval_bundle="eval_bundle",
        out_dir="out",
        post_processing_mode=POST_PROCESSING_EVAL_MODE_INFERENCE,
        split="train",
    )

    _, kwargs = _RendererFactory.calls[0]
    assert readiness_requirements == [True]
    assert kwargs["post_processing"] is restored
    assert kwargs["post_processing_source"] == (POST_PROCESSING_SOURCE_CHECKPOINT)
    assert kwargs["strict_post_processing_contract"] is True
    assert kwargs["split"] == "train"


def test_common_eval_camera_only_disables_controller_without_readiness_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostic mode keeps PPISP state but forces identity frame controls."""
    restored = torch.nn.Identity()
    restored.config = SimpleNamespace(use_controller=True)
    setattr(
        restored,
        "_hax_restoration_manifest",
        {"method": "ppisp", "exact": True},
    )
    readiness_requirements: list[bool] = []

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module:
        readiness_requirements.append(require_controller_ready)
        return restored

    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )

    result = render_common_eval._post_processing_for_mode(
        {"global_step": 42},
        POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC,
    )

    assert result is restored
    assert readiness_requirements == [False]
    assert restored.config.use_controller is False
    assert getattr(restored, "_hax_runtime_policy") == ("ppisp_camera_only_identity_exposure_color")


def test_common_eval_inference_rejects_missing_checkpoint_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested post-processing cannot silently become field-only."""

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module | None:
        return None

    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )

    with pytest.raises(ValueError, match="does not contain post-processing"):
        render_common_eval._build_renderer(
            {"global_step": 42},
            object(),
            eval_bundle="eval_bundle",
            out_dir="out",
            post_processing_mode=POST_PROCESSING_EVAL_MODE_INFERENCE,
        )


def test_sequence_metadata_inference_rejects_ppisp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPISP cannot masquerade as the temporal LuminanceAffine arm."""
    restored = torch.nn.Identity()

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module:
        return restored

    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )

    with pytest.raises(
        ValueError,
        match="requires a restored LuminanceAffine",
    ):
        render_common_eval._post_processing_for_mode(
            {"global_step": 42},
            POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
        )


def test_sequence_metadata_inference_accepts_exact_temporal_luminance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact temporal state enters only the explicit sequence mode."""
    restored = _restored_luminance_module(use_temporal_affine=True)
    readiness_requirements: list[bool] = []

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module:
        readiness_requirements.append(require_controller_ready)
        return restored

    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )

    result = render_common_eval._post_processing_for_mode(
        {"global_step": 42},
        POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
    )

    assert result is restored
    assert readiness_requirements == [False]
    assert restored._hax_runtime_policy == ("luminance_affine_non_rgb_sequence_metadata")


def test_sequence_metadata_inference_rejects_non_temporal_luminance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A camera-only affine checkpoint cannot enter the temporal arm."""
    restored = _restored_luminance_module(use_temporal_affine=False)

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module:
        return restored

    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )

    with pytest.raises(ValueError, match="use_temporal_affine=true"):
        render_common_eval._post_processing_for_mode(
            {"global_step": 42},
            POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
        )


def test_sequence_metadata_inference_rejects_legacy_camera_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even one-camera legacy state lacks durable mapping evidence."""
    restored = _restored_luminance_module(
        use_temporal_affine=True,
        durable_camera_mapping=False,
    )

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module:
        return restored

    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )

    with pytest.raises(ValueError, match="legacy numeric camera state"):
        render_common_eval._post_processing_for_mode(
            {"global_step": 42},
            POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
        )


def test_sequence_metadata_inference_rejects_transformed_restoration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transformed checkpoint tensors are not exact scientific state."""
    restored = _restored_luminance_module(use_temporal_affine=True)
    restored._hax_restoration_manifest["transformed_keys"] = ["temporal_log_gain_raw"]

    def restore(
        checkpoint: dict,
        *,
        require_controller_ready: bool,
    ) -> torch.nn.Module:
        return restored

    monkeypatch.setattr(
        render_common_eval,
        "load_checkpoint_post_processing",
        restore,
    )

    with pytest.raises(ValueError, match="exact LuminanceAffine"):
        render_common_eval._post_processing_for_mode(
            {"global_step": 42},
            POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
        )


def test_renderer_inference_seals_frame_sequence_and_exif() -> None:
    """Held-out frame identity and EXIF never reach canonical inference."""
    post_processing = _RecordingPostProcessing()
    renderer = object.__new__(Renderer)
    renderer.post_processing = post_processing
    renderer._post_processing_camera_index_mode = "dataset"
    renderer._post_processing_camera_mapping = [2, 0, 1]
    outputs = {"pred_rgb": torch.full((1, 1, 2, 3), 0.5)}
    exposure = torch.tensor([2.5])
    batch = SimpleNamespace(
        camera_idx=9,
        post_processing_camera_idx=0,
        frame_idx=37,
        sequence_idx=1234,
        pixel_coords=torch.zeros((1, 1, 2, 2)),
        exposure=exposure,
    )

    result = renderer._apply_inference_post_processing(outputs, batch)

    assert result is outputs
    assert post_processing.frame_idx == -1
    assert post_processing.sequence_idx == -1
    assert post_processing.exposure_prior is None
    assert post_processing.camera_idx == 2
    assert batch.sequence_idx == 1234
    assert batch.exposure is exposure


def test_renderer_sequence_metadata_passes_name_index_without_mutating_batch() -> None:
    """Only the parsed filename sequence reaches temporal inference."""
    post_processing = _RecordingTemporalLuminanceAffine()
    renderer = object.__new__(Renderer)
    renderer.post_processing = post_processing
    renderer.post_processing_mode = POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA
    renderer._post_processing_camera_index_mode = "dataset"
    renderer._post_processing_camera_mapping = [0]
    renderer.dataset = SimpleNamespace(
        _sequence_idx_from_path=ColmapDataset._sequence_idx_from_path,
    )
    outputs = {"pred_rgb": torch.full((1, 1, 2, 3), 0.5)}
    exposure = torch.tensor([2.5])
    batch = SimpleNamespace(
        camera_idx=9,
        post_processing_camera_idx=0,
        frame_idx=37,
        sequence_idx=64,
        image_path="front_0064.png",
        pixel_coords=torch.zeros((1, 1, 2, 2)),
        exposure=exposure,
    )

    result = renderer._apply_inference_post_processing(outputs, batch)

    assert result is outputs
    assert post_processing.received_frame_idx == -1
    assert post_processing.received_sequence_idx == 64
    assert post_processing.received_exposure_prior is None
    assert post_processing.received_camera_idx == 0
    assert batch.frame_idx == 37
    assert batch.sequence_idx == 64
    assert batch.exposure is exposure
    assert batch.image_path == "front_0064.png"


def test_renderer_sequence_metadata_rejects_unparseable_image_name() -> None:
    """An absent filename sequence cannot silently disable temporal terms."""
    renderer = object.__new__(Renderer)
    renderer.post_processing = _RecordingTemporalLuminanceAffine()
    renderer.post_processing_mode = POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA
    renderer._post_processing_camera_index_mode = "dataset"
    renderer._post_processing_camera_mapping = [0]
    renderer.dataset = SimpleNamespace(
        _sequence_idx_from_path=ColmapDataset._sequence_idx_from_path,
    )
    batch = SimpleNamespace(
        camera_idx=0,
        post_processing_camera_idx=0,
        frame_idx=0,
        sequence_idx=-1,
        image_path="front.png",
        pixel_coords=torch.zeros((1, 1, 2, 2)),
        exposure=None,
    )

    with pytest.raises(ValueError, match="could not parse"):
        renderer._apply_inference_post_processing(
            {"pred_rgb": torch.full((1, 1, 2, 3), 0.5)},
            batch,
        )


def test_camera_contract_maps_eval_order_to_checkpoint_order() -> None:
    """Physical names, not eval numeric order, select checkpoint slots."""
    renderer = _camera_contract_renderer(
        ["front", "left", "right"],
        ["right", "front", "left"],
    )

    assert renderer._post_processing_camera_mapping == [2, 0, 1]
    assert renderer._post_processing_camera_contract_status == ("validated_by_key")
    assert renderer._post_processing_camera_mapping_manifest()[0] == {
        "eval_index": 0,
        "eval_key": "right",
        "eval_frame_count": 2,
        "checkpoint_index": 2,
        "checkpoint_key": "right",
        "checkpoint_frame_count": 10,
    }


def test_camera_contract_rejects_multi_camera_key_mismatch() -> None:
    """Different physical rigs fail instead of using the wrong calibration."""
    with pytest.raises(ValueError, match="physical camera keys differ"):
        _camera_contract_renderer(
            ["front", "left", "right"],
            ["front", "left", "top"],
        )


def test_sequence_metadata_camera_contract_rejects_eval_key_mismatch() -> None:
    """Temporal metadata never crosses a differently named physical camera."""
    renderer = object.__new__(Renderer)
    renderer.post_processing = _restored_luminance_module(use_temporal_affine=True)
    renderer.post_processing_mode = POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA
    renderer.dataset = _CameraDataset(["left"], [2])
    renderer._post_processing_camera_index_mode = "dataset"
    renderer.strict_post_processing_contract = True

    with pytest.raises(ValueError, match="physical camera keys differ"):
        renderer._configure_post_processing_camera_contract()


def test_camera_contract_rejects_legacy_multi_camera_checkpoint() -> None:
    """Numeric-only legacy metadata is insufficient for a multi-camera rig."""
    with pytest.raises(ValueError, match="durable ordered camera keys"):
        _camera_contract_renderer(
            None,
            ["front", "left", "right"],
        )


def test_camera_contract_allows_proven_legacy_single_camera() -> None:
    """A one-slot module and one-camera eval are unambiguous."""
    renderer = _camera_contract_renderer(None, ["front"])

    assert renderer._post_processing_camera_mapping == [0]
    assert renderer._post_processing_camera_contract_status == ("legacy_single_camera_proven")


def test_ppisp_controller_readiness_requires_post_activation_scheduler_step() -> None:
    """Reaching, but not training past, activation is insufficient proof."""
    checkpoint = {
        "global_step": 100,
        "post_processing": {
            "schedulers": [{"last_epoch": 80}],
            "inference_contract": {
                "checkpoint_global_step": 100,
                "controller_activation_step": 80,
                "scheduler_last_epoch": 80,
                "controller_trained": True,
            },
        },
    }

    with pytest.raises(ValueError, match="requires proof"):
        _ppisp_controller_restore_manifest(
            checkpoint,
            use_controller=True,
            configured_activation_step=80,
            require_controller_ready=True,
        )


def test_ppisp_controller_readiness_records_durable_proof() -> None:
    """Matching activation and scheduler metadata proves controller use."""
    checkpoint = {
        "global_step": 100,
        "post_processing": {
            "schedulers": [{"last_epoch": 81}],
            "inference_contract": {
                "checkpoint_global_step": 100,
                "controller_activation_step": 80,
                "scheduler_last_epoch": 81,
                "controller_trained": True,
            },
        },
    }

    manifest = _ppisp_controller_restore_manifest(
        checkpoint,
        use_controller=True,
        configured_activation_step=80,
        require_controller_ready=True,
    )

    assert manifest["trained"] is True
    assert manifest["ready_for_controller_inference"] is True
    assert manifest["proof_source"] == "checkpoint_inference_contract"


def test_multiscale_readiness_requires_explicit_training_proof() -> None:
    """A local controller cannot inherit readiness from global PPISP alone."""
    checkpoint = {
        "global_step": 100,
        "post_processing": {
            "schedulers": [{"last_epoch": 81}],
            "inference_contract": {
                "checkpoint_global_step": 100,
                "controller_activation_step": 80,
                "scheduler_last_epoch": 81,
                "controller_trained": True,
                "multiscale_controller_trained": False,
            },
        },
    }

    with pytest.raises(ValueError, match="requires proof"):
        _ppisp_controller_restore_manifest(
            checkpoint,
            use_controller=True,
            configured_activation_step=80,
            require_controller_ready=True,
            use_multiscale_controller=True,
        )


def test_multiscale_readiness_accepts_matching_local_training_proof() -> None:
    """Global and local proof together authorize strict inference."""
    checkpoint = {
        "global_step": 100,
        "post_processing": {
            "schedulers": [{"last_epoch": 81}],
            "inference_contract": {
                "checkpoint_global_step": 100,
                "controller_activation_step": 80,
                "scheduler_last_epoch": 81,
                "controller_trained": True,
                "multiscale_controller_trained": True,
            },
        },
    }

    manifest = _ppisp_controller_restore_manifest(
        checkpoint,
        use_controller=True,
        configured_activation_step=80,
        require_controller_ready=True,
        use_multiscale_controller=True,
    )

    assert manifest["multiscale_trained"] is True
    assert manifest["multiscale_ready_for_inference"] is True


def test_view_context_readiness_requires_exact_versioned_contract() -> None:
    """Pose-conditioned inference must prove its exact feature semantics."""
    checkpoint = {
        "global_step": 100,
        "post_processing": {
            "schedulers": [{"last_epoch": 81}],
            "inference_contract": {
                "checkpoint_global_step": 100,
                "controller_activation_step": 80,
                "scheduler_last_epoch": 81,
                "controller_trained": True,
                "multiscale_controller_trained": True,
                "multiscale_view_context_enabled": True,
                "multiscale_view_context_trained": True,
                "multiscale_view_context_contract": (
                    view_context_inference_contract()
                ),
            },
        },
    }

    manifest = _ppisp_controller_restore_manifest(
        checkpoint,
        use_controller=True,
        configured_activation_step=80,
        require_controller_ready=True,
        use_multiscale_controller=True,
        use_view_context=True,
    )

    assert manifest["view_context_trained"] is True
    assert manifest["view_context_ready_for_inference"] is True
    assert manifest["view_context_contract"] == (
        view_context_inference_contract()
    )


def test_view_context_readiness_rejects_contract_drift() -> None:
    """Changed channel semantics cannot be accepted as exact inference."""
    checkpoint = {
        "global_step": 100,
        "post_processing": {
            "schedulers": [{"last_epoch": 81}],
            "inference_contract": {
                "checkpoint_global_step": 100,
                "controller_activation_step": 80,
                "scheduler_last_epoch": 81,
                "controller_trained": True,
                "multiscale_controller_trained": True,
                "multiscale_view_context_enabled": True,
                "multiscale_view_context_trained": True,
                "multiscale_view_context_contract": {
                    "schema_version": 999
                },
            },
        },
    }

    with pytest.raises(ValueError, match="exact versioned"):
        _ppisp_controller_restore_manifest(
            checkpoint,
            use_controller=True,
            configured_activation_step=80,
            require_controller_ready=True,
            use_multiscale_controller=True,
            use_view_context=True,
        )


def test_luminance_restore_is_exact_and_records_manifest() -> None:
    """An exact LuminanceAffine state round-trip records no transformations."""
    source = LuminanceAffine(num_cameras=1, num_frames=2)
    checkpoint = _luminance_checkpoint(source.state_dict())

    restored = load_checkpoint_post_processing(checkpoint, device="cpu")

    assert restored is not None
    manifest = getattr(restored, "_hax_restoration_manifest")
    assert manifest["restore_policy"] == "strict_exact"
    assert manifest["exact"] is True
    assert manifest["transformed_keys"] == []
    assert manifest["dropped_keys"] == []
    for key, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_luminance_restore_rejects_shape_mismatch() -> None:
    """Scientific evaluation never interpolates or drops learned state."""
    source = LuminanceAffine(num_cameras=1, num_frames=2)
    state = dict(source.state_dict())
    state["camera_bias"] = torch.zeros(2, 3)

    with pytest.raises(ValueError, match="Exact LuminanceAffine"):
        load_checkpoint_post_processing(
            _luminance_checkpoint(state),
            device="cpu",
        )


def test_restore_rejects_incomplete_durable_camera_metadata() -> None:
    """Camera keys cannot be accepted without counts and index semantics."""
    source = LuminanceAffine(num_cameras=1, num_frames=2)
    checkpoint = _luminance_checkpoint(source.state_dict())
    checkpoint["post_processing"].pop("camera_frame_counts")

    with pytest.raises(ValueError, match="both be present or both be absent"):
        load_checkpoint_post_processing(checkpoint, device="cpu")


def test_common_eval_main_disables_exif_and_preserves_training_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval config is sealed while original training provenance survives."""
    conf = OmegaConf.create(
        {
            "path": "training_bundle",
            "dataset": {
                "holdout_image_list_path": "old.txt",
                "shutter_type": "ROLLING",
                "load_exif": True,
                "sky_mask_folder": "sky_masks",
            },
            "loss": {"use_sky_opacity": True},
        }
    )
    checkpoint = {"config": conf, "global_step": 42}
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, model_conf: object) -> None:
            captured["model_conf"] = model_conf

        def init_from_checkpoint(
            self,
            loaded: dict,
            *,
            setup_optimizer: bool,
        ) -> None:
            captured["loaded"] = loaded

        def build_acc(self) -> None:
            return None

    class FakeRenderer:
        def render_all(self) -> None:
            captured["rendered"] = True

    def build_renderer(*args: object, **kwargs: object) -> FakeRenderer:
        captured["renderer_kwargs"] = kwargs
        return FakeRenderer()

    monkeypatch.setattr(
        render_common_eval,
        "_parse_args",
        lambda: SimpleNamespace(
            checkpoint="checkpoint.pt",
            eval_bundle="eval_bundle",
            holdout_list="holdout.txt",
            train_exclude_list=None,
            out_dir="out",
            split="val",
            post_processing_mode=POST_PROCESSING_EVAL_MODE_RAW,
        ),
    )
    monkeypatch.setattr(render_common_eval.torch, "load", lambda *a, **k: checkpoint)
    monkeypatch.setattr(
        render_common_eval,
        "create_gaussian_model",
        lambda *args, **kwargs: FakeModel(conf),
    )
    monkeypatch.setattr(render_common_eval, "_build_renderer", build_renderer)

    render_common_eval.main()

    assert conf.dataset.load_exif is False
    assert conf.path == "eval_bundle"
    assert conf.dataset.holdout_image_list_path == "holdout.txt"
    assert conf.dataset.train_exclude_image_list_path is None
    renderer_kwargs = captured["renderer_kwargs"]
    assert isinstance(renderer_kwargs, dict)
    assert renderer_kwargs["original_training_bundle"] == "training_bundle"
    assert renderer_kwargs["checkpoint_path"] == os.path.abspath("checkpoint.pt")


def test_provenance_hashes_checkpoint_sparse_model_and_camera_mapping(
    tmp_path: object,
) -> None:
    """The sidecar uniquely identifies checkpoint, eval cameras, and frames."""
    root = str(tmp_path)
    checkpoint_path = os.path.join(root, "checkpoint.pt")
    sparse_path = os.path.join(root, "eval", "sparse", "0")
    os.makedirs(sparse_path)
    with open(checkpoint_path, "wb") as handle:
        handle.write(b"checkpoint-bytes")
    with open(os.path.join(sparse_path, "cameras.txt"), "wb") as handle:
        handle.write(b"camera-model")
    holdout_path = os.path.join(root, "holdout.txt")
    with open(holdout_path, "w", encoding="utf-8") as handle:
        handle.write("front/0007.png\nleft/0007.png\n")

    renderer = _camera_contract_renderer(
        ["front", "left"],
        ["front", "left"],
    )
    renderer.conf = OmegaConf.create(
        {
            "path": os.path.join(root, "eval"),
            "dataset": {"load_exif": False},
            "post_processing": {"method": "ppisp"},
        }
    )
    renderer.dataset.holdout_image_list_path = holdout_path
    renderer.checkpoint_path = checkpoint_path
    renderer.original_training_bundle = "training_bundle"
    renderer.post_processing_mode = POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC
    renderer.post_processing_source = POST_PROCESSING_SOURCE_CHECKPOINT
    setattr(
        renderer.post_processing,
        "_hax_restoration_manifest",
        {
            "method": "ppisp",
            "restore_policy": "strict_exact",
            "exact": True,
        },
    )
    setattr(
        renderer.post_processing,
        "_hax_runtime_policy",
        "ppisp_camera_only_identity_exposure_color",
    )

    provenance = renderer._common_eval_contract_provenance(
        [
            {"image_relative_name": "front/0007.png"},
            {"image_relative_name": "left/0007.png"},
        ]
    )

    assert provenance["checkpoint_sha256"] == hashlib.sha256(b"checkpoint-bytes").hexdigest()
    assert provenance["eval_sparse_sha256"] is not None
    assert provenance["original_training_bundle"] == "training_bundle"
    assert provenance["evaluated_frame_names"] == [
        "front/0007.png",
        "left/0007.png",
    ]
    assert provenance["post_processing_camera_mapping"][1]["checkpoint_key"] == "left"
    assert provenance["post_processing_uses_sequence_idx"] is False
    assert provenance["post_processing_uses_exif_exposure"] is False
    assert provenance["post_processing_is_noncanonical_diagnostic"] is True


def test_sequence_metadata_provenance_labels_non_rgb_conditioning(
    tmp_path: object,
) -> None:
    """The sidecar pins the exact non-RGB sequence conditioning inputs."""
    root = str(tmp_path)
    first_path = os.path.join(root, "front_0064.png")
    second_path = os.path.join(root, "left_0064.png")

    renderer = _camera_contract_renderer(
        ["front", "left"],
        ["front", "left"],
    )
    renderer.conf = OmegaConf.create(
        {
            "path": root,
            "dataset": {"load_exif": False},
            "post_processing": {"method": "luminance_affine"},
        }
    )
    renderer.dataset.holdout_image_list_path = None
    renderer.checkpoint_path = None
    renderer.original_training_bundle = "training_bundle"
    renderer.post_processing_mode = POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA
    renderer.post_processing_source = POST_PROCESSING_SOURCE_CHECKPOINT
    setattr(
        renderer.post_processing,
        "_hax_restoration_manifest",
        {
            "method": "luminance_affine",
            "restore_policy": "strict_exact",
            "exact": True,
        },
    )
    setattr(
        renderer.post_processing,
        "_hax_runtime_policy",
        "luminance_affine_non_rgb_sequence_metadata",
    )
    evaluated_names = ["front_0064.png", "left_0064.png"]

    provenance = renderer._common_eval_contract_provenance(
        [
            {
                "image_relative_name": evaluated_names[0],
                "image_path": first_path,
                "sequence_idx": 64,
            },
            {
                "image_relative_name": evaluated_names[1],
                "image_path": second_path,
                "sequence_idx": 64,
            },
        ]
    )

    names_payload = json.dumps(
        evaluated_names,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert provenance["post_processing_conditioning_source"] == ("non_rgb_sequence_metadata")
    assert provenance["post_processing_uses_sequence_idx"] is True
    assert provenance["post_processing_uses_per_frame_latent"] is False
    assert provenance["post_processing_uses_exif_exposure"] is False
    assert provenance["post_processing_inference_contract"] == (
        "frame_idx_minus_one_sequence_idx_from_dataset_image_name_" "exposure_prior_none"
    )
    assert provenance["evaluated_frame_names"] == evaluated_names
    assert provenance["evaluated_frame_names_sha256"] == hashlib.sha256(names_payload).hexdigest()
    assert provenance["post_processing_sequence_metadata"] == [
        {"image_relative_name": "front_0064.png", "sequence_idx": 64},
        {"image_relative_name": "left_0064.png", "sequence_idx": 64},
    ]
