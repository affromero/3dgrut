"""Behavioral coverage for per-source-frame local projection fields."""

import copy

import torch
import pytest
from omegaconf import DictConfig, OmegaConf

from threedgrut.model.local_projection_field import LocalProjectionField
from threedgrut.native_ewa_renderer import Tracer
from threedgrut.optimizers.local_projection_field_adam import (
    LocalProjectionFieldAdam,
)
from threedgrut.render import (
    _restore_local_projection_field,
    _validate_local_projection_field_dataset,
)
from threedgrut.trainer import Trainer3DGRUT


def test_zero_local_projection_field_is_exact_identity() -> None:
    """Zero-initialized fields leave projected centers unchanged."""
    field = LocalProjectionField(num_source_frames=2)
    means2d = torch.tensor(((0.0, 0.0), (27.5, 13.0), (63.0, 31.0)))

    sampled = field.sample(
        source_frame_idx=0,
        means2d=means2d,
        resolution=(64, 32),
    )

    assert torch.equal(sampled, torch.zeros_like(means2d))


def test_local_projection_field_uses_the_selected_source_frame() -> None:
    """A sentinel field changes only the matching stable source frame."""
    field = LocalProjectionField(num_source_frames=2)
    with torch.no_grad():
        field.values[1].fill_(0.125)
    means2d = torch.tensor(((7.0, 9.0), (17.0, 11.0)))

    first = field.sample(
        source_frame_idx=0,
        means2d=means2d,
        resolution=(32, 32),
    )
    second = field.sample(
        source_frame_idx=1,
        means2d=means2d,
        resolution=(32, 32),
    )

    assert torch.equal(first, torch.zeros_like(means2d))
    assert torch.equal(second, torch.full_like(means2d, 0.125))


def test_local_projection_field_clamps_border_samples() -> None:
    """Out-of-range projected centers use the nearest field border."""
    field = LocalProjectionField(num_source_frames=1)
    with torch.no_grad():
        field.values[0, 0, 0] = torch.tensor((0.1, -0.2))
        field.values[0, -1, -1] = torch.tensor((-0.3, 0.4))
    means2d = torch.tensor(((-100.0, -100.0), (10_000.0, 10_000.0)))

    sampled = field.sample(
        source_frame_idx=0,
        means2d=means2d,
        resolution=(32, 32),
    )

    torch.testing.assert_close(
        sampled,
        torch.tensor(((0.1, -0.2), (-0.3, 0.4))),
    )


def test_local_projection_field_adam_updates_only_the_selected_row() -> None:
    """One field update preserves untouched source-frame state."""
    field = LocalProjectionField(num_source_frames=2)
    optimizer = LocalProjectionFieldAdam(field.values)
    gradient = torch.zeros_like(field.values)
    gradient[1].fill_(0.5)
    field.values.grad = gradient

    optimizer.step(1)

    expected = -0.001 * 0.25 * 0.5 / (0.5 + 2.0e-5)
    torch.testing.assert_close(
        field.values[0],
        torch.zeros_like(field.values[0]),
    )
    torch.testing.assert_close(
        field.values[1],
        torch.full_like(field.values[1], expected),
    )
    state = optimizer.state[field.values]
    assert torch.equal(
        state["source_frame_steps"],
        torch.tensor((0, 1), dtype=torch.int32),
    )


def test_local_projection_field_optimizer_resume_replays_next_update() -> None:
    """Restored per-frame moments reproduce the following selected update."""
    field = LocalProjectionField(num_source_frames=2)
    optimizer = LocalProjectionFieldAdam(field.values)
    first_gradient = torch.zeros_like(field.values)
    first_gradient[1].fill_(0.5)
    field.values.grad = first_gradient
    optimizer.step(1)

    resumed_field = LocalProjectionField(num_source_frames=2)
    resumed_field.load_state_dict(copy.deepcopy(field.state_dict()))
    resumed_optimizer = LocalProjectionFieldAdam(resumed_field.values)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    second_gradient = torch.zeros_like(field.values)
    second_gradient[1].fill_(-0.25)
    field.values.grad = second_gradient.clone()
    resumed_field.values.grad = second_gradient.clone()

    optimizer.step(1)
    resumed_optimizer.step(1)

    torch.testing.assert_close(resumed_field.values, field.values)
    source_steps = optimizer.state[field.values]["source_frame_steps"]
    resumed_steps = resumed_optimizer.state[resumed_field.values][
        "source_frame_steps"
    ]
    assert torch.equal(resumed_steps, source_steps)


def test_local_projection_field_optimizer_rejects_hyperparameter_drift() -> None:
    """A resume cannot silently replace the configured optimizer settings."""
    field = LocalProjectionField(num_source_frames=1)
    optimizer = LocalProjectionFieldAdam(field.values)
    checkpoint_state = copy.deepcopy(optimizer.state_dict())
    checkpoint_state["param_groups"][0]["lr"] = 0.002
    resumed = LocalProjectionFieldAdam(field.values)

    with pytest.raises(ValueError, match="hyperparameters changed"):
        resumed.load_state_dict(checkpoint_state)


def test_local_projection_field_rejects_nonfinite_checkpoint_values() -> None:
    """Invalid saved field state fails before it reaches the renderer."""
    field = LocalProjectionField(num_source_frames=1)
    with torch.no_grad():
        field.values[0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="must be finite"):
        field.validate_state()


def test_native_ewa_field_hook_is_identity_at_zero() -> None:
    """Binding a zero field does not change projected metadata."""
    tracer = object.__new__(Tracer)
    tracer.local_projection_field = LocalProjectionField(num_source_frames=1)
    means2d = torch.tensor(((3.0, 5.0), (7.0, 11.0)))
    intrinsics = torch.eye(3).reshape(1, 3, 3)
    intrinsics[0, 0, 0] = 100.0
    intrinsics[0, 1, 1] = 120.0
    gpu_batch = type("Batch", (), {"source_frame_idx": 0})()

    shifted = tracer._apply_local_projection_field(
        means2d=means2d,
        intrinsics=intrinsics,
        gpu_batch=gpu_batch,
        width=32,
        height=32,
    )

    assert torch.equal(shifted, means2d)


def test_native_ewa_field_hook_moves_only_the_selected_frame() -> None:
    """The hook converts normalized field offsets into focal-plane pixels."""
    tracer = object.__new__(Tracer)
    field = LocalProjectionField(num_source_frames=2)
    with torch.no_grad():
        field.values[1].fill_(0.01)
    tracer.local_projection_field = field
    means2d = torch.tensor(((3.0, 5.0), (7.0, 11.0)))
    intrinsics = torch.eye(3).reshape(1, 3, 3)
    intrinsics[0, 0, 0] = 100.0
    intrinsics[0, 1, 1] = 120.0
    first_batch = type("Batch", (), {"source_frame_idx": 0})()
    second_batch = type("Batch", (), {"source_frame_idx": 1})()

    first = tracer._apply_local_projection_field(
        means2d=means2d,
        intrinsics=intrinsics,
        gpu_batch=first_batch,
        width=32,
        height=32,
    )
    second = tracer._apply_local_projection_field(
        means2d=means2d,
        intrinsics=intrinsics,
        gpu_batch=second_batch,
        width=32,
        height=32,
    )

    assert torch.equal(first, means2d)
    torch.testing.assert_close(
        second,
        means2d + torch.tensor((1.0, 1.2)),
    )


class _Renderer:
    """Minimal native-EWA binding surface for trainer lifecycle coverage."""

    def __init__(self) -> None:
        self.field: LocalProjectionField | None = None

    def set_local_projection_field(
        self,
        field: LocalProjectionField | None,
    ) -> None:
        """Store the trainer-owned field for inspection."""
        self.field = field


class _Model:
    """Minimal model with the native-EWA renderer binding surface."""

    def __init__(self) -> None:
        self.renderer = _Renderer()


class _Dataset:
    """Minimal stable source-frame metadata provider."""

    def get_source_frame_count(self) -> int:
        """Return the fixed source-frame table size."""
        return 3

    def get_source_frame_manifest_hash(self) -> str:
        """Return a stable source-frame ordering marker."""
        return "test-source-manifest"


def _field_checkpoint(
    field: LocalProjectionField,
    *,
    source_frame_manifest_hash: str,
) -> dict:
    """Build the persisted field portion needed by checkpoint rendering."""
    return {
        "local_projection_field": {
            "format_version": field.checkpoint_format_version,
            "algorithm": field.checkpoint_algorithm,
            "source_frame_manifest_hash": source_frame_manifest_hash,
            "module": {
                name: value.detach().clone()
                for name, value in field.state_dict().items()
            },
        }
    }


def test_checkpoint_render_restore_binds_the_saved_field() -> None:
    """Checkpoint evaluation uses the persisted source-frame field."""
    source_field = LocalProjectionField(num_source_frames=3)
    with torch.no_grad():
        source_field.values[2].fill_(0.125)
    tracer = _Renderer()

    restored = _restore_local_projection_field(
        _field_checkpoint(
            source_field,
            source_frame_manifest_hash="test-source-manifest",
        ),
        renderer=tracer,
        device=torch.device("cpu"),
    )

    assert restored is not None
    field, source_frame_manifest_hash = restored
    assert tracer.field is field
    assert source_frame_manifest_hash == "test-source-manifest"
    torch.testing.assert_close(field.values, source_field.values)
    _validate_local_projection_field_dataset(
        field=field,
        source_frame_manifest_hash=source_frame_manifest_hash,
        dataset=_Dataset(),
    )


def test_checkpoint_render_restore_rejects_reordered_source_frames() -> None:
    """Checkpoint evaluation fails instead of remapping field rows by position."""
    field = LocalProjectionField(num_source_frames=3)
    dataset = _Dataset()

    with pytest.raises(ValueError, match="manifest mismatch"):
        _validate_local_projection_field_dataset(
            field=field,
            source_frame_manifest_hash="different-source-manifest",
            dataset=dataset,
        )


def _projection_field_conf(
    *,
    replay_native_distortion: bool = False,
) -> DictConfig:
    """Build the smallest trainer configuration for field lifecycle tests."""
    return OmegaConf.create(
        {
            "render": {"method": "native_ewa"},
            "post_processing": {
                "replay_native_distortion": replay_native_distortion,
            },
            "local_projection_field": {
                "enabled": True,
                "activation_start_step": -1,
                "activation_full_passes": 1,
            },
        }
    )


def _trainer_for_projection_field() -> Trainer3DGRUT:
    """Return an uninitialized trainer with only field dependencies present."""
    trainer = object.__new__(Trainer3DGRUT)
    trainer.device = torch.device("cpu")
    trainer.model = _Model()
    trainer.train_dataset = _Dataset()
    trainer.local_projection_field = None
    trainer.local_projection_field_optimizer = None
    trainer._local_projection_field_activation_step = -1
    return trainer


def test_trainer_binds_field_after_one_full_source_pass() -> None:
    """The default hypothesis delays updates until each frame was seen once."""
    trainer = _trainer_for_projection_field()
    conf = _projection_field_conf()

    trainer.init_local_projection_field(conf)

    assert trainer.local_projection_field is not None
    assert trainer.model.renderer.field is trainer.local_projection_field
    assert trainer._local_projection_field_activation_step == 3
    assert not trainer._local_projection_field_is_active(2)
    assert trainer._local_projection_field_is_active(3)


def test_trainer_rejects_double_projection_correction() -> None:
    """Trainable pre-raster fields cannot stack with post-render replay."""
    trainer = _trainer_for_projection_field()
    conf = _projection_field_conf(replay_native_distortion=True)

    with pytest.raises(ValueError, match="cannot combine"):
        trainer.init_local_projection_field(conf)
