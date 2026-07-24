from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.post_processing.luminance_affine import LuminanceAffine
from threedgrut.post_processing.native_appearance import (
    IndexedAppearanceAdam,
    NativeAppearanceGrid,
)
from threedgrut.trainer import (
    Trainer3DGRUT,
    _drop_shape_mismatched_optimizer_state,
    _validate_native_replay_configuration,
)
from threedgrut.utils.render import apply_post_processing


def test_zero_native_appearance_is_identity() -> None:
    module = NativeAppearanceGrid(num_frames=2)
    pred_rgb = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)
    pixel_coords = torch.tensor([[3.5, 2.5]], dtype=torch.float32)

    corrected = module(
        pred_rgb,
        pixel_coords,
        resolution=(8, 6),
        frame_idx=1,
    )

    assert torch.allclose(corrected, pred_rgb)


def test_native_appearance_applies_all_eight_channels() -> None:
    module = NativeAppearanceGrid(num_frames=1)
    values = torch.tensor(
        [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.1],
        dtype=torch.float32,
    )
    with torch.no_grad():
        grid = module.embedding.weight[0].reshape(33, 33, 8)
        grid.copy_(values.expand_as(grid))

    pred_rgb = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)
    pixel_coords = torch.tensor([[3.5, 2.5]], dtype=torch.float32)
    corrected = module(
        pred_rgb,
        pixel_coords,
        resolution=(8, 6),
        frame_idx=0,
    )

    gain = torch.exp(values[:3])
    transformed = pred_rgb * gain + 0.1 * values[3:6]
    transformed = transformed + 0.125 * values[6] * pred_rgb.square()
    transformed = transformed.clamp(0.0, 1.0)
    expected = 1.0 - torch.pow(1.0 - transformed, torch.exp(values[7]))
    assert torch.allclose(corrected, expected)


def test_native_appearance_saturated_pixels_use_native_gradients() -> None:
    """Saturated pixels retain the native ungated appearance gradients."""
    module = NativeAppearanceGrid(num_frames=1)
    with torch.no_grad():
        grid = module.embedding.weight[0].reshape(33, 33, 8)
        grid[..., :3] = 1.0
    pred_rgb = torch.full((1, 3), 0.8, requires_grad=True)
    corrected = module(
        pred_rgb,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        resolution=(1, 1),
        frame_idx=0,
    )

    corrected.sum().backward()

    gradient = (
        module.embedding.weight.grad.coalesce()
        .values()
        .reshape(
            1,
            33,
            33,
            8,
        )
        .sum(dim=(0, 1, 2))
    )
    gain = torch.exp(torch.tensor(1.0))
    assert torch.equal(corrected, torch.ones_like(corrected))
    assert torch.allclose(pred_rgb.grad, torch.ones_like(pred_rgb) * gain)
    assert torch.allclose(
        gradient[:3],
        torch.ones_like(gradient[:3]) * 0.8 * gain,
    )
    assert torch.allclose(
        gradient[3:6],
        torch.full_like(gradient[3:6], 0.1),
    )
    assert torch.allclose(gradient[6:7], torch.full_like(gradient[6:7], 0.24))
    assert torch.equal(gradient[7:8], torch.zeros_like(gradient[7:8]))
    assert torch.isfinite(gradient).all()


def test_native_appearance_interior_gradients_match_chain_rule() -> None:
    """The recovered gradients reduce to ordinary chain rule away from clamps."""
    module = NativeAppearanceGrid(num_frames=1)
    values = torch.tensor(
        [0.1, -0.2, 0.05, 0.2, -0.1, 0.15, 0.1, -0.2],
        dtype=torch.float32,
    )
    with torch.no_grad():
        grid = module.embedding.weight[0].reshape(33, 33, 8)
        grid.copy_(values.expand_as(grid))
    pred_rgb = torch.tensor(
        [[0.2, 0.3, 0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    corrected = module(
        pred_rgb,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        resolution=(1, 1),
        frame_idx=0,
    )
    corrected.sum().backward()
    gradient = (
        module.embedding.weight.grad.coalesce()
        .values()
        .reshape(
            1,
            33,
            33,
            8,
        )
        .sum(dim=(0, 1, 2))
    )

    reference_pred_rgb = pred_rgb.detach().clone().requires_grad_(True)
    reference_values = values.detach().clone().requires_grad_(True)
    reference_gain = torch.exp(reference_values[:3])
    reference_q = (
        reference_pred_rgb * reference_gain
        + 0.1 * reference_values[3:6]
        + 0.125 * reference_values[6] * reference_pred_rgb.square()
    )
    reference = 1.0 - torch.pow(
        1.0 - reference_q,
        torch.exp(reference_values[7]),
    )
    reference.sum().backward()

    assert torch.allclose(corrected, reference)
    assert torch.allclose(pred_rgb.grad, reference_pred_rgb.grad)
    assert torch.allclose(gradient, reference_values.grad)


def test_native_appearance_power_gradient_uses_recovered_floor() -> None:
    """The power lane uses the recovered 0.1 logarithm floor."""
    module = NativeAppearanceGrid(num_frames=1)
    pred_rgb = torch.full((1, 3), 0.95, requires_grad=True)
    corrected = module(
        pred_rgb,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        resolution=(1, 1),
        frame_idx=0,
    )

    corrected.sum().backward()

    gradient = (
        module.embedding.weight.grad.coalesce()
        .values()
        .reshape(
            1,
            33,
            33,
            8,
        )
        .sum(dim=(0, 1, 2))
    )
    expected_power_gradient = -3.0 * 0.05 * torch.log(torch.tensor(0.1))
    assert torch.allclose(gradient[7], expected_power_gradient)


def test_native_appearance_uses_pixel_center_bilinear_coordinates() -> None:
    module = NativeAppearanceGrid(num_frames=1)
    with torch.no_grad():
        grid = module.embedding.weight[0].reshape(33, 33, 8)
        grid[0, 0, 3] = 0.0
        grid[0, 1, 3] = 2.0
        grid[1, 0, 3] = 4.0
        grid[1, 1, 3] = 6.0

    corrected = module(
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        resolution=(32, 32),
        frame_idx=0,
    )

    assert torch.allclose(corrected[0, 0], torch.tensor(0.3))
    assert torch.equal(corrected[0, 1:], torch.zeros(2))


def test_unknown_native_appearance_frame_is_identity() -> None:
    module = NativeAppearanceGrid(num_frames=1)
    with torch.no_grad():
        module.embedding.weight.fill_(1.0)
    pred_rgb = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)

    corrected = module(
        pred_rgb,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        resolution=(1, 1),
        frame_idx=-1,
    )

    assert torch.equal(corrected, pred_rgb)


def test_native_appearance_preopt_count_matches_camera_multipliers() -> None:
    trainer = object.__new__(Trainer3DGRUT)
    trainer._post_processing_camera_index_mode = "dataset"
    conf = OmegaConf.create(
        {
            "post_processing": {
                "use_native_appearance_grid": True,
                "native_preoptimize": True,
                "native_max_appearance_iterations": 4,
                "native_iteration_multipliers": [3, 3, 1, 1],
            }
        }
    )

    first_camera = SimpleNamespace(camera_idx=0, post_processing_camera_idx=0)
    device_hd_camera = SimpleNamespace(
        camera_idx=2,
        post_processing_camera_idx=2,
    )

    assert (
        trainer._native_appearance_preopt_step_count(
            conf,
            first_camera,
        )
        == 1
    )
    assert (
        trainer._native_appearance_preopt_step_count(
            conf,
            device_hd_camera,
        )
        == 3
    )


def test_frozen_native_appearance_discards_persistent_updates() -> None:
    trainer = object.__new__(Trainer3DGRUT)
    trainer._post_processing_camera_index_mode = "dataset"
    grid = NativeAppearanceGrid(num_frames=1)
    trainer.post_processing_optimizers = [grid.create_optimizer(0.05)]
    trainer.post_processing_schedulers = []
    conf = OmegaConf.create(
        {
            "post_processing": {
                "use_native_appearance_grid": True,
                "native_preoptimize": True,
                "freeze_native_appearance": True,
            }
        }
    )
    batch = SimpleNamespace(camera_idx=0, post_processing_camera_idx=0)

    grid.embedding(torch.tensor([0])).sum().backward()
    original = grid.embedding.weight.detach().clone()
    trainer._step_post_processing_optimizers(
        conf=conf,
        native_appearance_preopt_steps=(trainer._native_appearance_preopt_step_count(conf, batch)),
    )

    assert torch.equal(grid.embedding.weight, original)
    assert grid.embedding.weight.grad is None


def test_training_allows_native_appearance_replay() -> None:
    _validate_native_replay_configuration(
        validate_only=False,
        native_checkpoint_path="native.msgpack",
        replay_native_appearance=True,
        replay_native_extrinsics=False,
        replay_native_distortion=False,
        apply_known_frame_in_eval=False,
    )


def test_training_allows_native_extrinsic_replay() -> None:
    _validate_native_replay_configuration(
        validate_only=False,
        native_checkpoint_path="native.msgpack",
        replay_native_appearance=False,
        replay_native_extrinsics=True,
        replay_native_distortion=False,
        apply_known_frame_in_eval=False,
    )


def test_training_allows_native_distortion_replay() -> None:
    _validate_native_replay_configuration(
        validate_only=False,
        native_checkpoint_path="native.msgpack",
        replay_native_appearance=False,
        replay_native_extrinsics=False,
        replay_native_distortion=True,
        apply_known_frame_in_eval=False,
    )


def test_known_frame_eval_remains_offline_only() -> None:
    with pytest.raises(ValueError, match="requires validate_only"):
        _validate_native_replay_configuration(
            validate_only=False,
            native_checkpoint_path="native.msgpack",
            replay_native_appearance=True,
            replay_native_extrinsics=False,
            replay_native_distortion=False,
            apply_known_frame_in_eval=True,
        )


@pytest.mark.parametrize(
    ("appearance", "extrinsics", "distortion"),
    ((True, False, False), (False, True, False), (False, False, True)),
)
def test_native_replay_requires_checkpoint_path(
    appearance: bool,
    extrinsics: bool,
    distortion: bool,
) -> None:
    with pytest.raises(ValueError, match="native_checkpoint_path"):
        _validate_native_replay_configuration(
            validate_only=True,
            native_checkpoint_path="",
            replay_native_appearance=appearance,
            replay_native_extrinsics=extrinsics,
            replay_native_distortion=distortion,
            apply_known_frame_in_eval=appearance,
        )


def test_known_native_appearance_uses_source_frame_index() -> None:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=2,
        use_native_appearance_grid=True,
    )
    with torch.no_grad():
        grid = module.native_appearance_grid.embedding.weight[1].reshape(33, 33, 8)
        grid[..., 0] = 0.2
    pred_rgb = torch.full((1, 2, 2, 3), 0.25)
    outputs = {"pred_rgb": pred_rgb}
    batch = SimpleNamespace(
        camera_idx=0,
        frame_idx=0,
        source_frame_idx=1,
        sequence_idx=0,
        exposure=None,
        pixel_coords=torch.tensor([[[[0.5, 0.5], [1.5, 0.5]], [[0.5, 1.5], [1.5, 1.5]]]]),
    )

    corrected = apply_post_processing(
        module,
        dict(outputs),
        batch,
        use_known_frame=True,
    )["pred_rgb"]
    novel_view = apply_post_processing(
        module,
        dict(outputs),
        batch,
        use_known_frame=False,
    )["pred_rgb"]

    assert torch.all(corrected[..., 0] > pred_rgb[..., 0])
    assert torch.equal(novel_view, pred_rgb)


def test_indexed_appearance_adam_counts_only_active_texels() -> None:
    embedding = torch.nn.Embedding(2, 16, sparse=True)
    torch.nn.init.zeros_(embedding.weight)
    optimizer = IndexedAppearanceAdam(embedding.weight, lr=0.01)

    first_loss = embedding(torch.tensor([0]))[0, 0]
    first_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    second_loss = embedding(torch.tensor([1]))[0, 0]
    second_loss.backward()
    optimizer.step()

    steps = optimizer.state[embedding.weight]["step"]
    assert torch.equal(steps, torch.tensor([[1, 0], [1, 0]]))
    assert embedding.weight[0, 0] < 0
    assert embedding.weight[1, 0] < 0
    assert torch.equal(embedding.weight[:, 8:], torch.zeros((2, 8)))


def test_indexed_appearance_adam_matches_native_first_step() -> None:
    embedding = torch.nn.Embedding(1, 8, sparse=True)
    torch.nn.init.constant_(embedding.weight, 0.5)
    optimizer = IndexedAppearanceAdam(embedding.weight, lr=0.01)

    embedding(torch.tensor([0]))[0, 0].backward()
    optimizer.step()

    first_moment = 0.2
    second_moment = 0.05
    bias_scale = (1.0 - 0.95) ** 0.5 / (1.0 - 0.8)
    step_schedule = 2.0 * 0.1 / (0.1**2 + 1.0)
    normalized_update = first_moment / (second_moment**0.5 + 1.024e-4)
    expected_first = 0.5 - (0.01 * bias_scale * step_schedule * (normalized_update + 0.1 * 0.5))
    expected_regularized = 0.5 - (0.01 * bias_scale * step_schedule * 0.1 * 0.5)

    assert torch.isclose(
        embedding.weight[0, 0],
        torch.tensor(expected_first),
    )
    assert torch.isclose(
        embedding.weight[0, 1],
        torch.tensor(expected_regularized),
    )


def test_native_appearance_gradient_is_sparse_by_frame() -> None:
    module = NativeAppearanceGrid(num_frames=3)
    pred_rgb = torch.full((4, 3), 0.5)
    pixel_coords = torch.tensor(
        [[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]],
        dtype=torch.float32,
    )

    module(
        pred_rgb,
        pixel_coords,
        resolution=(2, 2),
        frame_idx=2,
    ).sum().backward()

    gradient = module.embedding.weight.grad.coalesce()
    assert gradient.is_sparse
    assert torch.equal(gradient.indices(), torch.tensor([[2]]))


def test_native_appearance_replay_overwrites_resumed_weights(tmp_path) -> None:
    appearance = np.full((33, 33, 8), 0.25, dtype="<f2")
    distortion = np.zeros((12, 12, 2), dtype="<f4")
    checkpoint_path = tmp_path / "native.msgpack"
    with checkpoint_path.open("wb") as handle:
        msgpack.pack(
            {
                "images": [
                    {
                        "name": "frame.jpg",
                        "appearance": {"data": appearance.tobytes()},
                        "distortion": {"data": distortion.tobytes()},
                        "extrinsic": {
                            "qvec": [1.0, 0.0, 0.0, 0.0],
                            "tvec": [0.0, 0.0, 0.0],
                        },
                    }
                ]
            },
            handle,
        )

    trainer = object.__new__(Trainer3DGRUT)
    trainer.device = torch.device("cpu")
    trainer.train_dataset = SimpleNamespace(get_source_frame_names=lambda: ("frame.jpg",))
    trainer.post_processing = LuminanceAffine(
        num_cameras=1,
        num_frames=1,
        use_native_appearance_grid=True,
    )
    with torch.no_grad():
        trainer.post_processing.native_appearance_grid.embedding.weight.fill_(-0.5)
    conf = OmegaConf.create(
        {
            "post_processing": {
                "native_checkpoint_path": str(checkpoint_path),
                "replay_native_appearance": True,
                "use_native_appearance_grid": True,
            }
        }
    )

    trainer._replay_native_appearance(conf)

    expected = torch.full_like(
        trainer.post_processing.native_appearance_grid.embedding.weight,
        0.25,
    )
    assert torch.equal(
        trainer.post_processing.native_appearance_grid.embedding.weight,
        expected,
    )


def test_indexed_appearance_adam_restores_texel_counters() -> None:
    first_embedding = torch.nn.Embedding(2, 16, sparse=True)
    first_optimizer = IndexedAppearanceAdam(first_embedding.weight, lr=0.01)
    first_embedding(torch.tensor([1]))[0, 0].backward()
    first_optimizer.step()

    second_embedding = torch.nn.Embedding(2, 16, sparse=True)
    second_optimizer = IndexedAppearanceAdam(second_embedding.weight, lr=0.01)
    second_optimizer.load_state_dict(first_optimizer.state_dict())

    steps = second_optimizer.state[second_embedding.weight]["step"]
    assert torch.equal(steps, torch.tensor([[0, 0], [1, 0]]))


def test_indexed_appearance_adam_resume_keeps_texel_counters() -> None:
    first_embedding = torch.nn.Embedding(2, 16, sparse=True)
    first_optimizer = IndexedAppearanceAdam(first_embedding.weight, lr=0.01)
    first_embedding(torch.tensor([1]))[0, 0].backward()
    first_optimizer.step()

    second_embedding = torch.nn.Embedding(2, 16, sparse=True)
    second_optimizer = IndexedAppearanceAdam(second_embedding.weight, lr=0.01)
    second_optimizer.load_state_dict(first_optimizer.state_dict())

    dropped = _drop_shape_mismatched_optimizer_state(
        second_optimizer,
        label="post_processing",
    )

    steps = second_optimizer.state[second_embedding.weight]["step"]
    assert dropped == []
    assert torch.equal(steps, torch.tensor([[0, 0], [1, 0]]))
