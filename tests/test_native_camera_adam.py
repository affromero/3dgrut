"""Tests for the recovered per-image camera optimizer."""

import copy

import torch
from threedgrut.optimizers.native_camera_adam import NativeCameraAdam


def _parameters() -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    qvecs = torch.nn.Parameter(
        torch.tensor(
            ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            dtype=torch.float32,
        )
    )
    tvecs = torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    return qvecs, tvecs


def _optimizer(
    qvecs: torch.nn.Parameter,
    tvecs: torch.nn.Parameter,
) -> NativeCameraAdam:
    return NativeCameraAdam(
        (
            {"name": "qvecs", "params": [qvecs]},
            {"name": "tvecs", "params": [tvecs]},
        )
    )


def _set_gradients(
    qvecs: torch.nn.Parameter,
    tvecs: torch.nn.Parameter,
) -> None:
    qvecs.grad = torch.tensor(((1.0, -2.0, 3.0, -4.0), (5.0, 6.0, 7.0, 8.0)))
    tvecs.grad = torch.tensor(((1.0, -2.0, 3.0), (4.0, 5.0, 6.0)))


def test_native_camera_adam_updates_only_selected_image() -> None:
    """One launch updates one qvec/tvec row and its shared counter."""
    qvecs, tvecs = _parameters()
    optimizer = _optimizer(qvecs, tvecs)
    _set_gradients(qvecs, tvecs)

    optimizer.step(0)

    assert not torch.equal(
        qvecs[0],
        torch.tensor((1.0, 0.0, 0.0, 0.0)),
    )
    assert torch.equal(qvecs[1], torch.tensor((1.0, 0.0, 0.0, 0.0)))
    assert torch.equal(tvecs[1], torch.zeros(3))
    assert torch.linalg.vector_norm(qvecs[0]).item() == 1.0
    assert torch.equal(
        optimizer.state[qvecs]["image_steps"],
        torch.tensor((1, 0), dtype=torch.int32),
    )


def test_native_camera_adam_matches_recovered_first_step() -> None:
    """The first tvec update uses native bias correction and epsilon."""
    qvecs, tvecs = _parameters()
    optimizer = _optimizer(qvecs, tvecs)
    _set_gradients(qvecs, tvecs)

    optimizer.step(0)

    gradient = torch.tensor((1.0, -2.0, 3.0))
    first_moment = 0.2 * gradient
    second_moment = 0.05 * gradient.square()
    bias_scale = (1.0 - 0.95) ** 0.5 / (1.0 - 0.8)
    expected = (
        -0.001
        * bias_scale
        * first_moment
        / (torch.sqrt(second_moment) + 1.0e-7)
    )
    torch.testing.assert_close(tvecs[0], expected)


def test_native_camera_adam_resume_preserves_per_image_state() -> None:
    """Resume restores moments and independent image counters exactly."""
    qvecs, tvecs = _parameters()
    optimizer = _optimizer(qvecs, tvecs)
    _set_gradients(qvecs, tvecs)
    optimizer.step(0)

    resumed_qvecs = torch.nn.Parameter(qvecs.detach().clone())
    resumed_tvecs = torch.nn.Parameter(tvecs.detach().clone())
    resumed = _optimizer(resumed_qvecs, resumed_tvecs)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    _set_gradients(qvecs, tvecs)
    _set_gradients(resumed_qvecs, resumed_tvecs)
    optimizer.step(1)
    resumed.step(1)

    torch.testing.assert_close(resumed_qvecs, qvecs)
    torch.testing.assert_close(resumed_tvecs, tvecs)
    assert torch.equal(
        resumed.state[resumed_qvecs]["image_steps"],
        torch.tensor((1, 1), dtype=torch.int32),
    )
