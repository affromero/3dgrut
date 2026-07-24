"""Behavior tests for valid-center same-padding SSIM."""

import pytest
import torch

import threedgrut.model.losses as losses_module
from threedgrut.model.losses import masked_ssim_same_loss


class _RecordedSSIMMap:
    """Minimal fused-map boundary substitute for CPU behavior coverage."""

    prediction: torch.Tensor | None = None
    target: torch.Tensor | None = None
    padding: str | None = None
    train: bool | None = None

    @classmethod
    def apply(
        cls,
        c1: float,
        c2: float,
        prediction: torch.Tensor,
        target: torch.Tensor,
        padding: str,
        train: bool,
    ) -> torch.Tensor:
        """Record the fused-map contract and expose the masked prediction."""
        assert c1 == 0.01**2
        assert c2 == 0.03**2
        cls.prediction = prediction
        cls.target = target
        cls.padding = padding
        cls.train = train
        return prediction


def test_masked_ssim_same_loss_masks_inputs_and_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid pixels cannot affect the SSIM map or its RGB gradient."""
    monkeypatch.setattr(losses_module, "FusedSSIMMap", _RecordedSSIMMap)
    prediction = torch.tensor(
        [[[[0.2, 0.8], [0.4, 0.6]]]],
        requires_grad=True,
    )
    target = torch.tensor([[[[0.1, 0.7], [0.3, 0.5]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    loss = masked_ssim_same_loss(
        prediction=prediction,
        target=target,
        mask=mask,
        denominator=torch.tensor(8.0),
    )
    loss.backward()

    assert _RecordedSSIMMap.padding == "same"
    assert _RecordedSSIMMap.train is True
    torch.testing.assert_close(
        _RecordedSSIMMap.prediction,
        torch.tensor([[[[0.2, 0.0], [0.4, 0.0]]]]),
    )
    torch.testing.assert_close(
        _RecordedSSIMMap.target,
        torch.tensor([[[[0.1, 0.0], [0.3, 0.0]]]]),
    )
    torch.testing.assert_close(loss, torch.tensor(0.175))
    torch.testing.assert_close(
        prediction.grad,
        torch.tensor([[[[-0.125, 0.0], [-0.125, 0.0]]]]),
    )


def test_masked_ssim_same_loss_can_require_full_valid_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mask gap excludes adjacent centers but not zero-padded image edges."""
    monkeypatch.setattr(losses_module, "FusedSSIMMap", _RecordedSSIMMap)
    prediction = torch.ones((1, 1, 13, 13), requires_grad=True)
    target = torch.zeros((1, 1, 13, 13))
    mask = torch.ones((1, 1, 13, 13))
    mask[:, :, 6, 6] = 0.0

    loss = masked_ssim_same_loss(
        prediction=prediction,
        target=target,
        mask=mask,
        denominator=torch.tensor(169.0),
        require_full_valid_window=True,
    )
    loss.backward()

    expected_gradient = -1.0 / 169.0
    torch.testing.assert_close(
        prediction.grad[:, :, 0, 0],
        torch.tensor([[expected_gradient]]),
    )
    torch.testing.assert_close(
        prediction.grad[:, :, 1, 6],
        torch.tensor([[0.0]]),
    )
    torch.testing.assert_close(
        prediction.grad[:, :, 12, 12],
        torch.tensor([[expected_gradient]]),
    )
