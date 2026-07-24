"""Tests for the recovered visibility-adaptive low-rank color representation."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from threedgrut.model.model import decode_native_color_features
from threedgrut.strategy.base import BaseStrategy
from threedgrut.trainer import Trainer3DGRUT, _scale_parameter_gradient


def test_native_color_encoder_matches_explicit_linear_sh() -> None:
    """Low-rank decode produces the same SH values as explicit matmul."""
    generator = torch.Generator().manual_seed(7)
    latents = torch.randn((13, 8), generator=generator)
    encoder = torch.randn((45, 8), generator=generator)

    decoded = decode_native_color_features(latents, encoder)

    assert decoded.shape == (13, 45)
    torch.testing.assert_close(decoded, latents @ encoder.transpose(0, 1))


def test_native_color_encoder_backpropagates_to_latents_and_weights() -> None:
    """One image-space surrogate loss updates both native color factors."""
    generator = torch.Generator().manual_seed(11)
    latents = torch.randn(
        (17, 8), generator=generator, requires_grad=True
    )
    encoder = torch.randn(
        (45, 8), generator=generator, requires_grad=True
    )
    target = torch.randn((17, 45), generator=generator)

    loss = torch.nn.functional.huber_loss(
        decode_native_color_features(latents, encoder),
        target,
    )
    loss.backward()

    assert latents.grad is not None
    assert encoder.grad is not None
    assert torch.isfinite(latents.grad).all()
    assert torch.isfinite(encoder.grad).all()
    assert torch.count_nonzero(latents.grad) > 0
    assert torch.count_nonzero(encoder.grad) > 0


def test_densification_updates_only_per_gaussian_parameters() -> None:
    """Shared native encoder weights survive Gaussian shape mutations."""
    positions = torch.nn.Parameter(torch.ones((2, 3)))
    color_encoder = torch.nn.Parameter(torch.ones((45, 8)))
    optimizer = torch.optim.Adam(
        [
            {"params": [positions], "name": "positions"},
            {"params": [color_encoder], "name": "color_encoder"},
        ]
    )
    (positions.sum() + color_encoder.sum()).backward()
    optimizer.step()
    model = SimpleNamespace(
        positions=positions,
        color_encoder=color_encoder,
        optimizer=optimizer,
    )
    strategy = BaseStrategy(None, model)
    encoder_before = color_encoder.detach().clone()

    strategy._update_param_with_optimizer(
        lambda _name, value: torch.nn.Parameter(
            torch.cat((value, value[:1]), dim=0)
        ),
        lambda _key, value: torch.cat((value, value[:1]), dim=0),
    )

    assert model.positions.shape == (3, 3)
    assert model.color_encoder is color_encoder
    assert model.color_encoder.shape == (45, 8)
    torch.testing.assert_close(model.color_encoder, encoder_before)


def test_scale_parameter_gradient_handles_dense_and_sparse_grads() -> None:
    """Native RGB scaling supports dense and sparse gradients."""
    dense = torch.nn.Parameter(torch.ones((2, 3)))
    dense.grad = torch.full_like(dense, 2.0)

    _scale_parameter_gradient(dense, 4.0)

    torch.testing.assert_close(dense.grad, torch.full_like(dense, 8.0))

    embedding = torch.nn.Embedding(5, 2, sparse=True)
    embedding(torch.tensor([1, 3])).sum().backward()

    _scale_parameter_gradient(embedding.weight, 3.0)

    gradient = embedding.weight.grad
    assert gradient is not None
    assert gradient.is_sparse
    values = gradient.coalesce().values()
    torch.testing.assert_close(values, torch.full_like(values, 3.0))


def test_native_rgb_gradient_scale_skips_geometry_grads() -> None:
    """The recovered 255x RGB gap must not feed topology or geometry params."""
    trainer = object.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {"loss": {"radiance_gradient_scale": 5.0}}
    )
    positions = torch.nn.Parameter(torch.ones((2, 3)))
    albedo = torch.nn.Parameter(torch.ones((2, 3)))
    specular = torch.nn.Parameter(torch.ones((2, 8)))
    color_encoder = torch.nn.Parameter(torch.ones((45, 8)))
    for parameter, value in (
        (positions, 1.0),
        (albedo, 2.0),
        (specular, 3.0),
        (color_encoder, 4.0),
    ):
        parameter.grad = torch.full_like(parameter, value)
    appearance_weight = torch.nn.Parameter(torch.ones((4, 8)))
    appearance_weight.grad = torch.sparse_coo_tensor(
        torch.tensor([[1, 3]]),
        torch.ones((2, 8)),
        size=appearance_weight.shape,
    )
    trainer.model = SimpleNamespace(
        positions=positions,
        features_albedo=albedo,
        features_specular=specular,
        color_encoder=color_encoder,
    )
    trainer.post_processing = SimpleNamespace(
        native_appearance_grid=SimpleNamespace(
            embedding=SimpleNamespace(weight=appearance_weight)
        )
    )

    trainer._scale_radiance_gradients()

    torch.testing.assert_close(positions.grad, torch.ones_like(positions))
    torch.testing.assert_close(albedo.grad, torch.full_like(albedo, 10.0))
    torch.testing.assert_close(specular.grad, torch.full_like(specular, 15.0))
    torch.testing.assert_close(
        color_encoder.grad,
        torch.full_like(color_encoder, 20.0),
    )
    gradient = appearance_weight.grad
    assert gradient is not None
    torch.testing.assert_close(
        gradient.coalesce().values(),
        torch.full((2, 8), 5.0),
    )
