"""Behavioral coverage for the 3DGUT color/background contract."""

from types import SimpleNamespace

import torch

from threedgrut.model.features import Features
from threedgut_tracer.tracer import Tracer


class _Background:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        _camera_to_world: torch.Tensor,
        _rays_dir: torch.Tensor,
        rgb: torch.Tensor,
        opacity: torch.Tensor,
        _train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return rgb + (1.0 - opacity), opacity


class _Gaussians:
    def __init__(self, feature_type: Features.Type) -> None:
        self.feature_type = feature_type
        self.background = _Background()
        self.positions = torch.zeros((1, 3))
        self.num_gaussians = 1
        self.n_active_features = 0
        self.ray_feature_dim = 3

    def get_rotation(self) -> torch.Tensor:
        return torch.zeros((1, 4))

    def get_scale(self) -> torch.Tensor:
        return torch.zeros((1, 3))

    def get_density(self) -> torch.Tensor:
        return torch.zeros((1, 1))

    def get_features(self) -> torch.Tensor:
        return torch.zeros((1, 3))


def _batch() -> SimpleNamespace:
    rays = torch.zeros((1, 1, 1, 3))
    return SimpleNamespace(
        rays_ori=rays,
        rays_dir=rays,
        T_to_world=torch.eye(4).unsqueeze(0),
        T_to_world_end=None,
    )


def _trace_outputs(*_args: object) -> tuple[torch.Tensor, ...]:
    features_alpha = torch.tensor((((0.2, 0.3, 0.4, 0.25),),))
    scalar = torch.zeros((1, 1, 1))
    particle = torch.zeros((1, 1))
    position = torch.zeros((1, 2))
    conic = torch.zeros((1, 4))
    extent = torch.zeros((1, 2))
    return (
        features_alpha,
        scalar,
        scalar,
        particle,
        position,
        conic,
        extent,
        particle,
        particle,
        position,
    )


def _tracer(monkeypatch) -> Tracer:
    tracer = object.__new__(Tracer)
    tracer.tracer_wrapper = SimpleNamespace(collect_times=lambda: {})
    tracer.conf = SimpleNamespace()
    tracer.absolute_ray_gradient_diagnostics_enabled = False
    tracer.absolute_ray_gradient_densification_enabled = False
    tracer.cancellation_conditioned_split_enabled = False
    monkeypatch.setattr(
        Tracer,
        "_Tracer__create_camera_parameters",
        staticmethod(lambda _batch: (None, None)),
    )
    monkeypatch.setattr(
        Tracer,
        "_pad_particle_radiance",
        staticmethod(lambda features, _width: features),
    )
    monkeypatch.setattr(
        "threedgut_tracer.tracer.particle_radiance_width",
        lambda _conf: 3,
    )
    monkeypatch.setattr(Tracer._Autograd, "apply", _trace_outputs)
    return tracer


def test_3dgut_composites_background_for_sh(monkeypatch) -> None:
    """SH output is RGB, so the configured background fills transparency."""
    gaussians = _Gaussians(Features.Type.SH)

    outputs = _tracer(monkeypatch).render(gaussians, _batch())

    assert gaussians.background.calls == 1
    assert torch.allclose(
        outputs["pred_rgb"],
        torch.tensor([[[[0.95, 1.05, 1.15]]]]),
    )


def test_3dgut_keeps_nht_features_uncomposited(monkeypatch) -> None:
    """Latent NHT features reach their decoder without RGB background values."""
    gaussians = _Gaussians(Features.Type.NHT)

    outputs = _tracer(monkeypatch).render(gaussians, _batch())

    assert gaussians.background.calls == 0
    assert torch.allclose(
        outputs["pred_rgb"],
        torch.tensor([[[[0.2, 0.3, 0.4]]]]),
    )
