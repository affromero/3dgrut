"""Behavior tests for first-order Gaussian-Hermite carrier packing."""

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from threedgrut.model.model import (
    MixtureOfGaussians,
    _carrier_specular_dim,
    _initial_hermite_carrier_tail,
    sh_degree_to_specular_dim,
)
from threedgut_tracer.setup_3dgut import (
    particle_radiance_width,
    setup_3dgut,
)


def _conf(
    *,
    use_gabor_carrier: bool = False,
    use_hermite_carrier: bool = False,
    use_siren_carrier: bool = False,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "render": {"particle_radiance_sph_degree": 3},
            "model": {
                "use_gabor_carrier": use_gabor_carrier,
                "gabor_num_terms": 3,
                "use_hermite_carrier": use_hermite_carrier,
                "use_siren_carrier": use_siren_carrier,
                "siren_hidden_dim": 6,
            },
        }
    )


def _native_conf() -> OmegaConf:
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs")
    )
    with initialize_config_dir(
        config_dir=config_dir,
        version_base=None,
    ):
        conf = compose(config_name="apps/colmap_3dgut")
    conf.model.use_gabor_carrier = False
    conf.model.use_hermite_carrier = True
    conf.model.use_siren_carrier = False
    return conf


@pytest.mark.parametrize(
    ("carrier", "expected_width"),
    (
        ("sh", 48),
        ("hermite", 54),
        ("gabor", 66),
        ("siren", 111),
    ),
)
def test_model_and_native_layout_widths_agree(
    carrier: str,
    expected_width: int,
) -> None:
    """Model albedo plus specular width matches the compiled native row."""
    conf = _conf(
        use_gabor_carrier=carrier == "gabor",
        use_hermite_carrier=carrier == "hermite",
        use_siren_carrier=carrier == "siren",
    )
    model_width = (
        3
        + sh_degree_to_specular_dim(3)
        + _carrier_specular_dim(conf)
    )

    assert model_width == expected_width
    assert model_width == particle_radiance_width(conf)


def test_hermite_tail_initializes_to_exact_zero() -> None:
    """Hermite treatment is pixel-identical before its first update."""
    conf = _conf(use_hermite_carrier=True)

    tail = _initial_hermite_carrier_tail(
        num_gaussians=7,
        device="cpu",
        dtype=torch.float32,
        conf=conf,
    )

    assert tail.shape == (7, 6)
    assert torch.equal(tail, torch.zeros_like(tail))


@pytest.mark.parametrize(
    ("use_gabor_carrier", "use_hermite_carrier", "use_siren_carrier"),
    (
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ),
)
def test_model_rejects_conflicting_carrier_layouts(
    use_gabor_carrier: bool,
    use_hermite_carrier: bool,
    use_siren_carrier: bool,
) -> None:
    """Model packing rejects every pair of incompatible carrier tails."""
    conf = _conf(
        use_gabor_carrier=use_gabor_carrier,
        use_hermite_carrier=use_hermite_carrier,
        use_siren_carrier=use_siren_carrier,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        _carrier_specular_dim(conf)


def test_control_checkpoint_gains_zero_hermite_tail() -> None:
    """Control SH rows become Hermite rows without altering SH values."""
    conf = _conf(use_hermite_carrier=True)
    model = object.__new__(MixtureOfGaussians)
    model.conf = conf
    model.max_n_features = 3
    control = torch.randn((5, sh_degree_to_specular_dim(3)))

    treatment = MixtureOfGaussians._with_carrier_tail(model, control)

    assert treatment.shape == (5, 51)
    torch.testing.assert_close(treatment[:, :45], control)
    assert torch.equal(treatment[:, 45:], torch.zeros((5, 6)))


@pytest.mark.parametrize("foreign_width", (63, 108))
def test_foreign_carrier_checkpoint_fails_loudly(
    foreign_width: int,
) -> None:
    """Gabor and SIREN rows cannot be misread as Hermite checkpoints."""
    conf = _conf(use_hermite_carrier=True)
    model = object.__new__(MixtureOfGaussians)
    model.conf = conf
    model.max_n_features = 3
    foreign = torch.zeros((5, foreign_width))

    with pytest.raises(ValueError, match="Unexpected features_specular width"):
        MixtureOfGaussians._with_carrier_tail(model, foreign)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Native Hermite gradient validation requires CUDA.",
)
def test_native_hermite_gradient_matches_finite_difference() -> None:
    """Native backward matches a radiance-coefficient finite difference."""
    conf = _native_conf()
    plugin = setup_3dgut(conf)
    conf_dict = OmegaConf.to_container(conf)
    device = torch.device("cuda")
    height = 16
    width = 16
    particle_count = 32
    torch.manual_seed(42)
    particle_density = torch.zeros(
        (particle_count, 12),
        device=device,
    )
    particle_density[:, :3] = torch.randn(
        (particle_count, 3),
        device=device,
    ) * 0.3
    particle_density[:, 3] = 10.0
    particle_density[:, 4] = 1.0
    particle_density[:, 8:11] = 0.15
    particle_radiance = torch.zeros(
        (particle_count, particle_radiance_width(conf)),
        device=device,
    )
    ray_origin = torch.zeros(
        (1, height, width, 3),
        device=device,
    )
    ray_origin[..., 2] = 2.0
    ray_direction = torch.zeros(
        (1, height, width, 3),
        device=device,
    )
    for y in range(height):
        for x in range(width):
            direction = torch.tensor(
                [
                    (x - width / 2) / width,
                    (y - height / 2) / height,
                    -1.0,
                ],
                device=device,
            )
            ray_direction[0, y, x] = direction / direction.norm()
    ray_time = torch.zeros(
        (1, height, width, 1),
        dtype=torch.long,
        device=device,
    )
    sensor_params = plugin.fromOpenCVPinholeCameraModelParameters(
        [width, height],
        plugin.ShutterType.GLOBAL,
        [width / 2.0, height / 2.0],
        [width / 2.0, width / 2.0],
        [0.0] * 6,
        [0.0, 0.0],
        [0.0] * 4,
    )
    pose = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        device=device,
    )

    def render(radiance: torch.Tensor) -> torch.Tensor:
        raster = plugin.SplatRaster(conf_dict)
        return raster.trace(
            0,
            0,
            particle_density,
            radiance,
            ray_origin,
            ray_direction,
            ray_time,
            sensor_params,
            0,
            0,
            pose,
            pose,
        )[0]

    raster = plugin.SplatRaster(conf_dict)
    result = raster.trace(
        0,
        0,
        particle_density,
        particle_radiance,
        ray_origin,
        ray_direction,
        ray_time,
        sensor_params,
        0,
        0,
        pose,
        pose,
    )
    radiance_output_gradient = torch.zeros_like(result[0])
    radiance_output_gradient[..., :3] = 1.0
    density_gradient, radiance_gradient, _, _ = raster.trace_bwd(
        0,
        0,
        particle_density,
        particle_radiance,
        ray_origin,
        ray_direction,
        ray_time,
        sensor_params,
        0,
        0,
        pose,
        pose,
        result[0],
        radiance_output_gradient,
        result[1],
        torch.zeros_like(result[1]),
    )
    assert torch.isfinite(density_gradient).all()
    carrier_gradient = radiance_gradient[:, 48:54]
    assert torch.isfinite(carrier_gradient).all()
    assert bool((carrier_gradient != 0).any())
    flat_index = int(carrier_gradient.abs().argmax())
    particle_index = flat_index // 6
    carrier_index = flat_index % 6 + 48
    epsilon = 1.0e-3
    plus = particle_radiance.clone()
    minus = particle_radiance.clone()
    plus[particle_index, carrier_index] += epsilon
    minus[particle_index, carrier_index] -= epsilon
    finite_difference = (
        render(plus)[..., :3].sum() - render(minus)[..., :3].sum()
    ) / (2.0 * epsilon)
    torch.testing.assert_close(
        radiance_gradient[particle_index, carrier_index],
        finite_difference,
        rtol=0.02,
        atol=1.0e-4,
    )
