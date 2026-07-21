"""Tests for native particle-radiance feature packing."""

import pytest
from omegaconf import OmegaConf

from threedgut_tracer.setup_3dgut import particle_radiance_width


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


def test_sh_layout_retains_original_width() -> None:
    """Band-three SH retains its 16 RGB coefficient vectors."""
    assert particle_radiance_width(_conf()) == 48


def test_gabor_layout_includes_carrier_coefficients() -> None:
    """Three Gabor terms add six RGB coefficient vectors."""
    assert particle_radiance_width(
        _conf(use_gabor_carrier=True)
    ) == 66


def test_hermite_layout_includes_affine_coefficients() -> None:
    """Two Gaussian-Hermite terms add two RGB coefficient vectors."""
    assert particle_radiance_width(
        _conf(use_hermite_carrier=True)
    ) == 54


def test_siren_layout_includes_network_coefficients() -> None:
    """The packed six-unit SIREN adds 21 RGB coefficient vectors."""
    assert particle_radiance_width(
        _conf(use_siren_carrier=True)
    ) == 111


@pytest.mark.parametrize(
    ("use_gabor_carrier", "use_hermite_carrier", "use_siren_carrier"),
    (
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ),
)
def test_carrier_layouts_are_mutually_exclusive(
    use_gabor_carrier: bool,
    use_hermite_carrier: bool,
    use_siren_carrier: bool,
) -> None:
    """A native build cannot combine two incompatible carrier layouts."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        particle_radiance_width(
            _conf(
                use_gabor_carrier=use_gabor_carrier,
                use_hermite_carrier=use_hermite_carrier,
                use_siren_carrier=use_siren_carrier,
            )
        )
