import numpy as np

from threedgrut.datasets.utils import compute_fisheye_max_angle

# Real Insta360 X5 lens (KB4 re-fit at 3840x3840). The legacy
# `max_radius / focal` estimate returns 152.5 deg here, far past the KB4
# forward map's 116.7 deg turning point, which poisons ncore's backward
# polynomial and shrinks/veils the rendered fisheye periphery.
_X5_IMAGE = np.array([3840.0, 3840.0])
_X5_PP = np.array([1920.957142857143, 1916.2857142857144])
_X5_FOCAL = np.array([1021.3947697671305, 1021.2447527005324])
_X5_K = np.array(
    [0.016213675267435838, 0.026441762778483594, -0.0098534478683585097, 0.00054329697918768121]
)


def _kb4_forward_px(theta_deg: float, focal: float, k: np.ndarray) -> float:
    t = np.deg2rad(theta_deg)
    return float(focal * (t + k[0] * t**3 + k[1] * t**5 + k[2] * t**7 + k[3] * t**9))


def test_x5_max_angle_stays_inside_the_kb4_monotonic_domain() -> None:
    max_angle_deg = np.rad2deg(
        compute_fisheye_max_angle(_X5_IMAGE, _X5_PP, _X5_FOCAL, _X5_K)
    )
    # The image circle (keep-mask radius ~1868 px) subtends ~98 deg; the KB4
    # forward map peaks at 116.7 deg. A correct max_angle covers the content
    # yet stays below the turning point — never the legacy 152.5 deg.
    assert 98.0 < max_angle_deg < 116.7
    assert max_angle_deg < 152.0


def test_monotonic_lens_keeps_its_true_image_fov() -> None:
    # A purely equidistant lens never folds; max_angle must equal the angle
    # subtended by the farthest pixel, not an artificial cap.
    image = np.array([1920.0, 1920.0])
    pp = np.array([960.0, 960.0])
    focal = np.array([1000.0, 1000.0])
    k = np.zeros(4)
    max_angle = compute_fisheye_max_angle(image, pp, focal, k)
    corner_norm = float(np.linalg.norm((np.array([1920.0, 1920.0]) - pp) / focal))
    assert np.isclose(max_angle, corner_norm, atol=1e-6)


def test_folding_lens_is_capped_below_its_turning_point() -> None:
    # A lens whose KB4 folds well inside the frame must be capped below the
    # turning point so the backward-poly inversion never sees the fold.
    image = np.array([1920.0, 1920.0])
    pp = np.array([960.0, 960.0])
    focal = np.array([1000.0, 1000.0])
    k = np.array([0.05, 0.0, -0.02, 0.0])
    roots = np.roots([9 * k[3], 7 * k[2], 5 * k[1], 3 * k[0], 1.0])
    peak = min(np.sqrt(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 1e-12)
    max_angle = compute_fisheye_max_angle(image, pp, focal, k)
    assert max_angle < peak
    assert np.isclose(max_angle, 0.95 * peak, atol=1e-3)


def test_max_angle_radius_round_trips_through_the_kb4_forward() -> None:
    # The radius at max_angle must be a real forward radius (monotonic branch),
    # i.e. recoverable by the equidistant lower bound r >= focal * theta.
    max_angle = compute_fisheye_max_angle(_X5_IMAGE, _X5_PP, _X5_FOCAL, _X5_K)
    r_at_max = _kb4_forward_px(np.rad2deg(max_angle), float(_X5_FOCAL[0]), _X5_K)
    assert r_at_max > float(_X5_FOCAL[0]) * max_angle * 0.5
    # And it must exceed the keep-mask circle radius so real content is covered.
    assert r_at_max > 1868.0
