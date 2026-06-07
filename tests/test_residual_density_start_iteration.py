"""Tests for residual-density controller activation timing."""

from threedgrut.strategy.gs import GSStrategy


def test_residual_density_control_respects_start_iteration() -> None:
    """Late-start residual density control stays inactive before its start."""
    strategy = GSStrategy.__new__(GSStrategy)
    strategy.residual_density_control = {
        "enabled": True,
        "start_iteration": 15000,
    }

    assert not strategy.should_apply_residual_density_control(14999)
    assert strategy.should_apply_residual_density_control(15000)


def test_disabled_residual_density_control_stays_inactive() -> None:
    """Disabled residual density control ignores the timing gate."""
    strategy = GSStrategy.__new__(GSStrategy)
    strategy.residual_density_control = {
        "enabled": False,
        "start_iteration": 15000,
    }

    assert not strategy.should_apply_residual_density_control(20000)
