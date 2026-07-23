"""Regression tests for public GPU dynamics compatibility exports."""

from particula.gpu import dynamics, properties
from particula.gpu.dynamics import diffusion_coefficient_wp


def test_diffusion_coefficient_is_a_properties_compatibility_export() -> None:
    """Keep the public dynamics import path for the property helper."""
    assert "diffusion_coefficient_wp" in dynamics.__all__
    assert diffusion_coefficient_wp is properties.diffusion_coefficient_wp
