"""CPU reference regression for the wall-deposition example."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "docs/examples/processes/wall-loss/spherical_wall_loss_theory.py"
)


def test_surface_integral_retains_the_coupling_effect():
    """Check the reviewed one-hour observable without requiring CUDA in CI."""
    spec = importlib.util.spec_from_file_location("wall_theory_example", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference, additive, cpu = module.coefficients(np.array([1e-6]), 3.6)
    survival = np.exp(-reference * 3600)
    np.testing.assert_allclose(survival, [0.6841387762375624], rtol=1e-8)
    np.testing.assert_allclose(cpu, reference, rtol=0.005)
    assert 0.10 < float(survival[0] - np.exp(-additive[0] * 3600)) < 0.103
