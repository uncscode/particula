"""Tests for coagulation strategy helper utilities."""

import numpy as np
import pytest
from particula.dynamics.coagulation.coagulation_strategy import (
    coagulation_strategy_abc,
)


def test_resolve_radius_indices_unique_match() -> None:
    """Unique radius lookup should resolve indices deterministically."""
    particle_radius = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    lookup_values = np.array([1.0, 3.0], dtype=np.float64)

    indices = coagulation_strategy_abc._resolve_radius_indices(
        particle_radius=particle_radius,
        lookup_values=lookup_values,
    )

    np.testing.assert_array_equal(indices, np.array([0, 2], dtype=np.int64))


def test_resolve_radius_indices_missing_match() -> None:
    """Missing radius values should raise a clear error."""
    particle_radius = np.array([1.0, 2.0], dtype=np.float64)
    lookup_values = np.array([4.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Direct kernel lookup failed"):
        coagulation_strategy_abc._resolve_radius_indices(
            particle_radius=particle_radius,
            lookup_values=lookup_values,
        )


def test_resolve_radius_indices_ambiguous_match() -> None:
    """Duplicate radii should raise an ambiguity error."""
    particle_radius = np.array([1.0, 2.0, 2.0], dtype=np.float64)
    lookup_values = np.array([2.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Direct kernel lookup ambiguous"):
        coagulation_strategy_abc._resolve_radius_indices(
            particle_radius=particle_radius,
            lookup_values=lookup_values,
        )


def test_build_direct_kernel_index_func_uses_indexed_values() -> None:
    """Direct kernel should use indices for deterministic values."""
    particle_radius = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    particle_mass = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    particle_charge = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    def _kernel_builder(
        particle_radius: np.ndarray,
        particle_mass: np.ndarray,
        particle_charge: np.ndarray,
        temperature: float,
        pressure: float,
    ) -> np.ndarray:
        _ = temperature, pressure
        value = particle_radius[0] * 10 + particle_mass[1] + particle_charge[1]
        return np.array([[0.0, value], [value, 0.0]], dtype=np.float64)

    kernel_func = coagulation_strategy_abc._build_direct_kernel_index_func(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        kernel_builder=_kernel_builder,
        temperature=298.15,
        pressure=101325.0,
    )

    result = kernel_func(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    assert result.shape == (1,)
    assert result[0] == pytest.approx(31.0)
