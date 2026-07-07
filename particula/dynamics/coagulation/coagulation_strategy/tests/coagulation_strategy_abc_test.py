"""Tests for coagulation strategy helper utilities."""

import numpy as np
import pytest
from numpy.typing import NDArray
from particula.dynamics.coagulation.coagulation_strategy import (
    coagulation_strategy_abc,
)
from particula.particles import PresetParticleRadiusBuilder
from particula.particles.particle_data import ParticleData


class DummyCoagulationStrategy(coagulation_strategy_abc.CoagulationStrategyABC):
    """Minimal strategy for testing particle-resolved updates."""

    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return zeros for the dimensionless kernel in tests."""
        return np.zeros_like(diffusive_knudsen)

    def kernel(
        self,
        particle: coagulation_strategy_abc.ParticleLike,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Return a constant kernel matrix matching the input size."""
        _ = temperature, pressure
        particle_radius = coagulation_strategy_abc._get_radius(particle)
        size = particle_radius.size
        return np.ones((size, size), dtype=np.float64)


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


def test_get_mean_effective_density_particle_data_filters_zeros() -> None:
    """Mean effective density should ignore zero-mass particles."""
    data = ParticleData(
        masses=np.array([[[0.0], [1e-18], [2e-18]]]),
        concentration=np.array([[0.0, 1.0, 1.0]]),
        charge=np.zeros((1, 3)),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    expected = 1000.0
    result = coagulation_strategy_abc._get_mean_effective_density(data)
    assert result == pytest.approx(expected)


def test_get_mean_effective_density_particle_data_all_zero_returns_zero() -> (
    None
):
    """Mean effective density should be zero when no positive mass exists."""
    data = ParticleData(
        masses=np.array([[[0.0], [0.0]]]),
        concentration=np.array([[0.0, 0.0]]),
        charge=np.zeros((1, 2)),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )

    assert coagulation_strategy_abc._get_mean_effective_density(data) == 0.0


def test_is_particle_data_type_guard() -> None:
    """Type guard should discriminate ParticleData from representations."""
    data = ParticleData(
        masses=np.array([[[1e-18]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    representation = PresetParticleRadiusBuilder().build()

    assert coagulation_strategy_abc._is_particle_data(data) is True
    assert coagulation_strategy_abc._is_particle_data(representation) is False


def test_unwrap_particle_returns_same_instance() -> None:
    """Unwrap helper should return the original particle instance."""
    data = ParticleData(
        masses=np.array([[[1e-18]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    representation = PresetParticleRadiusBuilder().build()

    assert coagulation_strategy_abc._unwrap_particle(data) is data
    assert (
        coagulation_strategy_abc._unwrap_particle(representation)
        is representation
    )


def test_require_single_box_particle_data_accepts_single_box() -> None:
    """Single-box ParticleData should pass the shared validation helper."""
    data = ParticleData(
        masses=np.array([[[1e-18]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )

    result = coagulation_strategy_abc._require_single_box_particle_data(data)

    assert result is data


def test_require_single_box_particle_data_rejects_multi_box() -> None:
    """Multi-box ParticleData should raise a clear validation error."""
    data = ParticleData(
        masses=np.array([[[1e-18]], [[2e-18]]]),
        concentration=np.array([[1.0], [2.0]]),
        charge=np.array([[0.0], [1.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0, 2.0]),
    )

    with pytest.raises(ValueError, match=r"n_boxes=1.*n_boxes=2"):
        coagulation_strategy_abc._require_single_box_particle_data(data)


def test_getters_particle_data_match_arrays() -> None:
    """Adapter helpers should mirror ParticleData arrays."""
    data = ParticleData(
        masses=np.array([[[1e-18, 2e-18], [2e-18, 3e-18]]]),
        concentration=np.array([[1.0, 2.0]]),
        charge=np.array([[0.0, 1.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )

    np.testing.assert_allclose(
        coagulation_strategy_abc._get_radius(data),
        data.radii[0],
    )
    np.testing.assert_allclose(
        coagulation_strategy_abc._get_mass(data),
        data.total_mass[0],
    )
    np.testing.assert_allclose(
        coagulation_strategy_abc._get_concentration(data),
        data.concentration[0],
    )
    charge_array = coagulation_strategy_abc._get_charge(data)
    assert charge_array is not None
    assert charge_array.shape == data.charge[0].shape
    assert charge_array.tolist() == data.charge[0].tolist()
    np.testing.assert_allclose(
        coagulation_strategy_abc._get_effective_density(data),
        data.effective_density[0],
    )
    assert coagulation_strategy_abc._get_volume(data) == pytest.approx(
        float(data.volume[0])
    )


def test_getters_particle_data_reject_multi_box_inputs() -> None:
    """Adapter helpers should reject unsupported multi-box ParticleData."""
    data = ParticleData(
        masses=np.array(
            [
                [[1e-18, 2e-18], [2e-18, 3e-18]],
                [[9e-18, 8e-18], [7e-18, 6e-18]],
            ]
        ),
        concentration=np.array([[1.0, 2.0], [10.0, 20.0]]),
        charge=np.array([[0.0, 1.0], [4.0, 5.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0, 9.0]),
    )

    getters = (
        coagulation_strategy_abc._get_radius,
        coagulation_strategy_abc._get_mass,
        coagulation_strategy_abc._get_concentration,
        coagulation_strategy_abc._get_charge,
        coagulation_strategy_abc._get_volume,
        coagulation_strategy_abc._get_effective_density,
        coagulation_strategy_abc._get_mean_effective_density,
    )

    for getter in getters:
        with pytest.raises(ValueError, match=r"n_boxes=1.*n_boxes=2"):
            getter(data)


def test_getters_particle_representation_match_methods() -> None:
    """Adapter helpers should mirror ParticleRepresentation accessors."""
    representation = PresetParticleRadiusBuilder().build()

    np.testing.assert_allclose(
        coagulation_strategy_abc._get_radius(representation),
        representation.get_radius(),
    )
    np.testing.assert_allclose(
        coagulation_strategy_abc._get_mass(representation),
        representation.get_mass(),
    )
    np.testing.assert_allclose(
        coagulation_strategy_abc._get_concentration(representation),
        representation.concentration,
    )
    np.testing.assert_allclose(
        coagulation_strategy_abc._get_effective_density(representation),
        representation.get_effective_density(),
    )
    assert coagulation_strategy_abc._get_volume(
        representation
    ) == pytest.approx(representation.get_volume())


def test_get_charge_representation_none_when_unset() -> None:
    """Charge helper should mirror representation-provided charge values."""
    representation = PresetParticleRadiusBuilder().build()

    charge = representation.get_charge()
    result = coagulation_strategy_abc._get_charge(representation)
    if charge is None:
        assert result is None
    else:
        np.testing.assert_array_equal(result, charge)


def test_get_particle_resolved_kernel_radius_data_uses_bin_radius() -> None:
    """Helper should return provided bin radius unchanged."""
    data = ParticleData(
        masses=np.array([[[1e-18], [2e-18]]]),
        concentration=np.array([[1.0, 1.0]]),
        charge=np.zeros((1, 2)),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    bin_radius = np.array([1e-8, 2e-8], dtype=np.float64)
    result = coagulation_strategy_abc._get_particle_resolved_kernel_radius_data(
        data=data,
        bin_radius=bin_radius,
    )
    np.testing.assert_array_equal(result, bin_radius)


def test_get_particle_resolved_kernel_radius_data_raises_on_all_zero() -> None:
    """All-zero radii should raise a ValueError."""
    data = ParticleData(
        masses=np.zeros((1, 2, 1)),
        concentration=np.zeros((1, 2)),
        charge=np.zeros((1, 2)),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="Particle radius must be finite"):
        coagulation_strategy_abc._get_particle_resolved_kernel_radius_data(
            data=data,
        )


def test_get_particle_resolved_kernel_radius_data_raises_on_nonfinite() -> None:
    """Non-finite radii should raise a ValueError."""
    data = ParticleData(
        masses=np.array([[[np.inf]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="Particle radius must be finite"):
        coagulation_strategy_abc._get_particle_resolved_kernel_radius_data(
            data=data,
        )


def test_get_particle_resolved_kernel_radius_data_generates_requested_bins() -> (
    None
):
    """Radius helper should generate explicit or inferred logarithmic bins."""
    data = ParticleData(
        masses=np.array([[[1e-18], [8e-18], [27e-18]]]),
        concentration=np.array([[1.0, 1.0, 1.0]]),
        charge=np.array([[0.0, 0.0, 0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )

    fixed_bins = (
        coagulation_strategy_abc._get_particle_resolved_kernel_radius_data(
            data=data,
            total_bins=4,
        )
    )
    inferred_bins = (
        coagulation_strategy_abc._get_particle_resolved_kernel_radius_data(
            data=data,
            bins_per_radius_decade=5,
        )
    )

    assert fixed_bins.shape == (4,)
    assert inferred_bins.ndim == 1
    assert inferred_bins.size >= 1
    assert np.all(np.diff(fixed_bins) > 0)
    assert np.all(np.diff(inferred_bins) >= 0)


def test_get_binned_kernel_data_from_particle_data_aggregates_bins() -> None:
    """Binning helper should aggregate mass and concentration by bins."""
    data = ParticleData(
        masses=np.array([[[1e-18], [2e-18], [3e-18]]]),
        concentration=np.array([[1.0, 2.0, 3.0]]),
        charge=np.array([[0.0, 1.0, 2.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    radii = data.radii[0]
    kernel_radius = np.array(
        [radii.min() * 0.8, radii.mean(), radii.max() * 1.2], dtype=np.float64
    )
    binned = (
        coagulation_strategy_abc._get_binned_kernel_data_from_particle_data(
            data=data,
            kernel_radius=kernel_radius,
        )
    )
    bin_indexes = np.digitize(radii, kernel_radius)
    num_bins = kernel_radius.size
    valid_mask = bin_indexes < num_bins
    bin_indexes = bin_indexes[valid_mask]
    masses = data.masses[0][valid_mask]
    concentration = data.concentration[0][valid_mask]
    charge = data.charge[0][valid_mask]
    counts = np.bincount(bin_indexes, minlength=num_bins)
    concentration_sum = np.bincount(
        bin_indexes, weights=concentration, minlength=num_bins
    )
    new_masses = np.zeros((num_bins, data.n_species), dtype=np.float64)
    summed = np.bincount(bin_indexes, weights=masses[:, 0], minlength=num_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_mass = summed / counts
    mean_mass[counts == 0] = np.nan
    new_masses[:, 0] = mean_mass
    new_charge = np.full(num_bins, np.nan, dtype=np.float64)
    for bin_index in np.flatnonzero(counts):
        mask = bin_indexes == bin_index
        new_charge[bin_index] = np.median(charge[mask])
    new_charge = np.where(np.isnan(new_charge), 0, new_charge)
    new_concentration = np.where(
        np.isnan(concentration_sum), 0, concentration_sum
    )
    mask_nan_zeros = np.isnan(new_masses) | (new_masses == 0)
    mask_nan_zeros = ~np.any(mask_nan_zeros, axis=1)
    np.testing.assert_allclose(
        binned.masses[0], new_masses[mask_nan_zeros], rtol=1e-12
    )
    np.testing.assert_allclose(
        binned.concentration[0], new_concentration[mask_nan_zeros], rtol=1e-12
    )
    np.testing.assert_allclose(
        binned.charge[0], new_charge[mask_nan_zeros], rtol=1e-12
    )


def test_get_binned_kernel_data_from_particle_data_returns_empty_when_all_outside() -> (
    None
):
    """Binning helper should return an empty single-box container if needed."""
    data = ParticleData(
        masses=np.array([[[1e-18], [2e-18]]]),
        concentration=np.array([[1.0, 2.0]]),
        charge=np.array([[0.0, 1.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    kernel_radius = np.array([1e-12], dtype=np.float64)

    binned = (
        coagulation_strategy_abc._get_binned_kernel_data_from_particle_data(
            data=data,
            kernel_radius=kernel_radius,
        )
    )

    assert binned.masses.shape == (1, 0, 1)
    assert binned.concentration.shape == (1, 0)
    assert binned.charge.shape == (1, 0)
    np.testing.assert_allclose(binned.volume, np.array([1.0]))


def test_strategy_init_rejects_invalid_distribution_type() -> None:
    """Invalid distribution types should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid distribution type"):
        DummyCoagulationStrategy(distribution_type="invalid")


def test_loss_gain_and_net_rate_particle_data_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discrete and continuous ParticleData paths should use shared helpers."""
    data = ParticleData(
        masses=np.array([[[1e-18], [2e-18]]]),
        concentration=np.array([[3.0, 4.0]]),
        charge=np.array([[0.0, 1.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    kernel = np.array([[1.0, 0.5], [0.5, 2.0]], dtype=np.float64)

    discrete = DummyCoagulationStrategy(distribution_type="discrete")
    continuous = DummyCoagulationStrategy(distribution_type="continuous_pdf")
    monkeypatch.setattr(
        coagulation_strategy_abc.coagulation_rate,
        "get_coagulation_loss_rate_discrete",
        lambda concentration, kernel: np.array([10.0, 11.0]),
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.coagulation_rate,
        "get_coagulation_gain_rate_discrete",
        lambda radius, concentration, kernel: np.array([12.0, 13.0]),
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.coagulation_rate,
        "get_coagulation_loss_rate_continuous",
        lambda radius, concentration, kernel: np.array([14.0, 15.0]),
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.coagulation_rate,
        "get_coagulation_gain_rate_continuous",
        lambda radius, concentration, kernel: np.array([16.0, 17.0]),
    )

    np.testing.assert_allclose(discrete.loss_rate(data, kernel), [10.0, 11.0])
    np.testing.assert_allclose(discrete.gain_rate(data, kernel), [12.0, 13.0])
    np.testing.assert_allclose(continuous.loss_rate(data, kernel), [14.0, 15.0])
    np.testing.assert_allclose(continuous.gain_rate(data, kernel), [16.0, 17.0])

    net_strategy = DummyCoagulationStrategy(distribution_type="discrete")
    expected_net = np.array([2.0, 2.0])
    np.testing.assert_allclose(
        net_strategy.net_rate(data, temperature=298.15, pressure=101325.0),
        expected_net,
    )


def test_loss_gain_rate_reject_invalid_distribution_type() -> None:
    """Invalid distribution types should still fail in loss/gain helpers."""
    data = ParticleData(
        masses=np.array([[[1e-18]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    kernel = np.ones((1, 1), dtype=np.float64)
    strategy = DummyCoagulationStrategy(distribution_type="discrete")
    strategy.distribution_type = "invalid"

    with pytest.raises(ValueError, match="Invalid distribution type"):
        strategy.loss_rate(data, kernel)
    with pytest.raises(ValueError, match="Invalid distribution type"):
        strategy.gain_rate(data, kernel)


def test_step_discrete_particle_data_updates_concentration_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discrete ParticleData step should update concentration in place."""
    data = ParticleData(
        masses=np.array([[[1e-18], [2e-18]]]),
        concentration=np.array([[3.0, 4.0]]),
        charge=np.array([[0.0, 0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="discrete")
    monkeypatch.setattr(
        strategy,
        "net_rate",
        lambda particle, temperature, pressure: np.array(
            [3.0, -3.0], dtype=np.float64
        ),
    )

    result = strategy.step(
        particle=data,
        temperature=298.15,
        pressure=101325.0,
        time_step=2.0,
    )

    assert result is data
    np.testing.assert_allclose(data.concentration[0], np.array([9.0, -2.0]))


def test_step_discrete_representation_updates_concentration_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discrete legacy representations should use add_concentration."""
    particle = PresetParticleRadiusBuilder().build()
    strategy = DummyCoagulationStrategy(distribution_type="discrete")
    monkeypatch.setattr(
        strategy,
        "net_rate",
        lambda particle, temperature, pressure: np.full_like(
            particle.concentration,
            2.0,
            dtype=np.float64,
        ),
    )
    concentration_before = particle.concentration.copy()

    result = strategy.step(
        particle=particle,
        temperature=298.15,
        pressure=101325.0,
        time_step=3.0,
    )

    assert result is particle
    np.testing.assert_allclose(
        particle.concentration,
        concentration_before + np.full_like(concentration_before, 6.0),
    )


def test_step_particle_resolved_empty_loss_gain_indices_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Particle-resolved step should no-op when no collision pairs exist."""
    data = ParticleData(
        masses=np.array([[[1e-18, 1e-18], [2e-18, 2e-18]]]),
        concentration=np.array([[1.0, 1.0]]),
        charge=np.array([[0.0, 1.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="particle_resolved")
    strategy.particle_resolved_radius = np.array(
        [data.radii[0].min() * 0.8, data.radii[0].max() * 1.2],
        dtype=np.float64,
    )

    def _fake_step_func(**_: object) -> NDArray[np.int64]:
        return np.empty((0, 2), dtype=np.int64)

    monkeypatch.setattr(
        coagulation_strategy_abc.particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        _fake_step_func,
    )
    masses_before = data.masses.copy()
    concentration_before = data.concentration.copy()
    charge_before = data.charge.copy()

    result = strategy.step(
        particle=data,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )

    assert result is data
    np.testing.assert_allclose(data.masses, masses_before)
    np.testing.assert_allclose(data.concentration, concentration_before)
    np.testing.assert_allclose(data.charge, charge_before)


def test_step_particle_resolved_zero_kernel_radius_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Particle-resolved step should reject zero-valued kernel radii."""
    data = ParticleData(
        masses=np.array([[[1e-18, 1e-18], [2e-18, 2e-18]]]),
        concentration=np.array([[1.0, 1.0]]),
        charge=np.array([[0.0, 1.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    zero_kernel_data = ParticleData(
        masses=np.zeros((1, 1, 2), dtype=np.float64),
        concentration=np.ones((1, 1), dtype=np.float64),
        charge=np.zeros((1, 1), dtype=np.float64),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="particle_resolved")
    strategy.particle_resolved_radius = np.array([1e-9, 2e-9], dtype=np.float64)

    monkeypatch.setattr(
        coagulation_strategy_abc,
        "_get_binned_kernel_data_from_particle_data",
        lambda data, kernel_radius: zero_kernel_data,
    )

    with pytest.raises(ValueError, match="kernel radius cannot be zero"):
        strategy.step(
            particle=data,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_step_rejects_invalid_distribution_type() -> None:
    """Invalid distribution types should fail in step()."""
    data = ParticleData(
        masses=np.array([[[1e-18]]]),
        concentration=np.array([[1.0]]),
        charge=np.array([[0.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="discrete")
    strategy.distribution_type = "invalid"

    with pytest.raises(ValueError, match="Invalid distribution type"):
        strategy.step(
            data, temperature=298.15, pressure=101325.0, time_step=1.0
        )


def test_diffusive_knudsen_and_friction_factor_particle_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diffusive Knudsen path should compose friction-factor helpers."""
    data = ParticleData(
        masses=np.array([[[1e-18], [2e-18]]]),
        concentration=np.array([[1.0, 2.0]]),
        charge=np.array([[0.0, 1.0]]),
        density=np.array([1000.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="discrete")

    monkeypatch.setattr(
        coagulation_strategy_abc.gas,
        "get_dynamic_viscosity",
        lambda temperature: 1.8e-5,
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.gas,
        "get_molecule_mean_free_path",
        lambda temperature, pressure, dynamic_viscosity: 6.5e-8,
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.particles,
        "get_knudsen_number",
        lambda mean_free_path, particle_radius: np.full_like(
            particle_radius, 2.0
        ),
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.particles,
        "get_cunningham_slip_correction",
        lambda knudsen_number: np.full_like(knudsen_number, 3.0),
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.particles,
        "get_friction_factor",
        lambda particle_radius,
        dynamic_viscosity,
        slip_correction: np.full_like(particle_radius, 4.0),
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.particles,
        "get_diffusive_knudsen_number",
        lambda particle_radius,
        particle_mass,
        friction_factor,
        coulomb_potential_ratio,
        temperature: np.full_like(particle_radius, 5.0),
    )
    monkeypatch.setattr(
        coagulation_strategy_abc.particles,
        "get_coulomb_enhancement_ratio",
        lambda particle_radius, charge, temperature: np.full_like(
            particle_radius, 6.0
        ),
    )

    np.testing.assert_allclose(
        strategy.friction_factor(data, temperature=298.15, pressure=101325.0),
        np.array([4.0, 4.0]),
    )
    np.testing.assert_allclose(
        strategy.diffusive_knudsen(data, temperature=298.15, pressure=101325.0),
        np.array([5.0, 5.0]),
    )


def test_coulomb_potential_ratio_uses_zero_charge_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy representations with no charge should use zeros."""
    representation = PresetParticleRadiusBuilder().build()
    monkeypatch.setattr(representation, "get_charge", lambda: None)
    strategy = DummyCoagulationStrategy(distribution_type="discrete")

    def _fake_coulomb(
        particle_radius: NDArray[np.float64],
        charge: NDArray[np.float64],
        temperature: float,
    ) -> NDArray[np.float64]:
        _ = temperature
        np.testing.assert_allclose(charge, np.zeros_like(particle_radius))
        return np.full_like(particle_radius, 7.0)

    monkeypatch.setattr(
        coagulation_strategy_abc.particles,
        "get_coulomb_enhancement_ratio",
        _fake_coulomb,
    )

    np.testing.assert_allclose(
        strategy.coulomb_potential_ratio(representation, temperature=298.15),
        np.full_like(representation.get_radius(), 7.0),
    )


def test_step_particle_data_particle_resolved_updates_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ParticleData particle-resolved step should update arrays in place."""
    data = ParticleData(
        masses=np.array([[[1e-18, 1e-18], [2e-18, 2e-18], [3e-18, 3e-18]]]),
        concentration=np.array([[1.0, 1.0, 1.0]]),
        charge=np.array([[0.0, 1.0, -1.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="particle_resolved")
    strategy.particle_resolved_radius = np.array(
        [data.radii[0].min() * 0.8, data.radii[0].max() * 1.2],
        dtype=np.float64,
    )

    def _fake_step_func(**_: object) -> NDArray[np.int64]:
        return np.array([[0, 1]], dtype=np.int64)

    monkeypatch.setattr(
        coagulation_strategy_abc.particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        _fake_step_func,
    )
    strategy.step(
        particle=data,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )
    expected_masses = np.array(
        [[0.0, 0.0], [3e-18, 3e-18], [3e-18, 3e-18]], dtype=np.float64
    )
    expected_concentration = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    expected_charge = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    np.testing.assert_allclose(data.masses[0], expected_masses)
    np.testing.assert_allclose(data.concentration[0], expected_concentration)
    np.testing.assert_allclose(data.charge[0], expected_charge)


def test_step_particle_data_multi_box_raises_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Particle-resolved step should reject multi-box data before mutation."""
    data = ParticleData(
        masses=np.array(
            [
                [[1e-18, 1e-18], [2e-18, 2e-18], [3e-18, 3e-18]],
                [[9e-18, 8e-18], [7e-18, 6e-18], [5e-18, 4e-18]],
            ]
        ),
        concentration=np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
        charge=np.array([[0.0, 1.0, -1.0], [4.0, 5.0, 6.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0, 5.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="particle_resolved")
    strategy.particle_resolved_radius = np.array(
        [data.radii[0].min() * 0.8, data.radii[0].max() * 1.2],
        dtype=np.float64,
    )

    def _fake_step_func(**_: object) -> NDArray[np.int64]:
        return np.array([[0, 1]], dtype=np.int64)

    monkeypatch.setattr(
        coagulation_strategy_abc.particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        _fake_step_func,
    )
    masses_before = data.masses.copy()
    concentration_before = data.concentration.copy()
    charge_before = data.charge.copy()

    with pytest.raises(ValueError, match=r"n_boxes=1.*n_boxes=2"):
        strategy.step(
            particle=data,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )

    np.testing.assert_allclose(data.masses, masses_before)
    np.testing.assert_allclose(data.concentration, concentration_before)
    np.testing.assert_allclose(data.charge, charge_before)


def test_step_particle_data_zero_concentration_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero concentration should short-circuit particle-resolved updates."""
    data = ParticleData(
        masses=np.array([[[1e-18, 1e-18], [2e-18, 2e-18]]]),
        concentration=np.array([[0.0, 0.0]]),
        charge=np.array([[0.0, 0.0]]),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="particle_resolved")
    strategy.particle_resolved_radius = np.array(
        [data.radii[0].min() * 0.8, data.radii[0].max() * 1.2],
        dtype=np.float64,
    )

    def _fake_step_func(**_: object) -> NDArray[np.int64]:
        return np.array([[0, 1]], dtype=np.int64)

    monkeypatch.setattr(
        coagulation_strategy_abc.particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        _fake_step_func,
    )
    masses_before = data.masses.copy()
    concentration_before = data.concentration.copy()
    charge_before = data.charge.copy()
    strategy.step(
        particle=data,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )
    np.testing.assert_allclose(data.masses, masses_before)
    np.testing.assert_allclose(data.concentration, concentration_before)
    np.testing.assert_allclose(data.charge, charge_before)


def test_step_particle_data_empty_particle_resolved_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty single-box particle-resolved data should remain supported."""
    data = ParticleData(
        masses=np.zeros((1, 0, 2), dtype=np.float64),
        concentration=np.zeros((1, 0), dtype=np.float64),
        charge=np.zeros((1, 0), dtype=np.float64),
        density=np.array([1000.0, 1200.0]),
        volume=np.array([1.0]),
    )
    strategy = DummyCoagulationStrategy(distribution_type="particle_resolved")
    strategy.particle_resolved_radius = np.array([1e-9, 2e-9], dtype=np.float64)

    def _unexpected_step_func(**_: object) -> NDArray[np.int64]:
        raise AssertionError(
            "empty particle-resolved inputs should short-circuit"
        )

    monkeypatch.setattr(
        coagulation_strategy_abc.particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        _unexpected_step_func,
    )
    masses_before = data.masses.copy()
    concentration_before = data.concentration.copy()
    charge_before = data.charge.copy()

    result = strategy.step(
        particle=data,
        temperature=298.15,
        pressure=101325.0,
        time_step=1.0,
    )

    assert result is data
    np.testing.assert_allclose(data.masses, masses_before)
    np.testing.assert_allclose(data.concentration, concentration_before)
    np.testing.assert_allclose(data.charge, charge_before)
