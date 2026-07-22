"""Numerical-contract tests for chamber dilution helpers."""

import numpy as np
import numpy.testing as npt
import particula.dynamics as dynamics
import pytest
from particula.aerosol import Aerosol
from particula.dynamics.dilution import (
    dilute_aerosol,
    get_dilution_rate,
    get_dilution_step,
    get_volume_dilution_coefficient,
)
from particula.gas.atmosphere import Atmosphere
from particula.gas.species import GasSpecies
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.distribution_strategies import (
    MassBasedMovingBin,
    SpeciatedMassMovingBin,
)
from particula.particles.representation import ParticleRepresentation
from particula.particles.surface_strategies import SurfaceStrategyVolume


@pytest.mark.parametrize(
    "volume, flow_rate",
    [
        (10.0, 2.0),
        (np.array([10.0, 20.0]), np.array([2.0, 8.0])),
        (np.array([[10.0], [20.0]]), np.array([2.0, 8.0])),
    ],
)
def test_volume_dilution_coefficient_equation_and_broadcasting(
    volume, flow_rate
):
    """Coefficient follows Q / V with scalar and array return conventions."""
    result = get_volume_dilution_coefficient(volume, flow_rate)
    expected = np.asarray(flow_rate, dtype=np.float64) / np.asarray(
        volume,
        dtype=np.float64,
    )

    npt.assert_allclose(result, expected)
    if np.ndim(volume) == 0 and np.ndim(flow_rate) == 0:
        assert np.isscalar(result)
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape


@pytest.mark.parametrize(
    "coefficient, concentration",
    [
        (0.5, 10.0),
        (np.array([0.5, 1.0]), np.array([10.0, 4.0])),
        (np.array([[0.5], [1.0]]), np.array([10.0, 4.0])),
    ],
)
def test_dilution_rate_equation_and_broadcasting(coefficient, concentration):
    """Instantaneous rate follows -alpha c with broadcast output shape."""
    result = get_dilution_rate(coefficient, concentration)
    expected = -np.asarray(coefficient, dtype=np.float64) * np.asarray(
        concentration,
        dtype=np.float64,
    )

    npt.assert_allclose(result, expected)
    if np.ndim(coefficient) == 0 and np.ndim(concentration) == 0:
        assert np.isscalar(result)
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape


@pytest.mark.parametrize(
    "coefficient, concentration, time_step",
    [
        (0.5, 10.0, 2.0),
        (np.array([0.5, 1.0]), np.array([10.0, 4.0]), 2.0),
        (np.array([[0.5], [1.0]]), np.array([10.0, 4.0]), [1.0, 2.0]),
    ],
)
def test_dilution_step_equation_and_broadcasting(
    coefficient, concentration, time_step
):
    """Finite update follows the independent exact exponential expression."""
    result = get_dilution_step(coefficient, concentration, time_step)
    expected = np.asarray(concentration, dtype=np.float64) * np.exp(
        -np.asarray(coefficient, dtype=np.float64)
        * np.asarray(time_step, dtype=np.float64)
    )

    npt.assert_allclose(result, expected)
    if all(
        np.ndim(value) == 0 for value in (coefficient, concentration, time_step)
    ):
        assert np.isscalar(result)
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape


def test_dilution_step_extreme_finite_decay_is_warning_clean():
    """Finite coefficient-time overflow produces exact, finite zero decay."""
    result = get_dilution_step(
        np.finfo(np.float64).max,
        1.0,
        np.finfo(np.float64).max,
    )

    assert result == 0.0
    assert np.isfinite(result)
    assert result >= 0.0


@pytest.mark.parametrize(
    ("function", "arguments", "expected"),
    [
        (
            get_volume_dilution_coefficient,
            (np.array(10.0), 2.0),
            np.array(0.2),
        ),
        (get_dilution_rate, (np.array(0.5), 10.0), np.array(-5.0)),
        (get_dilution_step, (0.5, np.array(10.0), 2.0), np.array(10.0 / np.e)),
    ],
)
def test_dilution_helpers_preserve_zero_dimensional_array_results(
    function, arguments, expected
):
    """A zero-dimensional ndarray input produces a zero-dimensional ndarray."""
    result = function(*arguments)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    npt.assert_allclose(result, expected)


def test_dilution_rate_uint64_inputs_do_not_wrap_positive():
    """Unsigned inputs are converted before signed rate arithmetic."""
    coefficient = np.uint64(2**63)
    concentration = np.uint64(2)

    result = get_dilution_rate(coefficient, concentration)
    expected = -np.float64(coefficient) * np.float64(concentration)

    assert result <= 0.0
    npt.assert_allclose(result, expected)


def test_dilution_step_uint64_inputs_decay_without_overflow():
    """Unsigned inputs are converted before exponential step arithmetic."""
    coefficient = np.uint64(2**63)
    concentration = np.uint64(3)
    time_step = np.uint64(2)

    result = get_dilution_step(coefficient, concentration, time_step)
    expected = np.float64(concentration) * np.exp(
        -np.float64(coefficient) * np.float64(time_step)
    )

    assert np.isfinite(result)
    assert 0.0 <= result <= concentration
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (get_volume_dilution_coefficient, (0.0, 1.0), "must be positive"),
        (get_volume_dilution_coefficient, (-1.0, 1.0), "must be positive"),
        (get_volume_dilution_coefficient, (np.nan, 1.0), "must be finite"),
        (get_volume_dilution_coefficient, (np.inf, 1.0), "must be finite"),
        (get_volume_dilution_coefficient, (1.0, -1.0), "must be nonnegative"),
        (get_volume_dilution_coefficient, (1.0, np.nan), "must be finite"),
        (get_volume_dilution_coefficient, (1.0, np.inf), "must be finite"),
        (get_dilution_rate, (-1.0, 1.0), "must be nonnegative"),
        (get_dilution_rate, (np.nan, 1.0), "must be finite"),
        (get_dilution_rate, (np.inf, 1.0), "must be finite"),
        (get_dilution_rate, (1.0, -1.0), "must be nonnegative"),
        (get_dilution_rate, (1.0, np.nan), "must be finite"),
        (get_dilution_rate, (1.0, np.inf), "must be finite"),
        (get_dilution_step, (-1.0, 1.0, 1.0), "must be nonnegative"),
        (get_dilution_step, (np.nan, 1.0, 1.0), "must be finite"),
        (get_dilution_step, (np.inf, 1.0, 1.0), "must be finite"),
        (get_dilution_step, (1.0, -1.0, 1.0), "must be nonnegative"),
        (get_dilution_step, (1.0, np.nan, 1.0), "must be finite"),
        (get_dilution_step, (1.0, np.inf, 1.0), "must be finite"),
        (get_dilution_step, (1.0, 1.0, -1.0), "must be nonnegative"),
        (get_dilution_step, (1.0, 1.0, np.nan), "must be finite"),
        (get_dilution_step, (1.0, 1.0, np.inf), "must be finite"),
    ],
)
def test_dilution_helpers_reject_invalid_numeric_domains(
    function, arguments, message
):
    """Each numeric domain rejects zero where needed and nonfinite values."""
    with pytest.raises(ValueError, match=message):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (get_volume_dilution_coefficient, (np.array([1.0, np.nan]), 1.0)),
        (get_volume_dilution_coefficient, (1.0, np.array([1.0, -1.0]))),
        (get_dilution_rate, (np.array([1.0, np.inf]), 1.0)),
        (get_dilution_rate, (1.0, np.array([1.0, -1.0]))),
        (get_dilution_step, (np.array([1.0, -1.0]), 1.0, 1.0)),
        (get_dilution_step, (1.0, np.array([1.0, np.nan]), 1.0)),
        (get_dilution_step, (1.0, 1.0, np.array([1.0, np.inf]))),
    ],
)
def test_dilution_helpers_validate_every_array_element(function, arguments):
    """Invalid elements in otherwise compatible arrays fail validation."""
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments", "none_name"),
    [
        (get_volume_dilution_coefficient, (None, 1.0), "volume"),
        (get_volume_dilution_coefficient, (1.0, None), "input_flow_rate"),
        (get_dilution_rate, (None, 1.0), "coefficient"),
        (get_dilution_rate, (1.0, None), "concentration"),
        (get_dilution_step, (None, 1.0, 1.0), "coefficient"),
        (get_dilution_step, (1.0, None, 1.0), "concentration"),
        (get_dilution_step, (1.0, 1.0, None), "time_step"),
    ],
)
def test_dilution_helpers_reject_none(function, arguments, none_name):
    """Explicit None guards provide deterministic argument-specific errors."""
    with pytest.raises(
        TypeError, match=f"Argument '{none_name}' must not be None"
    ):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (get_volume_dilution_coefficient, ("one", 1.0)),
        (get_dilution_rate, (1.0, object())),
        (get_dilution_step, (1.0, "one", 1.0)),
    ],
)
def test_dilution_helpers_reject_unsupported_values(function, arguments):
    """Strings and object-valued inputs fail with TypeError."""
    with pytest.raises(TypeError):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (get_volume_dilution_coefficient, (np.ones((2, 2)), np.ones(3))),
        (get_dilution_rate, (np.ones((2, 2)), np.ones(3))),
        (get_dilution_step, (np.ones((2, 2)), np.ones(3), 1.0)),
    ],
)
def test_dilution_helpers_preflight_incompatible_shapes(function, arguments):
    """Incompatible operand shapes fail before arithmetic returns a result."""
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments", "expected"),
    [
        (
            get_volume_dilution_coefficient,
            (np.array([2.0, 3.0]), 0.0),
            [0.0, 0.0],
        ),
        (get_dilution_rate, (0.0, np.array([2.0, 3.0])), [0.0, 0.0]),
        (get_dilution_step, (1.0, np.array([0.0, 0.0]), 2.0), [0.0, 0.0]),
        (
            get_dilution_step,
            (np.array([1.0, 2.0]), [2.0, 3.0], 0.0),
            [2.0, 3.0],
        ),
    ],
)
def test_dilution_no_ops_are_exact(function, arguments, expected):
    """Zero flow, coefficient, concentration, or duration preserve exact no-ops."""
    result = function(*arguments)
    npt.assert_array_equal(result, np.asarray(expected, dtype=np.float64))


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (
            get_volume_dilution_coefficient,
            (np.array([2.0, 4.0]), np.array([1.0, 2.0])),
        ),
        (get_dilution_rate, (np.array([0.5, 1.0]), np.array([2.0, 4.0]))),
        (get_dilution_step, (np.array([0.5, 1.0]), np.array([2.0, 4.0]), 1.0)),
    ],
)
def test_dilution_helpers_do_not_mutate_successful_inputs(function, arguments):
    """Successful vectorized operations leave caller-owned arrays unchanged."""
    snapshots = [
        value.copy() for value in arguments if isinstance(value, np.ndarray)
    ]
    function(*arguments)
    for value, snapshot in zip(
        (value for value in arguments if isinstance(value, np.ndarray)),
        snapshots,
        strict=True,
    ):
        npt.assert_array_equal(value, snapshot)


def test_dilution_helpers_do_not_mutate_failed_inputs():
    """Validation and broadcast failures leave their input arrays unchanged."""
    invalid = np.array([1.0, np.nan])
    incompatible = np.ones((2, 2))
    valid = np.ones(3)
    snapshots = (invalid.copy(), incompatible.copy(), valid.copy())

    with pytest.raises(ValueError):
        get_dilution_rate(1.0, invalid)
    with pytest.raises(ValueError):
        get_dilution_step(incompatible, valid, 1.0)

    npt.assert_array_equal(invalid, snapshots[0])
    npt.assert_array_equal(incompatible, snapshots[1])
    npt.assert_array_equal(valid, snapshots[2])


def test_dilution_package_surface_remains_limited_to_existing_helpers():
    """P1 keeps the finite-step helper concrete-module-only."""
    assert (
        dynamics.get_volume_dilution_coefficient
        is get_volume_dilution_coefficient
    )
    assert dynamics.get_dilution_rate is get_dilution_rate
    assert not hasattr(dynamics, "get_dilution_step")
    assert not hasattr(dynamics, "dilute_aerosol")


def _make_aerosol(
    strategy: MassBasedMovingBin | SpeciatedMassMovingBin | None = None,
) -> Aerosol:
    """Build a container fixture with scalar and vector gas modes."""
    strategy = MassBasedMovingBin() if strategy is None else strategy
    distribution = np.array([1e-18, 2e-18])
    if isinstance(strategy, SpeciatedMassMovingBin):
        distribution = distribution[:, np.newaxis]
    particles = ParticleRepresentation(
        strategy=strategy,
        activity=ActivityIdealMass(),
        surface=SurfaceStrategyVolume(),
        distribution=distribution,
        density=np.array([1000.0]),
        concentration=np.array([4.0, 8.0]),
        charge=np.array([1.0, -1.0]),
        volume=2.0,
    )
    partitioning = GasSpecies(
        name="partitioning",
        molar_mass=0.1,
        concentration=3.0,
        partitioning=True,
    )
    gas_only = GasSpecies(
        name=np.array(["gas_a", "gas_b"]),
        molar_mass=np.array([0.02, 0.03]),
        concentration=np.array([5.0, 7.0]),
        partitioning=False,
    )
    atmosphere = Atmosphere(
        temperature=298.15,
        total_pressure=101325.0,
        partitioning_species=partitioning,
        gas_only_species=gas_only,
    )
    return Aerosol(atmosphere=atmosphere, particles=particles)


def test_dilute_aerosol_updates_physical_particle_and_gas_concentrations():
    """Container dilution uses physical particle concentrations and returns identity."""
    aerosol = _make_aerosol()
    particle = aerosol.particles
    partitioning = aerosol.atmosphere.partitioning_species
    gas_only = aerosol.atmosphere.gas_only_species
    particle_source = particle.get_concentration().copy()
    partitioning_source = partitioning.get_concentration()
    gas_only_source = gas_only.get_concentration().copy()
    decay = np.exp(-0.25 * 4.0)

    result = dilute_aerosol(aerosol, 0.25, 4.0)

    assert result is aerosol
    assert aerosol.particles is particle
    assert aerosol.atmosphere.partitioning_species is partitioning
    assert aerosol.atmosphere.gas_only_species is gas_only
    npt.assert_allclose(particle.get_concentration(), particle_source * decay)
    npt.assert_allclose(
        partitioning.get_concentration(), partitioning_source * decay
    )
    npt.assert_allclose(gas_only.get_concentration(), gas_only_source * decay)
    assert np.all(np.isfinite(particle.get_concentration()))
    assert np.all(particle.get_concentration() >= 0.0)


def test_dilute_aerosol_supports_speciated_particle_representation():
    """Dilution updates a supported speciated representation in physical units."""
    aerosol = _make_aerosol(SpeciatedMassMovingBin())
    source = aerosol.particles.get_concentration().copy()

    dilute_aerosol(aerosol, 0.5, 2.0)

    npt.assert_allclose(aerosol.particles.get_concentration(), source / np.e)


def test_dilute_aerosol_preserves_container_metadata_and_particle_state():
    """Dilution changes concentrations only, retaining container state."""
    aerosol = _make_aerosol()
    particle = aerosol.particles
    atmosphere = aerosol.atmosphere
    partitioning = atmosphere.partitioning_species
    gas_only = atmosphere.gas_only_species
    protected_particle_state = (
        particle.get_species_mass().copy(),
        particle.get_effective_density().copy(),
        particle.get_charge(clone=True),
        particle.get_volume(),
    )
    protected_gas_state = (
        partitioning.name,
        partitioning.molar_mass,
        partitioning.partitioning,
        gas_only.name.copy(),
        gas_only.molar_mass.copy(),
        gas_only.partitioning,
        atmosphere.temperature,
        atmosphere.total_pressure,
    )

    dilute_aerosol(aerosol, 0.25, 4.0)

    npt.assert_array_equal(
        particle.get_species_mass(), protected_particle_state[0]
    )
    npt.assert_array_equal(
        particle.get_effective_density(), protected_particle_state[1]
    )
    npt.assert_array_equal(particle.get_charge(), protected_particle_state[2])
    assert particle.get_volume() == protected_particle_state[3]
    assert partitioning.name == protected_gas_state[0]
    assert partitioning.molar_mass == protected_gas_state[1]
    assert partitioning.partitioning is protected_gas_state[2]
    npt.assert_array_equal(gas_only.name, protected_gas_state[3])
    npt.assert_array_equal(gas_only.molar_mass, protected_gas_state[4])
    assert gas_only.partitioning is protected_gas_state[5]
    assert atmosphere.temperature == protected_gas_state[6]
    assert atmosphere.total_pressure == protected_gas_state[7]


@pytest.mark.parametrize("coefficient, time_step", [(0.0, 1.0), (1.0, 0.0)])
def test_dilute_aerosol_zero_coefficient_or_duration_is_exact_no_op(
    coefficient, time_step
):
    """Zero coefficient or duration retains all concentration values exactly."""
    aerosol = _make_aerosol()
    sources = (
        aerosol.particles.get_concentration().copy(),
        np.asarray(
            aerosol.atmosphere.partitioning_species.get_concentration()
        ).copy(),
        aerosol.atmosphere.gas_only_species.get_concentration().copy(),
    )

    assert dilute_aerosol(aerosol, coefficient, time_step) is aerosol

    npt.assert_array_equal(aerosol.particles.get_concentration(), sources[0])
    npt.assert_array_equal(
        aerosol.atmosphere.partitioning_species.get_concentration(), sources[1]
    )
    npt.assert_array_equal(
        aerosol.atmosphere.gas_only_species.get_concentration(), sources[2]
    )


def test_dilute_aerosol_zero_source_is_an_exact_no_op():
    """A zero physical particle and gas source remains exactly zero."""
    aerosol = _make_aerosol()
    aerosol.particles.concentration = np.zeros(2)
    aerosol.atmosphere.partitioning_species.set_concentration(0.0)
    aerosol.atmosphere.gas_only_species.set_concentration(np.zeros(2))

    dilute_aerosol(aerosol, 1.0, 2.0)

    npt.assert_array_equal(aerosol.particles.get_concentration(), np.zeros(2))
    assert aerosol.atmosphere.partitioning_species.get_concentration() == 0.0
    npt.assert_array_equal(
        aerosol.atmosphere.gas_only_species.get_concentration(), np.zeros(2)
    )


@pytest.mark.parametrize(
    ("coefficient", "time_step", "exception", "message"),
    [
        (None, 1.0, TypeError, "coefficient.*numeric"),
        (1.0, None, TypeError, "time_step.*numeric"),
        ("invalid", 1.0, TypeError, "coefficient.*numeric"),
        (1.0, object(), TypeError, "time_step.*numeric"),
        (-1.0, 1.0, ValueError, "coefficient.*nonnegative"),
        (1.0, -1.0, ValueError, "time_step.*nonnegative"),
        (np.nan, 1.0, ValueError, "coefficient.*finite"),
        (1.0, np.nan, ValueError, "time_step.*finite"),
        (np.inf, 1.0, ValueError, "coefficient.*finite"),
        (1.0, np.inf, ValueError, "time_step.*finite"),
        (np.array([1.0]), 1.0, ValueError, "coefficient.*scalar"),
        (1.0, [1.0], ValueError, "time_step.*scalar"),
    ],
)
def test_dilute_aerosol_rejects_invalid_scalars_without_mutation(
    coefficient, time_step, exception, message
):
    """Scalar validation precedes every container access or assignment."""
    aerosol = _make_aerosol()
    snapshots = (
        aerosol.particles.get_concentration().copy(),
        np.asarray(
            aerosol.atmosphere.partitioning_species.get_concentration()
        ).copy(),
        np.asarray(
            aerosol.atmosphere.gas_only_species.get_concentration()
        ).copy(),
    )

    with pytest.raises(exception, match=message):
        dilute_aerosol(aerosol, coefficient, time_step)

    npt.assert_array_equal(aerosol.particles.get_concentration(), snapshots[0])
    npt.assert_array_equal(
        aerosol.atmosphere.partitioning_species.get_concentration(),
        snapshots[1],
    )
    npt.assert_array_equal(
        aerosol.atmosphere.gas_only_species.get_concentration(), snapshots[2]
    )


def test_dilute_aerosol_accepts_zero_dimensional_scalars_and_underflows():
    """Zero-dimensional numeric scalars are accepted and finite decay underflows."""
    aerosol = _make_aerosol()

    result = dilute_aerosol(
        aerosol,
        np.array(np.finfo(np.float64).max),
        np.array(np.finfo(np.float64).max),
    )

    assert result is aerosol
    npt.assert_array_equal(aerosol.particles.get_concentration(), np.zeros(2))
    assert aerosol.atmosphere.partitioning_species.get_concentration() == 0.0
    npt.assert_array_equal(
        aerosol.atmosphere.gas_only_species.get_concentration(), np.zeros(2)
    )


def test_dilute_aerosol_zero_dimensional_scalars_apply_normal_decay():
    """Zero-dimensional numeric inputs retain their documented scalar behavior."""
    aerosol = _make_aerosol()
    decay = np.exp(-np.float64(0.5) * np.float64(2.0))

    assert dilute_aerosol(aerosol, np.array(0.5), np.array(2.0)) is aerosol

    npt.assert_allclose(
        aerosol.particles.get_concentration(), np.array([2.0, 4.0]) * decay
    )
    npt.assert_allclose(
        aerosol.atmosphere.partitioning_species.get_concentration(), 3.0 * decay
    )
    npt.assert_allclose(
        aerosol.atmosphere.gas_only_species.get_concentration(),
        np.array([5.0, 7.0]) * decay,
    )


@pytest.mark.parametrize(
    "invalid_group", ["particle", "partitioning", "gas_only"]
)
def test_dilute_aerosol_invalid_source_preflight_is_atomic(invalid_group):
    """Nonfinite sources in every concentration domain fail before mutation."""
    aerosol = _make_aerosol()
    particle = aerosol.particles
    partitioning = aerosol.atmosphere.partitioning_species
    gas_only = aerosol.atmosphere.gas_only_species
    if invalid_group == "particle":
        particle.data.concentration[0, 0] = np.nan
    elif invalid_group == "partitioning":
        partitioning.data.concentration[0, 0] = np.nan
    else:
        gas_only.data.concentration[0, 0] = np.nan
    snapshots = (
        particle.get_concentration().copy(),
        np.asarray(partitioning.get_concentration()).copy(),
        gas_only.get_concentration().copy(),
    )

    with pytest.raises(ValueError):
        dilute_aerosol(aerosol, 0.5, 1.0)

    npt.assert_allclose(
        particle.get_concentration(), snapshots[0], equal_nan=True
    )
    npt.assert_allclose(
        partitioning.get_concentration(), snapshots[1], equal_nan=True
    )
    npt.assert_allclose(
        gas_only.get_concentration(), snapshots[2], equal_nan=True
    )


def test_dilute_aerosol_invalid_volume_preflight_is_atomic(monkeypatch):
    """Nonfinite representation volume fails before any concentration commit."""
    aerosol = _make_aerosol()
    particle = aerosol.particles
    sources = (
        particle.get_concentration().copy(),
        np.asarray(
            aerosol.atmosphere.partitioning_species.get_concentration()
        ).copy(),
        aerosol.atmosphere.gas_only_species.get_concentration().copy(),
    )
    monkeypatch.setattr(particle, "get_volume", lambda: np.nan)

    with pytest.raises(ValueError, match="stored particle concentration"):
        dilute_aerosol(aerosol, 0.5, 1.0)

    npt.assert_array_equal(particle.get_concentration(), sources[0])
    npt.assert_array_equal(
        aerosol.atmosphere.partitioning_species.get_concentration(), sources[1]
    )
    npt.assert_array_equal(
        aerosol.atmosphere.gas_only_species.get_concentration(), sources[2]
    )


def test_dilute_aerosol_rolls_back_particle_after_gas_setter_failure(
    monkeypatch,
):
    """A later commit failure restores the physical particle concentration."""
    aerosol = _make_aerosol()
    particle_snapshot = aerosol.particles.get_concentration().copy()
    gas_only_snapshot = (
        aerosol.atmosphere.gas_only_species.get_concentration().copy()
    )
    failing_gas = aerosol.atmosphere.partitioning_species

    def fail_setter(_):
        raise RuntimeError("setter failure")

    monkeypatch.setattr(failing_gas, "set_concentration", fail_setter)
    with pytest.raises(RuntimeError, match="setter failure"):
        dilute_aerosol(aerosol, 0.5, 1.0)

    npt.assert_array_equal(
        aerosol.particles.get_concentration(), particle_snapshot
    )
    npt.assert_array_equal(
        aerosol.atmosphere.gas_only_species.get_concentration(),
        gas_only_snapshot,
    )


def test_dilute_aerosol_rolls_back_first_gas_after_second_gas_failure(
    monkeypatch,
):
    """A later gas setter failure restores particle and prior gas updates."""
    aerosol = _make_aerosol()
    particle_snapshot = aerosol.particles.get_concentration().copy()
    partitioning_snapshot = (
        aerosol.atmosphere.partitioning_species.get_concentration()
    )
    gas_only_snapshot = (
        aerosol.atmosphere.gas_only_species.get_concentration().copy()
    )

    def fail_setter(_):
        raise RuntimeError("second setter failure")

    monkeypatch.setattr(
        aerosol.atmosphere.gas_only_species, "set_concentration", fail_setter
    )
    with pytest.raises(RuntimeError, match="second setter failure"):
        dilute_aerosol(aerosol, 0.5, 1.0)

    npt.assert_array_equal(
        aerosol.particles.get_concentration(), particle_snapshot
    )
    assert (
        aerosol.atmosphere.partitioning_species.get_concentration()
        == partitioning_snapshot
    )
    npt.assert_array_equal(
        aerosol.atmosphere.gas_only_species.get_concentration(),
        gas_only_snapshot,
    )
