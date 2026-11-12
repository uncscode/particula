"""Native gas construction retains explicit data and required-field checks."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas import GasData, GasDataBuilder
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


@pytest.mark.parametrize(
    ("name", "molar_mass", "concentration", "partitioning"),
    [
        ("Oxygen", 0.032, 1.0, True),
        ("Oxygen", 0.032, 1.2, False),
        ("Oxygen", 0.032, 1e-3, True),
        ("Preset100", 0.1, 1.0, False),
    ],
)
def test_native_gas_builder_retains_explicit_values(
    name,
    molar_mass,
    concentration,
    partitioning,
):
    """The former constructor inputs retain exact native storage values."""
    builder = GasDataBuilder()
    assert builder.set_names([name]) is builder
    assert builder.set_molar_mass([molar_mass], units="kg/mol") is builder
    assert builder.set_concentration([concentration], units="kg/m^3") is builder
    assert builder.set_partitioning([partitioning]) is builder
    data = builder.build()
    assert isinstance(data, GasData)
    assert data.name == [name]
    npt.assert_array_equal(data.molar_mass, [molar_mass])
    npt.assert_array_equal(data.concentration, [[concentration]])
    npt.assert_array_equal(data.partitioning, [partitioning])
    assert not hasattr(data, "pure_vapor_pressure_strategy")


@pytest.mark.parametrize(
    "field", ["names", "molar_mass", "concentration", "partitioning"]
)
def test_native_gas_builder_requires_each_field(field):
    """Native construction does not synthesize omitted species metadata."""
    values = dict(
        names=["Oxygen"],
        molar_mass=[0.032],
        concentration=[1.0],
        partitioning=[True],
    )
    builder = GasDataBuilder()
    for name, value in values.items():
        if name != field:
            getattr(builder, f"set_{name}")(value)
    with pytest.raises(ValueError, match=field):
        builder.build()


@pytest.mark.parametrize(
    ("field", "value"),
    [("molar_mass", -0.032), ("molar_mass", 0.0), ("concentration", -1.0)],
)
def test_native_gas_builder_rejects_invalid_fields(field, value):
    """Invalid molar mass and negative concentration reject without clamping."""
    source = np.array([value])
    with pytest.raises(ValueError):
        getattr(GasDataBuilder(), f"set_{field}")(source)
    npt.assert_array_equal(source, [value])


@pytest.mark.parametrize(
    ("names", "molar_mass", "concentration", "partitioning"),
    [
        (["Oxygen"], [0.032], [[1.2]], [False]),
        (["Oxygen", "Nitrogen"], [0.032, 0.028], [[1.2, 0.8]], [False, False]),
        (["TestSpecies"], [0.044], [[0.5]], [True]),
        (["ZeroConcentration"], [1e10], [[0.0]], [False]),
        (["A", "B"], [1.0, 1.0], [[0.0, 1.0]], [False, False]),
        (["Water"], [0.018], [[1e-6]], [True]),
        (["Water"], [0.018], [[2e-6], [4e-6]], [True]),
        (["Water"], [0.018], [[5e-6], [6e-6]], [True]),
        (["Water", "Ammonia"], [0.018, 0.017], [[1e-6, 2e-6]], [True, True]),
        (["Water", "Ammonia"], [0.018, 0.017], [[1e-6, 2e-6]], [True, False]),
    ],
)
def test_native_gas_preserves_species_facade_storage_cases(
    names, molar_mass, concentration, partitioning
):
    """Original facade values survive with explicit box and species axes."""
    data = (
        GasDataBuilder()
        .set_names(names)
        .set_molar_mass(molar_mass)
        .set_concentration(concentration)
        .set_partitioning(partitioning)
        .build()
    )
    assert data.name == names
    assert data.n_species == len(names)
    assert data.n_boxes == len(concentration)
    npt.assert_array_equal(data.molar_mass, molar_mass)
    npt.assert_array_equal(data.concentration, concentration)
    npt.assert_array_equal(data.partitioning, partitioning)


@pytest.mark.parametrize(
    ("pressure", "temperature"),
    [
        (101325.0, 298.0),
        (np.array([101325.0, 101325.0]), np.array([298.0, 300.0])),
    ],
)
def test_vapor_pressure_science_without_gas_facade(pressure, temperature):
    """The original constant-pressure assertions call the owning strategy."""
    strategy = ConstantVaporPressureStrategy(vapor_pressure=pressure)
    npt.assert_array_equal(strategy.pure_vapor_pressure(temperature), pressure)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("molar_mass", [-1.0, 1.0]),
        ("molar_mass", [0.0, 1.0]),
        ("concentration", [-1.0, 1.0]),
    ],
)
def test_native_gas_rejects_invalid_species_lane_without_clamping(
    field, values
):
    """One invalid species rejects instead of silently changing its inventory."""
    source = np.array(values, dtype=np.float64)
    with pytest.raises(ValueError):
        getattr(GasDataBuilder(), f"set_{field}")(source)
    npt.assert_array_equal(source, values)


@pytest.mark.parametrize(
    ("initial", "delta", "expected"),
    [
        ([[0.5]], [[0.5]], [[1.0]]),
        ([[0.5], [1.0]], [[0.5], [-1.0]], [[1.0], [0.0]]),
        ([[1e-6]], [[2e-6]], [[3e-6]]),
    ],
)
def test_native_gas_explicit_concentration_update(initial, delta, expected):
    """Valid native array updates retain original concentration arithmetic."""
    data = GasData(
        name=["TestSpecies"],
        molar_mass=np.array([0.044]),
        concentration=np.array(initial),
        partitioning=np.array([True]),
    )
    storage = data.concentration
    data.concentration[:] += delta
    assert data.concentration is storage
    npt.assert_allclose(data.concentration, expected)
    npt.assert_array_equal(data.molar_mass, [0.044])
    npt.assert_array_equal(data.partitioning, [True])


@pytest.mark.parametrize(
    ("names", "masses", "concentrations"),
    [
        (["A", "B"], [0.01, 0.01], [1.0, 1.0]),
        (["Water", "Ammonia"], [0.018, 0.017], [1e-6, 2e-6]),
    ],
)
def test_native_gas_explicit_species_combination(names, masses, concentrations):
    """Explicit combined storage preserves both original species containers."""
    originals = [
        GasData(
            name=[name],
            molar_mass=np.array([mass]),
            concentration=np.array([[concentration]]),
            partitioning=np.array([True]),
        )
        for name, mass, concentration in zip(
            names, masses, concentrations, strict=True
        )
    ]
    combined = GasData(
        name=[name for gas in originals for name in gas.name],
        molar_mass=np.concatenate([gas.molar_mass for gas in originals]),
        concentration=np.concatenate(
            [gas.concentration for gas in originals], axis=1
        ),
        partitioning=np.concatenate([gas.partitioning for gas in originals]),
    )
    assert combined.name == names
    assert combined.n_species == 2
    npt.assert_array_equal(combined.molar_mass, masses)
    npt.assert_array_equal(combined.concentration, [concentrations])
    for index, original in enumerate(originals):
        assert combined is not original
        assert original.name == [names[index]]
        npt.assert_array_equal(original.molar_mass, [masses[index]])
        npt.assert_array_equal(
            original.concentration, [[concentrations[index]]]
        )
        assert not np.shares_memory(
            combined.concentration, original.concentration
        )
