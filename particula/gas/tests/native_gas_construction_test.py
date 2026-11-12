"""Native gas construction retains explicit data and required-field checks."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas import GasData, GasDataBuilder


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
