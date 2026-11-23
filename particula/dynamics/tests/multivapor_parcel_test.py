"""Species budgets, common surface heat balance and limiting parcel physics."""

from dataclasses import replace

import numpy as np
import pytest
from particula.dynamics.multivapor_parcel import (
    MultiVaporParcelInitialState,
    ParcelTendencies,
    ParcelVapor,
)
from particula.dynamics.parcel import CP, RD, run_parcel
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.util.constants import GAS_CONSTANT, STANDARD_GRAVITY


def parcel():
    """Return an ideal test mixture, independent of the optional backend."""
    return MultiVaporParcelInitialState(
        masses=np.array([[1.5e-14, 1e-15], [4e-14, 2e-15]]),
        number=np.array([2e7, 1e7]),
        vapor=np.array([0.02, 2e-6]),
        material_names=("water", "organic"),
        density=np.array([1000.0, 1100.0]),
        vapors=(
            ParcelVapor("water", 0, 0.01801528, 3169.0, 298.15, 2.5e6, 2.11e-5),
            ParcelVapor("organic", 1, 0.09, 0.73, 298.4, 8.7e5, 7e-6),
        ),
        activity_strategy=ActivityIdealMass(),
        pressure=101325.0,
        temperature=298.15,
    )


def test_no_exchange_ascent():
    """Moist gas plus unchanging condensate follows the dry-heat-capacity adiabat."""
    state = parcel()
    hist = run_parcel(
        state, updraft=1.0, timestep=10.0, steps=15, phase_exchange=False
    )
    loading = (
        1
        + state.vapor.sum()
        + state.number @ state.masses.sum(axis=1) / state.dry_air_mass
    )
    expected = state.temperature - loading * STANDARD_GRAVITY * hist.height / CP
    np.testing.assert_allclose(hist.temperature, expected, atol=2e-11, rtol=0)
    rmix = RD + sum(
        q * GAS_CONSTANT / v.molar_mass
        for q, v in zip(state.vapor, state.vapors, strict=True)
    )
    np.testing.assert_allclose(
        hist.pressure,
        state.pressure * (expected / state.temperature) ** (CP / rmix),
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        hist.vapor, np.broadcast_to(state.vapor, hist.vapor.shape), atol=1e-20
    )
    np.testing.assert_allclose(
        hist.masses,
        np.broadcast_to(state.masses, hist.masses.shape),
        rtol=1e-13,
    )


def test_only_organic_exchanges_and_heats():
    """A fixed-height closed parcel warms by the organic latent heat alone."""
    state = parcel()
    state = replace(
        state, vapors=(replace(state.vapors[0], enabled=False), state.vapors[1])
    )
    hist = run_parcel(state, updraft=0.0, timestep=0.1, steps=10)
    delta = hist.liquid - hist.liquid[0]
    assert delta[-1, 1] > 0
    np.testing.assert_allclose(
        hist.temperature - hist.temperature[0],
        state.vapors[1].latent_heat * delta[:, 1] / CP,
        rtol=1e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        hist.vapor + hist.liquid,
        np.tile(hist.vapor[0] + hist.liquid[0], (11, 1)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        hist.masses[:, :, 0],
        np.broadcast_to(state.masses[:, 0], (11, 2)),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(hist.pressure, np.full(11, state.pressure))


def test_water_only_continuum_limit():
    """Recover the legacy single-vapor resistances in the plane-water limit."""
    old = parcel()
    state = replace(
        old,
        masses=old.masses[:, :1],
        material_names=("water",),
        density=old.density[:1],
        vapor=old.vapor[:1],
        vapors=old.vapors[:1],
        surface_tension=0.0,
    )
    law = ParcelTendencies(state, 0)
    _, _, _, flux = law.rates(
        state.pressure, state.temperature, state.vapor, state.masses
    )
    v = state.vapors[0]
    rv = GAS_CONSTANT / v.molar_mass
    t = state.temperature
    rh = (
        state.pressure
        * state.vapor[0]
        / (RD / rv + state.vapor[0])
        / v.reference_pressure
    )
    diffusion = v.diffusivity * (t / 273.15) ** 1.94 * 101325 / state.pressure
    resistance = state.density[0] * (
        rv * t / (diffusion * v.reference_pressure)
        + v.latent_heat
        / (get_thermal_conductivity(t) * t)
        * (v.latent_heat / (rv * t) - 1)
    )
    radius = np.cbrt(3 * state.masses[:, 0] / (4 * np.pi * state.density[0]))
    np.testing.assert_allclose(
        flux[:, 0] / (2 * np.pi * radius * state.density[0]),
        2 * (rh - 1) / resistance,
        rtol=5e-14,
    )


def test_freeze_and_reject_unsupported_layouts():
    """Valid initial inputs are immutable and invalid target maps fail early."""
    state = parcel()
    with pytest.raises(ValueError, match="invalid parcel vapor layout"):
        replace(
            state,
            vapors=(
                state.vapors[0],
                replace(state.vapors[1], material_index=0),
            ),
        )
    with pytest.raises(ValueError, match="continuum"):
        replace(state.vapors[0], accommodation=0.5)
    with pytest.raises(ValueError):
        state.vapor[0] = 0.01
