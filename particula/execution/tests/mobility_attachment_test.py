"""Independent limits, CPU/device agreement and prepared ion continuation."""

from typing import no_type_check

import numpy as np
import pytest

from particula.dynamics.ions.coagulation import (
    POSITIVE_ION_MASS,
    AtmosphericCoagulation,
)
from particula.dynamics.ions.mobility_attachment import mobility_attachment
from particula.execution.tests.prepared_atmospheric_test import (  # noqa: F401, F811
    device,
    pair_case,
)
from particula.util.constants import BOLTZMANN_CONSTANT, ELEMENTARY_CHARGE_VALUE

wp = pytest.importorskip("warp")
from particula.gpu.dynamics.mobility_attachment import (  # noqa: E402
    mobility_attachment_wp,
)


@no_type_check
@wp.kernel
def evaluate(
    values: wp.array2d(dtype=wp.float64), rates: wp.array(dtype=wp.float64)
):
    """Evaluate each independent test pair on the selected device."""
    i = wp.tid()
    rates[i] = mobility_attachment_wp(
        values[i, 0],
        values[i, 1],
        values[i, 2],
        values[i, 3],
        values[i, 4],
        values[i, 5],
        values[i, 6],
        values[i, 7],
        values[i, 8],
    )


def test_parity_and_polarities(device):  # noqa: F811
    """Both signs, neutral/repulsive/attractive, full scan T/P envelope."""
    rows = np.array(
        [
            [
                r,
                4 * np.pi / 3 * r**3 * 1400,
                q,
                t,
                p,
                POSITIVE_ION_MASS,
                1.4e-4,
                sign,
                0.25e-9,
            ]
            for r in [2e-9, 35e-9, 1e-6]
            for q in [-5, 0, 5]
            for t in [187.0, 278.15, 298.15]
            for p in [0.373, 70000.0, 101325.0, 1013250.0]
            for sign in [-1, 1]
        ]
    )
    expected = mobility_attachment(*rows.T)
    output = wp.empty(len(rows), dtype=wp.float64, device=device)
    wp.launch(
        evaluate,
        len(rows),
        inputs=[wp.array(rows, dtype=wp.float64, device=device), output],
        device=device,
    )
    np.testing.assert_allclose(output.numpy(), expected, rtol=2e-12, atol=0)
    rates = mobility_attachment(
        35e-9,
        2e-19,
        np.array([-1, 0, 1]),
        298.15,
        101325.0,
        POSITIVE_ION_MASS,
        1.4e-4,
        1,
    )
    assert rates[0] > rates[1] > rates[2] > 0


def test_independent_neutral_limits():
    """Ballistic area times thermal relative speed; continuum ion diffusion."""
    r, m, mi, t, z, ri = (
        35e-9,
        2e-19,
        POSITIVE_ION_MASS,
        298.15,
        1.4e-4,
        0.25e-9,
    )
    ballistic = (
        np.pi
        * (r + ri) ** 2
        * np.sqrt(8 * BOLTZMANN_CONSTANT * t / np.pi * (1 / m + 1 / mi))
    )
    actual = mobility_attachment(r, m, 0, t, 1e-8, mi, z, 1, ri)
    np.testing.assert_allclose(actual, ballistic, rtol=2e-8)
    # Very small mobility makes ion diffusion negligible; use analytic Stokes
    # drag for the aerosol in the high-pressure continuum limit.
    p = 1e12
    mu = 1.716e-5 * (t / 273.15) ** 1.5 * (273.15 + 110.4) / (t + 110.4)
    zi = z * 101325 / p
    expected = (
        4
        * np.pi
        * (r + ri)
        * BOLTZMANN_CONSTANT
        * t
        * (zi / ELEMENTARY_CHARGE_VALUE + 1 / (6 * np.pi * mu * r))
    )
    actual = mobility_attachment(r, m, 0, t, p, mi, z, 1, ri)
    # Aerosol Brownian ballistic correction remains at this small radius.
    np.testing.assert_allclose(actual, expected, rtol=0.005)


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 0),
        (1, -1),
        (2, 0.5),
        (3, np.nan),
        (4, 0),
        (6, -1),
        (7, 0),
        (8, np.inf),
    ],
)
def test_invalid_inputs(index, value):
    """Reject nonphysical or nonfinite input before evaluating rates."""
    args = [
        35e-9,
        2e-19,
        0,
        298.15,
        101325.0,
        POSITIVE_ION_MASS,
        1.4e-4,
        1,
        0.25e-9,
    ]
    args[index] = value
    with pytest.raises(ValueError, match="invalid mobility"):
        mobility_attachment(*args)


def test_variable_environment_capture_restart(device):  # noqa: F811
    """A non-reference attachment preserves exact mass/charge and continuation."""
    initial = pair_case((0, 1))
    initial.environment.temperature[:] = 278.15
    initial.environment.pressure[:] = 70000.0
    rate = float(
        mobility_attachment(
            10e-9,
            initial.particles.masses[0, 0].sum(),
            0,
            278.15,
            70000.0,
            POSITIVE_ION_MASS,
            1.4e-4,
            1,
        )
    )
    process = AtmosphericCoagulation(0, 1, attachment_model="mobility")
    with process.prepare(initial, device=device, time_step=2e-12 / rate) as sim:
        before = sim.checkpoint()
        if device != "cpu":
            sim.capture()
        sim.step()
        result = sim.snapshot()
        majorant = sim._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        np.testing.assert_allclose(majorant, rate, rtol=2e-12)
    assert result.particles.concentration.sum() == 1
    assert result.particles.charge.sum() == 1
    np.testing.assert_array_equal(
        result.particles.masses.sum(axis=1),
        initial.particles.masses.sum(axis=1),
    )
    with process.resume(before) as restarted:
        restarted.step()
        np.testing.assert_array_equal(
            restarted.snapshot().particles.masses, result.particles.masses
        )


def test_environment_update_refreshes_attachment(device):  # noqa: F811
    """Captured environment forcing refreshes the cached pair rates."""
    initial = pair_case((0, 1))
    process = AtmosphericCoagulation(0, 1, attachment_model="mobility")
    with process.prepare(initial, device=device, time_step=1e-20) as sim:
        if device != "cpu":
            sim.capture()
        sim.step()
        before = sim._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        sim.update_environment(temperature=278.15, pressure=70000.0)
        sim.step()
        sim.synchronize()
        after = sim._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        expected = mobility_attachment(
            10e-9,
            initial.particles.masses[0, 0].sum(),
            0,
            278.15,
            70000.0,
            POSITIVE_ION_MASS,
            1.4e-4,
            1,
        )
        assert after != before
        np.testing.assert_allclose(after, expected, rtol=2e-12)


@pytest.mark.parametrize(
    "changes",
    [
        {"attachment_model": "unknown"},
        {"positive_mobility": 0},
        {"negative_mobility": np.nan},
        {"ion_radius": -1},
        {"omitted": "lower", "attachment_model": "mobility"},
    ],
)
def test_invalid_model_configuration(changes):
    """Reject invalid physical configuration before upload."""
    with pytest.raises(ValueError):
        AtmosphericCoagulation(0, 1, **changes).prepare(
            pair_case(), device="cpu", time_step=1.0
        )
