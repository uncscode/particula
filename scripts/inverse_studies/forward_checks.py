"""Compare the differentiable profile, ordinary native step and radius ODE."""

import numpy as np
from particula.dynamics import CondensationIsothermal
from particula.execution.inverse_condensation import terminal_radius
from scripts import assess_native_condensation as case
from scripts.assess_native_adaptive import VISCOSITY, local_reference
from scripts.inverse_studies.distribution import OPTIONS


def assess(device):
    """Refine three trajectories and their initial-radius derivatives."""
    rows = []
    for radius in (0.8, 1.0, 1.2):

        def reference(r):
            (m, g), _ = local_reference(r, case.VAPOR, 1.4, 0.0)
            return np.cbrt(m / case.MASS), g

        expected, vapor = reference(radius)
        slope = (
            reference(radius + 1e-4)[0] - reference(radius - 1e-4)[0]
        ) / 2e-4
        for steps in (16, 64, 256):
            x, grad = terminal_radius(
                [[radius]],
                [[case.NUMBER]],
                [case.VAPOR],
                steps=steps,
                adjoint=[[1.0]],
                device=device,
                **OPTIONS,
            )
            particle, gas, strategy = case.initial(CondensationIsothermal, 0.0)
            particle.masses[0, 0, 0] = case.MASS * radius**3
            for _ in range(steps):
                strategy.step(
                    particle,
                    gas,
                    case.TEMPERATURE,
                    case.PRESSURE,
                    1.4 / steps,
                    VISCOSITY,
                )
            final = float(x[0, 0])
            final_vapor = case.VAPOR + case.NUMBER * case.MASS * (
                radius**3 - final**3
            )
            np.testing.assert_allclose(
                final, particle.radii[0, 0] * 1e6, rtol=1e-11
            )
            np.testing.assert_allclose(
                final_vapor, gas.concentration[0, 1], rtol=1e-11
            )
            np.testing.assert_array_equal(
                [gas.concentration[0, 0], particle.masses[0, 1, 0]], [0.1, 0.0]
            )
            error = abs(float(grad[0, 0]) / slope - 1)
            if steps == 256 and (
                error >= 1e-3 or abs(final - expected) >= 0.001
            ):
                raise ArithmeticError(
                    "refined trajectory or gradient misses target"
                )
            rows.append(
                dict(
                    initial_um=radius,
                    steps=steps,
                    device=device,
                    radius_error_nm=1000 * abs(final - expected),
                    vapor_error_kg_m3=abs(final_vapor - vapor),
                    slope_relative_error=error,
                )
            )
    return rows
