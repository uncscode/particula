"""Whole-trajectory gradients for coupled gas, particle and wall budgets."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")
from particula.dynamics.chemistry.discovery import compile_network  # noqa: E402
from particula.dynamics.chemistry.observations import (  # noqa: E402
    ObservationSeries,  # noqa: E402
)
from particula.execution.inverse_chemistry import (  # noqa: E402
    SeededChamber,
    VaporExchange,
    simulate_chamber,
)


def coupled_case():
    """Build A→V in air and V→P in particles with an inert seed core."""
    names = ("gas:A", "gas:V", "particle:V", "particle:P", "particle:seed")
    mech = compile_network(
        ((-1, 1, 0, 0, 0), (0, 0, -1, 1, 0)),
        {
            n: {"molar_mass_kg_per_mol": 0.1, "elements": {"X": 1}}
            for n in names
        },
        [0.5, 0.2],
    )
    return SeededChamber(
        mech,
        {n: 1000.0 for n in names[2:]},
        {"particle:seed": 0.95, "particle:V": 0.05},
        (VaporExchange("gas:V", "particle:V", 1e-5, 6e-8),),
        ("particle:V", "particle:P"),
        ("gas:A", "gas:V"),
        (80e-9, 120e-9),
        0.12,
        3,
    )


@pytest.mark.warp
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
)
def test_coupled_gradients(device):
    """Check conservation and each control against finite differences."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    model = coupled_case()
    times = np.array([0.0, 0.8, 2.0])
    pars = dict(
        dose=np.array([1e-7, 1e-9]),
        rates=np.array([0.5, 0.2]),
        saturation=np.array([2e-8]),
        seed=np.array([1e10, 100e-9, 1.4, 0.04]),
    )

    def solve(**kwargs):
        return simulate_chamber(
            model, times, max_step=0.1, device=device, **kwargs
        )

    forward = solve(**pars)
    total = forward.final_state[:, :-1].sum()
    initial_organic = forward.predictions[0, 2] / 0.1
    expected = pars["dose"].sum() + initial_organic / 0.05
    np.testing.assert_allclose(total, expected, rtol=1e-12)
    scale = np.max(forward.predictions, axis=0)
    measured = forward.predictions * 0.9
    obs = ObservationSeries.create(times, measured, np.diag(scale**2))
    fitted = solve(**pars, observations=obs, gradient=True)
    for group, values in pars.items():
        for j, value in enumerate(values):
            delta = value * 1e-4
            left = values.copy()
            right = values.copy()
            left[j] -= delta
            right[j] += delta
            # Difference predictions first to avoid cancellation in log densities.
            slope = (
                solve(**{**pars, group: right}).predictions
                - solve(**{**pars, group: left}).predictions
            ) / (2 * delta)
            expected = np.sum(obs.loss(forward.predictions)[1] * slope)
            np.testing.assert_allclose(
                fitted.gradients[group][j] * value,
                expected * value,
                rtol=1e-5,
                atol=1e-9,
                err_msg=f"{group}[{j}]",
            )
