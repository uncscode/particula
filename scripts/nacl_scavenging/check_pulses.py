"""Check pulse pair physics against CPU APIs and a two-particle survival law."""

from types import SimpleNamespace

import numpy as np
import warp as wp
from other_kernels import CHARGED, mechanical
from particula import dynamics as dyn
from pulse_pairs import MODELS, PulsePairs, refresh_pairs
from study import DENSITY, PRESSURE, TEMP

DEVICE = "cuda:0"


def state(ds, charges):
    """Create one tagged tracer and one collector in each independent box."""
    masses = np.pi / 6 * DENSITY * (ds * 1e-9) ** 3
    n = len(masses)
    m = np.zeros((n, 2, 2))
    m[:, 0, 0] = masses[:, 0]
    m[:, 1, 1] = masses[:, 1]
    return SimpleNamespace(
        masses=wp.array(m, dtype=wp.float64, device=DEVICE),
        concentration=wp.ones((n, 2), dtype=wp.float64, device=DEVICE),
        charge=wp.array(charges, dtype=wp.float64, device=DEVICE),
    )


def check_pairs():
    """Check all thermal laws, mechanical increments and fallback columns."""
    ds = np.array([[4, 100], [4, 100], [4, 100], [4, 160], [4, 100], [10, 95]])
    charges = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [1, -1], [-1, 0]])
    laws = list(CHARGED.values())
    for model, (law, a, eps, settled, shear, ao) in MODELS.items():
        s = state(ds, charges)
        clock = PulsePairs(model, len(ds), 2, DEVICE, 1)
        clock.left.fill_(1)
        fault = wp.zeros(1, dtype=int, device=DEVICE)
        wp.launch(
            refresh_pairs,
            clock.rates.shape,
            inputs=[
                s.masses,
                s.charge,
                clock.pairs,
                clock.left,
                clock.old,
                clock.rates,
                *clock.options,
                fault,
            ],
            device=DEVICE,
        )
        actual = clock.rates.numpy()[:, 0]
        np.testing.assert_array_equal(fault.numpy(), 0)
        for i, (d, z) in enumerate(zip(ds, charges, strict=True)):
            r = d * 0.5e-9
            m = 4 * np.pi / 3 * DENSITY * r**3
            tail = law == -1 and i in (3, 4)
            if law == -2:
                thermal = dyn.get_brownian_kernel_via_system_state(
                    r, m, TEMP, PRESSURE
                )
            elif law == -1 and not tail:
                thermal = dyn.get_van_der_waals_kernel_via_system_state(
                    r, m, z, TEMP, PRESSURE, a, eps, exclude_self_pairs=True
                )
            else:
                q = np.zeros_like(z) if model.endswith("_neutral") else z
                thermal = laws[max(law, 0)](r, m, q, TEMP, PRESSURE)
            expected = thermal[0, 1]
            np.testing.assert_allclose(
                actual[i, 0], expected, rtol=1e-8, atol=0
            )
            np.testing.assert_allclose(
                actual[i, 2], expected if tail else 0, rtol=1e-8, atol=0
            )
            increments = mechanical(r)
            mech = 0.0
            if settled:
                mech += increments["settling"][0, 1]
            if shear:
                mech += increments[f"shear_{shear:g}"][0, 1]
            if ao:
                mech += increments["ao2008_0.01"][0, 1]
            # Only the exploratory AO law uses size interpolation.
            np.testing.assert_allclose(
                actual[i, 1], mech, rtol=0.001 if ao else 1e-10, atol=0
            )
    print(f"Validated {len(ds) * len(MODELS)} pair/model combinations")


def check_survival():
    """Compare independent box mergers with 1-exp(-K t/V)."""
    n = 32768
    d = np.array([4.0, 100.0])
    s = state(np.tile(d, (n, 1)), np.zeros((n, 2)))
    clock = PulsePairs("full_settling_shear_10", n, 2, DEVICE, 1e10)
    r = d * 0.5e-9
    m = 4 * np.pi / 3 * DENSITY * r**3
    _, a, eps, *_ = MODELS["full_settling_shear_10"]
    rate = dyn.get_van_der_waals_kernel_via_system_state(
        r, m, [0, 0], TEMP, PRESSURE, a, eps, exclude_self_pairs=True
    )[0, 1]
    increments = mechanical(r)
    rate += increments["settling"][0, 1] + increments["shear_10"][0, 1]
    p = -np.expm1(-rate * 1e10 * 300)
    fault = wp.zeros(1, dtype=int, device=DEVICE)
    clock.step(s, 300, 887, fault)
    observed = clock.events.numpy().mean()
    np.testing.assert_array_equal(fault.numpy(), 0)
    np.testing.assert_allclose(
        observed, p, rtol=0, atol=5 * np.sqrt(p * (1 - p) / n)
    )
    print(f"Two-particle capture fraction {observed:.5f}, analytical {p:.5f}")


if __name__ == "__main__":
    check_pairs()
    check_survival()
