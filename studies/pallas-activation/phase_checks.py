"""Compare sorbitol water activities with Table 1 of Ninni et al. (2000) and check the oil reservoir."""

import json

import numpy as np
from observations import P
from particula_aiomfac._equilibrium import ConvergenceError
from scipy.optimize import least_squares
from tables import DENS, prepare


def main():
    model = prepare("cpu")
    checks = []
    for pct, obs in [(9.93, 0.992), (29.80, 0.960), (64.57, 0.803)]:
        a = model.evaluate(np.array([100 - pct, pct, 0.0, 0.0, 0.0]), 298.15)
        checks.append(
            dict(
                wt_pct=pct,
                observed=obs,
                calculated=a.neutral_activities[a.neutral_ids.index("water")],
            )
        )
    # Compare pure-oil reservoir approximation with the library's unrestricted LLE solver
    # for two dilute growth states of equal salt, sorbitol and oil dry volumes.
    phases = []
    for wr in (50.0, 500.0):
        mass = np.array(
            [0.997 * wr, DENS[0] / 3, DENS[1] / 3, 1.77 / 6, 2.165 / 6]
        )
        try:
            eq = model.phase_equilibrium(mass, 298.15, 0)
            state = dict(
                two_liquids=eq.beta is not None,
                full_aw=eq.alpha.neutral_activities[
                    eq.alpha.neutral_ids.index("water")
                ],
            )
        except ConvergenceError as err:
            state = dict(full_aw=None, failure=str(err))
        wet = mass.copy()
        wet[2] = 0.997 * wr * 1e-12
        for _ in range(3):
            v = model.evaluate(wet, 298.15)
            wet[2] = min(
                mass[2],
                wet[2] / v.neutral_activities[v.neutral_ids.index("squalane")],
            )
        a = model.evaluate(wet, 298.15)
        phases.append(
            dict(
                water_dry=wr,
                **state,
                reservoir_aw=a.neutral_activities[a.neutral_ids.index("water")],
            )
        )

    def binary(log_ratios):
        oil_in_water, water_in_oil = np.exp(log_ratios)
        mass = np.array(
            [
                [1.0, 0.0, oil_in_water, 0.0, 0.0],
                [water_in_oil, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        val = model.evaluate(mass, 298.15)
        a = val.neutral_activities[
            :, [val.neutral_ids.index(x) for x in ["water", "squalane"]]
        ]
        return np.log(a[0] / a[1])

    pair = least_squares(
        binary,
        [-40.0, -8.0],
        bounds=([-200.0, -50.0], [-2.0, -2.0]),
        gtol=1e-14,
    )
    assert np.max(abs(binary(pair.x))) < 1e-6
    equilibrium = dict(
        oil_to_water_mass=np.exp(pair.x[0]),
        water_to_oil_mass=np.exp(pair.x[1]),
        max_log_activity_mismatch=max(abs(binary(pair.x))),
    )
    (P / "physical-checks.json").write_text(
        json.dumps(
            dict(sorbitol=checks, phases=phases, binary=equilibrium), indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
