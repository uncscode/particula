# %%
"""Check coupled wall-loss histories against an independent CPU reference."""

import json

import numpy as np
import pandas as pd
from engine import CONDITIONS, ORDER, RESULT, TIMES, Engine
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state,
)
from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient_via_system_state,
)
from scipy.integrate import solve_ivp
from wall_fit import OUT, WallObjective, simulate


def seed_rhs(_, state, temp, eddy, collisions):
    """Use native NumPy transport and the seed/number moment equations."""
    seed, number = state
    mass = seed * 1e-9 / number
    radius = (3 * mass / (4 * np.pi * 1770)) ** (1 / 3)
    k = float(
        get_rectangle_wall_loss_coefficient_via_system_state(
            eddy, radius, 1770, temp, 101325.0, (3, 2, 3)
        )
    )
    pair = 0.0
    if collisions:
        pair = get_brownian_kernel_via_system_state(
            np.array([radius]), np.array([mass]), temp, 101325.0
        )[0, 0]
    return [-k * seed, -k * number - 0.5 * pair * number**2]


if __name__ == "__main__":
    original = np.array(
        json.loads((RESULT / "single_nox.json").read_text())["physical"]
    )
    models = json.loads((OUT / "restarts.json").read_text())
    e = Engine(3, wall_interval_s=60)
    p = original.copy()
    p[0] = 0
    values = []
    for collisions in (False, True):
        soa, frac = e(p, eddy_s=5.0, brownian=collisions, return_numbers=True)
        np.testing.assert_array_equal(soa, 0.0)
        for i, r in enumerate(ORDER[:4]):
            _, seed, n, temp = CONDITIONS[r]
            ref = solve_ivp(
                seed_rhs,
                (0.0, 18000.0),
                [seed, n * 1e6],
                args=(temp + 273.15, 5.0, collisions),
                t_eval=TIMES,
                rtol=1e-10,
                atol=1e-10,
            )
            if not ref.success:
                raise RuntimeError(ref.message)
            delta = np.max(abs(frac[0, i] - ref.y[1] / (n * 1e6)))
            values.append(
                dict(
                    run=r,
                    model="eddy_brownian" if collisions else "eddy",
                    seed_oracle_count_fraction=delta,
                )
            )
    pd.DataFrame(values).to_csv(OUT / "seed_oracle.csv", index=False)
    if max(row["seed_oracle_count_fraction"] for row in values) > 0.001:
        raise ArithmeticError("GPU and independent CPU seed histories disagree")
    rows = []
    scorer = WallObjective()
    for name in {r["model"] for r in models}:
        rec = min(
            (
                r
                for r in models
                if r["model"] == name and r["objective"] == "joint"
            ),
            key=lambda r: r["score"],
        )
        coarse = simulate(Engine(3), rec["coords"], name, original)
        fine = simulate(e, rec["coords"], name, original)
        m, n = scorer.errors(*coarse)
        mf, nf = scorer.errors(*fine)
        np.testing.assert_allclose(
            max(m.max(), n.max()), rec["score"], atol=1e-9
        )
        for i, r in enumerate(ORDER[:4]):
            rows.append(
                dict(
                    model=name,
                    run=r,
                    refine_mass_ug_m3=np.max(
                        abs(coarse[0][0, i] - fine[0][0, i])
                    ),
                    refine_number_fraction=np.max(
                        abs(coarse[1][0, i] - fine[1][0, i])
                    ),
                    refined_mass_nrmse=mf[0, i],
                    refined_number_nrmse=nf[0, i],
                )
            )
    pd.DataFrame(rows).sort_values(["model", "run"]).to_csv(
        OUT / "wall_interval_check.csv", index=False
    )
