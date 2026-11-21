"""Check mass-action kinetics and an unobservable particle reaction."""

import numpy as np
import warp as wp
from fit import save
from reactions import CANDIDATES
from scipy.integrate import solve_ivp
from smoke import DOSES, OUT, start

from gpu import F64, V3, Model, State, advance_channel


@wp.kernel
def pair_test(
    a: wp.int32, b: wp.int32, steps: wp.int32, result: wp.array(dtype=F64)
):
    """Replay an isolated heterogeneous or self-reaction for one hour."""
    x = State()
    x[a] = F64(40.0)
    if b != a:
        x[b] = F64(17.0)
    for _step in range(steps):
        x, transfer = advance_channel(
            x,
            a,
            b,
            9,
            F64(3.0),
            F64(1.0) / F64(steps),
            V3(F64(184.0), F64(140.0), F64(118.0)),
        )
    result[0] = x[a]
    result[1] = x[b]
    result[2] = x[9]


def main():
    """Verify isolated rate integration and a label-only negative control."""
    m = Model()
    p, k, base = start(m, "mass")
    h = next(i for i, r in enumerate(CANDIDATES) if r.key == "aa_particle")
    k[h] = 4.0
    d = m.run(p, k, [h], DOSES, "mass", sub=2, diagnostics=True)
    gap = np.abs(base[1][0, :, :2] - d[1][0, :, :2]).max()
    peak = np.abs(base[1][0, :, 3] - d[1][0, :, 3]).max()
    if gap > 1e-10 or peak > 1e-8 or d[3][0, :, h].max() < 1:
        raise ArithmeticError(
            "Unexpected result from the stabilization-only control"
        )
    errors = []
    for b in [0, 1]:
        if b == 0:
            single = 40.0 / (1.0 + 2.0 * 3.0 * 40.0 / 184.0)
            truth = [single, single, 40.0 - single]
        else:

            def rhs(t, x):
                v = 3.0 * x[0] / 184.0 * x[1] / 140.0
                return [-184 * v, -140 * v, (184 + 140) * v]

            truth = solve_ivp(
                rhs, [0.0, 1.0], [40.0, 17.0, 0.0], rtol=1e-12, atol=1e-12
            ).y[:, -1]
        for n in [120, 240, 480]:
            out = wp.zeros(3, dtype=F64, device="cuda:0")
            wp.launch(pair_test, dim=1, inputs=[0, b, n, out], device="cuda:0")
            errors.append(
                dict(
                    partner=b,
                    steps=n,
                    max_abs_ug_m3=float(np.abs(out.numpy() - truth).max()),
                )
            )
    if (
        errors[-1]["max_abs_ug_m3"] > 0.01
        or errors[-1]["max_abs_ug_m3"] >= errors[-3]["max_abs_ug_m3"]
    ):
        raise ArithmeticError(
            "Cross-reaction extent failed its refinement check"
        )
    _, _, joint = start(m, "joint")
    save(
        OUT / "numerical_checks.json",
        dict(
            stabilization_max_nrmse_change=gap,
            stabilization_max_peak_change_ug_m3=peak,
            stabilization_max_gross_ug_m3=float(d[3][0, :, h].max()),
            mass_null=base[0][0],
            joint_null=joint[0][0],
            pair_tests=errors,
            versions=dict(
                warp=wp.__version__, gpu=wp.get_device("cuda:0").name
            ),
        ),
    )


if __name__ == "__main__":
    main()
