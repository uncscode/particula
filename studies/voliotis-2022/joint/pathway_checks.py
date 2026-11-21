# %%
"""Check the two-sink gas reservoir and species-selective diversion."""

import json

import numpy as np
import warp as wp
from pathways import DEFAULT, OUT, Campaign, anchors
from scipy.linalg import expm
from wall_campaign import F64, vapor_step


@wp.kernel
def reservoir(rows: wp.array2d(dtype=F64), out: wp.array2d(dtype=F64)):
    """Evaluate the two competing sinks for each test case."""
    i = wp.tid()
    x = vapor_step(rows[i, 0], rows[i, 1], rows[i, 2], rows[i, 3], rows[i, 4])
    for j in range(3):
        out[i, j] = x[j]


def main():
    """Use an independent matrix exponential and a no-isoprene control."""
    rng = np.random.default_rng(451)
    n = 256
    data = np.column_stack(
        (
            rng.uniform(0, 100, (n, 2)),
            10 ** rng.uniform(-6, 1, (n, 2)),
            10 ** rng.uniform(-1, 2, n),
        )
    )
    input_gpu = wp.array(data, dtype=F64, device="cuda:0")
    out = wp.empty((n, 3), dtype=F64, device="cuda:0")
    wp.launch(reservoir, dim=n, inputs=[input_gpu, out], device="cuda:0")
    oracle = []
    for old, source, gas_gain, gas_loss, dt in data:
        generator = np.array(
            [
                [-gas_gain - gas_loss, 0, 0, source / dt],
                [gas_gain, 0, 0, 0],
                [gas_loss, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        oracle.append((expm(dt * generator) @ np.array([old, 0, 0, 1]))[:3])
    error = np.abs(np.array(oracle) - out.numpy()).max()
    np.testing.assert_allclose(oracle, out.numpy(), atol=1e-8, rtol=1e-8)
    c = Campaign()
    base, *_ = anchors()
    p = np.r_[base, DEFAULT]
    ref = c.evaluate(base, True, "mass", np.array([1.0]), sub=2, traces=True)[
        3
    ][0]
    p[12] = 3.0
    changed = c.evaluate(
        p, True, "mass", np.array([1.0]), sub=2, pathway=6, traces=True
    )[3][0]
    pure = [j for j, run in enumerate(c.runs) if run[4][:, [2, 5]].sum() == 0]
    error_pure = np.abs(ref[pure] - changed[pure]).max()
    np.testing.assert_allclose(ref[pure], changed[pure], atol=1e-12)
    r24 = [r[0] for r in c.runs].index(24)
    if changed[r24, 1] >= ref[r24, 1] or changed[r24, 5] <= 0:
        raise ValueError("Expected diverted mass and reduced peak in run 24")
    (OUT / "checks.json").write_text(
        json.dumps(
            dict(
                reservoir_cases=n,
                reservoir_max_abs_ug_m3=float(error),
                isoprene_free_run_count=len(pure),
                isoprene_free_max_change=float(error_pure),
                reference_nominal_peak24=ref[r24, 1],
                diverted_nominal_peak24=changed[r24, 1],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
