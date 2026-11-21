"""Check nested wall-model agreement and GPU inventory before searching."""

import json
from pathlib import Path

import numpy as np
from reactions import CANDIDATES

from gpu import Model

OUT = Path(__file__).resolve().parents[1] / "results/discovery"
DOSES = np.linspace(0.85, 1.15, 61)


def start(model, obj):
    """Recover the archived optimum and verify the zero-channel replay."""
    previous = json.loads(
        (OUT.parent / "pathways/wider_bounds.json").read_text()
    )
    p = np.array(
        min(
            (r for r in previous if r["objective"] == obj),
            key=lambda r: r["score"],
        )["physical"]
    )
    k = np.zeros(len(CANDIDATES))
    new = model.run(p, k, [], DOSES, obj, sub=2)
    old = model.evaluate(p, True, obj, DOSES, sub=2, pathway=6, outcomes=True)
    np.testing.assert_allclose(
        new[1][0, :, :2], old[1][0], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        new[1][0, :, 3], old[3][0, :, 0], rtol=1e-10, atol=1e-10
    )
    return p, k, new


if __name__ == "__main__":
    m = Model()
    for obj in ["mass", "joint"]:
        p, k, res = start(m, obj)
        print(obj, res[0], res[1][0, :, 2].max(), flush=True)
    k[:] = 0.4
    print(
        m.run(
            p,
            k,
            list(range(len(k))),
            DOSES[::10],
            "joint",
            sub=4,
            diagnostics=True,
        )[0],
        flush=True,
    )
