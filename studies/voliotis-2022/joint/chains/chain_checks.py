"""Check nesting, positivity, conservation and chain activation on CUDA."""

import json

import numpy as np
from bank import BANK, DOSES, OUT, PRIOR
from reactions import Reaction

from gpu import Model


def starting(obj):
    """Use the previously replayed optimum, with zero chain rates."""
    k = np.zeros(len(BANK))
    k[:20] = [v[f"{obj}_fit_rate"] for v in PRIOR["reactions"]]
    return np.r_[PRIOR["baseline_parameters"][obj], k]


def main():
    """Verify that tagging changes no observations before new chemistry."""
    old = Model(
        [
            Reaction(**{k: r[k] for k in Reaction.__dataclass_fields__})
            for r in PRIOR["reactions"]
        ]
    )
    model = Model(BANK)
    checks = {}
    for obj in ("mass", "joint"):
        x = starting(obj)
        orig = old.run(x[:16], x[16:36], list(range(20)), DOSES, obj, sub=4)
        nested = model.run(x[:16], x[16:], list(range(20)), DOSES, obj, sub=4)
        np.testing.assert_allclose(orig[1], nested[1], atol=1e-9)
        checks[obj] = float(nested[0][0])
    x = starting("mass")
    x[36:] = 0.5
    ids = list(range(len(BANK)))
    _, stats, _, flux = model.run(
        x[:16], x[16:], ids, [1.0], "mass", sub=4, diagnostics=True
    )
    keys = [r.key for r in BANK]
    for key in ["AA_oxidize", "AA_oxsplit", "AA_gasage"]:
        if flux[0, :, keys.index(key)].max() <= 0:
            raise ArithmeticError(f"No transfer through {key}")
    checks.update(
        mass_closure=float(stats[..., 2].max()),
        minimum_pool=float(stats[..., 6].min()),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(checks, flush=True)


if __name__ == "__main__":
    main()
