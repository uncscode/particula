"""Refit the mechanisms reported in SiMBA Tables 1–3 to the pinned CSVs."""

import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from reference_inputs import CASES, REVISION
from scipy.integrate import odeint
from scipy.optimize import least_squares

# Letters repeated on a side denote stoichiometric coefficients.
TABLES = {
    "hypoth": (
        ("AA>B AA>C", 1139.86),
        ("AA>D AA>B D>C", 106.28),
        ("AA>B A>D D>E AE>C", -317.99),
        ("AA>B AA>D D>EE E>F EF>C", 390.92),
    ),
    "aldol_condensation": (
        ("AB>CD", None),
        ("A>E BE>CD", 866.58),
        ("A>E BE>F F>CD", -351.17),
        ("A>E EB>FG F>C G>D", -349.21),
    ),
    "fruc_HMF2": (
        ("A>BD D>BE E>BC", -166.18),
        ("A>D D>BE E>BF F>BC", -169.40),
        ("A>BD D>EF EF>D D>BG G>BC", -161.52),
    ),
}


def refit(task, *, initial_override=None, data_override=None, upper=10.0):
    """Fit jointly and estimate one noise variance per channel."""
    path, case, iteration, generator_convention = task
    reaction, published_aic = TABLES[case][iteration - 1]
    _, duration, _, observed, initials = CASES[case]
    sides = [word.split(">") for word in reaction.split()]
    names = sorted(set("".join(a + b for a, b in sides)))
    change = np.array(
        [[b.count(n) - a.count(n) for n in names] for a, b in sides]
    )
    if generator_convention:
        change = np.sign(change)
    first = [names.index(a[0]) for a, _ in sides]
    second = [
        names.index(a[1]) if len(a) == 2 else len(names) for a, _ in sides
    ]
    initial = np.zeros((len(initials), len(names)))
    initial[:, :observed] = np.array(initials)[:, :observed]
    if initial_override is not None:
        initial[3, :observed] = initial_override
    times = np.linspace(0, duration, 30)
    with tarfile.open(path) as source:
        data = np.stack(
            [
                np.loadtxt(
                    source.extractfile(
                        f"auto_react_mech_construct-{REVISION}/exp_data_{case}/exp_{i}.csv"
                    ),
                    delimiter=",",
                    skiprows=1,
                )
                for i in range(1, len(initials) + 1)
            ],
            axis=1,
        )

    if data_override is not None:
        data[:, 3, :] = data_override

    def predict(rates):
        def tendency(flat, time):
            values = flat.reshape(initial.shape)
            augmented = np.column_stack((values, np.ones(len(values))))
            flux = rates * values[:, first] * augmented[:, second]
            return (flux @ change).ravel()

        return odeint(
            tendency, initial.ravel(), times, rtol=1e-9, atol=1e-10
        ).reshape(len(times), *initial.shape)[:, :, :observed]

    def residual(rates):
        return (predict(rates) - data).ravel()

    fits = [
        least_squares(
            residual,
            np.full(len(sides), start),
            bounds=(0.0, upper),
            diff_step=1e-4,
            max_nfev=160,
        )
        for start in (0.1, 1.0)
    ]
    fit = min(fits, key=lambda x: x.cost)
    variance = np.mean(fit.fun.reshape(data.shape) ** 2, axis=(0, 1))
    n = times.size * len(initials)
    aic = n * np.sum(1 + np.log(2 * np.pi * variance)) + 2 * len(sides)
    return {
        "case": case,
        "iteration": iteration,
        "reactions": reaction,
        "stoichiometry": "unit signed change"
        if generator_convention
        else "molecular",
        "rates": fit.x.tolist(),
        "rmse": np.sqrt(variance).tolist(),
        "predicted": predict(fit.x).round(7).tolist(),
        "aic": float(aic),
        "published_aic": published_aic,
        "optimizer_status": [int(f.status) for f in fits],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tasks = [
        (args.source, case, i, sign)
        for case, rows in TABLES.items()
        for i in range(1, len(rows) + 1)
        for sign in ((False, True) if case == "hypoth" else (False,))
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(refit, tasks))
    import gzip

    with gzip.GzipFile(
        filename=str(args.output.with_suffix(".json.gz")), mode="wb", mtime=0
    ) as out:
        out.write(
            json.dumps(
                [r.pop("predicted") for r in rows], separators=(",", ":")
            ).encode()
        )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
