"""Check refinements and export a reconstruction that can be replayed alone."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from analyze import HERE, LIMITS, METRICS, continuous_scores, first_pass
from compression import tree
from particula.optics import get_mie_efficiencies
from particula.particles import get_mixing_state_index
from physics import DENSITY, EDGES, SS, evolve, make_optics, responses
from run import SEEDS


def compare_trees(carriers, generic):
    """Report first calibration pass and its independent holdout performance."""
    rows = []
    for method, folder in [("carrier split", carriers), ("generic", generic)]:
        scores = np.stack(
            [np.load(folder / f"{s}.npz")["scores"] for s in SEEDS]
        )
        train = scores[:8].max(axis=(0, 1))
        hold = scores[8:].max(axis=(0, 1))
        name = "generic" if method == "generic" else "carrier"
        checked = pd.read_csv(HERE / f"continuum_{name}.csv")
        continuous_train = continuous_scores(train, checked, "calibration")
        continuous_hold = continuous_scores(hold, checked, "held-out")
        for thresholds in [[3, 7.5, 10], LIMITS, [1, 2.5, 2.5]]:
            cap = first_pass(continuous_train, np.array(thresholds))
            rows.append(
                dict(
                    method=method,
                    **dict(
                        zip(
                            ["limit_rmse", "limit_channel", "limit_abs"],
                            thresholds,
                            strict=True,
                        )
                    ),
                    calibration_cap=cap,
                    grid_only_calibration=first_pass(
                        train, np.array(thresholds)
                    ),
                    holdout_only_cap=first_pass(
                        continuous_hold, np.array(thresholds)
                    ),
                    continuous_max_pp=continuous_hold[cap - 1, 1],
                    **dict(zip(METRICS, hold[cap - 1], strict=True)),
                )
            )
    pd.DataFrame(rows).to_csv(
        HERE / "tolerances.csv", index=False, float_format="%.8g"
    )


def refinements(main, fine, half, cap):
    """Quantify trajectory and compression sensitivity on one history."""
    base = np.load(main / "1199.npz")
    rows = []
    for label, folder in [
        ("2.5 minute aging", fine),
        ("50000 particles per parcel", half),
    ]:
        d = np.load(folder / "1199.npz")
        weights = base["counts"] / base["counts"].sum(axis=1, keepdims=True)
        rows.append(
            dict(
                case=label,
                max_reference_activation_change_pp=100
                * abs(d["ccn"] - base["ccn"]).max(),
                max_reference_absorption_change_pct=100
                * abs(d["babs"] / base["babs"] - 1).max(),
                max_reference_chi_rmse=np.sqrt(
                    np.nansum(weights * (d["chi"] - base["chi"]) ** 2, axis=1)
                ).max(),
                **dict(
                    zip(
                        METRICS,
                        d["scores"][:, cap - 1].max(axis=0),
                        strict=True,
                    )
                ),
            )
        )
    pd.DataFrame(rows).to_csv(
        HERE / "refinement.csv", index=False, float_format="%.8g"
    )


def export_example(folder, cap):
    """Save the representatives at the largest held-out channel discrepancy."""
    s = pd.read_csv(HERE / "selected_scores.csv").query("split=='held-out'")
    case = s.loc[s.activation_max_pp.idxmax()]
    data = np.load(folder / f"{case.seed}.npz")
    j = 2 * int(case.hour) + int(case.parcel == "B")
    state = next(
        state
        for hour, state, *_ in evolve(int(case.seed), int(data["particles"]))
        if hour == case.hour
    )
    vol = state[int(case.parcel == "B")]
    dia = np.cbrt(vol.sum(axis=1))
    groups = np.searchsorted(EDGES[1:-1], dia)
    feat = np.column_stack(
        [
            np.log(dia) / np.log(EDGES[1] / EDGES[0]),
            vol / vol.sum(axis=1, keepdims=True),
        ]
    )
    rows = []
    for b in np.unique(groups):
        take = groups == b
        # Fixed sizes, pure or internally mixed recipes: independently verify
        # that the alternative used by the shape analysis matches inventories
        # and the public within-bin chi, even though its responses differ.
        species = vol[take].sum(axis=0) * DENSITY
        target = data["chi"][j, b]
        entropy = data["bulk_entropy"][j, b]
        f = np.log1p(target * np.expm1(entropy)) / entropy
        alternative = np.vstack([f * species, (1 - f) * np.diag(species)])
        np.testing.assert_allclose(alternative.sum(axis=0), species, rtol=1e-12)
        np.testing.assert_allclose(
            get_mixing_state_index(alternative), target, atol=1e-12
        )
        weights, means, splits = tree(vol[take], feat[take], cap, True)
        leaves = {0}
        for parent, left, right in splits:
            leaves.remove(parent)
            leaves.update([left, right])
        for i in sorted(leaves):
            rows.append(
                [b, weights[i], *(means[i] * DENSITY * np.pi / 6 * 1e-27)]
            )
    names = [
        "bin",
        "count",
        "salt_kg",
        "primary_organic_kg",
        "oxidized_organic_kg",
        "bc_kg",
    ]
    path = HERE / "representatives.csv"
    pd.DataFrame(rows, columns=names).to_csv(
        path, index=False, float_format="%.17g"
    )
    saved = pd.read_csv(path, float_precision="round_trip")
    masses = saved[names[2:]].to_numpy()
    mult = saved["count"].to_numpy()
    recipe = masses / (DENSITY * np.pi / 6 * 1e-27)
    np.testing.assert_equal(mult.sum(), len(vol))
    np.testing.assert_allclose(
        mult @ masses, vol.sum(axis=0) * DENSITY * np.pi / 6 * 1e-27, rtol=1e-11
    )
    critical, area = responses(recipe, make_optics())
    counts = mult @ (critical[:, None] <= SS)
    np.testing.assert_allclose(
        counts / mult.sum(), data["rep_ccn"][j, cap - 1], atol=1e-12
    )

    def direct_abs(v, weights):
        summed = v.sum(axis=1)
        d = np.cbrt(summed)
        m = 1.5 + (0.45 + 0.79j) * v[:, 3] / summed
        q = get_mie_efficiencies(d * 5e-10, 550e-9, m).absorption
        return np.average(np.pi * d**2 / 4 * q, weights=weights) * 1e-3

    full = direct_abs(vol, None)
    reduced = direct_abs(recipe, mult)
    print(
        "export",
        int(case.seed),
        case.parcel,
        int(case.hour),
        len(rows),
        "rows; absorption full/reduced direct:",
        full,
        reduced,
        "lookup:",
        data["babs"][j],
        np.average(area, weights=mult) * 1e-3,
    )
    np.testing.assert_allclose(full, data["babs"][j], rtol=0.001, atol=0)
    np.testing.assert_allclose(
        reduced, np.average(area, weights=mult) * 1e-3, rtol=0.001, atol=0
    )


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description=__doc__)
    for key in ["carriers", "generic", "fine", "half"]:
        cmd.add_argument(key, type=Path)
    args = cmd.parse_args()
    compare_trees(args.carriers, args.generic)
    table = pd.read_csv(HERE / "tolerances.csv")
    cap = int(
        table[
            (table.method == "carrier split") & (table.limit_rmse == 2)
        ].calibration_cap.iloc[0]
    )
    refinements(args.carriers, args.fine, args.half, cap)
    export_example(args.carriers, cap)
