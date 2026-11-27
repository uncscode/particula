"""Quantify loss of a physical mixing-state signal after compression."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stage3_evidence import write


def series(matches):
    """Return both wet and gas trajectories in concentration units."""
    return np.stack([np.column_stack((s["wet"], s["gas"])) for s in matches])


def collect(folder):
    """Join uniform and segregated states only when their input matches."""
    pool = defaultdict(lambda: defaultdict(dict))
    for p in sorted(folder.glob("*.npz")):
        with np.load(p) as d:
            c = tuple(int(d[k]) for k in ("n", "s", "f", "d", "c"))
            opts = (
                str(d["grouping"]),
                int(d["dry_bins"]),
                str(d["method"]),
                float(d["tolerance"]),
            )
            r = {
                k: d[k]
                for k in (
                    "wet",
                    "gas",
                    "v",
                    "t",
                    "bulk",
                    "exact_flux",
                    "group_flux",
                    "groups",
                    "elapsed_s",
                    "exact_flux_s",
                )
            }
            pool[c][opts][int(d["mix"])] = r
    return pool


def score(cache):
    """Measure phase and mixing-state contrast errors."""
    rows, curves = [], []
    for cfg, group in collect(cache / "mix-groups").items():
        full = {key: (v[0], v[1]) for key, v in group.items() if len(v) == 2}
        sizes = sorted(
            v for v in full if v[0] == "composition" and v[2] == "DOP853"
        )
        if len(sizes) < 2:
            continue
        larger, smaller = sizes[-1], sizes[-2]
        units = [k for k in full if k[0] == "unit" and k[2] == "DOP853"]
        refkey = units[0] if units else larger
        target = series(full[refkey])
        coarse = series(full[larger if units else smaller])
        delta = target[1] - target[0]
        residual = abs((coarse[1] - coarse[0]) - delta)
        norm = np.maximum((abs(target[1]) + abs(target[0])) / 2, 1e-11)
        scale = np.maximum(abs(target), 1e-11)
        # Cross-integrator check is explicitly local to the same quadrature.
        common = [
            k
            for k in full
            if k[0] == "composition"
            and k[2] == "BDF"
            and (k[0], k[1], "DOP853", 2e-8) in full
        ]
        cross = 0.0
        for k in common:
            b = series(full[k])
            a = series(full[(k[0], k[1], "DOP853", 2e-8)])
            cross = max(
                cross, 100 * np.max(abs(a - b) / np.maximum(abs(b), 1e-11))
            )
        n, s, f, d, c = cfg
        largest = (
            1
            + np.unravel_index(
                np.argmax(abs(delta[:, 1:s]) / norm[:, 1:s]),
                delta[:, 1:s].shape,
            )[1]
        )
        for opts, pair in sorted(full.items()):
            a, b = pair
            np.testing.assert_allclose(
                a["bulk"], b["bulk"], rtol=1e-11, atol=1e-22
            )
            np.testing.assert_allclose(a["v"], b["v"], rtol=1e-14, atol=0)
            x = series(pair)
            dif = x[1] - x[0]
            flux = b["exact_flux"] - a["exact_flux"]
            estimated = b["group_flux"] - a["group_flux"]
            fluxnorm = max(abs(flux).max(), np.finfo(float).tiny)
            for j in (0, largest):
                idx = int(np.argmax(abs(delta[:, j]) / norm[:, j]))
                sig = delta[idx, j]
                resolved = abs(sig) > max(
                    5 * residual[idx, j], 1e-10 * norm[idx, j]
                )
                row = dict(
                    n=n,
                    species=s,
                    family=f,
                    density_class=d,
                    chemistry=c,
                    number_density_m3=n / float(a["v"]),
                    grouping=opts[0],
                    dry_bins=opts[1],
                    ode_method=opts[2],
                    tol=opts[3],
                    uniform_groups=int(a["groups"]),
                    segregated_groups=int(b["groups"]),
                    reference_grouping=refkey[0],
                    reference_dry_bins=refkey[1],
                    reference_segregated_groups=int(full[refkey][1]["groups"]),
                    field="water" if j == 0 else f"organic_{j}",
                    peak_time_s=a["t"][idx],
                    reference_difference_kg_m3=sig,
                    group_difference_kg_m3=dif[idx, j],
                    reference_peak_pct=100 * sig / norm[idx, j],
                    contrast_refinement_pct=100
                    * residual[idx, j]
                    / norm[idx, j],
                    contrast_resolved=bool(resolved),
                    recovered_fraction=dif[idx, j] / sig if resolved else "",
                    max_phase_error_pct=100 * np.max(abs(x - target) / scale),
                    max_phase_size_refinement_pct=100
                    * np.max(abs(target - coarse) / scale),
                    max_same_grid_BDF_DOP853_pct=cross if common else "",
                    initial_flux_contrast_error_pct_of_largest=100
                    * abs(estimated - flux).max()
                    / fluxnorm,
                    uniform_direct_million_or_unit_flux_s=float(
                        a["exact_flux_s"]
                    ),
                    segregated_advance_s=float(b["elapsed_s"]),
                )
                rows.append(row)
                if (
                    n == 1000000
                    and s == 32
                    and f == 86011
                    and d == 8
                    and opts[2] == "DOP853"
                    and opts[1] in (32, 128)
                ):
                    for i, h in enumerate(a["t"]):
                        curves.append(
                            dict(
                                time_s=h,
                                field=row["field"],
                                grouping=opts[0],
                                dry_bins=opts[1],
                                groups=row["segregated_groups"],
                                reference_pct=100 * delta[i, j] / norm[i, j],
                                grouped_pct=100 * dif[i, j] / norm[i, j],
                            )
                        )
    return rows, curves


def chart(rows, path):
    """Plot matched pulse contrasts for dry-only and composition grouping."""
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.1), layout="constrained")
    fields = sorted({r["field"] for r in rows}, key=lambda v: v != "water")
    for ax, field in zip(axs, fields, strict=True):
        rr = [r for r in rows if r["field"] == field]
        for tag, label, col, ls in [
            ("size", "Dry size only", "#a85628", ":"),
            ("composition", "Dry size and composition", "#277d69", "--"),
        ]:
            v = [r for r in rr if r["grouping"] == tag and r["dry_bins"] == 32]
            if not v:
                continue
            ts = [1000 * r["time_s"] for r in v]
            if tag == "size":
                ax.plot(
                    ts,
                    [r["reference_pct"] for r in v],
                    label="Fine physical reference",
                    color="#193b5a",
                    lw=2.3,
                )
            ax.plot(
                ts,
                [r["grouped_pct"] for r in v],
                label=label,
                color=col,
                ls=ls,
                lw=2,
            )
        ax.set_title(
            "Water"
            if field == "water"
            else field.replace("_", " ").capitalize()
        )
        ax.set_xlabel("Time (ms)")
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("Segregated minus uniform particulate mass (%)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Which particle properties preserve the response to a vapor pulse?"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    rows, curves = score(a.cache)
    write(a.results / "mixture-grouping.csv", rows)
    write(a.results / "mixture-group-response.csv", curves)
    chart(curves, a.results / "grouped-mixture-pulse.png")
