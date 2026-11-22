"""Joint input propagation for conditional cluster growth and buffered neutral dimers.

Run from the packet with the repository interpreter. Results are independent of CWD.
"""

import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from design import draws

sys.path.insert(0, str(ROOT / "analysis"))
from chemistry import STATES, SPECIES, THERMO, DENS, DIAM, kernel, solve
from dimers import calculation
from reference import ACID, T, AIR, RH, NH3, WINDOW, Q, dump_csv, cold_condition_label

THERMOS = ("optimized", "posterior_median")
ATTRACTION = (0.0, 6.4e-20)
NAMES = (
    "pentamer_probability",
    "monomer_exit_probability",
    "chamber_loss_probability",
    "lifetime_s",
)
QUANT = [0.05, 0.5, 0.95]


def tagged(temp, acid, base, wall, chemical, hamaker):
    """Solve all design members together for common state and collision structure."""
    n = len(acid)
    ng = len(STATES)
    gen = np.zeros((n, ng, ng))
    out = np.zeros((n, ng, 4))
    k = kernel(temp, hamaker)
    std = 101325 / (1.380649e-23 * temp) * 1e-6
    for i, (a, b) in enumerate(STATES):

        def transfer(target, rate):
            if target in STATES:
                gen[:, i, STATES.index(target)] += rate
            else:
                out[:, i, 0 if target[0] == 5 else 1] += rate

        transfer((a + 1, b), k[2 + i, 0] * acid)
        if b < a:
            transfer((a, b + 1), k[2 + i, 1] * base)
        parent = (a, b - 1) if a == b else (a - 1, b)
        release = int(a == b)
        h, s = THERMO[a, b][2 * chemical : 2 * chemical + 2]
        evap = (
            k[SPECIES.index(parent), release]
            * std
            * np.exp((-1000 * h + temp * s) / (1.987204258 * temp))
        )
        transfer(parent, evap)
        out[:, i, 2] = 0.001 * wall / (DIAM[2 + i] * 1e9) + 9.6e-5
        gen[:, i, i] = -gen[:, i].sum(axis=1) - out[:, i, :3].sum(axis=1)
    out[:, :, 3] = 1
    solved = np.linalg.solve(-gen, out)
    closure = np.max(np.abs(solved[:, :, :3].sum(axis=2) - 1))
    if (
        not np.all(np.isfinite(solved))
        or np.min(solved) < -1e-12
        or closure > 2e-8
    ):
        raise ValueError(("first-passage", closure))
    return solved[:, 0], closure


@lru_cache(None)
def dimer_kernel(temp, hamaker):
    # Match dimers.calculation's acid/base masses exactly.
    import particula as par

    m = (
        np.array([98.079, 17.03052, 115.10952, 196.158, 213.18852, 230.21904])
        * 1.66053906660e-27
    )
    d = (6 * m / (np.pi * DENS)) ** (1 / 3) * 1e9
    p = par.particles.ParticleData(
        masses=m[None, :, None],
        density=np.array([DENS]),
        charge=np.zeros((1, 6)),
        concentration=np.ones((1, 6)),
        volume=np.ones(1),
        distribution_type="discrete",
    )
    strategy = par.dynamics.CoagulationFactory().get_strategy(
        "van_der_waals",
        dict(
            hamaker_constant=hamaker,
            distribution_type="discrete",
            relative_permittivity=1.0,
        ),
    )
    return strategy.kernel(p, temp, 1.005 * 101325) * 1e6, d


def dimer_equilibrium(
    temp, rh, acid, base, wall, chemical, hamaker, thermo_shift
):
    """Eliminate linear dimer balances and bracket the AB balance. Units cm^-3 and s."""
    k, diam = dimer_kernel(temp, hamaker)
    rates = []
    for cluster, pair in zip(
        ((1, 1), (2, 1), (2, 2)), (k[0, 1], k[0, 2], k[1, 4])
    ):
        idx = int(chemical == "posterior_median") * 2
        h, s = THERMO[cluster][idx : idx + 2]
        if chemical == "2015_adduct" and cluster == (1, 1):
            h, s = 16.1 + 0.6 * thermo_shift, 26.4 - 2.6 * thermo_shift
        std = 101325 / (1.380649e-23 * temp) / 1e6
        rates.append(
            pair * std * np.exp((-1000 * h + temp * s) / (1.987204258 * temp))
        )
    ke0, ke1, ke2 = rates
    e0 = (
        0.5
        * k[0, 0]
        / (1.380649e-23 * temp * 1e6 * np.exp(10109 / temp - 35.03))
        * (20 / rh) ** 0.5
    )
    loss = 0.001 * wall[:, None] / diam + 9.6e-5

    def balance(ab):
        a = acid - ab
        s0, s1, s2 = (
            0.5 * k[0, 0] * a * a,
            k[0, 2] * a * ab,
            0.5 * k[2, 2] * ab * ab,
        )
        b0, b1 = k[3, 1] * base, k[4, 1] * base
        sink = k[3:, 0] * a[:, None] + k[3:, 2] * ab[:, None] + loss[:, 3:]
        x = s0 / (e0 + b0 + sink[:, 0])
        v = ke2 + sink[:, 2]
        # Schur complement in cancellation-free form (b1 >= 0, all sinks positive).
        y1 = (s1 + b0 * x + ke2 * s2 / v) / (
            ke1 + sink[:, 1] + b1 * sink[:, 2] / v
        )
        y2 = (s2 + b1 * y1) / v
        d = np.column_stack([x, y1, y2])
        source = k[0, 1] * a * base + ke1 * y1
        debit = (ke0 + loss[:, 2] + d @ k[3:, 2]) * ab + s1 + 2 * s2
        return source - debit, d, source + debit

    lo, hi = np.zeros_like(acid), acid.copy()
    assert (balance(lo)[0] >= 0).all() and (balance(hi)[0] <= 0).all()
    for _ in range(80):
        mid = (lo + hi) / 2
        rising = balance(mid)[0] > 0
        lo, hi = np.where(rising, mid, lo), np.where(rising, hi, mid)
    residual, d, turnover = balance((lo + hi) / 2)
    error = np.max(abs(residual) / turnover)
    if error > 1e-8 or not np.isfinite(d).all() or d.min() <= 0:
        raise ValueError(("dimer balance", error))
    return d.sum(axis=1), error


def fields(records, key):
    return np.array([r[key] for r in records])


def main():
    common = draws()
    n = len(common)
    ids, blocks = fields(common, "id"), fields(common, "block")
    rows = []
    for block in np.unique(blocks):
        seed = 92401 + int(block) * 13
        u = qmc.Sobol(7, scramble=True, seed=seed).random_base2(
            int(np.log2((blocks == block).sum()))
        )
        for i, z in zip(ids[blocks == block], 2 * u - 1):
            rows.append(
                dict(
                    id=int(i),
                    block=int(block),
                    seed=seed,
                    cloud5_acid=1 + 0.30 * z[0],
                    cloud5_dimer=1 + 0.30 * z[0] + 0.20 * z[1],
                    cloud5_ammonia=2 ** z[2],
                    cloud5_wall=2 ** z[3],
                    thermo_shift=z[4],
                    external_acid=1.5 ** z[5],
                    external_ammonia=1.5 ** z[6],
                )
            )
    dump_csv(HERE / "cluster_draws.csv", rows)
    acids = fields(common, "acid")
    walls = fields(common, "wall")
    li = np.flatnonzero(WINDOW & (Q == 0))
    first = np.empty((len(li), len(THERMOS), len(ATTRACTION), n, 4))
    summary = []
    checks = dict(
        max_closure=0.0,
        max_tagged_nominal_rel=0.0,
        max_bdf_relative=0.0,
        max_dimer_balance=0.0,
        seed_count=int(len(np.unique(blocks))),
        draws=n,
    )

    def save_summary(case, chem, ham, name, value, table):
        quant = np.quantile(value, QUANT)
        block_quant = np.array(
            [np.quantile(value[blocks == b], QUANT) for b in np.unique(blocks)]
        )
        table.append(
            dict(
                case=case,
                chemical=chem,
                hamaker_J=ham,
                quantity=name,
                q05=quant[0],
                q50=quant[1],
                q95=quant[2],
                block_median_min=min(
                    np.median(value[blocks == b]) for b in np.unique(blocks)
                ),
                block_median_max=block_quant[:, 1].max(),
                block_q05_min=block_quant[:, 0].min(),
                block_q05_max=block_quant[:, 0].max(),
                block_q95_min=block_quant[:, 2].min(),
                block_q95_max=block_quant[:, 2].max(),
            )
        )

    for j, i in enumerate(li):
        bracket = (
            0
            if NH3[i] < 1
            else 1
            if NH3[i] <= 10
            else 3
            if NH3[i] >= 100
            else 2
        )
        acid = ACID[i] * 1e6
        base = NH3[i] * AIR[i] * 1e-12
        for c, chem in enumerate(THERMOS):
            for h, ham in enumerate(ATTRACTION):
                x, err = tagged(
                    T[i],
                    np.r_[acid * acids, acid],
                    np.r_[base * fields(common, f"base{bracket}"), base],
                    np.r_[walls, 1],
                    c,
                    ham,
                )
                reference, tau = solve(T[i], acid, base, c, ham)
                nominal_rel = np.max(abs(x[-1] / np.r_[reference, tau] - 1))
                checks["max_tagged_nominal_rel"] = max(
                    checks["max_tagged_nominal_rel"], nominal_rel
                )
                checks["max_closure"] = max(checks["max_closure"], err)
                first[j, c, h] = x[:-1]
                for m, name in enumerate(NAMES):
                    save_summary(i + 2, chem, ham, name, x[:-1, m], summary)
    dump_csv(HERE / "cluster_first_passage.csv", summary)
    np.savez_compressed(
        HERE / "cluster_first_passage.npz",
        values=first,
        lines=li + 2,
        chemical=THERMOS,
        hamaker_J=ATTRACTION,
        draw_id=ids,
        quantities=NAMES,
    )
    companion = np.genfromtxt(
        ROOT / "rate_sources/kurten_fig9_248K_dimers.csv",
        names=True,
        delimiter=",",
    )
    scenarios = []
    for j, x in enumerate(companion):
        t, rh, nh, ac, obs = list(x)[:5]
        nz = nh * 1.005 * 101325 / (1.380649e-23 * t) * 1e-18
        scenarios.append(
            (
                f"CLOUD5_{j + 1}",
                t,
                rh,
                ac,
                nz,
                obs,
                fields(rows, "cloud5_acid"),
                fields(rows, "cloud5_ammonia"),
                fields(rows, "cloud5_wall"),
                fields(rows, "cloud5_dimer"),
            )
        )
    scenarios.append(
        (
            "flowtube_265K",
            265.0,
            10.0,
            1.9e9,
            3.8e9,
            1.1e7,
            fields(rows, "external_acid"),
            fields(rows, "external_ammonia"),
            np.ones(n),
            np.ones(n),
        )
    )
    for line in (332, 334, 336, 338, 344, 352, 366):
        i = line - 2
        bracket = (
            0
            if NH3[i] < 1
            else 1
            if NH3[i] <= 10
            else 3
            if NH3[i] >= 100
            else 2
        )
        scenarios.append(
            (
                f"archive_{line}",
                T[i],
                RH[i],
                ACID[i] * 1e6,
                NH3[i] * AIR[i] * 1e-12,
                None,
                acids,
                fields(common, f"base{bracket}"),
                walls,
                np.ones(n),
            )
        )
    dim_rows, summary = [], []
    for case, temp, rh, ac, nh, obs, fa, fb, fw, fd in scenarios:
        for chem in (*THERMOS, "2015_adduct"):
            for ham in ATTRACTION:
                sizes, err = dimer_equilibrium(
                    temp,
                    rh,
                    np.r_[ac * fa, ac],
                    np.r_[nh * fb, nh],
                    np.r_[fw, 1],
                    chem,
                    ham,
                    np.r_[fields(rows, "thermo_shift"), 0],
                )
                checks["max_dimer_balance"] = max(
                    checks["max_dimer_balance"], err
                )
                bdf = calculation(temp, rh, ac, nh, chem, ham)["dimer_cm3"]
                checks["max_bdf_relative"] = max(
                    checks["max_bdf_relative"], abs(bdf / sizes[-1] - 1)
                )
                if case in ("CLOUD5_1", "archive_336") and chem in THERMOS:
                    for j in (3, 47):
                        bdf = calculation(
                            temp,
                            rh,
                            ac * fa[j],
                            nh * fb[j],
                            chem,
                            ham,
                            0.001 * fw[j],
                        )["dimer_cm3"]
                        checks["max_bdf_relative"] = max(
                            checks["max_bdf_relative"], abs(bdf / sizes[j] - 1)
                        )
                save_summary(case, chem, ham, "dimer_cm3", sizes[:-1], summary)
                if obs:
                    save_summary(
                        case,
                        chem,
                        ham,
                        "model_over_observation",
                        sizes[:-1] / (obs * fd),
                        summary,
                    )
                for did, vals in enumerate(sizes[:-1]):
                    dim_rows.append(
                        dict(
                            case=case,
                            chemical=chem,
                            hamaker_J=ham,
                            id=did,
                            acid_cm3=ac * fa[did],
                            ammonia_cm3=nh * fb[did],
                            wall_scale=fw[did],
                            dimer_cm3=vals,
                            comparison_cm3=obs * fd[did] if obs else "",
                        )
                    )
    assert (
        checks["max_tagged_nominal_rel"] < 1e-8
        and checks["max_bdf_relative"] < 1e-5
    ), checks
    dump_csv(HERE / "cluster_dimers.csv", dim_rows)
    dump_csv(HERE / "cluster_dimer_summary.csv", summary)
    paired = []
    for h, ham in enumerate(ATTRACTION):
        for j, i in enumerate(li):
            save_summary(
                i + 2,
                "optimized_over_marginal_medians",
                ham,
                "paired_pentamer_ratio",
                first[j, 0, h, :, 0] / first[j, 1, h, :, 0],
                paired,
            )
        for chem in (*THERMOS, "2015_adduct"):
            selected = [
                r
                for r in dim_rows
                if r["case"].startswith("CLOUD5")
                and r["chemical"] == chem
                and r["hamaker_J"] == ham
            ]
            ratio = np.array(
                [r["dimer_cm3"] / r["comparison_cm3"] for r in selected]
            ).reshape(3, n)
            factor = np.exp(np.mean(np.log(ratio) ** 2, axis=0) ** 0.5)
            save_summary(
                "CLOUD5_all_three",
                chem,
                ham,
                "rms_log_error_factor",
                factor,
                paired,
            )
            if chem == "optimized":
                optimized = factor
            else:
                checks[
                    f"count_optimized_lower_rms_than_{chem}_hamaker_{ham}"
                ] = int((optimized < factor).sum())
    dump_csv(HERE / "cluster_paired_summary.csv", paired)
    (HERE / "cluster_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print(json.dumps(checks), flush=True)
    plot()


def plot():
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")

    def records(name):
        with (HERE / name).open() as f:
            return list(csv.DictReader(f))

    data = records("cluster_first_passage.csv")
    dimer = records("cluster_dimer_summary.csv")
    colors = ["#216675", "#bb5236", "#635482"]
    labs = ["2019 optimized", "2019 marginal medians", "2015 adduct"]
    labels = [(str(i), cold_condition_label(i, separator="\n")) for i in [366, 332, 334, 344, 336, 338]]
    channels = [
        (labels, data, "pentamer_probability"),
        (
            [
                (f"CLOUD5_{i}", s)
                for i, s in enumerate(["7.3 pptv", "4.4 pptv", "2.3 pptv"], 1)
            ]
            + [("flowtube_265K", "265 K")],
            dimer,
            "model_over_observation",
        ),
    ]
    for ax, (ticks, rows, stat) in zip(axes, channels):
        chems = THERMOS if rows is data else (*THERMOS, "2015_adduct")
        for c, chem in enumerate(chems):
            for j, (key, label) in enumerate(ticks):
                item = next(
                    r
                    for r in rows
                    if r["case"] == key
                    and r["chemical"] == chem
                    and float(r["hamaker_J"]) > 0
                    and r["quantity"] == stat
                )
                lo, med, hi = (float(item[k]) for k in ("q05", "q50", "q95"))
                ax.errorbar(
                    j + (c - (len(chems) - 1) / 2) * 0.18,
                    med,
                    yerr=[[med - lo], [hi - med]],
                    fmt="o",
                    ms=5,
                    capsize=3,
                    color=colors[c],
                    lw=1.5,
                )
        ax.set(
            xticks=range(len(ticks)),
            xticklabels=[s for _, s in ticks],
            yscale="log",
        )
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set(
        title="A   Growth from an acid–ammonia pair",
        xlabel="Temperature and H₂SO₄ (10⁶ molecules cm⁻³)",
        ylabel="Probability of reaching five acids",
    )
    axes[1].set(
        title="B   Neutral dimer abundance",
        xlabel="248 K CLOUD ammonia or flow-tube temperature",
        ylabel="Predicted / measured dimer concentration",
    )
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].axhline(1, color=".3", ls="--", lw=1)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=col, ls="", label=lab)
            for col, lab in zip(colors, labs)
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(HERE / "cluster-bounds.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
