"""Sample individual sizes and mixing states in paired CUDA parcels."""

import argparse
import csv
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from campaign import (
    HERE,
    RESULTS,
    V0,
    VOL,
    Case,
    build_cuda,
    call_gpu,
    parcel,
    prepare,
    state,
    write,
)
from scipy.special import ndtr, ndtri

MASTER_SEED = 53947
REGIMES = ("marine", "smoke")
RECIPE_SAMPLING = ("common",)
MIXING = ("homogeneous", "external")


@dataclass(frozen=True)
class Sample:
    """Continuous-size sample for one tagged population."""

    diameter: np.ndarray
    kappa: np.ndarray
    weight_cm3: np.ndarray


def sample_sizes(mode, u):
    """Inverse-transform the same truncated lognormal law as campaign.Mode."""
    low, high = ndtr(
        np.log(np.asarray(mode.limits) / mode.dg) / np.log(mode.sd)
    )
    return mode.dg * np.exp(ndtri(low + (high - low) * u) * np.log(mode.sd))


def raw_samples(groups, draws, replicate, regime):
    """Return shared random numbers across refinements and paired treatments."""

    def rng(stream):
        return np.random.default_rng(
            np.random.SeedSequence([MASTER_SEED, replicate, stream])
        )

    target_modes = groups[:2]
    first = sample_sizes(target_modes[0], rng(0).random(draws))
    second = sample_sizes(target_modes[1], rng(1).random(draws))
    backgrounds = []
    for j, mode in enumerate(groups[2:]):
        stream = 100 * (1 + REGIMES.index(regime)) + 2 * j
        d = sample_sizes(mode, rng(stream).random(draws))
        # Give twins a common diameter. Their symmetric κ values conserve both
        # number- and volume-weighted mean κ at this diameter, exactly.
        deviation = rng(stream + 1).uniform(0.25, 0.75, size=draws)
        spread = np.r_[
            mode.kappa * (1 - deviation), mode.kappa * (1 + deviation)
        ]
        diam = np.tile(d, 2)
        wt = np.full(2 * draws, mode.count / (2 * draws))
        homogeneous = Sample(diam, np.full_like(diam, mode.kappa), wt)
        external = Sample(diam, spread, wt)
        if not np.allclose(
            (spread[:draws] + spread[draws:]) / 2,
            mode.kappa,
            rtol=1e-15,
            atol=0,
        ):
            raise ValueError("External mixing changed the mean κ")
        backgrounds.append((homogeneous, external))
    return first, second, backgrounds


def carriers(mode, sample, case, nodes):
    """Build campaign CUDA particles and growth tables at sampled diameters."""
    a = sample.diameter[:, None] * 0.5e-9
    log_growth = np.linspace(0.0, np.log(1500.0), nodes)
    wet = a * np.exp(np.maximum(log_growth, 1e-9))
    if mode.recipe == "bg":
        water = np.expm1(3 * log_growth)
        aw = water / (water + sample.kappa[:, None])
        sigma = np.full_like(wet, parcel.SIGMA)
    else:
        aw, sigma_mn, *_ = state(a, wet, mode.recipe, "finite", case.scenario)
        sigma = sigma_mn * 0.001
    table = np.stack([aw, sigma], axis=-1)
    table[:, 0, 0] = 0.0
    saturation = table[:, :, 0] * np.exp(
        2 * sigma / (parcel.RV * 1000 * 285 * wet)
    )
    above = saturation >= 1.0
    if not np.all(above.any(axis=1)):
        raise ValueError("No stable haze branch")
    i = above.argmax(axis=1)
    j = np.arange(i.size)
    lo = saturation[j, i - 1]
    fraction = (1 - lo) / (saturation[j, i] - lo)
    r_haze = a[:, 0] * np.exp(log_growth[i - 1] + log_growth[1] * fraction)
    density, molar_mass = (
        (1500.0, 0.15) if mode.recipe == "bg" else (1665.0, 0.192124)
    )
    n_abs = VOL * a[:, 0] ** 3 * density * mode.absorb / molar_mass
    zero = np.zeros_like(r_haze)
    seed = np.column_stack(
        (a[:, 0], sample.weight_cm3 * 1e6 * V0, r_haze, zero, n_abs, zero)
    )
    return seed, table, sample.diameter


def run_seed(func, case, n, index, nodes, dt, detail=False):
    """Run all four matched sampling × mixing treatments for a single seed."""
    ms, _, params, wind = prepare(case, 1, nodes, dt, {})
    d0, d1, bg = raw_samples(ms, n, index, case.regime)
    zero = np.zeros(n)
    target_weight = np.full(n, ms[0].count / n)
    octyl = Sample(d0, zero, target_weight)
    dodecyl = [Sample(d, zero, target_weight) for d in (d0, d1)]
    t0 = carriers(ms[0], octyl, case, nodes)
    t1 = [carriers(ms[1], part, case, nodes) for part in dodecyl]
    bgs = [
        [
            carriers(mode, pair[k], case, nodes)
            for pair, mode in zip(bg, ms[2:], strict=True)
        ]
        for k in range(2)
    ]
    raw, batch, labels = [], [], []
    for s, sampling in enumerate(RECIPE_SAMPLING):
        for k, mixing in enumerate(MIXING):
            raw.append([octyl, dodecyl[s]] + [pair[k] for pair in bg])
            batch.append((ms, [t0, t1[s], *bgs[k]], params, wind))
            labels.append((sampling, mixing))
    air, particles, _ = call_gpu(func, batch, nodes, len(wind))
    rows, individual = [], []
    for i, (sampling, mixing) in enumerate(labels):
        gas = air[i, 3] * 1e9
        key = dict(
            regime=case.regime,
            draws=n,
            seed=index,
            sampling=sampling,
            mixing=mixing,
        )
        row = dict(
            **key,
            carriers=particles.shape[1],
            nodes=nodes,
            dt_s=dt,
            w_m_s=case.w,
            peak_pct=air[i, 0],
            height_m=air[i, 6],
            gas_ug_kg=gas,
            water_error_kg_kg=air[i, 1],
            organic_error_kg_kg=air[i, 2],
            heat_error_J_kg=air[i, 10],
            rh_solver_error=air[i, 11],
        )
        counts = {}
        condensed = 0.0
        offset = 0
        bg_count = 0.0
        for mode, sample in zip(ms, raw[i], strict=True):
            p = particles[i, offset : offset + len(sample.diameter)]
            offset += len(p)
            active = p[:, 1] > 0
            if (
                np.any(p[:, 0] <= 0)
                or np.any(p[:, 3] < 0)
                or np.any(p[:, 2] <= 0)
            ):
                raise ValueError("Invalid particle weight, mass or radius")
            counts[mode.label] = float(p[active, 0].sum())
            condensed += float(np.sum(p[:, 3] * p[:, 4]) * 1e9)
            if mode.recipe == "bg":
                bg_count += counts[mode.label]
            if detail:
                for j in range(len(p)):
                    individual.append(
                        dict(
                            **key,
                            population=mode.label,
                            marker=j,
                            dry_nm=sample.diameter[j],
                            kappa_bg=sample.kappa[j]
                            if mode.recipe == "bg"
                            else "",
                            initial_cm3=sample.weight_cm3[j],
                            final_cm3=p[j, 0],
                            wet_final_um=p[j, 2] * 1e6,
                            organic_final_kg=p[j, 3],
                            final_persistence_log_margin=p[j, 1],
                            activated=int(active[j]),
                        )
                    )
        row.update(
            octyl_cm3=counts["octyl"],
            dodecyl_cm3=counts["dodecyl"],
            background_cm3=bg_count,
            total_cm3=sum(counts.values()),
            tagged_difference_cm3=counts["dodecyl"] - counts["octyl"],
            particle_organic_ug_kg=condensed,
        )
        recovered = abs(gas + condensed - air[i, 4] * 1e9)
        if (
            abs(row["height_m"] - 100) > 1e-7
            or row["water_error_kg_kg"] > 1e-11
            or row["organic_error_kg_kg"] > 1e-14
            or recovered > 1e-6
        ):
            raise ValueError(
                ("Particle-resolved parcel failed conservation", row)
            )
        rows.append(row)
    return rows, individual


def describe(x, seed):
    """Report draw-to-draw spread and bootstrap CI on the mean independently."""
    x = np.asarray(x, float)
    b = (
        np.random.default_rng(seed)
        .choice(x, (5000, len(x)), replace=True)
        .mean(axis=1)
    )
    return dict(
        mean=x.mean(),
        sd=x.std(ddof=1),
        p025=np.quantile(x, 0.025),
        p975=np.quantile(x, 0.975),
        mean_ci_lo=np.quantile(b, 0.025),
        mean_ci_hi=np.quantile(b, 0.975),
        positive=int((x > 1e-9).sum()),
        tied=int((np.abs(x) <= 1e-9).sum()),
        negative=int((x < -1e-9).sum()),
    )


def summarize(rows):
    """Aggregate replicate endpoints and seed-wise external-mixing contrasts."""
    groups = {}
    for r in rows:
        key = (r["regime"], r["draws"], r["sampling"], r["mixing"])
        groups.setdefault(key, {})[r["seed"]] = r
    summaries = []
    for (region, n, sampling, mix), group in groups.items():
        base = groups[region, n, sampling, "homogeneous"]
        for field in (
            "octyl_cm3",
            "dodecyl_cm3",
            "background_cm3",
            "total_cm3",
            "tagged_difference_cm3",
            "peak_pct",
            "particle_organic_ug_kg",
        ):
            values = [r[field] for r in group.values()]
            key = dict(
                regime=region,
                draws=n,
                sampling=sampling,
                mixing=mix,
                field=field,
                replicates=len(group),
            )
            summaries.append(
                dict(**key, contrast="level", **describe(values, 711))
            )
            if mix == "external":
                d = [r[field] - base[seed][field] for seed, r in group.items()]
                summaries.append(
                    dict(
                        **key,
                        contrast="external_minus_homogeneous",
                        **describe(d, 711),
                    )
                )
    return summaries


def main():
    """Run and archive the stochastic ensemble."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--draws", nargs="+", type=int, default=[120])
    parser.add_argument("--nodes", type=int, default=513)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.replicates < 2 or any(n < 2 or 6 * n > 1024 for n in args.draws):
        raise ValueError(
            "Need at least two replicates and 2–170 draws per population"
        )
    args.out.mkdir(exist_ok=True, parents=True)
    source_hashes = {
        p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
        for p in ("cloud.cu", "organic.cuh", "campaign.py", "resolved.py")
    }
    func = build_cuda()
    rows, markers = [], []
    start = time.monotonic()
    for n in args.draws:
        # Keep the original stream index (smoke is the second reference regime).
        for region in ("smoke",):
            case = Case(regime=region, w=1.0, gas=1.0, entr=0.0)
            for j in range(args.replicates):
                rr, mm = run_seed(
                    func, case, n, j, args.nodes, args.dt, detail=j == 0
                )
                rows.extend(rr)
                markers.extend(mm)
                if j % 8 == 0:
                    print(
                        region,
                        n,
                        j,
                        f"{time.monotonic() - start:.1f}s",
                        flush=True,
                    )
    base = args.out / ("gpu_resolved" + args.tag)
    write(base.with_suffix(".csv"), rows)
    write(base.with_name(base.name + "_summary.csv"), summarize(rows))
    with gzip.open(
        base.with_name(base.name + "_markers.csv.gz"), "wt", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=markers[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(markers)
    metadata = dict(
        replicates=args.replicates,
        draws=args.draws,
        nodes=args.nodes,
        dt_s=args.dt,
        master_seed=MASTER_SEED,
        parcel_count=len(rows),
        elapsed_s=time.monotonic() - start,
        first_seed_markers=len(markers),
        max_water_error_kg_kg=max(r["water_error_kg_kg"] for r in rows),
        max_organic_error_kg_kg=max(r["organic_error_kg_kg"] for r in rows),
        max_rh_solver_error=max(r["rh_solver_error"] for r in rows),
        sha256=source_hashes,
    )
    base.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
