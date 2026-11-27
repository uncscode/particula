"""Compute simultaneous paired family and collision-seed uncertainty bands."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from scipy.stats import t as student
from stage2_accuracy import NEAR_ZERO, write_rows
from stage2_score_split import arrays


def grouped_observables(v, s):
    """Append particle and whole-system mass to the phase totals."""
    cols = v[..., : 2 * s + 3]
    return np.concatenate(
        (
            cols,
            cols[..., 1 : s + 3].sum(-1, keepdims=True),
            cols[..., 1:].sum(-1, keepdims=True),
        ),
        axis=-1,
    )


def panel(path, root, draws, seed, prefix=None):
    """Resample complete families and paired event histories."""
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        keys = data["keys"]
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError("Duplicated event streams require nested sampling")
        keep = np.ones(len(keys), dtype=bool)
        if prefix is not None:
            seeds = np.unique(keys[:, 3])
            if prefix < 2 or prefix > len(seeds):
                raise ValueError("Requested seed prefix is outside the archive")
            keep = np.isin(keys[:, 3], seeds[:prefix])
            meta["timed_forecast_trajectories"] = len(keys)
            meta["timing_scope"] = "complete archived forecast"
            keys = keys[keep]
        hi, lo, nref = arrays({"keys": keys}, root, meta)
        if nref < 2:
            raise ValueError(
                "Two independent reference tolerances are required"
            )
        times = data["t"]
        names = list(data["names"])
        s = sum(k.startswith("gas_") for k in names)
        fields = names[: 2 * s + 3] + ["particle_mass_total", "mass_total"]
        main = grouped_observables(data["obs"][keep], s)
        hi, lo = [grouped_observables(x, s) for x in (hi, lo)]
        wall = float(data["seconds"].sum())
        drift = float(data["max_drift"])
    groups = np.unique(keys[:, :3], axis=0)
    runs = np.unique(keys[:, 3])
    idx = np.stack([np.flatnonzero((keys[:, :3] == g).all(1)) for g in groups])
    idx = np.take_along_axis(idx, np.argsort(keys[idx, 3], axis=1), axis=1)
    np.testing.assert_array_equal(
        keys[idx, 3], np.broadcast_to(runs, idx.shape)
    )
    if len(runs) < 2:
        raise ValueError("Stochastic validation requires independent seeds")
    reference = hi[idx]
    err = (main - hi)[idx]
    numerical = abs(hi - lo)[idx].mean(1)
    mean = reference.mean(1)
    scales = np.maximum(abs(mean), NEAR_ZERO)
    center = err.mean(1) / scales
    families = np.unique(groups[:, 0])
    rng = np.random.default_rng(seed)
    # A shared seed can drive different dry families. Resample it jointly.
    # Resample complete dry families with both mixtures and both densities.
    max_seed = []
    max_family = []
    strata = np.unique(groups[:, 1:], axis=0)
    family_idx = np.stack([np.flatnonzero(groups[:, 0] == f) for f in families])
    np.testing.assert_array_equal(
        groups[family_idx][:, :, 1:],
        np.broadcast_to(strata, (len(families), *strata.shape)),
    )
    family_mean = center[family_idx].mean(0)
    family_margins = []
    normalized = err / scales[:, None, :, :]
    influence = err - np.where(
        abs(mean[:, None]) > NEAR_ZERO, center[:, None] * reference, 0.0
    )
    ratio_se = influence.std(1, ddof=1) / (np.sqrt(len(runs)) * scales)
    for start in range(0, draws, 32):
        chunk = min(32, draws - start)
        weights = rng.multinomial(
            len(runs), np.full(len(runs), 1 / len(runs)), size=chunk
        ) / len(runs)
        sample = np.einsum("cn,gntd->cgtd", weights, err, optimize=True)
        denom = np.einsum("cn,gntd->cgtd", weights, reference, optimize=True)
        ratio = sample / np.maximum(abs(denom), NEAR_ZERO)
        dev = abs(ratio - center[None])
        dev = dev / np.maximum(ratio_se[None], 1e-14)
        for i in range(chunk):
            ix = rng.choice(families, len(families), replace=True)
            select = np.isin(groups[:, 0], ix)
            pick = family_idx[np.searchsorted(families, ix)]
            sample_mean = ratio[i, pick].mean(0)
            family_margins.append(abs(sample_mean - family_mean).max((0, 1)))
            max_seed.append([dev[i, :, :, 0].max(), dev[i, :, :, 1:].max()])
            max_family.append(
                [dev[i, select, :, 0].max(), dev[i, select, :, 1:].max()]
            )
    family_margin = np.quantile(family_margins, 0.95, axis=0)
    bs = np.quantile(max_seed, 0.95, axis=0)
    bf = np.quantile(max_family, 0.95, axis=0)
    # The paired t envelope additionally handles nonzero but infrequent deltas
    # and a noisy resolved reference denominator. Fields and time share alpha.
    alpha = 0.05 / (len(groups) * len(fields) * len(times))
    quant = student.ppf(1 - alpha / 2, len(runs) - 1)
    se = err.std(1, ddof=1) / np.sqrt(len(runs))
    sy = reference.std(1, ddof=1) / np.sqrt(len(runs))
    lower = np.where(abs(mean) <= NEAR_ZERO, NEAR_ZERO, abs(mean) - quant * sy)
    ubt = np.divide(
        abs(err.mean(1)) + quant * se + numerical,
        lower,
        out=np.full_like(mean, np.inf),
        where=lower > 0,
    )
    output = []
    for j, field in enumerate(fields):
        category = int(j != 0)
        threshold = 0.01 if j == 0 else 0.001
        raw = abs(center[:, :, j])
        boot = (
            raw
            + max(bs[category], bf[category]) * ratio_se[:, :, j]
            + numerical[:, :, j] / scales[:, :, j]
        )
        upper = np.maximum(boot, ubt[:, :, j])
        exact = (normalized[:, :, :, j] == 0).all(1) & (numerical[:, :, j] == 0)
        # The model and reference have identical initial and reset number.
        # A sampled zero difference alone cannot prove a future bound.
        upper[:, 0] = np.where(exact[:, 0], 0, upper[:, 0])
        passing = (upper <= threshold if j == 0 else upper < threshold).all(0)
        failed = (raw > threshold if j == 0 else raw >= threshold).any(0)
        bad = np.flatnonzero(~passing)
        i = bad[0] if len(bad) else len(times)
        diff = abs(err[:, :, :, j]) > 1e-12 * np.maximum(
            abs(reference[:, :, :, j]), NEAR_ZERO
        )
        miss = diff.any(2).sum(1)
        rare = np.where(
            miss == len(runs),
            1,
            beta.ppf(0.95, miss + 1, np.maximum(1, len(runs) - miss)),
        )
        k = np.unravel_index(raw.argmax(), raw.shape)
        output.append(
            dict(
                file=path.name,
                **meta,
                field=field,
                dry_families=len(families),
                settings=len(groups),
                independent_seeds=len(runs),
                refinements=nref,
                resamples=draws,
                worst_family=int(groups[k[0], 0]),
                worst_mix=int(groups[k[0], 1]),
                worst_density=int(groups[k[0], 2]),
                worst_time_s=times[k[1]],
                worst_mean_pct=100 * raw.max(),
                equal_family_mean_worst_pct=100
                * abs(family_mean[..., j]).max(),
                equal_family_mean_upper_pct=100
                * (
                    abs(family_mean[..., j]).max()
                    + family_margin[j]
                    + (numerical[:, :, j] / scales[:, :, j]).max()
                ),
                family_worst_error_p95_pct=100
                * np.quantile(raw[family_idx].max((1, 2)), 0.95),
                worst_t_upper_pct=100 * ubt[:, :, j].max(),
                seed_simultaneous_margin_pp=100
                * bs[category]
                * ratio_se[:, :, j].max(),
                family_seed_simultaneous_margin_pp=100
                * bf[category]
                * ratio_se[:, :, j].max(),
                worst_upper_pct=100 * upper.max(),
                number_of_unqualified_frames=int((~passing).sum()),
                mean_threshold_exceedances=int(failed.sum()),
                first_unqualified_s=times[i] if i < len(times) else "",
                last_contiguous_s=times[i - 1] if i else "",
                end_s=times[-1],
                max_reference_change_pct=100
                * (numerical[:, :, j] / scales[:, :, j]).max(),
                max_probability_any_discrepancy_upper95=rare.max(),
                max_relative_ledger=drift,
                forecast_seconds=wall,
            )
        )
    return output


def run(a):
    """Write an uncertainty-aware qualification record for every forecast."""
    rows = []
    for p in sorted(a.folder.glob(a.pattern)):
        rows.extend(panel(p, a.reference, a.draws, a.seed, a.prefix))
        write_rows(a.out, rows)
        print(p.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("reference", "folder", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--draws", type=int, default=999)
    p.add_argument("--seed", type=int, default=61731)
    p.add_argument("--pattern", default="*.npz")
    p.add_argument("--prefix", type=int)
    run(p.parse_args())
