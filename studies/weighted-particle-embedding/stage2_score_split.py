"""Score stochastic ensembles against a refined continuous-event ODE."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from stage2_accuracy import NEAR_ZERO, write_rows


@lru_cache(maxsize=40000)
def reference_row(
    root, family, mix, density, seed, slots, process, charge, frames, legs
):
    """Load independent reference refinements for one stochastic realization."""
    base = (
        f"f{family}-m{mix}-n{slots}-d{density}-{process}-{charge}-r{seed}-full"
    )
    versions = []
    for tol in ("1e-10", "1e-09", "1e-08", "1e-07"):
        file = root / f"{base}-t{tol}-LSODA-f{frames}-l{legs}.npz"
        if file.exists():
            with np.load(file) as z:
                versions.append(z["obs"])
    if not versions:
        raise FileNotFoundError(base)
    return versions[0], versions[-1], len(versions)


def arrays(d, root, meta):
    """Align complete reference trajectories with the stored forecast keys."""
    hi, lo, count = zip(
        *(
            reference_row(
                root,
                *map(int, k),
                meta["slots"],
                meta["process"],
                meta["charge"],
                meta["frames"],
                meta["legs"],
            )
            for k in d["keys"]
        ),
        strict=True,
    )
    return np.stack(hi), np.stack(lo), min(count)


def summarize_group(
    file, meta, k, select, p, ref, lo, t, names, settings, drift, seconds, nref
):
    """Compute paired uncertainty bounds for each count and phase observable."""
    rows = []
    n = len(p)
    s = sum(name.startswith("gas_") for name in names)
    cols = [
        (idx, name)
        for idx, name in enumerate(names)
        if idx == 0 or name.startswith(("particle_", "gas_"))
    ]
    cols.extend([(None, "mass_total"), (None, "particle_mass_total")])
    # Conservative simultaneous bands across observations, stated observables
    # and groups in this evaluation panel. Paired seed differences reduce noise.
    alpha = 0.05 / max(1, len(t) * len(cols) * settings)
    quant = student_t.ppf(1 - alpha / 2, n - 1) if n > 1 else 0.0
    for j, name in cols:

        def field(a, j=j, name=name):
            if j is not None:
                return a[:, :, j]
            return a[
                :, :, 1 : 2 * s + 3 if name == "mass_total" else s + 3
            ].sum(-1)

        x, y, z = map(field, (p, ref, lo))
        denom = np.maximum(abs(y.mean(0)), NEAR_ZERO)
        err = x - y
        diff = abs(err.mean(0))
        numerical = abs(y - z).mean(0)
        se = err.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(t))
        sy = y.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(len(t))
        # A ratio confidence bound is unbounded if its resolved reference
        # denominator can reach zero. Near-zero species use the absolute gate.
        lower_reference = np.where(
            abs(y.mean(0)) <= NEAR_ZERO, NEAR_ZERO, abs(y.mean(0)) - quant * sy
        )
        numerator = diff + quant * se + numerical
        upper = np.divide(
            numerator,
            lower_reference,
            out=np.full_like(diff, np.inf),
            where=lower_reference > 0,
        )
        lower = np.maximum(0, diff - quant * se - numerical) / (
            denom + quant * sy
        )
        limit = 0.01 if name == "number_m3" else 0.001
        qualified = upper <= limit if name == "number_m3" else upper < limit
        failed = lower > limit if name == "number_m3" else lower >= limit
        bad = np.flatnonzero(~qualified)
        first = bad[0] if len(bad) else len(t)
        any_member = np.any(
            abs(err) > 1e-12 * np.maximum(abs(y), NEAR_ZERO), axis=1
        )
        # A paired t band collapses for a count difference never observed.
        # Record an independent, one-sided 95% bound on the probability that
        # an additional trajectory exhibits any count discrepancy.
        from scipy.stats import beta

        misses = int(any_member.sum())
        rare = (
            float(beta.ppf(0.95, misses + 1, n - misses)) if misses < n else 1.0
        )
        rows.append(
            dict(
                file=file,
                **meta,
                family=int(k[0]),
                mix=int(k[1]),
                density=int(k[2]),
                seeds=n,
                reference_refinements=nref,
                field=name,
                worst_mean_pct=100 * np.max(diff / denom),
                worst_upper_pct=100 * upper.max(),
                worst_reference_sem_pct=100 * np.max(sy / denom),
                worst_numerical_pct=100 * np.max(numerical / denom),
                worst_member_pct=100
                * np.max(abs(err) / np.maximum(abs(y), NEAR_ZERO)),
                first_unqualified_s=t[first] if first < len(t) else "",
                last_contiguous_s=t[first - 1] if first else "",
                end_s=t[-1],
                passing_frames=int(qualified.sum()),
                failed_frames=int(failed.sum()),
                ambiguous_frames=int((~(qualified | failed)).sum()),
                differing_trajectories=misses,
                probability_any_difference_upper95=rare,
                drift_relative=drift,
                forecast_seconds=seconds,
            )
        )
    return rows


def score(reference, folder, out):
    """Score fixed families and retain unresolved observations."""
    allrows = []
    for file in sorted(folder.glob("*.npz")):
        with np.load(file) as d:
            meta = json.loads(str(d["meta"]))
            hi, lo, nref = arrays(d, reference, meta)
            keys = d["keys"]
            groups = np.unique(keys[:, :3], axis=0)
            obs = d["obs"]
            for k in groups:
                selection = np.all(keys[:, :3] == k, axis=1)
                allrows.extend(
                    summarize_group(
                        file.name,
                        meta,
                        k,
                        selection,
                        obs[selection],
                        hi[selection],
                        lo[selection],
                        d["t"],
                        d["names"],
                        len(groups),
                        float(d["max_drift"]),
                        float(d["seconds"].sum()),
                        nref,
                    )
                )
    if not allrows:
        raise ValueError("No completed forecasts")
    write_rows(out, allrows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "folder", "out"):
        p.add_argument(name, type=Path)
    score(**vars(p.parse_args()))
