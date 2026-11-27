"""Bound mean count error from the frequency of any paired path disagreement."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import beta
from stage2_accuracy import write_rows
from stage2_score_split import reference_row


def bound(prediction, fine, coarse, slots, looks, alpha=0.05):
    """Use time-uniform Bernoulli and bounded final-count confidence limits."""
    if prediction.shape != fine.shape or coarse.shape != fine.shape:
        raise ValueError("Paired particle counts require identical shapes")
    for x in (prediction, fine, coarse):
        if x.min() < 1 or x.max() > slots or np.any(np.diff(x, axis=1) > 0):
            raise ValueError("Expected a closed pure-merger count history")
    n = len(fine)
    discord = np.any((prediction != fine) | (prediction != coarse), axis=1)
    k = int(discord.sum())
    rate = alpha / (2 * looks)
    upper = 1.0 if k == n else float(beta.ppf(1 - rate, k + 1, n - k))
    final = np.minimum(fine[:, -1], coarse[:, -1])
    lower = max(
        1.0, final.mean() - (slots - 1) * np.sqrt(np.log(1 / rate) / (2 * n))
    )
    return dict(
        seeds=n,
        slots=slots,
        positive_discordant_paths=k,
        observed_any_difference_pct=100 * k / n,
        any_difference_upper_pct=100 * upper,
        min_reference_final_mean_count=final.mean(),
        lower_expected_reference_final_count=lower,
        observed_worst_mean_number_difference_pct=100
        * np.max(abs(prediction.mean(0) - fine.mean(0)) / fine.mean(0)),
        simultaneous_number_upper_pct=100 * (slots - 1) * upper / lower,
        maximum_observed_member_count_difference=int(
            abs(prediction - fine).max()
        ),
        alpha=alpha,
        registered_looks=looks,
    )


def qualify(reference, forecasts, output, looks=3):
    """Evaluate complete prefixes with both physical reference refinements."""
    rows = []
    for file in sorted(forecasts.glob("adaptive-kinetics-n*-t1e-09.npz")):
        with np.load(file) as z:
            n = len(z["keys"])
            if n not in (256, 512, 1024):
                raise ValueError("Unregistered sample count")
            obs = z["obs"]
            keys = z["keys"]
            np.testing.assert_array_equal(
                keys[:, 3], np.arange(185001, 185001 + n)
            )
            if np.any(keys[:, :3] != [74001, 1, 10**11]):
                raise ValueError("Unexpected qualification family")
            refs = [
                reference_row(
                    reference, *k, 16, "dilute-coupled", "zero", 13, 4
                )
                for k in keys
            ]
            if min(v[2] for v in refs) != 2:
                raise ValueError("Physical references require two refinements")
            fine, coarse = (np.stack([r[i] for r in refs]) for i in range(2))

            def counts(arr):
                x = arr[:, :, 0] * (16 / 1e11)
                np.testing.assert_allclose(x, x.round(), rtol=0, atol=1e-12)
                return x.round().astype(int)

            result = bound(counts(obs), counts(fine), counts(coarse), 16, looks)
            rows.append(
                dict(
                    family=74001,
                    mix=1,
                    density_m3=10**11,
                    horizon_s=z["t"][-1],
                    observations=len(z["t"]),
                    **result,
                )
            )
    if rows:
        write_rows(output, sorted(rows, key=lambda r: r["seeds"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "forecasts", "output"):
        p.add_argument(key, type=Path)
    p.add_argument("--looks", type=int, default=3)
    a = p.parse_args()
    qualify(a.reference, a.forecasts, a.output, a.looks)
