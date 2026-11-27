"""Family-block uncertainty for paired aerosol forecasts."""

import numpy as np


def intervals(pred, truth, bound, times, clusters, draws=1000, seed=4117):
    """Resample entire families, keeping paired trajectories together."""
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[1] != 2:
        raise ValueError("Expected memberwise family trajectories")
    if bound.shape != (len(pred), pred.shape[2]) or len(clusters) != len(pred):
        raise ValueError("Incompatible uncertainty or cluster labeling")
    keep = np.diff(times, prepend=times[0]) > 0
    t = times[keep]
    if not len(t):
        raise ValueError("At least one advancing observation is required")
    a = pred[:, 1, keep] - pred[:, 0, keep]
    y = truth[:, 1, keep] - truth[:, 0, keep]
    mask = np.abs(y) > 4 * bound[:, keep]
    unique, group = np.unique(clusters, return_inverse=True)
    resample = np.random.default_rng(seed).multinomial(
        len(unique), np.ones(len(unique)) / len(unique), size=draws
    )
    weights = np.vstack((np.ones(len(unique)), resample))

    def sums(x):
        joined = np.zeros((len(unique), x.shape[1]))
        np.add.at(joined, group, x)
        return weights @ joined

    n = sums(mask)
    signal = sums(abs(y) * mask)
    err = sums(abs(a - y) * mask)
    sign = sums((a * y > 0) & mask)
    member = sums(np.abs(pred - truth)[:, :, keep].mean(1) * 1e6)
    size = sums(np.ones((len(pred), 1)))[:, 0]
    passed = (n == 0) | ((sign >= 0.90 * n) & (err <= 0.25 * signal))
    chain = np.logical_and.accumulate(passed, axis=1)
    horizon = np.where(chain & (n > 0), t[None], 0).max(1)
    result = {"bootstrap_clusters": len(unique), "bootstrap_draws": draws}

    def add(label, x):
        values = x[1:][np.isfinite(x[1:])]
        lo, hi = (
            np.quantile(values, [0.025, 0.975]) if values.size else ("", "")
        )
        result.update(
            {
                label + "_estimate": x[0] if np.isfinite(x[0]) else "",
                label + "_lo95": lo,
                label + "_hi95": hi,
            }
        )

    add("paired_horizon_s", horizon)
    add("member_mae_mg_m3", member.mean(1) / size)

    def quotient(x, y):
        return np.divide(x, y, out=np.full_like(x, np.nan), where=y > 0)

    add("resolved_sign", quotient(sign.sum(1), n.sum(1)))
    add(
        "resolved_contrast_error_over_signal",
        quotient(err.sum(1), signal.sum(1)),
    )
    return result
