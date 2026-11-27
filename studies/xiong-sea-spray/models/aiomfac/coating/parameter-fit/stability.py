# %% Global tangent-plane stability scan on a humidity-decreasing homogeneous branch.
from dataclasses import dataclass
import numpy as np
from trial import MW_WATER, feed_ratio, prepare, states


@dataclass
class Grid:
    salt: str
    molality: np.ndarray
    feed_mu: np.ndarray
    trial_coordinates: np.ndarray
    energies: np.ndarray
    fraction: float


def grid(salt, ovf, temp, model, refine=1, maxmol=80.0, density=1.266):
    f = feed_ratio(salt, ovf, density)
    # m is total analytical charge equivalents per kg water. Allow independently
    # partitioned counterions in the reciprocal AS system.
    composition = np.unique(
        np.r_[np.linspace(0, 1, 12 * refine + 1), 0.002, 0.02, 0.98, 0.998, f]
    )
    na = composition if salt == "AS" else [1.0]
    mgrid = np.geomspace(0.025, maxmol, 45 * refine)
    m, u, v = np.meshgrid(mgrid, na, composition, indexing="ij")
    w = 1 / (MW_WATER * m)
    counts, mu = states(model, salt, w, u, v, temp)
    energy = np.einsum("ij,ij->i", counts, mu)
    ms = np.geomspace(0.04, 18.0, 90 * refine)
    _, feed = states(
        model, salt, 1 / (MW_WATER * ms), f if salt == "AS" else 1.0, f, temp
    )
    return Grid(salt, ms, feed, counts, energy, f)


def stability(g, other=None, delta=0.0):
    feed = (
        g.feed_mu if other is None else g.feed_mu + delta * (other.feed_mu - g.feed_mu)
    )
    energy = (
        g.energies
        if other is None
        else g.energies + delta * (other.energies - g.energies)
    )
    return feed, energy


def onset(g, other=None, delta=0.0, tol=1e-5):
    feed, energy = stability(g, other, delta)
    # At every sampled feed state look for any electroneutral phase with lower
    # tangent-plane distance. Track the first split when concentrating from water.
    values = energy[:, None] - g.trial_coordinates @ feed.T
    best = np.argmin(values, axis=0)
    d = values[best, np.arange(len(feed))]
    hits = np.flatnonzero(d < -tol)
    if not len(hits):
        return np.nan, np.nan, False
    j = hits[0]
    # A thermodynamically consistent trivial trial has TPD near zero for every
    # feed. Refine the first off-feed witness instead of interpolating the trivial TPD.
    witness = best[j]
    i = max(0, j - 1)
    a = values[witness, i]
    b = values[witness, j]
    t = np.clip((a + tol) / max(a - b, 1e-15), 0, 1)
    logaw = (1 - t) * feed[i, 0] + t * feed[j, 0]
    m_witness = 1 / (MW_WATER * g.trial_coordinates[witness, 0])
    return 100 * np.exp(min(logaw, 2)), m_witness, j == 0


def cloud_curve(salt, ovfs, tail, sodium, delta, **kwargs):
    model = prepare(salt, tail, sodium, delta)
    return [onset(grid(salt, v, 275.15, model, **kwargs)) for v in ovfs]
