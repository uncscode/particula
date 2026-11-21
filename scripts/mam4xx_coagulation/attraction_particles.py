"""Continuous-time Monte Carlo sampling of size-dependent collision tables."""

import numpy as np
from scipy.special import ndtri
from scripts.mam4xx_coagulation.modal_attraction import HIST_EDGES, HOURS


def trajectory(modes, edges, kernel, box_cm3, seed):
    """Sample unordered pairs at K/V and add their actual dry volumes."""
    rng = np.random.default_rng(seed)
    volumes = []
    for diameter, sigma, number in modes:
        count = round(number * box_cm3)
        np.testing.assert_allclose(count, number * box_cm3, atol=2e-10)
        samples = diameter**3 * np.exp(
            3 * np.log(sigma) * ndtri((np.arange(count) + 0.5) / count)
        )
        volumes.extend(
            samples
            * (diameter**3 * np.exp(4.5 * np.log(sigma) ** 2) / samples.mean())
        )
    # Track D³ in µm³; convert to physical volume at observation times.
    initial = np.asarray(volumes)
    start_volume = initial.sum()
    minimum = initial.min() ** (1 / 3) * 1e3
    maximum = start_volume ** (1 / 3) * 1e3
    first = np.searchsorted(edges, minimum, side="right") - 1
    last = np.searchsorted(edges, maximum, side="right")
    if first < 0 or last >= len(edges):
        raise ValueError("Particle volume would exceed the kernel grid")
    edges, kernel = edges[first : last + 1], kernel[first:last, first:last]
    bins = [[] for _ in range(len(kernel))]
    for b, v in zip(
        np.searchsorted(edges, np.cbrt(initial) * 1e3) - 1, initial, strict=True
    ):
        bins[b].append(v)
    counts = np.array([len(bucket) for bucket in bins])
    diagonal = kernel.diagonal()
    time = 0.0
    obs = []
    spectra = []

    def observe():
        v = np.concatenate([bucket for bucket in bins if bucket])
        d = 1e3 * np.cbrt(v)
        high = d >= 100
        np.testing.assert_allclose(v.sum(), start_volume, rtol=5e-13)
        hist = np.histogram(d, bins=np.r_[0, HIST_EDGES, np.inf])[0]
        np.testing.assert_equal(hist.sum(), counts.sum())
        obs.append(
            [
                (~high).sum() / box_cm3,
                high.sum() / box_cm3,
                v[high].sum() * np.pi / (6 * box_cm3),
            ]
        )
        spectra.append(hist / box_cm3)

    def remove(bucket, uniform):
        pos = int(uniform * len(bucket))
        value = bucket[pos]
        bucket[pos] = bucket[-1]
        bucket.pop()
        return value

    for end in HOURS * 3600:
        while time < end:
            # Sum over oriented pairs, excluding self and halving for K_ij/V.
            first_weights = counts * (kernel @ counts - diagonal)
            cumulative = np.cumsum(first_weights)
            total = cumulative[-1] / (2 * box_cm3)
            if total <= 0:
                time = end
                break
            next_time = time + rng.exponential(1 / total)
            if next_time >= end:
                time = end
                break
            time = next_time
            u = rng.random(4)
            i = np.searchsorted(cumulative, u[0] * cumulative[-1], side="right")
            second = kernel[i] * counts
            second[i] -= diagonal[i]
            cum2 = np.cumsum(second)
            j = np.searchsorted(cum2, u[1] * cum2[-1], side="right")
            volume = remove(bins[i], u[2]) + remove(bins[j], u[3])
            k = np.searchsorted(edges, np.cbrt(volume) * 1e3, side="right") - 1
            bins[k].append(volume)
            counts[i] -= 1
            counts[j] -= 1
            counts[k] += 1
        observe()
    return obs, spectra


def ensemble(modes, edges, kernel, box_cm3, seeds):
    """Save endpoint replicates and pointwise distribution quantiles."""
    endpoints, spectra = zip(
        *(trajectory(modes, edges, kernel, box_cm3, s) for s in seeds),
        strict=True,
    )
    return dict(
        seeds=list(seeds),
        box_cm3=box_cm3,
        endpoints=np.asarray(endpoints).tolist(),
        mean=np.mean(spectra, axis=0).tolist(),
        p16=np.quantile(spectra, 0.16, axis=0).tolist(),
        p84=np.quantile(spectra, 0.84, axis=0).tolist(),
    )
