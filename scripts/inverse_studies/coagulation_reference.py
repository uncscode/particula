"""Compare a constant-kernel mean spectrum and resolved ensembles."""

import numpy as np


def assess():
    """Measure cluster-size counts in independent closed, well-mixed boxes.

    Sizes are integer multiples of the monomer mass. Every unordered pair has
    hazard K/V in s^-1. The deterministic discrete Smoluchowski solution gives
    n_k(t)/n(0) = tau^(k-1)/(1+tau)^(k+1), tau = K n(0) t / 2.
    """
    initial = 256
    kernel = 1e-15  # m³/s
    concentration = 1e11  # m⁻³
    duration = 1e4  # s
    volume = initial / concentration
    samples = []
    for seed in range(256):
        rng = np.random.default_rng(seed)
        clusters = np.ones(initial, dtype=int)
        count = initial
        clock = 0.0
        while count > 1:
            clock += rng.exponential(
                2 * volume / (kernel * count * (count - 1))
            )
            if clock > duration:
                break
            left, right = sorted(rng.choice(count, size=2, replace=False))
            clusters[left] += clusters[right]
            clusters[right] = clusters[count - 1]
            count -= 1
        live = clusters[:count]
        if live.sum() != initial:
            raise ArithmeticError("resolved merger lost monomer mass")
        bins = np.bincount(np.minimum(live, 4), minlength=5)[1:]
        samples.append(bins / initial)
    tau = kernel * concentration * duration / 2
    ratio = tau / (1 + tau)
    theory = np.array(
        [ratio**i / (1 + tau) ** 2 for i in range(3)] + [ratio**3 / (1 + tau)]
    )
    mean = np.mean(samples, axis=0)
    stderr = np.std(samples, axis=0, ddof=1) / np.sqrt(len(samples))
    return dict(
        initial=initial,
        boxes=len(samples),
        kernel_m3_s=kernel,
        volume_m3=volume,
        time_s=duration,
        labels=["1", "2", "3", "4+"],
        mean_fraction=mean.tolist(),
        standard_error=stderr.tolist(),
        deterministic=theory.tolist(),
        max_z=float(np.max(abs(mean - theory) / stderr)),
    )
