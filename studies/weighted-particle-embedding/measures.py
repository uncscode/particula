"""Count-weighted diagnostics from native snapshots in SI units."""

import numpy as np


def safe_ratio(numerator, denominator):
    """Divide elementwise, defining the ratio of empty populations as zero."""
    result = np.zeros(np.broadcast_shapes(numerator.shape, denominator.shape))
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def observables(mass, counts, density, volume, gas, grid="pulse"):
    """Input axes (box,time,slot,material), counts in represented particles."""
    counts = np.broadcast_to(counts, mass.shape[:-1])
    dry = mass[..., 1:].sum(-1)
    diameter = 2e6 * np.cbrt((mass / density).sum(-1) * 3 / (4 * np.pi))
    dry_diameter = 2e6 * np.cbrt(
        (mass[..., 1:] / density[1:]).sum(-1) * 3 / (4 * np.pi)
    )
    fraction = safe_ratio(mass[..., 2], dry)
    water_fraction = safe_ratio(mass[..., 0], mass.sum(-1))
    x = np.stack(
        [
            np.log(np.maximum(dry_diameter, 0.001)),
            np.log(np.maximum(diameter, 0.001)),
            fraction,
            water_fraction,
        ],
        axis=-1,
    )
    total = counts.sum(-1, keepdims=True)
    weights = safe_ratio(counts, total)
    number = total[..., 0] / volume
    inventory = (counts[..., None] * mass).sum(-2) / volume
    physical = np.concatenate(
        [number[..., None], inventory, gas[..., None]], axis=-1
    )
    # Soft bins are defined once in physical units across all splits.
    if grid == "pulse":
        centers = np.linspace(np.log(0.4), np.log(4.0), 12)
        bandwidth = 0.17
        species = (2,)
    elif grid in ("broad", "tails"):
        centers = np.linspace(np.log(0.045), np.log(4.5), 20)
        bandwidth = 0.20
        species = (0, 1, 2)
    else:
        raise ValueError(f"Unknown diagnostic grid {grid}")
    logits = -0.5 * ((x[..., 1, None] - centers) / bandwidth) ** 2
    if grid in ("broad", "tails"):
        logits -= logits.max(-1, keepdims=True)
    response = np.exp(logits)
    response /= np.maximum(response.sum(-1, keepdims=True), 1e-30)
    histograms = [(weights[..., None] * response).sum(-2)]
    for lane in species:
        contained = (counts[..., None] * mass[..., lane, None] * response).sum(
            -2
        ) / volume
        histograms.append(safe_ratio(contained, inventory[..., lane, None]))
    if grid == "tails":
        rich = fraction >= 0.8
        for measure in (counts, counts * dry, counts * mass[..., 0]):
            total_measure = measure.sum(-1, keepdims=True)
            histograms.append(
                safe_ratio(
                    (measure * rich).sum(-1, keepdims=True), total_measure
                )
            )
    moments = np.stack(
        [diameter, diameter**2, fraction**2, water_fraction * fraction], axis=-1
    )
    diag = np.concatenate(
        [*histograms, (weights[..., None] * moments).sum(-2)], axis=-1
    )
    return x, physical, diag
