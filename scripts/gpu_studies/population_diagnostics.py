"""Compact number, size and charge distributions for resolved box ensembles."""

import numpy as np

DIAMETER_EDGES = np.geomspace(1e-9, 10e-6, 81)


def population_summary(aerosol, groups, *, material=None):
    """Reduce snapshots without retaining individual particle trajectories.

    Diameters are volume-equivalent wet diameters [m]. Histograms contain
    actual particle counts, not densities. Unit active multiplicities are
    required; histogram totals and net charge are independently checked.
    """
    p = aerosol.particles
    if p.n_boxes % groups:
        raise ValueError("groups must partition boxes equally")
    np.testing.assert_equal(np.isfinite(p.masses).all(), True)
    np.testing.assert_equal((p.masses >= 0).all(), True)
    np.testing.assert_equal(np.isin(p.concentration, [0, 1]).all(), True)
    active = p.concentration.reshape(groups, -1) == 1
    if material is not None:
        active &= p.masses[..., material].reshape(groups, -1) > 0
    diameter = (
        2 * np.cbrt(3 / (4 * np.pi) * np.sum(p.masses / p.density, axis=-1))
    ).reshape(groups, -1)
    charge = p.charge.reshape(groups, -1)
    np.testing.assert_equal(np.isfinite(charge[active]).all(), True)
    values = np.union1d([0], np.unique(charge[active]))
    np.testing.assert_array_equal(values, np.rint(values))
    charge_counts = {str(int(q)): [] for q in values}
    histograms, quantiles, number, net_charge = [], [], [], []
    for group in range(groups):
        d = diameter[group, active[group]]
        q = charge[group, active[group]]
        hist = np.histogram(d, DIAMETER_EDGES)[0]
        np.testing.assert_equal(hist.sum(), len(d))
        histograms.append(hist.tolist())
        quantiles.append(
            np.quantile(d, [0.1, 0.5, 0.9]).tolist()
            if len(d)
            else [0.0, 0.0, 0.0]
        )
        number.append(len(d))
        net_charge.append(int(q.sum()))
        for value in values:
            charge_counts[str(int(value))].append(
                int(np.count_nonzero(q == value))
            )
    np.testing.assert_array_equal(
        np.sum(list(charge_counts.values()), axis=0), number
    )
    np.testing.assert_array_equal(
        sum(int(q) * np.array(n) for q, n in charge_counts.items()), net_charge
    )
    return {
        "number": number,
        "pooled_diameter_quantiles": (
            np.quantile(diameter[active], [0.1, 0.5, 0.9]).tolist()
            if active.any()
            else [0.0, 0.0, 0.0]
        ),
        "net_charge": net_charge,
        "diameter_histogram": histograms,
        "diameter_quantiles": quantiles,
        "charge_counts": charge_counts,
    }


def diagnostic_steps(duration, time_step):
    """Include early water-growth transients and ten full-duration intervals."""
    steps = round(duration / time_step)
    early = [
        round(t / time_step) for t in (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
    ]
    return sorted(
        set(range(0, steps + 1, steps // 10))
        | {s for s in early if 0 < s < steps}
    )
