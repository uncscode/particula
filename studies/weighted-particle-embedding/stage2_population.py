"""Matched, closed 293 K boxes for conservative process-operator experiments."""

from dataclasses import dataclass

import numpy as np
from particula.particles import get_binned_number_spectrum
from coupled_events import KT, MFP_AIR, MU_AIR, pair_rates
from vapor_properties import Materials, materials


@dataclass
class Box:
    """Store the physical state and forcing of one aerosol box."""

    mass: np.ndarray  # kg per representative, volatile species first
    charge: np.ndarray  # signed elementary charges per particle
    gas: np.ndarray  # kg/m3
    volume: float  # m3, effective equal-weight simulation volume
    spec: Materials
    levels: np.ndarray  # imposed gas concentrations per plateau
    masks: np.ndarray
    t: np.ndarray  # s; repeated values are forcing boundaries

    def copy(self):
        """Clone all state that a physical process can change."""
        from dataclasses import replace

        return replace(
            self,
            mass=self.mass.copy(),
            charge=self.charge.copy(),
            gas=self.gas.copy(),
        )

    @property
    def concentration(self):
        """Return represented particle number per cubic metre."""
        return len(self.mass) / self.volume

    def inventory(self):
        """Return each conserved material concentration across both phases."""
        total = self.mass.sum(0) / self.volume
        total[: self.spec.species] += self.gas
        return total


def initial_loadings(dry, initial, mix, contrasts, tail, slots):
    """Construct paired volatile loadings at fixed dry sizes."""
    s = len(initial)
    loading = dry[:, None] * initial
    if s > 1:
        loading[:, 1:] *= 1 + 0.45 * mix * contrasts[:, None]
    if tail:
        alternate = np.tile([-0.87, 0.71, -0.13, 0.29], slots // 4)
        loading[:, 0] *= 1 + 0.65 * mix * alternate
        if s > 1:
            # Each loading sums to its original within-size inventory.
            loading[:, 1:] = (
                dry[:, None]
                * initial[None, 1:]
                * (1 + 0.8 * mix * alternate[:, None])
            )
    return loading


def fixture(
    family,
    mix,
    slots=16,
    density=1,
    process="both",
    charge="zero",
    frames=8,
    legs=4,
    duration=0.04,
    chemistry=None,
):
    """Generate matched bulk inventories with distinct mixtures."""
    if slots < 4 or slots % 4:
        raise ValueError("Use four particles per dry size")
    if mix not in (0, 1):
        raise ValueError("Use uniform (0) or segregated (1) composition")
    rng = np.random.default_rng([82739, family])
    spec = materials(2) if chemistry is None else chemistry
    s = spec.species
    mean_radius = rng.uniform(45e-9, 105e-9)
    base = mean_radius * np.exp(
        np.linspace(-1, 1, slots // 4) * rng.uniform(0.27, 0.65)
    )
    # Development and reserved challenge IDs use four distinct compositions
    # per size and a large collector tail. Their family seeds remain disjoint.
    tail = any(
        start <= family < start + 128
        for start in (
            21001,
            41001,
            51001,
            55001,
            57001,
            59001,
            61001,
            61129,
            61257,
            68001,
            70001,
            72001,
            74001,
        )
    )
    if tail:
        extra = np.random.default_rng([49823, family])
        base[-1] *= extra.uniform(1.8, 2.8)
    dry = 4 * np.pi / 3 * 1400 * np.repeat(base, 4) ** 3
    center = rng.uniform(0.36, 0.64)
    contrasts = np.tile([-1.0, -1.0, 1.0, 1.0], slots // 4)
    if tail:
        contrasts = np.tile([-1.0, -0.22, 0.33, 0.89], slots // 4)
    comp = center + mix * rng.uniform(0.16, 0.34) * contrasts
    initial = rng.uniform(0.07, 0.38, s)
    loading = initial_loadings(dry, initial, mix, contrasts, tail, slots)
    mass = np.column_stack((loading, dry * (1 - comp), dry * comp))
    forcing = rng.uniform(0.18, 0.94, (4, s))
    forcing[:, 0] = [
        rng.uniform(0.83, 0.98),
        0.65,
        rng.uniform(0.22, 0.45),
        0.9,
    ]
    levels = forcing * spec.saturation
    masks = np.ones((4, s), dtype=bool)
    masks[1, 0] = s == 1
    if s > 1:
        masks[2, 1:] = False
    if process in ("water", "organic"):
        if process == "organic" and s < 2:
            raise ValueError("Organic-only experiment requires an organic lane")
        selected = 0 if process == "water" else 1
        keep = [selected, s, s + 1]
        spec = Materials(
            (spec.names[selected],),
            spec.molar[keep],
            spec.density[keep],
            spec.pressure[[selected]],
            spec.diffusivity[[selected]],
            spec.accommodation[[selected]],
        )
        mass = mass[:, keep]
        levels, masks = levels[:, [selected]], masks[:, [selected]]
    levels = levels[np.arange(legs) % 4]
    masks = masks[np.arange(legs) % 4]
    # Carry time exactly at boundaries, including both sides of each reset.
    t = np.concatenate(
        [
            (i * frames + np.arange(frames + 1)) * (duration / frames)
            for i in range(legs)
        ]
    )
    volume = slots / (density * 10 ** rng.uniform(14.05, 14.50))
    patterns = dict(
        zero=np.zeros(slots),
        like=np.ones(slots),
        opposite=np.tile([-1, 1, -1, 1], slots // 4),
        segregated=np.repeat([-1, 1], slots // 2),
        bipolar2=2 * np.tile([-1, 1, -1, 1], slots // 4),
        bipolar4=4 * np.tile([-1, 1, -1, 1], slots // 4),
        bipolar8=8 * np.tile([-1, 1, -1, 1], slots // 4),
        segregated8=8 * np.repeat([-1, 1], slots // 2),
        like8=np.full(slots, 8),
    )
    return Box(
        mass,
        patterns[charge].astype(int),
        levels[0].copy(),
        volume,
        spec,
        levels,
        masks,
        t,
    )


def decoded(box, representation):
    """Pool composition while retaining the original collision indices."""
    out = box.copy()
    if representation == "full":
        return out
    if representation not in ("size", "conditional"):
        raise ValueError("Unknown population decoder")
    for start in range(0, len(out.mass), 4):
        indices = np.arange(start, start + 4)
        groups = [indices] if representation == "size" else np.split(indices, 2)
        for group in groups:
            out.mass[group] = out.mass[group].mean(axis=0)
    return out


def wet_radius(mass, spec):
    """Convert additive component volumes to wet particle radii."""
    return np.cbrt(3 * (mass / spec.density).sum(-1) / (4 * np.pi))


def brownian(radius, total_mass, i, j):
    """Fuchs transition kernel with radius in m and particle mass in kg."""
    kn = MFP_AIR / radius
    slip = 1 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
    diffusivity = KT * slip / (6 * np.pi * MU_AIR * radius)
    speed = np.sqrt(8 * KT / (np.pi * total_mass))
    length = 8 * diffusivity / (np.pi * speed)
    collection = (
        (2 * radius + length) ** 3 - (4 * radius**2 + length**2) ** 1.5
    ) / (6 * radius * length) - 2 * radius
    a = np.stack((radius, diffusivity, collection, speed))
    return pair_rates(a[:, i], a[:, j])


def collision_kernel(mass, z, spec, kind, i, j):
    """Evaluate the requested native or analytic pair kernel in m³/s."""
    r = wet_radius(mass, spec)
    total = mass.sum(-1)
    if kind == "brownian":
        return brownian(r, total, i, j)
    if kind == "constant":
        return np.full(len(i), 2e-15)
    if kind in ("charged", "hard-sphere"):
        from particula.dynamics import (
            get_coulomb_kernel_gopalakrishnan2012_via_system_state,
            get_hard_sphere_kernel_via_system_state,
        )

        fn = (
            get_coulomb_kernel_gopalakrishnan2012_via_system_state
            if kind == "charged"
            else get_hard_sphere_kernel_via_system_state
        )
        return fn(r, total, z, 293.0, 101325.0)[i, j]
    raise ValueError(f"Unknown kernel {kind}")


def observables(box):
    """Measure number, phases, wet-size occupancy and signed charge."""
    radius = wet_radius(box.mass, box.spec)
    # Number and signed charge densities on the same wet-radius classes.
    bins = np.array([0.0, 50e-9, 90e-9, 150e-9, np.inf])
    interior, low, high = get_binned_number_spectrum(
        2 * radius, np.full(radius.shape, 1 / box.volume), 2 * bins[1:-1],
        include_upper_edge=False,
    )
    count = np.r_[low, interior, high]
    charge = np.histogram(radius, bins, weights=box.charge)[0] / box.volume
    return np.r_[
        box.concentration,
        box.mass.sum(0) / box.volume,
        box.gas,
        radius.mean() if len(radius) else 0.0,
        box.charge.sum() / box.volume,
        count,
        charge,
        (abs(box.charge)).sum() / box.volume,
    ]


def observable_names(spec):
    """Name the output fields in their serialized order."""
    return [
        "number_m3",
        *[f"particle_{x}" for x in (*spec.names, "dry_a", "dry_b")],
        *[f"gas_{x}" for x in spec.names],
        "mean_wet_radius_m",
        "net_charge_e_m3",
        *[f"count_bin_{i}" for i in range(4)],
        *[f"charge_bin_{i}" for i in range(4)],
        "abs_charge_e_m3",
    ]


def merge(box, i, j):
    """Combine two equal-weight particles and remove one event slot."""
    box.mass[i] += box.mass[j]
    box.charge[i] += box.charge[j]
    box.mass = np.delete(box.mass, j, axis=0)
    box.charge = np.delete(box.charge, j)
