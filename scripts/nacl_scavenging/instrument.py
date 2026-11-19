"""Inputs and forward DMA response for the Cai NaCl pulse study."""

import csv

import numpy as np
from charge_study import BJERRUM_NM
from particula import gas, particles
from particula.dynamics.ions.attachment import attachment_coefficient
from scipy.constants import elementary_charge
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, curve_fit
from study import OUT, PRESSURE, ROOT, TEMP, archive_table

HERE = OUT / "resolved"

D = np.geomspace(1.6, 600, 1601)
LOG_D = np.log(D)
STATES = np.arange(-8, 9)
SCAN_CENTERS = (3.59, 3.72, 3.85, 4.00, 4.14, 4.29, 4.45)
TARGETS = (3.6, 3.7, 4.0, 5.0, 10.0)
DILUTION = 0.6 / 3600
MU = gas.get_dynamic_viscosity(TEMP)
MFP = gas.get_molecule_mean_free_path(
    temperature=TEMP, pressure=PRESSURE, dynamic_viscosity=MU
)


def mobility(d):
    """Single-charge electrical mobility in air, diameter in nm."""
    si = np.asarray(d) * 1e-9
    slip = particles.get_cunningham_slip_correction(2 * MFP / si)
    return elementary_charge * particles.get_aerodynamic_mobility(
        si / 2, slip, MU
    )


def probabilities(d, omitted):
    """Wiedensohler fits with LF flux ratios for rare small multiple charges."""
    d = np.asarray(d)
    p = np.zeros((d.size, len(STATES)))
    mid = len(STATES) // 2
    for z in (0, -1, 1):
        p[:, mid + z] = particles.get_bipolar_charge_probability(d * 1e-9, z)
    for sign in (-1, 1):
        for k in range(2, mid + 1):
            z = sign * k
            multi = d >= 20 if k == 2 else d >= 50
            if k == 2:
                p[multi, mid + z] = particles.get_bipolar_charge_probability(
                    d[multi] * 1e-9, z
                )
            else:
                var = d[multi] / (2 * BJERRUM_NM)
                p[multi, mid + z] = np.exp(
                    -((z - var * np.log(0.875)) ** 2) / (2 * var)
                ) / np.sqrt(2 * np.pi * var)
            sub = ~multi
            if np.any(sub):
                radius = d[sub] * 0.5e-9
                a = attachment_coefficient(
                    radius, z - sign, sign, omitted=omitted
                )
                b = attachment_coefficient(radius, z, -sign, omitted=omitted)
                p[sub, mid + z] = p[sub, mid + z - sign] * a / b
    p /= p.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
    return p


def transfer(ratio, width, blur):
    """Balanced-flow triangle convolved with a Gaussian in relative mobility."""
    axis = np.linspace(0, 2, 4001)
    tri = np.maximum(0, 1 - abs(axis - 1) / width)
    if blur:
        tri = gaussian_filter1d(
            tri, blur / (axis[1] - axis[0]), mode="constant"
        )
    return np.interp(ratio, axis, tri, left=0, right=0)


def response(target, width, blur, p, sign, max_z=3):
    """Weight each true diameter by charge fraction and DMA transmission."""
    rel = mobility(D) / mobility(target)
    return np.column_stack(
        [
            p[:, 8 + sign * z] * transfer(z * rel, width, blur)
            for z in range(1, max_z + 1)
        ]
    )


def selected(
    target,
    p,
    *,
    large=False,
    mode_ratio=1.0,
    gsd=1.6,
    width=None,
    blur=None,
    sign=1,
    max_z=3,
    ideal=False,
):
    """Return joint probabilities for physical size and classifier charge."""
    if ideal:
        out = np.zeros((D.size, max_z))
        out[np.abs(LOG_D - np.log(target)).argmin(), 0] = 1.0
        return out
    if width is None:
        width = 0.1 if large else 0.35
    if blur is None:
        blur = 0.01 if large else 0.03
    parent = np.exp(
        -0.5 * (np.log(D / (target * mode_ratio)) / np.log(gsd)) ** 2
    )
    joint = parent[:, None] * response(target, width, blur, p, sign, max_z)
    return joint / joint.sum()


def condition_scan(joint, p, tube):
    """Tilt the tracer size prior to match the pooled detected mean channel.

    This one-parameter sensitivity retains the classifier branches, inlet
    neutralization, tube losses and the seven-channel detector response.
    """
    observed = np.genfromtxt(
        ROOT / "docs/data/nacl-scavenging/archived-smps-counts.csv",
        names=True,
        delimiter=",",
    )
    centers = np.asarray(SCAN_CENTERS)
    np.testing.assert_array_equal(observed["channel_nm"], centers)
    centroid = np.average(centers, weights=observed["pooled_raw_counts"])
    sensor = np.stack(
        [response(d, 0.30, 0.03, p, 1).sum(axis=1) for d in centers], axis=1
    )
    transmitted = p[:, 8] + tube * (1 - p[:, 8])
    base = joint.sum(axis=1) * transmitted
    log_d = np.log(D / 4.0)

    def error(slope):
        signals = (base * np.exp(slope * log_d)) @ sensor
        return np.average(centers, weights=signals) - centroid

    slope = brentq(error, -40, 40)
    tilted = joint * np.exp(slope * log_d)[:, None]
    return tilted / tilted.sum(), slope


def fit_walls():
    """Fit the six published blank decays and subtract dilution."""
    t = archive_table("Fig4_OverallLifetime")
    out = []
    for d, tc, nc in [
        (3, 2, 7),
        (4, 2, 6),
        (5, 2, 5),
        (7, 2, 4),
        (10, 2, 3),
        (100, 0, 1),
    ]:
        good = np.isfinite(t[:, tc]) & np.isfinite(t[:, nc]) & (t[:, nc] > 0)
        time, y = t[good, tc], t[good, nc]
        (amp, k), _ = curve_fit(
            lambda x, a, b: a * np.exp(-b * x),
            time,
            y,
            p0=(1.0, 5.0),
            bounds=(0, np.inf),
        )
        out.append(
            dict(
                d_nm=d,
                points=len(y),
                amplitude=amp,
                total_h=k,
                wall_h=k - 0.6,
                rmse=np.sqrt(np.mean((amp * np.exp(-k * time) - y) ** 2)),
            )
        )
    return out


def wall_table(fits):
    """Interpolate the fitted wall rates onto the diameter grid in s⁻¹."""
    return (
        np.exp(
            np.interp(
                LOG_D,
                np.log([r["d_nm"] for r in fits]),
                np.log([r["wall_h"] for r in fits]),
            )
        )
        / 3600
    )


def sample(joint, p, n, rng, tube):
    """Recondition selected particles and sample those passing the tube."""
    # Reweight diameters for size-dependent mean survival through the tube.
    transmitted = p.copy()
    transmitted[:, :8] *= tube
    transmitted[:, 9:] *= tube
    passage = transmitted.sum(axis=1)
    joint = joint * passage[:, None]
    joint /= joint.sum()
    k = rng.choice(joint.size, n, p=joint.ravel())
    ix, selection_charge = np.unravel_index(k, joint.shape)
    cdf = np.cumsum(transmitted / passage[:, None], axis=1)
    charge = np.sum(rng.random(n)[:, None] > cdf[ix], axis=1) - 8
    return D[ix], charge, selection_charge + 1


def aliases(d):
    """Find the diameters at the center of the first three mobility branches."""
    return [
        brentq(lambda x, z=z: z * mobility(x) - mobility(d), d, 6 * d)
        for z in (1, 2, 3)
    ]


def vdw_table():
    """Added neutral dispersion rate (cm³/s), Cai supplement S7–S11."""
    from particula.dynamics import get_hard_sphere_kernel_via_system_state
    from scipy.constants import Boltzmann

    d = D[::8] * 1e-9
    m = 2160 * np.pi / 6 * d**3
    base = (
        get_hard_sphere_kernel_via_system_state(
            d / 2, m, np.zeros_like(d), TEMP, PRESSURE
        )
        * 1e6
    )
    diffusion = particles.get_diffusion_coefficient_via_system_state(
        d / 2, TEMP, PRESSURE
    )
    speed = particles.get_mean_thermal_speed(m, TEMP)
    diam = d[:, None] + d
    a = 8.9e-20 / (Boltzmann * TEMP) * 4 * d[:, None] * d / diam**2
    log = np.log1p(a)
    ec = 1 + 0.0757 * log + 0.0015 * log**3
    ef = (
        1
        + np.sqrt(a / 3) / (1 + 0.0151 * np.sqrt(a))
        - 0.186 * log
        - 0.0163 * log**3
    )
    x = (
        8
        * (diffusion[:, None] + diffusion)
        / (diam * np.hypot(speed[:, None], speed))
    )
    incr = base * ((1 + x) / (1 / ec + x / ef) - 1)
    np.testing.assert_equal(
        np.all(incr > 0) and np.allclose(incr, incr.T, atol=0), True
    )
    return np.log(incr)


def save(name, rows):
    """Write rows from one resolved calculation to its output directory."""
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / name).open("w") as stream:
        w = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)


def ion_density(p, mix, number, q):
    """Solve the two steady ion balances for concentrations in cm⁻³."""
    avg = [
        sum(
            np.sum(
                mix
                * p[:, j]
                * attachment_coefficient(D * 0.5e-9, z, pol, omitted="lower")
            )
            for j, z in enumerate(STATES)
        )
        for pol in [-1, 1]
    ]
    sink = DILUTION + number * 1e6 * np.array(avg)
    ratio = sink[0] / sink[1]
    neg = 2 * q / (sink[0] + np.sqrt(sink[0] ** 2 + 4 * 2.3e-6 * ratio * q))
    return np.array([neg, neg * ratio])
