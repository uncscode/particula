# %% [markdown]
# # Charge conditioning and NaCl scavenging in the Cai chamber
# See the NaCl scavenging validation record for data and equations.
# %%
"""Charge-state sensitivity for the Cai et al. NaCl capture coefficients."""

import argparse
import json

import matplotlib
import numpy as np
from particula.particles import get_bipolar_charge_probability
from particula.util.lf2013_coagulation import lf2013_coag_full
from scipy.constants import Boltzmann, elementary_charge, epsilon_0
from scipy.integrate import simpson
from scipy.linalg import expm
from study import OUT, TEMP, read, revision, write

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BJERRUM_NM = (
    elementary_charge**2 * 1e9 / (4 * np.pi * epsilon_0 * Boltzmann * TEMP)
)
HAMAKER_KT = 8.9e-20 / (Boltzmann * TEMP)
KN = {3.6: 1.72, 3.7: 1.70, 4.0: 1.63, 5.0: 1.45, 10.0: 0.99}
SMALL = np.arange(-1, 2)
LARGE = np.arange(-5, 6)


def outlet(d, charges):
    """Return raw fits and normalized probabilities for diameter d in nm."""
    var = d / (2 * BJERRUM_NM)
    mean = var * np.log(0.875)
    raw = np.array(
        [
            get_bipolar_charge_probability(d * 1e-9, z)
            if abs(z) <= 2
            else np.exp(-((z - mean) ** 2) / (2 * var))
            / np.sqrt(2 * np.pi * var)
            for z in charges
        ]
    )
    return raw, raw / raw.sum()


def chamber_ions(q, collector_number, weights):
    """Solve the steady balance for negative and positive ions (cm⁻³)."""
    alpha = 2.3e-6  # cm3/s, Franchin et al. at 20 C
    if not collector_number:
        return np.full(2, np.sqrt(q / alpha))
    rates = lf2013_coag_full(
        radius_vals=np.array([50e-9]), charge_vals=list(LARGE)
    )
    losses = np.array(
        [
            0.6 / 3600 + collector_number * 1e6 * (rate[0] @ weights)
            for rate in rates
        ]
    )
    # Equal pair sources imply the two first-order sink fluxes are equal.
    ratio = losses[0] / losses[1]
    neg = 2 * q / (losses[0] + np.sqrt(losses[0] ** 2 + 4 * alpha * ratio * q))
    ions = neg * np.array([1, ratio])
    np.testing.assert_allclose(losses * ions + alpha * ions.prod(), q)
    return ions


def recharge(d, ions, seconds):
    """Evolve neutral tracers in a fixed bipolar ion bath."""
    neg, pos = lf2013_coag_full(
        radius_vals=np.array([d * 0.5e-9]), charge_vals=[-1, 0, 1]
    )
    # Table in m3/s at 298.15 K. Transitions to |z|=2 are omitted.
    attach_neg, discharge_pos = neg[0, 1:] * 1e6 * ions[0]
    discharge_neg, attach_pos = pos[0, :2] * 1e6 * ions[1]
    generator = np.array(
        [
            [-discharge_neg, attach_neg, 0],
            [discharge_neg, -attach_neg - attach_pos, discharge_pos],
            [0, attach_pos, -discharge_pos],
        ]
    )
    np.testing.assert_allclose(generator.sum(axis=0), 0, atol=1e-15)
    return expm(seconds * generator)[:, 1]


def conductor_energy(a, b, r):
    """Coefficients of z1², z1*z2, z2² in interaction energy/kT (lengths in nm).

    Lekner (2012), Eqs. 1.1–1.6. Sum only where n*u < 36. Outside this
    range successive terms decay exponentially. Isolated self energies cancel.
    """
    u = np.arccosh((r * r - a * a - b * b) / (2 * a * b))
    factor = a * b * np.sinh(u)
    caa, cbb = np.full_like(r, a), np.full_like(r, b)
    cab = np.zeros_like(r)
    for n in range(1, 1 + int(36 / u[0])):
        end = np.searchsorted(u, 36 / n)
        sn = np.sinh(n * u[:end])
        sn1 = np.sinh((n + 1) * u[:end])
        caa[:end] += factor[:end] / (a * sn + b * sn1)
        cbb[:end] += factor[:end] / (b * sn + a * sn1)
        cab[:end] -= factor[:end] / (r[:end] * sn)
    det = caa * cbb - cab * cab
    return (
        0.5 * BJERRUM_NM * (cbb / det - 1 / a),
        -BJERRUM_NM * cab / det,
        0.5 * BJERRUM_NM * (caa / det - 1 / b),
    )


def limits(s, potential, energy):
    """Compute continuum and ballistic factors for an absorbing surface."""
    # Integrate continuum resistance over contact distance/r from zero to one.
    x = np.r_[0, (1 / s)[::-1], 1]
    boltz = np.exp(potential)
    ec = 1 / np.trapezoid(np.r_[1, boltz[::-1], boltz[0]], x)
    # Flux-weighted Maxwell average of the critical impact area. Potential
    # at the first grid point extends over the unresolved final surface gap.
    # Below the radial barrier even head-on trajectories cannot collide.
    energy = energy + max(0.0, potential.max())
    samples = []
    for energies in np.array_split(energy, 12):
        barrier = (energies[:, None] - potential) * s**2
        samples.extend(
            np.maximum(
                0, np.minimum(barrier.min(axis=1), energies - potential[0])
            )
        )
    ef = simpson(np.exp(-energy) * samples, x=energy)
    return ec, ef


def pair_factors(d, fine):
    """Compute pair factors with electrostatics alone and with dispersion."""
    a, b = d / 2, 50.0
    s = 1 + np.geomspace(2.5e-6 if fine else 1e-5, 1e5, 1800 if fine else 900)
    r = (a + b) * s
    e = np.r_[0, np.geomspace(1e-7, 44, 600 if fine else 300)]
    u1, cross, u2 = conductor_energy(a, b, r)
    np.testing.assert_allclose(cross[-1] * r[-1], BJERRUM_NM, rtol=1e-8)
    # Hamaker (1937) two-sphere potential with h = 4ab/(r²-(a+b)²).
    h = 4 * a * b / ((a + b) ** 2 * (s * s - 1))
    vdw = -HAMAKER_KT / 6 * (h / 2 + h / (2 * (1 + h)) - np.log1p(h))
    # Analytic limiting kernels for a pure +/-1/r potential.
    for strength in [-1.0, 0.0, 1.0]:
        calc = limits(s, strength / s, e)
        exact = [
            strength / np.expm1(strength) if strength else 1,
            np.exp(-strength) if strength > 0 else 1 - strength,
        ]
        np.testing.assert_allclose(calc, exact, rtol=0.001)
    x = (4 / 3) * KN[d]  # supplement S5, S6, S11
    factors = np.zeros((2, len(SMALL), len(LARGE)))
    for i, z in enumerate(SMALL):
        for j, w in enumerate(LARGE):
            electro = z * z * u1 + z * w * cross + w * w * u2
            for k, potential in enumerate([electro, electro + vdw]):
                ec, ef = limits(s, potential, e)
                factors[k, i, j] = (1 + x) / (1 / ec + x / ef)
    np.testing.assert_allclose(factors, factors[:, ::-1, ::-1])
    np.testing.assert_equal(np.isfinite(factors).all(), True)
    np.testing.assert_array_less(0, factors)
    return factors


def plot():
    """Replot the comparison from the saved scenario table."""
    rows = read("charge-scenarios.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for name, label, mark in [
        ("measured", "Measured", "o"),
        ("authors_enhanced", "Published van der Waals", "s"),
        ("bipolar_outlet", "Charge only, equilibrium", "^"),
    ]:
        sub = [v for v in rows if v["scenario"] == "bipolar_outlet"]
        x = [float(v["d_nm"]) for v in sub]
        y = [
            float(v["charge_only_factor"])
            if name == "bipolar_outlet"
            else float(v[name]) / float(v["authors_hard_sphere"])
            for v in sub
        ]
        ax.plot(
            x,
            y,
            marker=mark,
            ls="none" if name == "measured" else "-",
            label=label,
        )
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set(ylabel="Rate / published hard-sphere rate", ylim=(0.95, 1.82))
    for name, label in [
        ("collector_only", "Filtered inlet"),
        ("natural", "Natural ion production"),
        ("elevated", "Elevated ion production"),
        ("bipolar_outlet", "Outlet distribution"),
    ]:
        sub = [v for v in rows if v["scenario"] == name]
        bx.plot(
            x,
            [100 * (float(v["joint_over_vdw"]) - 1) for v in sub],
            ".-",
            label=label,
        )
    bx.set(ylabel="Charge increment over van der Waals (%)")
    for axis in (ax, bx):
        axis.set(xlabel="Small NaCl diameter (nm)")
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "charge-comparison.png", dpi=180)
    plt.close(fig)


def run():
    """Run all sizes, verify the numerical checks, and save the results."""
    fractions, pairs, scenarios = [], [], []
    raw, collectors = outlet(100, LARGE)
    for z, f, p in zip(LARGE, raw, collectors, strict=True):
        fractions.append(dict(d_nm=100, charge_e=z, raw_fit=f, probability=p))
    neutral = np.array([0.0, 1.0, 0.0])
    collector_zero = (LARGE == 0).astype(float)
    max_change = 0.0
    for row in read("coefficients.csv"):
        d = float(row["d_nm"])
        meas, hs, vdw, par = (
            float(row[k])
            for k in (
                "measured",
                "authors_hard_sphere",
                "authors_enhanced",
                "particula",
            )
        )
        raw, small = outlet(d, SMALL)
        for z, f, p in zip(SMALL, raw, small, strict=True):
            fractions.append(dict(d_nm=d, charge_e=z, raw_fit=f, probability=p))
        fine = pair_factors(d, True)
        coarse = pair_factors(d, False)
        max_change = max(max_change, np.max(abs(coarse / fine - 1)))
        neutral_vdw = fine[1, 1, LARGE == 0].item()
        for i, z in enumerate(SMALL):
            for j, w in enumerate(LARGE):
                pairs.append(
                    dict(
                        d_nm=d,
                        small_charge_e=z,
                        collector_charge_e=w,
                        charge_only_factor=fine[0, i, j],
                        joint_factor=fine[1, i, j],
                        neutral_vdw_factor=neutral_vdw,
                    )
                )
        cases = [
            ("collector_only", neutral, collectors, 0, [0, 0], 0, 0),
            ("small_only", small, collector_zero, 0, [0, 0], 0, 0),
            ("bipolar_outlet", small, collectors, 0, [0, 0], 0, 0),
            (
                "all_opposite",
                np.array([0, 0, 1]),
                (LARGE == -1).astype(float),
                0,
                [0, 0],
                0,
                0,
            ),
        ]
        for name, q in [("natural", 2.0), ("elevated", 100.0)]:
            for number, suffix in [(0, ""), (1e4, "_with_collectors")]:
                ions = chamber_ions(q, number, collectors)
                p = recharge(d, ions, 300)
                cases.append(
                    (name + suffix, p, collectors, q, ions, 300, number)
                )
        for name, p, c, q, ions, t, number in cases:
            np.testing.assert_allclose([p.sum(), c.sum()], 1, atol=1e-12)
            np.testing.assert_equal(min(p) >= 0 and min(c) >= 0, True)
            charge, joint = p @ fine @ c
            incr = joint / neutral_vdw
            required_hs = required_vdw = None
            if name == "bipolar_outlet":
                start_charge, start_joint = neutral @ fine @ c
                f = 1 - p[1]
                required_hs = (
                    f * (meas / hs - start_charge) / (charge - start_charge)
                )
                start_ratio = start_joint / neutral_vdw
                required_vdw = (
                    f * (meas / vdw - start_ratio) / (incr - start_ratio)
                )
            scenarios.append(
                dict(
                    d_nm=d,
                    required_fraction_vs_hs=required_hs,
                    required_fraction_vs_vdw=required_vdw,
                    scenario=name,
                    ion_pairs_cm3_s=q,
                    negative_ions_cm3=ions[0],
                    positive_ions_cm3=ions[1],
                    ion_sink_collectors_cm3=number,
                    exposure_s=t,
                    p_negative=p[0],
                    p_neutral=p[1],
                    p_positive=p[2],
                    charge_only_factor=charge,
                    joint_over_vdw=incr,
                    measured=meas,
                    authors_hard_sphere=hs,
                    authors_enhanced=vdw,
                    charge_only_on_authors_hs_cm3_s=hs * charge,
                    charge_only_on_particula_cm3_s=par * charge,
                    joint_on_authors_vdw_cm3_s=vdw * incr,
                )
            )
    np.testing.assert_array_less(max_change, 0.002)
    write("charge-fractions.csv", fractions)
    write("charge-pairs.csv", pairs)
    write("charge-scenarios.csv", scenarios)
    metadata = dict(
        revision=revision(),
        max_relative_grid_change=max_change,
        attachment_temperature_K=298.15,
        collision_temperature_K=TEMP,
        bjerrum_length_nm=BJERRUM_NM,
        source_run=json.loads((OUT / "run.json").read_text())["revision"],
    )
    (OUT / "charge-run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    if parser.parse_args().replot:
        plot()
    else:
        run()
