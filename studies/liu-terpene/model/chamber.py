# %%
"""Independent chamber reconstruction and held-out surrogate tests.

Run with the Particula checkout on PYTHONPATH. Inputs are copies
of the digitized Figure 4 observations and normalized Table S4/S5 weights.
Time in minutes, initial mobility diameter in nm, density in kg/m3.
"""

import csv
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import particula as par
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import softmax

HERE = Path(__file__).resolve().parent
DENSITY = dict(
    pinene=1240.0, ocimene=1350.0, caryophyllene=1170.0, farnesene=1200.0
)
SIZE = dict(pinene=165.0, ocimene=220.0, caryophyllene=165.0, farnesene=140.0)
CYCLIC = dict(
    pinene="pinene",
    ocimene="pinene",
    caryophyllene="caryophyllene",
    farnesene="caryophyllene",
)
CSTAR = np.power(10.0, np.arange(-4, 3))


def reported_fractions():
    """Normalize each author's seven rounded fractions once, in C* order."""
    path = HERE.parent / "sources" / "fractions.csv"
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["precursor"], row["rh"])
            values.setdefault(key, []).append(
                (
                    float(row["cstar_ug_m3"]),
                    float(row["mass_fraction_as_printed"]),
                )
            )
    result = {}
    for key, series in values.items():
        stars, weights = zip(*sorted(series))
        np.testing.assert_allclose(stars, CSTAR, atol=0, rtol=1e-14)
        assert abs(sum(weights) - 1.0) < 0.01
        result[key] = np.array(weights) / sum(weights)
    return result


REPORTED = reported_fractions()
GAS_D = 6e-6  # m2/s, scenario for all organic surrogates
MW = 0.20  # kg/mol, common organic-bin surrogate
TEMP = 297.15
MFP = 3 * GAS_D / np.sqrt(8 * 8.314462618 * TEMP / (np.pi * MW))


@dataclass(frozen=True)
class Scenario:
    alpha: float = 1.0
    sigma: float = 0.03  # N/m
    water: bool = False
    density_offset: float = 0.0
    skip_shape: bool = False
    marker_shift: float = 0.0
    time_shift: float = 0.0


SCENARIOS = {
    "primary": Scenario(),
    "low_accommodation": Scenario(alpha=0.1),
    "high_tension": Scenario(sigma=0.045),
    "zero_curvature": Scenario(sigma=0.0),
    "humid_water": Scenario(water=True),
    "shape_after30": Scenario(skip_shape=True),
    "opposed_reading_up": Scenario(marker_shift=0.01, time_shift=2.0),
    "opposed_reading_down": Scenario(marker_shift=-0.01, time_shift=-2.0),
    "opposed_density": Scenario(density_offset=10.0),
}


def particle_properties(name, rh, s):
    opposite = 1 if CYCLIC[name] == name else -1
    rho = DENSITY[name] + s.density_offset * opposite
    water_vol = 0.0
    if s.water and rh == "wet":
        kappa = 0.10 if CYCLIC[name] == "pinene" else 0.05
        water_vol = kappa * 0.75 / (1 - 0.75)
    # Constant water:organic ratio; miniSPLAT retains the dry organic signal.
    organic_mole_share = 1 / (1 + water_vol * 1000 * MW / (rho * 0.018015))
    return rho, water_vol, organic_mole_share


def kinetics(r, rho, s):
    transition = par.particles.get_vapor_transition_correction(MFP / r, s.alpha)
    k = par.dynamics.get_first_order_mass_transport_k(r, transition, GAS_D)
    kelvin_r = (
        0.0
        if s.sigma == 0
        else par.particles.get_kelvin_radius(s.sigma, rho, MW, TEMP)
    )
    kelvin = par.particles.get_kelvin_term(r, kelvin_r)
    return k, kelvin


def predict(
    f, name, rh, diam, times, scenario, *, n=1300, speed=1.0, hardening=0.0
):
    # Equal MW across bins reduces seven coupled ODEs to one exposure q.
    # m_i(q)/m0 = f_i exp(-C_i* q). Integrate dt/dq, then invert once.
    q = np.r_[0.0, np.geomspace(1e-7, 2e4, n)]
    mass = np.exp(-q[:, None] * CSTAR) @ f
    rho, water, mole_share = particle_properties(name, rh, scenario)
    r0 = diam * 0.5e-9
    mass0 = 4 * np.pi * r0**3 * rho / 3
    r = r0 * np.cbrt(mass * (1 + water))
    k, kelvin = kinetics(r, rho, scenario)
    time = cumulative_trapezoid(
        mass0
        * mass
        / (k * kelvin * mole_share * 1e-9 * speed * mass**hardening * 60),
        q,
        initial=0.0,
    )
    if np.max(times) > time[-1]:
        raise ValueError("time range for exposure integral too short")
    return np.interp(times, time, mass)


def fit_weights(obs, prior, name, rh, sc, *, n=1300):
    # Softmax with a fixed seventh logit and weak ridge on fractions. The
    # reported precursor-specific weights provide the regularization target.
    # A cyclic fit sees only its own observed curve and reported weights.
    def residual(z):
        f = softmax(np.r_[z, 0.0])
        pred = predict(f, name, rh, SIZE[name], obs.time_min, sc, n=n)
        return np.r_[pred - obs.vfr, 0.055 * (f - prior)]

    result = least_squares(
        residual,
        np.log(prior[:-1] / prior[-1]),
        bounds=(-22, 22),
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=180,
    )
    f = softmax(np.r_[result.x, 0.0])
    return f


def perturb(obs, name, sc):
    data = obs.copy()
    sign = 1 if CYCLIC[name] == name else -1
    sampled = data.time_min > 0
    data.loc[sampled, "time_min"] += sc.time_shift * sign
    data.loc[sampled, "vfr"] += sc.marker_shift * sign
    if name == "farnesene" and sc.skip_shape:
        data = data[(data.time_min == 0) | (data.time_min >= 30)]
    return data


def speed_fit(f, name, rh, diam, data, sc):
    objective = lambda z: np.mean(
        (
            predict(f, name, rh, diam, data.time_min, sc, speed=np.exp(z))
            - data.vfr
        )
        ** 2
    )
    return np.exp(
        minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded").x
    )


def finite_sink(
    f, name, rh, diam, times, sc, tau_s, number_cm3, *, rtol=1e-8, speed=1.0
):
    # Coupled dimensionless particle and gas inventories relative to initial
    # particle mass per air volume. Gas starts clean. Walls irreversibly take
    # up well-mixed organic vapor on the specified first-order lifetime.
    rho, water, mole_share = particle_properties(name, rh, sc)
    r0 = diam * 0.5e-9
    m0 = 4 * np.pi * r0**3 * rho / 3
    number = number_cm3 * 1e6
    mfp = MFP
    kr = (
        0.0 if sc.sigma == 0 else 2 * sc.sigma * MW / (8.314462618 * TEMP * rho)
    )

    def rhs(t, y):
        p = y[:7]
        gas = y[7:14]
        total = p.sum()
        r = r0 * np.cbrt(max(total, (0.5e-9 / r0) ** 3) * (1 + water))
        # Scalar equivalent of the Particula kernels, tested independently
        # against predict() with zero vapor and compared in the saved oracle.
        kn = mfp / r
        corr = (
            0.75
            * sc.alpha
            * (1 + kn)
            / (kn**2 + kn + 0.283 * sc.alpha * kn + 0.75 * sc.alpha)
        )
        k = 4 * np.pi * r * GAS_D * corr
        transfer = (
            k
            * speed
            * (
                CSTAR * 1e-9 * p / total * np.exp(kr / r) * mole_share / m0
                - gas * number
            )
        )
        if tau_s == 0:
            return np.r_[-transfer * 60, np.zeros(7), transfer * 60]
        sink = gas / tau_s
        return np.r_[-transfer * 60, (transfer - sink) * 60, sink * 60]

    unique, inv = np.unique(np.asarray(times), return_inverse=True)

    def disappeared(t, y):
        return y[:7].sum() - (0.5e-9 / r0) ** 3

    disappeared.terminal = True
    disappeared.direction = -1
    sol = solve_ivp(
        rhs,
        (0, max(unique)),
        np.r_[f, np.zeros(14)],
        t_eval=unique,
        method="LSODA",
        rtol=rtol,
        atol=rtol * 1e-3,
        events=disappeared,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.y
    budget = np.max(np.abs(y[:7] + y[7:14] + y[14:] - f[:, None]))
    assert budget < 1e-7 and y.min() > -1e-7, (budget, y.min())
    # Zero denotes loss below 1 nm physical diameter. Post-loss gas histories
    # are not used for the particle survival comparison.
    surviving = np.zeros(len(unique))
    surviving[: len(sol.t)] = y[:7].sum(axis=0)
    return surviving[inv], budget


# %%
def run(source, out, mode):
    out.mkdir(exist_ok=True, parents=True)
    observed = pd.read_csv(source)
    groups = {
        (p, rh): rows.sort_values("time_min")
        for (p, rh), rows in observed.groupby(["precursor", "rh"])
    }
    assert set(groups) == set(REPORTED)
    all_names = list(DENSITY)
    fits, matched, series, age, hold, cons, refine = [], [], [], [], [], [], []
    diameters = np.unique(
        np.r_[np.geomspace(65.0, 300.0, 47), 140.0, 165.0, 220.0]
    )
    treatment_names = ["primary"] if mode == "pilot" else list(SCENARIOS)
    for model in treatment_names:
        sc = SCENARIOS[model]
        fitted = {}
        for key, data in groups.items():
            name, rh = key
            reference = REPORTED[key]
            data = perturb(data, name, sc)
            fitted[key] = fit_weights(data, reference, name, rh, sc)
            for cstar, pub, own in zip(
                CSTAR, reference, fitted[key], strict=True
            ):
                fits.append((model, name, rh, cstar, pub, own))
        for key, obs in groups.items():
            name, rh = key
            parent = CYCLIC[name]
            f = fitted[key]
            cy = fitted[parent, rh]
            times = obs.time_min.to_numpy()
            own = predict(f, name, rh, SIZE[name], times, sc)
            tran = predict(cy, name, rh, SIZE[name], times, sc)
            pub = predict(REPORTED[key], name, rh, SIZE[name], times, sc)
            for t, y, b, c, a in zip(
                times, obs.vfr, pub, tran, own, strict=True
            ):
                series.append((model, name, rh, t, y, b, c, a))
            pred = lambda weights, p, d: float(
                predict(weights, p, rh, d, [200.0], sc)[0]
            )
            matched.append(
                (
                    model,
                    name,
                    rh,
                    pred(f, name, SIZE[name]),
                    pred(cy, name, SIZE[name]),
                    pred(f, name, 165.0),
                    pred(cy, parent, 165.0),
                    pred(cy, name, 165.0),
                    pred(REPORTED[key], name, SIZE[name]),
                )
            )
            for diam in diameters:
                age.append((name, rh, diam, pred(f, name, diam), model))
            if model != "primary":
                continue
            # Challenge whether a changed emission rate alone at unchanged
            # cyclic composition predicts the last record after an early fit.
            early = obs[(obs.time_min > 0) & (obs.time_min <= 300)]
            if not len(early):
                continue
            speed = speed_fit(cy, name, rh, SIZE[name], early, sc)
            late = obs[obs.time_min > 300]
            for t, y in zip(late.time_min, late.vfr):
                a = predict(cy, name, rh, SIZE[name], [t], sc, speed=speed)[0]
                orig = predict(cy, name, rh, SIZE[name], [t], sc)[0]
                hold.append((name, rh, len(early), speed, t, y, orig, a))
            challenge_times = np.unique(np.r_[times, 200.0])
            for lifetime, count in [
                (0, 150),
                (600, 100),
                (3600, 200),
                (86400, 200),
            ]:
                finite, b = finite_sink(
                    cy,
                    name,
                    rh,
                    SIZE[name],
                    challenge_times,
                    sc,
                    lifetime,
                    count,
                )
                if lifetime == 0:
                    direct = predict(
                        cy, name, rh, SIZE[name], challenge_times, sc
                    )
                    fine = predict(
                        cy, name, rh, SIZE[name], challenge_times, sc, n=5200
                    )
                    for t, co, fi, ode in zip(
                        challenge_times, direct, fine, finite
                    ):
                        refine.append((name, rh, t, co, fi, ode))
                for t, value in zip(challenge_times, finite):
                    cons.append((name, rh, lifetime, count, t, value, b))

    def write(filename, columns, rows):
        pd.DataFrame(rows, columns=columns.split()).to_csv(
            out / filename, index=False, float_format="%.10g"
        )

    write(
        "fits.csv",
        "model precursor rh cstar_ug_m3 published_fraction fitted_fraction",
        fits,
    )
    write(
        "series.csv",
        "model precursor rh time_min observed_vfr published_reconstruction cyclic_prediction precursor_fit",
        series,
    )
    write(
        "matched.csv",
        "model precursor rh own_native cyclic_native own_165 cyclic_reference_165 cyclic_at_acyclic_density_165 published_native",
        matched,
    )
    write(
        "ageing.csv",
        "precursor rh initial_diameter_nm volume_fraction_200min model",
        age,
    )
    write(
        "late.csv",
        "precursor rh early_points fitted_rate_ratio time_min observed_vfr unchanged_rate held_rate_only",
        hold,
    )
    write(
        "sink.csv",
        "precursor rh gas_wall_lifetime_s particles_cm3 time_min cyclic_vfr species_budget_max",
        cons,
    )
    write(
        "refinement.csv",
        "precursor rh time_min primary refined ode_oracle",
        refine,
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument(
        "--observations",
        type=Path,
        default=HERE.parent / "sources" / "observations.csv",
    )
    p.add_argument("--out", type=Path, default=HERE / "results")
    p.add_argument("--mode", choices=["pilot", "full"], default="full")
    a = p.parse_args()
    run(a.observations, a.out, a.mode)
