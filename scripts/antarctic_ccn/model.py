"""Size-integrated CCN calculations for the King Sejong comparison."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from particula.particles import get_kelvin_radius
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

DATA = Path(__file__).resolve().parents[2] / "docs/data/antarctic-ccn"
WATER_DENSITY = 997.1
A = 2e9 * get_kelvin_radius(0.072, WATER_DENSITY, 0.018015, 298.15)
SALT_DENSITIES = {"NaCl": 2165.0, "AS": 1770.0}
BANDS = [(35, 40), (40, 60), (60, 75), (75, 165)]
DIAM = np.geomspace(30.0, 185.0, 641)

FILM_THICKNESS = 0.30  # nm
FILM_SIGMA = 0.035  # N/m; adipic-acid analogue adopted by Davies et al.


@dataclass
class SaltFilm:
    """A fixed aqueous core plus a water-free organic phase of fixed volume."""

    salt: str
    temperature: float
    lab: bool = False

    def __post_init__(self):
        """Load the salt water-activity curve at the chosen temperature."""
        table = pd.read_csv(DATA / "activities.csv")
        table = table[
            (table.salt == self.salt)
            & (table.temperature_K == self.temperature)
        ]
        self.bounds = (table.ratio.min(), table.ratio.max())
        self.log_aw = PchipInterpolator(
            np.log(table.ratio), np.log(table.aw), extrapolate=False
        )
        self.sigma = 0.07275 if self.lab else 0.072
        self.water_density = 997.0 if self.lab else 997.1
        self.solid = 1770.0 if self.salt == "AS" else 2165.0
        self.liquid = 1550.0 if self.lab else self.solid
        self.kelvin = 2e9 * get_kelvin_radius(
            self.sigma, self.water_density, 0.018015, self.temperature
        )

    def growth(self, ratio, salt_fraction):
        """Return the wet to dry diameter ratio at fixed dry inventory."""
        # Volumes relative to total anhydrous dry particle volume.
        return np.cbrt(
            1
            - salt_fraction
            + salt_fraction
            * self.solid
            * (1 / self.liquid + ratio / self.water_density)
        )

    def fraction_covered(self, diameter, growth, salt_fraction):
        """Return the fraction covered by the available organic volume."""
        wet = diameter * growth
        # Stable expansion of the relative shell volume.
        x = np.minimum(2 * FILM_THICKNESS / wet, 1.0)
        shell_volume = growth**3 * x * (3 - 3 * x + x * x)
        return np.minimum(1.0, (1 - salt_fraction) / shell_volume)

    def saturation(self, ratio, diameter, salt_fraction, film=True):
        """Return log equilibrium saturation at the wet surface."""
        g = self.growth(ratio, salt_fraction)
        coverage = (
            self.fraction_covered(diameter, g, salt_fraction) if film else 0.0
        )
        tension = self.sigma + coverage * (FILM_SIGMA - self.sigma)
        return self.log_aw(np.log(ratio)) + self.kelvin * tension / (
            self.sigma * diameter * g
        )

    def peaks(self, diameters, fraction, resolution, film=True):
        """Find the critical supersaturation on the salt activity table."""
        diameters = np.atleast_1d(diameters)
        logratio = np.linspace(*np.log(self.bounds), resolution)
        ratio = np.exp(logratio)
        # Guard PCHIP domain against an endpoint moved by exp/log round-off.
        ratio = np.clip(ratio, *self.bounds)
        peaks = []
        for start in range(0, len(diameters), 256):
            d = diameters[start : start + 256]
            vals = self.saturation(ratio[None, :], d[:, None], fraction, film)
            i = vals.argmax(axis=1)
            if np.any(i == 0) or np.any(i == len(ratio) - 1):
                raise ValueError(
                    "Finite table did not bracket the selected maximum"
                )
            low = logratio[i - 1]
            high = logratio[i + 1]
            for _ in range(23):
                a = (2 * low + high) / 3
                b = (low + 2 * high) / 3
                right = self.saturation(
                    np.exp(a), d, fraction, film
                ) < self.saturation(np.exp(b), d, fraction, film)
                low = np.where(right, a, low)
                high = np.where(right, high, b)
            optimum = np.exp((low + high) / 2)
            sat = self.saturation(optimum, d, fraction, film)
            peaks.extend(100 * np.expm1(sat))
        return np.array(peaks)


def integrate_spectrum(data, mass, cutoffs, band=(20.0, 1000.0)):
    """Integrate activation with piecewise-linear weights in log diameter."""
    sizes = np.log10(data["sizes"])
    centers = np.log10(data["diam"])
    edges = np.r_[
        centers[0] - (centers[1] - centers[0]) / 2,
        (centers[1:] + centers[:-1]) / 2,
        centers[-1] + (centers[-1] - centers[-2]) / 2,
    ]
    limits = np.log10(band)
    segments = np.unique(
        np.clip(
            np.r_[edges, sizes, limits],
            max(limits[0], edges[0]),
            min(limits[1], edges[-1]),
        )
    )
    left, right = segments[:-1], segments[1:]
    idx = np.searchsorted(edges, (left + right) / 2) - 1
    density = data["density"][:, idx]
    values = np.zeros((len(density), cutoffs.shape[-1]))
    for a in range(len(sizes)):
        hat = np.eye(len(sizes))[a]
        w1, w2 = np.interp(left, sizes, hat), np.interp(right, sizes, hat)
        slope = (w2 - w1) / (right - left)
        for j in range(mass.shape[-1]):
            if not mass[:, a, j].any():
                continue
            lower = np.clip(np.log10(cutoffs[a, j])[:, None], left, right)
            weights = (right - lower) * (w2 + (lower - right) * slope / 2)
            values += mass[:, a, j, None] * (density @ weights.T)
    return values


def critical_k(k, ss, kelvin):
    """Invert the κ-Köhler barrier for critical dry diameter at each setting."""
    log_s = np.log1p(ss / 100)
    ks = k[..., None]
    lo = np.full((*k.shape, len(ss)), 1e-10)
    hi = np.full_like(lo, 10.0)

    def diameter(x):
        return kelvin / (np.exp(x) * (log_s + np.log1p(ks / np.expm1(3 * x))))

    for _ in range(70):
        x = (2 * lo + hi) / 3
        y = (lo + 2 * hi) / 3
        increasing = diameter(x) < diameter(y)
        lo = np.where(increasing, x, lo)
        hi = np.where(increasing, hi, y)
    return np.where(ks > 0, diameter((lo + hi) / 2), kelvin / log_s)


def _match_fraction(film, growth, diameter, rh, active):
    water_per_dry_volume = (growth**3 - 1) * film.water_density / film.solid
    minimum = water_per_dry_volume / film.bounds[1]
    maximum = min(1.0, water_per_dry_volume / film.bounds[0])
    if minimum > maximum or minimum <= 0:
        return None

    def residual(fraction):
        ratio = np.clip(water_per_dry_volume / fraction, *film.bounds)
        return film.saturation(ratio, diameter, fraction, active) - np.log(rh)

    grid = np.linspace(minimum, maximum, 65)
    error = residual(grid)
    exact = grid[error == 0]
    brackets = np.flatnonzero(error[:-1] * error[1:] < 0)
    if len(exact) + len(brackets) != 1:
        return None
    if len(exact):
        return exact[0]
    i = brackets[0]
    return brentq(residual, grid[i], grid[i + 1])


def _salt_activity_tables():
    water = pd.read_csv(DATA / "water-activity.csv")
    fun = []
    limits = []
    for s in ["NaCl", "AS"]:
        t = water[water.salt == s].sort_values("aw")
        x = np.log(-np.log(t.aw.values))
        fun.append(
            PchipInterpolator(
                x[::-1],
                np.log(t.ratio.values * SALT_DENSITIES[s] / WATER_DENSITY)[
                    ::-1
                ],
                extrapolate=False,
            )
        )
        limits.append([x.min(), x.max()])
    x = np.linspace(max(v[0] for v in limits), min(v[1] for v in limits), 4097)
    volumes = np.exp([f(x) for f in fun])
    return fun, x, volumes


def make_thresholds(data, settings, gf_scale=1.0, rh=0.90, res=513):
    """Infer composition from growth and return activation cutoffs in nm."""
    gf, group = np.unique(data["gf"], return_inverse=True)
    mass = np.stack(
        [data["raw_pdf"][:, :, group == i].sum(2) for i in range(len(gf))],
        axis=2,
    )
    mass /= mass.sum(axis=2, keepdims=True)
    anchors = data["sizes"]
    g = np.maximum(1, gf * gf_scale)
    k = (g**3 - 1) * np.expm1(A / (anchors[:, None] * g) - np.log(rh))
    base = critical_k(k, settings, A)
    cuts = [base.copy() for i in range(7)]
    paper_kelvin = 4 * 0.018015 * 0.072 / (8.315 * 298.15 * 997.1) * 1e9
    cuts[0][:] = paper_kelvin * np.cbrt(
        4
        / (
            27
            * np.where(g >= 1.85, 1.1, 0.6)[:, None]
            * np.log1p(settings / 100) ** 2
        )
    )
    fun, x, volumes = _salt_activity_tables()
    denom = np.log1p(settings[:, None] / 100) + np.exp(x)
    film = SaltFilm("AS", 298.15)
    ok = np.zeros((3, *k.shape), bool)
    for a, d in enumerate(anchors):
        for j, gg in enumerate(g):
            if gg <= 1 or not mass[:, a, j].any():
                continue
            x0 = np.log(-np.log(rh) + A / (d * gg))
            fractions = (gg**3 - 1) / np.exp([f(x0) for f in fun])
            if ((fractions > 0) & (fractions <= 1)).all():
                for n, f in enumerate(fractions):
                    cuts[2 + n][a, j] = np.max(
                        A / (np.cbrt(1 + f * volumes[n]) * denom), axis=1
                    )

            for active in [True, False]:
                f = _match_fraction(film, gg, d, rh, active)
                if f is None:
                    continue
                for idx, surf in (
                    [(0, True), (2, False)] if active else [(1, False)]
                ):
                    crit = film.peaks(DIAM, f, res, surf)
                    if np.diff(crit).max() >= 2e-5:
                        raise ValueError("Nonmonotone activation threshold")
                    cuts[4 + idx][a, j] = np.exp(
                        np.interp(
                            settings,
                            crit[::-1],
                            np.log(DIAM[::-1]),
                            left=np.log(1e6),
                            right=np.log(0.01),
                        )
                    )
                    ok[idx, a, j] = True
    return np.stack(cuts), ok, mass


def responses(data, thresholds, diameter_scale=1.0, number_scale=1.0):
    """Return hourly counts and coating effects by dry size (CCN/cm³)."""
    cuts, support, mass = thresholds
    shifted = dict(data, diam=data["diam"] * diameter_scale)
    band = (20 * diameter_scale, 1000.0)

    def integrate(c, weights, region):
        return number_scale * integrate_spectrum(shifted, weights, c, region)

    salt = sum(integrate(c, mass, band) for c in cuts[2:4]) / 2

    def differences(layer):
        group = mass * support[layer]
        return np.stack(
            [
                integrate(cuts[4 + layer], group, b)
                - sum(integrate(c, group, b) for c in cuts[2:4]) / 2
                for b in BANDS
            ]
        )

    return (
        integrate(cuts[0], mass, band),
        integrate(cuts[1], mass, band),
        salt,
        differences(0),
        differences(1),
        differences(2),
    )
