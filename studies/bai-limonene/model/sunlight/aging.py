"""Paired solar exposure, absorptive gas exchange and size-resolved source labels."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
RHO = 1250.0
MW = 0.250
# Fresh MV is resolved into three volatility bands; aged products in two groups.
J = pd.read_csv(ROOT / "solar/rates.csv").set_index("species").standard_j_per_s
PROFILES = pd.read_csv(ROOT / "inflow/profiles.csv")


@dataclass(frozen=True)
class Exposure:
    """Condensed inventory, volatility and photochemical rate assignments."""

    cstar_lv: float = 0.1
    cstar_aged_mv: float = 100.0
    fresh_mv: float = 0.45
    inventory: str = "bai"
    rate_assignment: str = "ranked"
    spread_aged: bool = False
    common_fresh_k: bool = False


@dataclass
class Population:
    original: np.ndarray  # diameter nm
    diameter: np.ndarray  # diameter nm
    number: np.ndarray  # counts/cm3 at reference density
    kappa: np.ndarray
    tag: np.ndarray  # 0 emitted seeds, 1 field background
    edges: tuple
    med: float
    audit: dict


def inflow(profile, bins=48, sub=2, loading=300.0, med=100.0, background_k=1.0):
    z = np.linspace(-4.0, 4.0, bins + 1)
    edges = med * 1.5**z
    diam = np.sqrt(edges[1:] * edges[:-1])
    n = loading * np.diff(ndtr(z)) / (ndtr(4) - ndtr(-4))
    q = PROFILES[PROFILES.profile == profile]
    if q.empty:
        raise ValueError(profile)
    log_edges = np.log(np.r_[q.lower_nm.iloc[0], q.upper_nm])
    bedges = np.exp(
        np.interp(
            np.arange(len(q) * sub + 1) / sub, np.arange(len(q) + 1), log_edges
        )
    )
    bd = np.sqrt(bedges[1:] * bedges[:-1])
    bn = np.repeat(q.count_cm3.to_numpy() / sub, sub)
    # Smooth interpolation of the published daily kappa(D) between measured centers.
    bk = np.interp(np.log(bd), np.log(q.diameter_nm), q.kappa) * background_k
    if profile == "clean" and background_k == 0:
        bn[:] = 0
        bk[:] = 0.30
    d = np.r_[diam, bd]
    return Population(
        d,
        d.copy(),
        np.r_[n, bn],
        np.r_[np.full(bins, 0.0465), bk],
        np.r_[np.zeros(bins, int), np.ones(len(bd), int)],
        (edges, bedges),
        med,
        {},
    )


@lru_cache
def optical_table(name):
    return pd.read_csv(ROOT / "optics" / name)


def species(case):
    if case.inventory == "bai":
        slow = np.array([1 - case.fresh_mv])
        fast = case.fresh_mv * np.array([0.270, 0.176, 0.008]) / 0.454
        cslow = np.array([case.cstar_lv])
        cfast = np.array([100.0, 1000.0, 10000.0])
    else:
        # Wilson et al. Table S1 is an effective volatility basis inferred from
        # dark suspended particles. It does not resolve Bai's photoactive groups.
        weights = np.array(
            {
                "wilson": [0.24, 0.10, 0.18, 0.44, 0.04],
                "wilson_dry": [0.26, 0.20, 0.18, 0.36, 0.0],
            }[case.inventory]
        )
        cslow = cfast = 10.0 ** np.arange(-3, 2)
        if case.rate_assignment == "literal":
            fast = np.zeros(5)
        elif case.rate_assignment == "independent":
            fast = case.fresh_mv * weights
        elif case.rate_assignment == "inverse":
            fast = np.minimum(
                weights,
                np.maximum(
                    0, case.fresh_mv - np.r_[0, np.cumsum(weights)[:-1]]
                ),
            )
        elif case.rate_assignment == "ranked":
            above = np.cumsum(weights[::-1])[::-1] - weights
            fast = np.minimum(weights, np.maximum(0, case.fresh_mv - above))
        else:
            raise ValueError(case.rate_assignment)
        slow = weights - fast
    aged_mv = case.cstar_aged_mv * 10.0 ** np.arange(
        4 if case.spread_aged else 1
    )
    c = np.r_[cslow, cfast, case.cstar_lv, aged_mv]
    ns = len(slow)
    nf = len(fast)
    k = np.r_[
        np.full(ns, 0.060),
        np.full(nf, 0.030),
        0.095,
        np.full(len(aged_mv), 0.150),
    ]
    if case.common_fresh_k:
        k[: ns + nf] = 0.0465
    return c, k, np.r_[slow, fast], slice(0, ns), slice(ns, ns + nf), ns + nf
