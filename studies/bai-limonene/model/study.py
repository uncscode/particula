# %% Imports and constants
"""Reproduce matched Particula parcels for the Bai photochemical aging study."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from particula.dynamics import parcel as par

HERE = Path(__file__).resolve().parent
T0, P0, RH0 = 285.0, 90000.0, 0.999
sat0 = par._saturation_pressure(T0)
QREF = par.RD / par.RV * RH0 * sat0 / (P0 - RH0 * sat0)
VREF = (par.RD + par.RV * QREF) * T0 / P0


# %% Case setup and observation conventions
def setup(
    n,
    k,
    f,
    median=100.0,
    loading=300.0,
    size_slope=0.0,
    probes=(),
    returned_share=0.0,
    path_nm=75.0,
    alpha=1.0,
):
    edges = np.linspace(-4, 4, n + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = np.diff(ndtr(edges))
    weights /= weights.sum()
    centers = np.r_[centers, probes]
    weights = np.r_[weights, np.zeros(len(probes))]
    ix = np.argsort(centers, kind="stable")
    centers, weights = centers[ix], weights[ix]
    dd = median * 1e-9 * 1.5**centers
    # Enforce the measured bulk mass fraction even when per-size losses differ.
    if returned_share:
        from redistribution import redeposit

        fraction = redeposit(dd, weights, f, returned_share, path_nm, alpha)
    elif f == 1 or size_slope == 0:
        fraction = np.full_like(dd, f)
    else:

        def fractions(scale):
            return np.minimum(1, scale * (dd / (median * 1e-9)) ** size_slope)

        scale = brentq(
            lambda x: np.average(fractions(x), weights=weights * dd**3) - f,
            0,
            50,
        )
        fraction = fractions(scale)
    dry = dd / 2 * np.cbrt(fraction)
    number = weights * loading * 1e6 * VREF

    def state(q):
        return par.ParcelInitialState(dry, number, k, P0, T0, q)

    def residual(q):
        wet = par._initial_radii(state(q))
        return q + par.WATER_FACTOR * (number @ (wet**3 - dry**3)) - QREF

    q = brentq(residual, QREF * 0.98, QREF, xtol=1e-16)
    return state(q), centers


def observe(tr, centers, initial):
    dry, k = initial.dry_radius, initial.kappa

    def counts(i):
        x = tr.wet_radius[i] / dry
        u = x**3 - 1
        a = 2 * par.SIGMA / (par.RV * par.RHO_W * tr.temperature[i] * dry)
        # log ratio changes sign exactly at the critical wet radius.
        margin = np.log(a * u * (u + k) / (3 * k * x**4))
        yes = margin >= 0
        assert np.all(np.diff(yes.astype(int)) >= 0), (
            "nonmonotone activation requires a different integration rule"
        )
        conc = sum(initial.number) * 1e-6 / tr.volume[i]
        raw = np.average(yes, weights=initial.number) * conc
        if yes.all() or not yes.any():
            return raw, raw, 0.0, float(4 if not yes.any() else -4)
        j = np.argmax(yes)
        cut = np.interp(0.0, margin[j - 1 : j + 1], centers[j - 1 : j + 1])
        smooth = (ndtr(4) - ndtr(cut)) / (ndtr(4) - ndtr(-4)) * conc
        width = (
            (ndtr(centers[j]) - ndtr(centers[j - 1]))
            / (ndtr(4) - ndtr(-4))
            * conc
        )
        return smooth, raw, width, cut

    c, raw, width, cut = counts(-1)
    wet = tr.wet_radius[-1]
    geom_cut = np.interp(1e-6, wet, centers, left=-4.0, right=4.0)
    geometric = (
        (ndtr(4) - ndtr(geom_cut))
        / (ndtr(4) - ndtr(-4))
        * sum(initial.number)
        * 1e-6
        / tr.volume[-1]
    )
    water = tr.vapor + tr.liquid
    return dict(
        count_cm3=c,
        discrete_cm3=raw,
        boundary_bin_cm3=width,
        cutoff_z=cut,
        geometric_um_cm3=geometric,
        peak_ss_pct=100 * np.max(tr.supersaturation),
        peak_z_m=tr.height[np.argmax(tr.supersaturation)],
        cloud_base_m=np.interp(
            0.0,
            tr.supersaturation[: np.argmax(tr.supersaturation) + 1],
            tr.height[: np.argmax(tr.supersaturation) + 1],
        ),
        water_residual=float(np.max(abs(water - QREF))),
        liquid_gkg=1000 * tr.liquid[-1],
        final_vol_m3=tr.volume[-1],
        min_wet_dry=float(np.min(tr.wet_radius / dry)),
    )


def simulate(n, k, f, w=1.0, device="cpu", tol=1e-5, cadence_m=1.0, **kw):
    initial, centers = setup(n, k, f, **kw)
    tr = par.run_parcel(
        initial,
        updraft=w,
        timestep=cadence_m / w,
        steps=round(100 / cadence_m),
        device=device,
        adaptive=None if device == "cpu" else par.ParcelROS3(rtol=tol),
    )
    return observe(tr, centers, initial)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("device")
    p.add_argument("out", type=Path)
    p.add_argument("--bins", type=int, default=16)
    a = p.parse_args()
    rows = []
    for k, f in [(0.047, 1.0), (0.110, 0.309)]:
        start = time.monotonic()
        row = dict(
            kappa=k,
            f=f,
            n=a.bins,
            device=a.device,
            **simulate(a.bins, k, f, device=a.device),
            elapsed=time.monotonic() - start,
        )
        print(row, flush=True)
        rows.append(row)
    a.out.write_text(json.dumps(rows, indent=2) + "\n")
