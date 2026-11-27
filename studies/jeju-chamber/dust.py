"""Reconstruct pre-cloud run-5 aerosol from supplier volume and run-4 sulfate."""

import os

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar, nnls
from scipy.special import rel_entr


from observations import START, source

# Powder Technology Inc., ISO 12103-1 A2 Fine Test Dust, cumulative dry
# volume (%). Rows contain diameter in µm and lower/upper volume limits.
# https://www.powdertechnologyinc.com/product/iso-12103-1-a2-fine-test-dust/
MANUFACTURER = np.array(
    [
        [0.97, 4.5, 5.5],
        [1.38, 8, 9.5],
        [2.75, 21.3, 23.3],
        [5.5, 39.5, 42.5],
        [11, 57, 59.5],
        [22, 73.5, 76],
        [44, 89.5, 91.5],
        [88, 97.9, 98.9],
        [124.5, 99, 100],
        [176, 100, 100],
    ]
)
if "JEJU_A2_TABLE" in os.environ:
    MANUFACTURER = pd.read_csv(os.environ["JEJU_A2_TABLE"]).to_numpy()
NODES, WEIGHTS = leggauss(12)
BANDS = np.array([0.0, 2.0, 5.0, 10.0, 20.0, np.inf])
DUST_FLOOR = 0.35
# Detection starts at the first channel with signal in the archived six runs.
FIRST = 11


def extract_observations():
    windows, air = [], []
    for run in (4, 5):
        raw = pd.read_csv(
            source(f"OPC_exp{run}.csv"), index_col=0, parse_dates=[0]
        )
        minute0 = pd.Timestamp(f"{raw.index[0].date()} {START[run]}")
        t = (raw.index - minute0).total_seconds().to_numpy()
        center = raw.columns.astype(float).to_numpy()
        logd = np.log(center)
        edge = np.exp(
            np.r_[
                logd[0] - np.diff(logd)[0] / 2,
                (logd[:-1] + logd[1:]) / 2,
                logd[-1] + np.diff(logd)[-1] / 2,
            ]
        )
        per_bin = raw.to_numpy() * np.diff(np.log10(edge))
        assert np.all(per_bin[:, :FIRST] == 0)
        minute = [per_bin[(t >= 60 * j) & (t < 60 * (j + 1))] for j in range(5)]
        assert all(55 <= len(a) <= 65 for a in minute)
        windows.append(np.stack([a.mean(axis=0) for a in minute]))
        met = pd.read_csv(
            source(f"Chamber_P,T_exp{run}.csv"), index_col=0, parse_dates=[0]
        )
        lic = pd.read_csv(source(f"LICOR_H2O_exp{run}.csv")).dropna(
            subset=["Date", "Time"]
        )
        lic.index = pd.to_datetime(lic.Date + " " + lic.Time.str[:8])
        index = pd.date_range(minute0, periods=300, freq="s")
        temp = met.loc[index, "Tair"].to_numpy() + 273.15
        vapor = lic.iloc[:, -1].reindex(index).to_numpy() * 0.01801528
        es = np.exp(
            54.842763
            - 6763.22 / temp
            - 4.210 * np.log(temp)
            + 0.000367 * temp
            + np.tanh(0.0415 * (temp - 218.8))
            * (
                53.878
                - 1331.22 / temp
                - 9.44523 * np.log(temp)
                + 0.014025 * temp
            )
        )
        rh = vapor * 0.001 * 461.52 * temp / es
        assert np.nanmax(rh) < 1
        air.append(
            [
                run,
                minute0,
                np.nanmin(rh),
                np.nanmax(rh),
                np.min(temp) - 273.15,
                np.max(temp) - 273.15,
            ]
        )
    return center, edge, np.array(windows), air


def quadrature(edges, envelope, optical, floor):
    """Split at all source and instrument boundaries then integrate in ln D.

    Basis normalizes to 1 µm³ dry powder volume. Per-node number times
    individual sphere volume recovers each input interval's volume fraction.
    """
    d = MANUFACTURER[:, 0]
    cumulative = {
        "middle": MANUFACTURER[:, 1:].mean(axis=1),
        "lower": MANUFACTURER[:, 1],
        "upper": MANUFACTURER[:, 2],
    }[envelope] / 100
    table_edges = np.r_[floor, d]
    fractions = np.diff(np.r_[0.0, cumulative])
    sub_edges = np.geomspace(floor, edges[FIRST] / optical, 9)
    split = np.unique(np.r_[table_edges, sub_edges, edges[FIRST:] / optical])
    split = split[(split >= floor) & (split <= d[-1])]
    lo, hi = np.log(split[:-1]), np.log(split[1:])
    diameter = np.exp(
        ((hi - lo)[:, None] * NODES + (hi + lo)[:, None]) / 2
    ).ravel()
    interval = np.searchsorted(d, diameter)
    volume = (((hi - lo)[:, None] / 2) * WEIGHTS).ravel()
    volume *= fractions[interval] / np.diff(np.log(table_edges))[interval]
    number = volume / (np.pi / 6 * diameter**3)
    back = np.cumsum(
        np.bincount(
            interval, weights=number * np.pi / 6 * diameter**3, minlength=len(d)
        )
    )
    err_pct = np.max(abs(back - cumulative)) * 100
    assert err_pct < 1e-9
    cell = (
        np.searchsorted(
            np.r_[sub_edges, edges[FIRST + 1 :] / optical], diameter
        )
        - 1
    )
    # 8 sub-Welas classes, 59 on-grid classes, 1 overflow.
    return diameter, number, volume, cell, err_pct


def spectrum(dry, number, volume, cell, d50, edges, optical, floor):
    efficiency = (
        np.ones_like(dry) if np.isinf(d50) else 1 / (1 + (dry / d50) ** 2)
    )
    n = np.bincount(cell, weights=number * efficiency, minlength=68)
    v = np.bincount(cell, weights=volume * efficiency, minlength=68)
    sub = np.geomspace(floor, edges[FIRST] / optical, 9)
    limits = np.r_[sub, edges[FIRST + 1 :] / optical, MANUFACTURER[-1, 0]]
    representative = (
        6
        * np.divide(
            v, n, out=np.pi / 6 * (limits[:-1] * limits[1:]) ** 1.5, where=n > 0
        )
        / np.pi
    ) ** (1 / 3)
    return n, v, representative


def reconstruct(
    observed,
    sulfate,
    response,
    center,
    edges,
    envelope,
    optical,
    floor,
    delivery,
    sulfate_cap=np.inf,
):
    dry, num, vol, cell, closure = quadrature(edges, envelope, optical, floor)
    valid = np.arange(len(center)) >= FIRST
    band = np.digitize(center, BANDS) - 1
    weights = np.zeros(len(center))
    for group in np.unique(band[valid]):
        take = valid & (band == group)
        weights[take] = 1 / np.sqrt(take.sum() * 5)
    # Unknown common sample volume cancels when conditioning on total number.
    # Treat this as a descriptive multinomial discrepancy, without sampling CIs.
    observed_total = observed.sum()
    observed_shape = observed / observed_total
    sulfate_shape = (response @ sulfate) / sulfate.sum()

    def profile(d50):
        n, v, d = spectrum(dry, num, vol, cell, d50, edges, optical, floor)
        unit_dust = np.r_[np.zeros(FIRST), n[8:-1]]
        dust_shape = response @ unit_dust / unit_dust.sum()

        def divergence(f):
            expected = f * dust_shape + (1 - f) * sulfate_shape
            return np.sum(
                rel_entr(observed_shape, np.maximum(expected, 1e-100))
            )

        low = max(0.0, 1.0 - sulfate_cap * sulfate.sum() / observed_total)
        scan = minimize_scalar(
            divergence,
            method="bounded",
            bounds=(low, 1.0),
            options={"xatol": 1e-12},
        )
        f = min([low, scan.x, 1.0], key=divergence)
        return divergence(f), f, n, v, d, unit_dust

    d50 = np.inf
    if delivery == "coarse":
        opt = minimize_scalar(
            lambda x: profile(np.exp(x))[0],
            method="bounded",
            bounds=(np.log(1), np.log(1e4)),
            options={"xatol": 1e-8},
        )
        d50 = min([1.0, np.exp(opt.x), 1e4], key=lambda diam: profile(diam)[0])
    deviance, fraction, n, v, d, unit_dust = profile(d50)
    a = observed_total * fraction / unit_dust.sum()
    b = observed_total * (1 - fraction) / sulfate.sum()
    dust, sulf = unit_dust * a, sulfate * b
    model = response @ (dust + sulf)
    assert np.isclose(model.sum(), observed_total)
    n, v = n * a, v * a
    output = dict(
        dust=dust,
        sulfate=sulf,
        dust_dry_um=np.r_[center[:FIRST] / optical, d[8:-1]],
        subdust_number=n[:8],
        subdust_dry_um=d[:8],
        overflow_number=n[-1],
        overflow_dry_um=d[-1],
        original_below097_number=a
        * np.sum(num[dry < 0.97] / (1 + (dry[dry < 0.97] / d50) ** 2)),
        fitted_measurement=model,
    )
    captured = float(np.sum(n * np.pi / 6 * d**3))
    assert np.isclose(captured, v.sum(), rtol=1e-11)
    row = dict(
        envelope=envelope,
        optical_per_dry=optical,
        floor_um=floor,
        delivery=delivery,
        D50_um=d50,
        source_volume_um3_cm3=a,
        airborne_volume_um3_cm3=v.sum(),
        dust_above_cm3=dust.sum(),
        sulfate_above_cm3=sulf.sum(),
        run4_multiplier=b,
        subdust_cm3=n[:8].sum(),
        below097_cm3=output["original_below097_number"],
        overflow_cm3=n[-1],
        overflow_volume_um3_cm3=v[-1],
        sulfate_cap=sulfate_cap,
        multinomial_divergence=deviance,
        score=np.linalg.norm(weights * (np.log1p(model) - np.log1p(observed))),
        mae_cm3_bin=np.mean(abs(model - observed)),
        powder_back_conversion_max_abs_pct=closure,
        dry_volume_closure_um3_cm3=abs(captured - v.sum()),
    )
    return output, row, weights


def source_arrays():
    """Airborne midpoint population selected by the pre-cloud fit in runs 4 and 5."""
    center, edges, windows, _ = extract_observations()
    template = windows[0].mean(axis=0)
    response = np.zeros((len(center), len(center)))
    response[FIRST:, FIRST:] = gaussian_filter1d(
        np.eye(len(center) - FIRST), 0.6, axis=0, mode="reflect"
    )
    sulfate = np.r_[
        np.zeros(FIRST), nnls(response[FIRST:, FIRST:], template[FIRST:])[0]
    ]
    sulfate *= template.sum() / sulfate.sum()
    ref = reconstruct(
        windows[1, 0],
        sulfate,
        response,
        center,
        edges,
        "middle",
        1.0,
        DUST_FLOOR,
        "flat",
    )[0]
    material = np.ones((4, len(center) + 9))
    material[0] = 0
    material[0, : len(center)] = ref["sulfate"]
    material[1] = np.r_[
        ref["dust"], ref["subdust_number"], ref["overflow_number"]
    ]
    dry = np.r_[
        ref["dust_dry_um"], ref["subdust_dry_um"], ref["overflow_dry_um"]
    ]
    material[2] = np.where(np.isfinite(dry) & (dry > 0), dry, 1.0)
    material[3, : len(center)] = center / material[2, : len(center)]
    return material
