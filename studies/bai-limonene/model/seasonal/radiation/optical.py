"""Aggregate class-resolved parcel optics and solve non-grey cloud transfer."""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import roots_legendre

HERE = Path(__file__).resolve().parent
TMP = Path(os.environ["SEASONAL_TMP"]) / "radiation"
TABLE_TAG = os.environ.get("BAI_SPHERE_SET", "")
TABLE_SUFFIX = "_" + TABLE_TAG if TABLE_TAG else ""
PTR = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


@lru_cache(maxsize=1)
def library():
    """Load the compiled C++ adding–doubling solver with typed array inputs."""
    native = ctypes.CDLL(str(TMP / "transfer.so")).solve
    native.argtypes = [ctypes.c_int] * 5 + [PTR] * 6 + [ctypes.c_double, PTR]
    native.restype = ctypes.c_int
    return native


@lru_cache(maxsize=1)
def table():
    """Load the chosen, independently regenerable water-sphere optical grid."""
    with np.load(TMP / f"sphere_table{TABLE_SUFFIX}.npz") as data:
        return data["radius"], data["wave"], data["eff"]


@lru_cache(maxsize=12)
def kernel(bands, order, step):
    """Select radius nodes, waveband weights and unnormalized optical moments."""
    radius, wave, eff = table()
    bands_table = pd.read_csv(HERE / f"spectral_bands{TABLE_SUFFIX}.csv")
    if bands == 0:
        js = np.arange(3)
        beam = sky = np.ones(3)
    else:
        this = bands_table[bands_table.n == bands]
        js = np.argmin(
            np.abs(wave[:, None] - this.nm.to_numpy()[None, :]), axis=0
        )
        if len(js) != bands:
            raise ValueError(bands)
        beam, sky = this.beam.to_numpy(), this.sky.to_numpy()
    ri = np.unique(np.r_[np.arange(0, len(radius), step), len(radius) - 1])
    k = np.ascontiguousarray(
        eff[ri[:, None], js[None, :], : order + 2]
    ).reshape(len(ri), -1)
    return radius[ri], wave[js], k, beam, sky


def quadrature(centers, layers):
    """Height integration per top-down layer; shared boundary weights sum properly."""
    if not (
        np.all(np.diff(centers) > 0)
        and abs(centers[0]) < 1e-8
        and centers[-1] > 0
    ):
        raise ValueError(
            "Expected increasing physical grid from cloud base to top"
        )
    bounds = np.round(np.linspace(0, len(centers) - 1, layers + 1)).astype(int)
    if len(set(bounds)) != len(bounds):
        raise ValueError("Too many optical layers")
    pieces = []
    for bot, top in reversed(list(zip(bounds[:-1], bounds[1:]))):
        z = centers[bot : top + 1]
        w = np.gradient(z)
        w[[0, -1]] *= 0.5
        pieces.append((slice(bot, top + 1), w))
    return pieces


def optical_column(radius, number, height, bands=48, order=8, layers=4, step=1):
    """Integrate individual geometric areas times interpolated Mie efficiencies."""
    grid, wave, k, _, _ = kernel(bands, order, step)
    rr = np.asarray(radius, dtype=float)
    nn = np.asarray(number, dtype=float)
    if (
        np.any(rr < grid[0] * (1 - 1e-7))
        or np.any(rr > grid[-1])
        or np.min(nn) < 0
        or not np.isfinite(rr + nn).all()
    ):
        raise ValueError(
            ("Optical radius/weight range", rr.min(), rr.max(), grid[[0, -1]])
        )
    locations = np.clip(np.searchsorted(grid, rr) - 1, 0, len(grid) - 2)
    right = np.clip(
        (rr - grid[locations]) / (grid[locations + 1] - grid[locations]), 0, 1
    )
    area = nn * np.pi * rr**2
    ncase = rr.shape[0]
    rows = []
    for s, w in quadrature(height, layers):
        offsets = (
            locations[:, s] + (len(grid) * np.arange(ncase))[:, None, None]
        )
        right_weight = right[:, s]
        mass = area[:, s] * w[None, :, None]
        left = np.bincount(
            offsets.ravel(),
            weights=(mass * (1 - right_weight)).ravel(),
            minlength=ncase * len(grid),
        )
        adj = np.bincount(
            (offsets + 1).ravel(),
            weights=(mass * right_weight).ravel(),
            minlength=ncase * len(grid),
        )
        rows.append((left + adj).reshape(ncase, len(grid)))
    hist = (
        np.stack(rows, axis=1)
        .reshape(ncase * layers, len(grid))
        .astype(np.float32)
    )
    # Dense SGEMM handles whole independent profile batches at once. Raw moment
    # coefficients remain positive/negative as specified by Mie theory.
    result = hist @ k
    return np.ascontiguousarray(
        result.reshape(ncase, layers, len(wave), order + 2).transpose(
            0, 2, 1, 3
        )
    )


def bulk_moments(radius, number, height):
    # Geometric relation tau_geo = 3 LWP/(2 rho_water r_eff); the actual
    # wavelength-specific Mie extinction includes size-dependent Q_ext.
    """Return wet-equivalent path, column effective radius and geometric depth."""
    rr = np.asarray(radius, dtype=float)
    nn = np.asarray(number, dtype=float)
    m = [
        np.trapezoid(np.sum(nn * rr**k, axis=-1), height, axis=1)
        for k in (2, 3)
    ]
    wet_volume = m[1] * 4 * np.pi / 3
    return np.stack(
        (wet_volume * 1000 * 1000, m[1] / m[0] * 1e6, 2 * np.pi * m[0]), axis=-1
    )


def response(optics, mu, streams=4, order=8, thin=2e-5):
    """Return direct, diffuse top/bottom (R,T) for all columns and wavelengths."""
    moments = np.ascontiguousarray(optics, dtype=float)
    nc, nw, nl, nk = moments.shape
    if len(mu) != nc or np.max(mu) > 1 or np.min(mu) <= 0 or nk < order + 2:
        raise ValueError("Invalid beam or angular job")
    nodes, weights = roots_legendre(streams)
    pos = (nodes + 1) / 2
    weights = weights / 2
    leg = np.ascontiguousarray(np.polynomial.legendre.legvander(pos, order - 1))
    cos = np.repeat(mu, nw).astype(float)
    sun = np.ascontiguousarray(np.polynomial.legendre.legvander(cos, order - 1))
    out = np.empty((nc * nw, 6))
    ret = library()(
        streams,
        nc * nw,
        nl,
        nk,
        order,
        moments,
        cos,
        pos,
        weights,
        leg,
        sun,
        thin,
        out,
    )
    if ret or not np.isfinite(out).all():
        raise ValueError(("transfer failed", ret))
    return out.reshape(nc, nw, 6)


def spectral_boundaries(wave, beam, sky, colors):
    """Match hourly three-band energies while preserving the two measured fluxes."""
    group = np.where(wave < 400, 0, np.where(wave < 700, 1, 2))
    shape = np.stack((beam, sky), axis=1)
    native = np.stack(
        [shape[:, :, group == k].sum(axis=2) for k in range(3)], axis=2
    )
    given = native.sum(axis=2)
    total = given.sum(axis=1)
    ensemble = native.sum(axis=1)
    uv, par = (
        colors.solar_uv_fraction.to_numpy(),
        colors.solar_par_fraction.to_numpy(),
    )
    wanted = total[:, None] * np.column_stack((uv, par, 1 - uv - par))
    wanted = np.where(
        colors.ceres_color_valid.to_numpy()[:, None], wanted, ensemble
    )
    energy = native.copy()
    for _ in range(24):
        energy *= np.divide(
            wanted,
            energy.sum(axis=1),
            where=energy.sum(axis=1) > 0,
            out=np.ones_like(wanted),
        )[:, None, :]
        energy *= np.divide(
            given,
            energy.sum(axis=2),
            where=energy.sum(axis=2) > 0,
            out=np.ones_like(given),
        )[:, :, None]
    if (
        max(
            np.max(abs(energy.sum(axis=1) - wanted)),
            np.max(abs(energy.sum(axis=2) - given)),
        )
        > 1e-5
    ):
        raise ValueError(
            "Direct/diffuse or wavelength-band energy budget failed"
        )
    factor = np.divide(
        energy, native, where=native > 0, out=np.ones_like(energy)
    )
    adjusted = shape * factor[:, :, group]
    return adjusted[:, 0], adjusted[:, 1]


def radiance(optics, forcing, bands, streams, order, thin=2e-5):
    """Broadband up, down and absorbed fluxes scaled to measured local downwelling."""
    _, wave, _, beam_w, sky_w = kernel(bands, order, 1)
    n = len(forcing)
    outputs = np.zeros((n, 17))
    day = forcing.ghi_wm2.to_numpy() > 0
    if not day.any():
        return outputs
    beam = forcing.direct_horizontal_wm2.to_numpy()[day]
    sky = forcing.dhi_wm2.to_numpy()[day]
    sun_mu = forcing.mu0.to_numpy()[day]
    mu = np.where(sun_mu > 0, sun_mu, 0.5)
    q = response(optics[day], mu, streams, order, thin)
    weighted_beam = beam[:, None] * beam_w
    weighted_sky = sky[:, None] * sky_w
    up = q[:, :, 0] * weighted_beam + q[:, :, 2] * weighted_sky
    down = q[:, :, 1] * weighted_beam + q[:, :, 3] * weighted_sky

    def reflected(albedo):
        return (
            up + down * albedo * q[:, :, 5] / (1 - albedo * q[:, :, 4])
        ).sum(axis=1)

    outputs[day, 0] = up.sum(axis=1)
    outputs[day, 1] = down.sum(axis=1)
    outputs[day, 2] = reflected(0.2)
    outputs[day, 3] = reflected(0.8)
    outputs[day, 4] = np.sum(q[:, :, 0] * weighted_beam, axis=1)
    outputs[day, 5] = np.sum(q[:, :, 2] * weighted_sky, axis=1)
    outputs[day, 6] = np.sum(
        np.maximum(q[:, :, 0] + q[:, :, 1] - 1, 0) * weighted_beam
        + np.maximum(q[:, :, 2] + q[:, :, 3] - 1, 0) * weighted_sky,
        axis=1,
    )
    outputs[day, 7] = np.sum(
        np.abs(q[:, :, 3] - q[:, :, 5]) * (weighted_beam + weighted_sky), axis=1
    )
    outputs[day, 8] = (q[:, :, 2] * sky_w).sum(axis=1)
    if "site_albedo" in forcing:
        albedo = forcing.site_albedo.to_numpy()[day, None]
        outputs[day, 9] = reflected(albedo)
        # Deliberately exchange the AM1.5 direct and hemispheric normalized
        # spectral shapes as a broadband spectral-partition sensitivity.
        trial_up = q[:, :, 0] * (beam[:, None] * sky_w) + q[:, :, 2] * (
            sky[:, None] * beam_w
        )
        trial_down = q[:, :, 1] * (beam[:, None] * sky_w) + q[:, :, 3] * (
            sky[:, None] * beam_w
        )
        outputs[day, 10] = trial_up.sum(axis=1)
        outputs[day, 11] = (
            trial_up
            + trial_down * albedo * q[:, :, 5] / (1 - albedo * q[:, :, 4])
        ).sum(axis=1)
        # Transfer one quarter of observed diffuse horizontal energy into a
        # circumsolar direction, keeping its diffuse spectral distribution.
        circumsolar = np.where(sun_mu > 0, 0.25, 0.0)[:, None] * weighted_sky
        alt_up = up + circumsolar * (q[:, :, 0] - q[:, :, 2])
        alt_down = down + circumsolar * (q[:, :, 1] - q[:, :, 3])
        outputs[day, 12] = alt_up.sum(axis=1)
        outputs[day, 13] = (
            alt_up + alt_down * albedo * q[:, :, 5] / (1 - albedo * q[:, :, 4])
        ).sum(axis=1)
        # A two-band surface sensitivity conserves the observed unclouded
        # broadband input albedo at every hour. The contrast is a scenario.
        short = (wave < 700).astype(float)
        incident = weighted_beam + weighted_sky
        frac = np.sum(incident * short, axis=1) / incident.sum(axis=1)
        contrast = np.where(albedo >= 0.4, 0.25, -0.12)
        spectral_surface = albedo + contrast * (short - frac[:, None])
        if np.min(spectral_surface) < 0 or np.max(spectral_surface) > 1:
            raise ValueError("Two-band surface exceeded physical limits")
        outputs[day, 14] = reflected(spectral_surface)
        if "ceres_color_valid" in forcing:
            cbeam, csky = spectral_boundaries(
                wave, weighted_beam, weighted_sky, forcing.loc[day]
            )
            cup = q[:, :, 0] * cbeam + q[:, :, 2] * csky
            cdown = q[:, :, 1] * cbeam + q[:, :, 3] * csky
            outputs[day, 15] = cup.sum(axis=1)
            outputs[day, 16] = (
                cup + cdown * albedo * q[:, :, 5] / (1 - albedo * q[:, :, 4])
            ).sum(axis=1)
    if np.min(outputs[:, :6]) < -1e-5 or np.any(
        outputs[:, 0] + outputs[:, 1] > forcing.ghi_wm2 + 1.0e-3
    ):
        raise ValueError(
            (
                "Flux budget",
                outputs[:, :6].min(),
                (outputs[:, 0] + outputs[:, 1] - forcing.ghi_wm2).max(),
            )
        )
    return outputs
