"""Replay matched TOA flux and radiance comparisons with the public solver."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import (
    sensor_band_radiance,
    solve_shortwave,
)
from particula.optics.solar import earth_reference
from particula.optics.surface import (
    dusty_ground,
    ocean_surface,
    ross_li_surface,
)
from scene_profiles import interfaces

HERE = Path(__file__).resolve().parent


def surface_for(row, wave, ideal):
    """Prescribe surface reflection independently of TOA measurements."""
    if row["surface"] == "ocean":
        return ocean_surface(
            wave, wind_speed=None if ideal else row["wind_m_s"]
        )
    if ideal:
        return dusty_ground(wave)
    # Retain the mineral reference's shape outside the three Deep Blue bands.
    anchors = np.array([412.0, 470.0, 659.0]) * 1e-9
    measured = np.array(row["land_surface_reflectance"])
    shape = dusty_ground(wave).coefficients[0]
    ratios = measured / dusty_ground(anchors).coefficients[0]
    reflectance = shape * np.interp(wave, anchors, ratios)
    # NASA HLS Table 1 lists f_iso, f_geo, f_vol, in that order.
    hls = np.array(
        [
            [470, 0.0774, 0.0079, 0.0372],
            [555, 0.1306, 0.0178, 0.0580],
            [645, 0.1690, 0.0227, 0.0574],
            [858, 0.3093, 0.0330, 0.1535],
            [1640, 0.3430, 0.0453, 0.1154],
            [2130, 0.2658, 0.0387, 0.0639],
        ]
    )
    geo = np.interp(wave, hls[:, 0] * 1e-9, hls[:, 2] / hls[:, 1])
    vol = np.interp(wave, hls[:, 0] * 1e-9, hls[:, 3] / hls[:, 1])
    shape_model = ross_li_surface(wave, 1, vol, geo)
    brf = np.pi * shape_model.brdf(
        np.cos(np.deg2rad(row["solar_zenith_deg"])),
        np.cos(np.deg2rad(row["view_zenith_deg"])),
        np.deg2rad(row["relative_azimuth_deg"]),
    )
    amp = reflectance / brf
    return ross_li_surface(
        wave,
        amp,
        amp * vol,
        amp * geo,
        source="Deep Blue directional amplitude; HLS global kernel ratios",
    )


def solve(row, pressure, gas, wave, controls, streams):
    """Build the constrained column and calculate its reflected light."""
    z = interfaces(row)
    ref = earth_reference(
        row["latitude"],
        row["longitude"],
        datetime.fromisoformat(row["time"]),
        z,
        wave,
        pressure_pa=pressure,
        gas_optical_depth=gas,
        profile_name="SSF pressure and water, AFGL shape",
    )
    sun = replace(ref.sun, mu0=np.cos(np.deg2rad(row["solar_zenith_deg"])))
    col = ref.column
    w = col.wavelength_m
    alpha = row["angstrom"] if row["angstrom"] is not None else 1.0
    is_water = row["surface"] == "ocean"
    ssa = 0.99 if is_water else (row["land_ssa659"] or 0.95)
    depth_scale, ideal, g = controls
    depth = np.zeros_like(col.optical_depth)
    depth[-1] = row["aod550"] * (w / 550e-9) ** (-alpha) * depth_scale
    moments = np.broadcast_to(
        g ** np.arange(2 * streams + 1), (*depth.shape, 2 * streams + 1)
    )
    aerosol = SpectralColumn(
        w, z, depth, np.broadcast_to(ssa, depth.shape), moments
    )
    return solve_shortwave(
        col.plus(aerosol),
        sun,
        surface_for(row, w, ideal),
        streams=streams,
        view_zenith_deg=row["view_zenith_deg"] if wave is not None else None,
        relative_azimuth_deg=row["relative_azimuth_deg"],
    ), ref


def compute(job):
    """Return the three observed quantities and modeled sensitivity cases."""
    i, row, streams, primary_only = job
    with np.load(HERE / "scene-profiles.npz") as profiles:
        if (
            row["footprint"] != profiles["footprint"][i]
            or row["source"] != profiles["source"][i]
        ):
            raise ValueError("Profile and observation ordering differ")
        p, gas, band_gas = [
            profiles[k][i]
            for k in ["pressure_pa", "broad_gas_tau", "band_gas_tau"]
        ]
    curve = np.loadtxt(HERE / "terra-response.csv", delimiter=",", skiprows=1)
    w = curve[:, 1] * 1e-9
    out = {
        "source": row["source"],
        "footprint": row["footprint"],
        "surface": row["surface"],
        "stratum": row["stratum"],
        "sza_deg": row["solar_zenith_deg"],
        "vza_deg": row["view_zenith_deg"],
        "glint_deg": row["glint_angle_deg"],
        "aod550": row["aod550"],
        "observed_sw": row["ceres_sw_w_m2"],
        "observed_b1": row["modis_radiance_w_m2_sr_um"][0],
        "observed_b2": row["modis_radiance_w_m2_sr_um"][1],
    }
    choices = [
        ("scene", (1.0, False, 0.70)),
        ("ideal_surface", (1.0, True, 0.70)),
        ("lower_aod", (0.75, False, 0.70)),
        ("higher_aod", (1.25, False, 0.70)),
        ("lower_g", (1.0, False, 0.60)),
        ("higher_g", (1.0, False, 0.80)),
    ]
    for label, controls in choices[:1] if primary_only else choices:
        broad, ref = solve(row, p, gas, None, controls, streams)
        narrow, _ = solve(row, p, band_gas, w, controls, streams)
        out[label + "_sw"] = float(broad.integrate(broad.up[0]))
        for b in (1, 2):
            response = np.where(curve[:, 0] == b, curve[:, 2], 0)
            out[f"{label}_b{b}"] = float(
                sensor_band_radiance(narrow, response) * 1e-6
            )
        if label == "scene":
            out["incident_sw"] = float(broad.integrate(broad.direct_down[0]))
            out["ephemeris_sza_deg"] = ref.position.zenith_deg
            out["distance_au"] = ref.position.distance_au
    return out


def main():
    """Replay every footprint in the portable input packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    rows = json.loads((HERE / "footprints.json").read_text())["footprints"]
    with (
        ProcessPoolExecutor(args.workers) as pool,
        args.output.open("w") as out,
    ):
        for i, row in enumerate(
            pool.map(
                compute,
                [
                    (i, r, args.streams, args.primary_only)
                    for i, r in enumerate(rows)
                ],
            )
        ):
            if i == 0:
                writer = csv.DictWriter(
                    out, fieldnames=row.keys(), lineterminator="\n"
                )
                writer.writeheader()
            writer.writerow(row)
            out.flush()


if __name__ == "__main__":
    main()
