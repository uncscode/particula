"""Extract CERES Terra Ed4A footprints and colocated MODIS L1B radiance means.

Usage: python matchups.py output.json /path/to/CER_SSF_Terra*
Requires pyhdf and NumPy. The original HDF4 files come from the CERES archive.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pyhdf.SD import SD, SDC


def extract(path):
    """Return clear footprints, sequential rejection counts and source."""
    f = SD(str(path), SDC.READ)
    try:
        return read_fields(f, path)
    finally:
        f.end()


def band_indices(wavelengths, radiances):
    """Locate valid red and near-infrared bands in this footprint."""
    b = []
    for nominal in (0.645, 0.858):
        j = int(np.argmin(np.abs(wavelengths - nominal)))
        if abs(wavelengths[j] - nominal) < 0.02 and 0 < radiances[j] < 1000:
            b.append(j)
    return b


def stratify(rows):
    """Select four footprints per date and surface/view stratum."""
    # Equal-spaced time ranks within each independently specified stratum.
    groups = defaultdict(list)
    for row in rows:
        if row["surface"] == "bare":
            group = (
                "land_low_view"
                if row["view_zenith_deg"] < 35
                else "land_high_view"
            )
        else:
            group = (
                "ocean_near_glint"
                if row["glint_angle_deg"] < 40
                else "ocean_off_glint"
            )
        row["stratum"] = group
        groups[group].append(row)
    selected = []
    for group in sorted(groups):
        eligible = groups[group]
        indices = np.floor(
            (np.arange(min(4, len(eligible))) + 0.5)
            * len(eligible)
            / min(4, len(eligible))
        ).astype(int)
        if group == "ocean_near_glint":
            # Include the closest-to-specular eligible observation on each date.
            indices[0] = np.argmin([r["glint_angle_deg"] for r in eligible])
            indices = np.unique(indices)
        selected.extend(eligible[i] for i in indices)
    return selected, groups


def ocean_donor(i, donors, points, location, times):
    """Locate the nearest eligible ocean AOD retrieval in space and time."""
    if not donors.size:
        return None, np.inf
    offsets = np.linalg.norm(points - location, axis=1)
    nearest = np.argmin(offsets)
    donor = donors[nearest]
    distance = 2 * 6371 * np.arcsin(min(1, offsets[nearest] / 2))
    if distance > 250 or abs(times[i] - times[donor]) * 86400 > 120:
        return None, distance
    return donor, distance


def angstrom(surface, land_exponent, aod550, aod865):
    """Use the land retrieval or the measured two-band ocean depth ratio."""
    if surface == "ocean":
        return np.log(aod550 / aod865) / np.log(0.865 / 0.55)
    return land_exponent


def read_fields(f, path):
    """Read the SSF Edition 4 fields listed in its data product catalog."""

    def v(name):
        d = f.select(name)
        try:
            return np.asarray(d[:], dtype=float)
        finally:
            d.endaccess()

    jd = v("Time of observation")
    lat = 90.0 - v("Colatitude of CERES FOV at surface")
    lon_raw = v("Longitude of CERES FOV at surface")
    lon = (lon_raw + 180) % 360 - 180
    solar = v("CERES solar zenith at surface")
    view = v("Imager viewing zenith over CERES FOV")
    # CERES convention puts the Sun at azimuth 180. Particula uses zero.
    raw_azimuth = v("Imager relative azimuth over CERES FOV")
    azimuth = (raw_azimuth - 180) % 360
    heights = v("Altitude of surface above sea level")
    cloudfree = v("Clear/layer/overlap percent coverages")[:, 0]
    imager_coverage = v("Imager percent coverage")
    type_ids = v("Surface type index")
    coverage = v("Surface type percent coverage")
    wind = np.hypot(v("Surface wind - U-vector"), v("Surface wind - V-vector"))
    band_waves = v("Imager channel central wavelength")
    band_radiances = v("Mean imager radiances over full CERES FOV")
    band_spread = v("Stddev of imager radiances over full CERES FOV")
    toa = v("CERES SW TOA flux - upwards")
    land_aod = v("PSF-wtd MOD04 deep blue aerosol optical depth land (0.550)")
    water_aod = v("PSF-wtd MOD04 effective optical depth average ocean (0.550)")
    land_alpha = v("PSF-wtd MOD04 deep blue angstrom exponent land")
    water_aod865 = v(
        "PSF-wtd MOD04 effective optical depth average ocean (0.865)"
    )
    qa = v("Radiance and Mode flags").astype(int)
    instrument_ok = (qa & 15) == 0
    coverage_land = v("Percentage of CERES FOV with MODIS deep blue aerosol")
    coverage_water = v("Percentage of CERES FOV with MODIS ocean aerosol")
    land_reflectance = np.column_stack(
        [
            v(f"PSF-wtd MOD04 deep blue surface reflectance land ({band})")
            for band in ("0.412", "0.470", "0.659")
        ]
    )
    land_ssa = v(
        "PSF-wtd MOD04 deep blue single scattering albedo land (0.659)"
    )
    azimuth_valid = (raw_azimuth >= 0) & (raw_azimuth <= 360)
    pressure = v("Surface pressure")
    water = v("Precipitable water")
    candidates = np.ones(jd.shape, dtype=bool)
    counts = {"total": len(jd)}
    outcomes = np.full(jd.size, "eligible_unsampled", dtype=object)
    for reason, keep in [
        (
            "outside_subtropical_sample",
            (abs(lat) <= 40) & (lon_raw >= 0) & (lon_raw <= 360),
        ),
        ("outside_daylight_angle_range", (solar >= 0) & (solar <= 65)),
        ("ceres_instrument_or_earth_view", instrument_ok & azimuth_valid),
        ("missing_imager", (imager_coverage >= 99) & (imager_coverage <= 100)),
        ("cloud", (cloudfree >= 99) & (cloudfree <= 100)),
        (
            "mixture",
            np.isin(type_ids[:, 0], [16, 17]) & (coverage[:, 0] == 100),
        ),
        (
            "missing_flux_or_view",
            (toa > 0) & (toa < 1400) & (view >= 0) & (view <= 65),
        ),
    ]:
        outcomes[candidates & ~keep] = reason
        counts[reason] = int(np.count_nonzero(candidates & ~keep))
        candidates &= keep
    # A nearby same-orbit retrieval may constrain ocean aerosol in sunglint.
    donors = np.flatnonzero(
        candidates
        & (type_ids[:, 0] == 17)
        & (water_aod > 0)
        & (water_aod <= 1.5)
        & (coverage_water >= 50)
        & (coverage_water <= 100)
        & (water_aod865 > 0)
        & (water_aod865 <= 2)
    )

    def xyz(lat_deg, lon_deg):
        la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
        return np.column_stack(
            [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        )

    for reason, keep in [
        (
            "outside_ocean_wind_domain",
            (type_ids[:, 0] != 17) | ((wind >= 1) & (wind <= 14)),
        ),
        (
            "outside_profile_domain",
            (water >= 0.3)
            & (water <= 5)
            & (pressure >= 700)
            & (pressure <= 1050)
            & (heights >= -400)
            & (heights <= 2500),
        ),
    ]:
        outcomes[candidates & ~keep] = reason
        counts[reason] = int(np.count_nonzero(candidates & ~keep))
        candidates &= keep
    donor_xyz = xyz(lat[donors], lon[donors])
    rows = []
    for i in np.flatnonzero(candidates):
        surface = "ocean" if type_ids[i, 0] == 17 else "bare"
        aod = water_aod[i] if surface == "ocean" else land_aod[i]
        donor = i
        distance = 0.0
        if surface == "ocean":
            donor, distance = ocean_donor(
                i, donors, donor_xyz, xyz(lat[i], lon[i]), jd
            )
            if donor is None:
                outcomes[i] = "missing_aod"
                counts["missing_aod"] = counts.get("missing_aod", 0) + 1
                continue
            aod = water_aod[donor]
        elif not (
            0 < aod <= 1.5
            and 50 <= coverage_land[i] <= 100
            and np.all((land_reflectance[i] > 0) & (land_reflectance[i] < 1))
        ):
            outcomes[i] = "missing_aod_or_land_reflectance"
            counts["missing_aod_or_land_reflectance"] = (
                counts.get("missing_aod_or_land_reflectance", 0) + 1
            )
            continue
        b = band_indices(band_waves[i], band_radiances[i])
        if len(b) != 2:
            outcomes[i] = "missing_bands"
            counts["missing_bands"] = counts.get("missing_bands", 0) + 1
            continue
        alpha = angstrom(surface, land_alpha[i], aod, water_aod865[donor])
        sz, vz, az = np.deg2rad([solar[i], view[i], azimuth[i]])
        glint = np.rad2deg(
            np.arccos(
                np.clip(
                    np.cos(sz) * np.cos(vz)
                    - np.sin(sz) * np.sin(vz) * np.cos(az),
                    -1,
                    1,
                )
            )
        )
        rows.append(
            {
                "source": path.name,
                "footprint": int(i),
                "surface": surface,
                "time": datetime.fromtimestamp(
                    (jd[i] - 2440587.5) * 86400, UTC
                ).isoformat(),
                "latitude": lat[i],
                "longitude": lon[i],
                "elevation_m": heights[i],
                "solar_zenith_deg": solar[i],
                "view_zenith_deg": view[i],
                "relative_azimuth_deg": azimuth[i],
                "clear_percent": cloudfree[i],
                "wind_m_s": wind[i],
                "aod550": aod,
                "angstrom": float(alpha) if -0.5 <= alpha <= 3.5 else None,
                "aod_donor_footprint": int(donor),
                "aod_donor_distance_km": distance,
                "glint_angle_deg": glint,
                "land_surface_reflectance": land_reflectance[i].tolist()
                if surface == "bare"
                else None,
                "land_ssa659": land_ssa[i]
                if surface == "bare" and 0 < land_ssa[i] <= 1
                else None,
                "pressure_hpa": pressure[i],
                "precipitable_water_cm": water[i],
                "ceres_sw_w_m2": toa[i],
                "modis_radiance_w_m2_sr_um": band_radiances[i, b].tolist(),
                "modis_spatial_sd_w_m2_sr_um": band_spread[i, b].tolist(),
            }
        )
    counts["retained"] = len(rows)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    selected, groups = stratify(rows)
    outcomes[[r["footprint"] for r in selected]] = "selected"
    return (
        outcomes,
        selected,
        {
            "granule": path.name,
            "sha256": digest,
            "counts": counts,
            "eligible_strata": {k: len(v) for k, v in groups.items()},
        },
    )


def main():
    """Extract the named SSF granules to a portable footprint packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("granules", type=Path, nargs="+")
    parser.add_argument("--screening", type=Path)
    args = parser.parse_args()
    rows, sources = [], []
    screenings = {}
    for path in args.granules:
        outcomes, selected, source = extract(path)
        labels, codes = np.unique(outcomes.astype(str), return_inverse=True)
        suffix = path.name.split(".")[-1]
        screenings["reason_" + suffix] = labels
        screenings["index_" + suffix] = codes.astype(np.uint8)
        rows.extend(selected)
        sources.append(source)
    if args.screening:
        np.savez_compressed(args.screening, **screenings)
    args.output.write_text(
        json.dumps(
            {"sources": sources, "footprints": rows}, indent=2, allow_nan=False
        )
        + "\n"
    )
    print(f"{len(rows)} matched footprints from {len(sources)} granules")


if __name__ == "__main__":
    main()
