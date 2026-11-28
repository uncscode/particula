"""Compare the SSF PSF-weighted radiances with nearby Terra MODIS L1B pixels."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = "https://terrafusiondatasampler.s3.us-west-2.amazonaws.com/P169/"
ORBIT = [
    ("TERRA_BF_L1B_O10135_20011113072222_F000_V001.h5", "2001317"),
    ("TERRA_BF_L1B_O10368_20011129072222_F000_V001.h5", "2001333"),
]
LAT = "_1KM/Geolocation/Latitude"
LON = "_1KM/Geolocation/Longitude"
RAD = "_1KM/Data_Fields/EV_250_Aggr1km_RefSB"
UNC = RAD + "_Uncert_Indexes"
FIELDS = (
    LAT,
    LON,
    RAD,
    UNC,
    "SolarZenith",
    "SolarAzimuth",
    "SensorZenith",
    "SensorAzimuth",
)


def cache_granule(cache, orbit, minute):
    """Cache selected public TerraFusion arrays from one five-minute granule."""
    fname, day = ORBIT[orbit]
    path = cache / f"rtv-prune-{orbit}-{minute}.h5"
    if path.exists():
        return path
    with (
        fsspec.open(
            BASE + fname, block_size=100000, cache_type="readahead"
        ) as remote,
        h5py.File(remote) as f,
    ):
        group = f[f"MODIS/granule_{day}_{minute}"]
        pending = path.with_suffix(".partial.h5")
        with h5py.File(pending, "w") as out:
            for key in FIELDS:
                out.create_dataset(
                    key,
                    data=group[key][:],
                    compression="gzip",
                    compression_opts=1,
                )
            for k, v in group.attrs.items():
                out.attrs[k] = v
        pending.replace(path)
    return path


def aperture_fields(path, rows):
    """Average valid Level-1B samples in a circular 10 km aperture."""
    out = []
    with h5py.File(path) as f:
        lat, lon = np.deg2rad(f[LAT][:]), np.deg2rad(f[LON][:])
        l1b = f.attrs["GranuleName_1KM"].decode()
        time = datetime.strptime(
            f.attrs["GranuleTime"].decode(), "%Y%j.%H%M"
        ).replace(tzinfo=UTC)
        radiance = f[RAD][:]
        uncertainty = f[UNC][:]
        geom = {k: f[k][:] for k in FIELDS[4:]}
        for r in rows:
            # Allow 90 s on either side of each five-minute scan.
            dt = (datetime.fromisoformat(r["time"]) - time).total_seconds()
            if not -90 <= dt <= 390:
                continue
            la, lo = np.deg2rad([r["latitude"], r["longitude"]])
            half = (
                np.sin((lat - la) / 2) ** 2
                + np.cos(lat) * np.cos(la) * np.sin((lon - lo) / 2) ** 2
            )
            pix = half <= np.sin(10 / (2 * 6371)) ** 2
            valid = pix & np.all(
                (radiance > 0)
                & (radiance < 1000)
                & (uncertainty > 0)
                & (uncertainty <= 5),
                axis=0,
            )
            if not np.any(valid):
                continue
            dist = 2 * 6371 * np.arcsin(np.sqrt(half[valid].min()))
            angles = (
                geom["SensorAzimuth"][valid] - geom["SolarAzimuth"][valid]
            ) % 360
            relative = float(
                np.mod(
                    np.rad2deg(
                        np.angle(np.exp(1j * np.deg2rad(angles)).mean())
                    ),
                    360,
                )
            )
            out.append(
                dict(
                    footprint=r["footprint"],
                    source=r["source"],
                    l1b=l1b,
                    geolocation=f.attrs["GranuleName_MOD03"].decode(),
                    granule_start_offset_s=dt,
                    min_distance_km=dist,
                    pixel_count=int(valid.sum()),
                    candidate_pixels=int(pix.sum()),
                    b1=float(radiance[0][valid].mean()),
                    b2=float(radiance[1][valid].mean()),
                    b1_uncertainty_percent=float(uncertainty[0][valid].mean()),
                    b2_uncertainty_percent=float(uncertainty[1][valid].mean()),
                    solar_zenith_deg=float(geom["SolarZenith"][valid].mean()),
                    view_zenith_deg=float(geom["SensorZenith"][valid].mean()),
                    relative_azimuth_deg=relative,
                )
            )
    return out


def main():
    """Match both public orbit strips to the archived CERES centroids."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    records = json.loads((HERE / "footprints.json").read_text())["footprints"]
    found = {}
    for orbit, (orbit_name, _) in enumerate(ORBIT):
        rows = [
            r
            for r in records
            if r["source"].endswith(orbit_name.split("_")[4][:8] + "08")
        ]
        for minute in ("0800", "0805", "0810", "0815", "0820"):
            path = cache_granule(args.cache, orbit, minute)
            for row in aperture_fields(path, rows):
                k = (row["source"], row["footprint"])
                if (
                    k not in found
                    or row["pixel_count"] > found[k]["pixel_count"]
                ):
                    found[k] = row
    with args.output.open("w") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=next(iter(found.values())).keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(found[k] for k in sorted(found))


if __name__ == "__main__":
    main()
