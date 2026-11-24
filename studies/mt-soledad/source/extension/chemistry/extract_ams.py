"""Extract May–June ambient NR chemistry from Berta et al.'s calibrated products."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

SPECIES = ("organics", "nitrate", "sulfate", "ammonium", "chloride")
START = pd.Timestamp("2023-05-01", tz="UTC").timestamp()
STOP = pd.Timestamp("2023-07-01", tz="UTC").timestamp()
HASHES = {
    "ams": "8ed0442af97ac227ae3cdb6f49b8b491ce47aef14f359ce504cae2217b01ee53",
    "kappa": "cd3167e127762b8873a792f1190a00cb099240a2270ba809d2cc75b18fff531d",
    "ccn": "9f94d7f3e1e66bbb7d5149fd3b9e3f2eea776dd9f898f3af6d185ceca9c53800",
}
QC_GAPS = [
    ("2023-05-23 05:30", "2023-05-25 08:30"),
    ("2023-06-02 01:30", "2023-06-08 03:00"),
]


def read(path, field, directory, dtype="<f8"):
    with tempfile.NamedTemporaryFile(dir=directory) as f:
        subprocess.run(
            ["h5dump", "-d", field, "-b", "LE", "-o", f.name, str(path)],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        a = np.fromfile(f.name, dtype).astype(float)
    a[np.abs(a) > 1e30] = np.nan
    return a


def ambient_flags(archive, directory):
    times, flags = [], []
    with ZipFile(archive) as z:
        day = directory / "day.cdf"
        for member in sorted(m for m in z.namelist() if m.endswith(".cdf")):
            day.write_bytes(z.read(member))
            times.extend(read(day, "time", directory, "<i4"))
            flags.extend(read(day, "cvi_flag", directory))
    times, flags = np.array(times), np.array(flags)
    assert (np.diff(times) > 0).all()
    return times, flags


def eligible(t, times, flags):
    idx = np.clip(np.searchsorted(times, t), 1, len(times) - 1)
    ok = (
        (t >= START)
        & (t < STOP)
        & (t - times[idx - 1] <= 300)
        & (times[idx] - t <= 300)
        & (flags[idx - 1] == 0)
        & (flags[idx] == 0)
    )
    for left, right in QC_GAPS:
        a, b = [pd.Timestamp(x, tz="UTC").timestamp() for x in (left, right)]
        ok &= (t < a) | (t > b)
    return ok


def frame(t):
    t = np.asarray(t, dtype="int64")
    return pd.DataFrame(
        {
            "time_utc": pd.to_datetime(t, utc=True, unit="s").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "time_unix": t,
        }
    )


def save(df, path):
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def main(ams, kappa, ccn, output):
    for name, path in dict(ams=ams, kappa=kappa, ccn=ccn).items():
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != HASHES[name]:
                raise ValueError("Source hash differs: " + name)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        times, flags = ambient_flags(ccn, temp)
        ktime = read(kappa, "time", temp)
        sel = eligible(ktime, times, flags)
        kt = frame(ktime[sel])
        for col in ("kappa_sol_01", "kappa_sol_OC", "kappa_sol_f44"):
            kt[col] = read(kappa, col, temp)[sel]
        kt = kt.loc[kt[list(kt.columns[2:])].notna().any(axis=1)]
        save(kt, output / "ams-kappa.csv.gz")

        t = read(ams, "time", temp)
        sel = eligible(t, times, flags)
        ts = t[sel]
        bulk = frame(ts)
        n = len(t)
        for sp in SPECIES:
            bulk[sp + "_ug_m3"] = read(ams, sp, temp)[sel]
        b = bulk[[s + "_ug_m3" for s in SPECIES]].to_numpy()
        valid = (np.isfinite(b) & (b >= 0)).all(axis=1) & (b.sum(axis=1) > 0)
        for j, sp in enumerate(SPECIES):
            bulk[sp + "_fraction"] = np.where(
                valid, b[:, j] / np.where(valid, b.sum(axis=1), 1), np.nan
            )
        save(bulk, output / "ams.csv.gz")

        d = read(ams, "diameter", temp)
        ld = np.log10(d)
        assert len(d) == 60 and np.allclose(np.diff(ld), 1 / 30, atol=1e-5)
        midpoint = 0.5 * (ld[:-1] + ld[1:])
        edges = np.r_[2 * ld[0] - midpoint[0], midpoint, 2 * ld[-1] - midpoint[-1]]
        weight = np.minimum(edges[1:], np.log10(180.0)) - np.maximum(
            edges[:-1], np.log10(50.0)
        )
        ib = weight > 1e-10
        weights = weight[ib]
        assert np.isclose(weights.sum(), np.log10(180 / 50))
        vals = np.stack(
            [
                read(ams, sp + "_sizeresolved", temp).reshape(len(d), n)[ib][:, sel]
                for sp in SPECIES
            ]
        )
        # AMS blanks, including partially missing spectra, are never filled in the central estimate.
        records, coverage = [], []
        floors = (ts.astype("int64") // 21600) * 21600
        for start in sorted(set(floors)):
            mask = floors == start
            a = vals[:, :, mask]
            scan_count = int(mask.sum())
            present = np.isfinite(a)
            counts = present.sum(axis=2)
            joint = present.all(axis=0)
            common_count = joint.sum(axis=1)
            shared = np.where(joint[None, :, :], a, 0).sum(axis=2)
            total = np.nansum(a, axis=2)
            base_ok = scan_count >= 24 and np.all(counts >= max(3, 0.25 * scan_count))
            simultaneous_ok = bool(base_ok and np.all(common_count >= 3))
            row = dict(
                time_unix=int(start),
                time_end_unix=int(start + 21600),
                n_ambient=scan_count,
                min_coverage=float(counts.min() / scan_count),
                min_joint=int(common_count.min()),
                coverage_pass=bool(base_ok),
                joint_pass=simultaneous_ok,
            )
            for tag, num, den, ok in [
                ("finite", total, counts, base_ok),
                ("joint", shared, common_count[None, :], simultaneous_ok),
                ("zero", total, np.ones_like(counts) * scan_count, base_ok),
            ]:
                means = num / np.maximum(den, 1)
                masses = means @ weights if ok else np.full(len(SPECIES), np.nan)
                frac = (
                    masses / np.sum(masses)
                    if np.isfinite(masses).all()
                    and (masses >= 0).all()
                    and masses.sum() > 0
                    else np.full(len(SPECIES), np.nan)
                )
                for s, m, f in zip(SPECIES, masses, frac, strict=True):
                    row[s + "_" + tag + "_ug_m3"] = m
                    row[s + "_" + tag + "_fraction"] = f
            for j, size in enumerate(d[ib]):
                cr = dict(
                    time_unix=int(start),
                    diameter_dva_nm=size,
                    width_log10_dva=weights[j],
                    n_ambient=scan_count,
                    n_common=int(common_count[j]),
                )
                cr.update({s + "_n": int(counts[i, j]) for i, s in enumerate(SPECIES)})
                coverage.append(cr)
            records.append(row)
        bands = pd.DataFrame(records)
        save(bands, output / "ams-band-6h.csv.gz")
        save(pd.DataFrame(coverage), output / "ams-band-coverage.csv.gz")
        result = dict(
            doi="10.6075/J0WM1DSC",
            raw_family_doi="10.6075/J0ZS2WQ0",
            source_hashes=HASHES,
            eligible_ams_scans=len(ts),
            complete_nonnegative_bulk=int(valid.sum()),
            kappa_rows=len(kt),
            band_minimum_nm=float(d[ib].min()),
            band_maximum_nm=float(d[ib].max()),
            bins=int(ib.sum()),
            six_hour_windows=len(bands),
            eligible_windows=int(bands.coverage_pass.sum()),
            eligible_joint=int(bands.joint_pass.sum()),
            min_finite_differential=float(np.nanmin(vals)),
            negative_finite_differential=int(np.sum(vals < 0)),
            finite_differential=int(np.isfinite(vals).sum()),
            all_species_all_bin_complete_scans=int(
                np.isfinite(vals).all(axis=(0, 1)).sum()
            ),
        )
        (output / "ams-source.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("ams", "kappa", "ccn", "output"):
        p.add_argument(name, type=Path)
    main(**vars(p.parse_args()))
