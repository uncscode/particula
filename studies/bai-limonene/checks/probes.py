"""Replay the paired cloud-probe comparison using PaCE 2015 spectra."""

import argparse
import hashlib
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.io import netcdf_file

ARCHIVE = "https://fmi.b2share.csc.fi/api/records/va6sm-a6894/files/"
# MD5 values are the file fixity metadata published with this versioned record.
CHECKSUMS = {
    "FSSP": {
        "csv": "82f922d5de609d014ff424eaf0c8c367",
        "nc": "75b3776d7ed7de6ce7c0db82d217c6e7",
    },
    "CAS": {
        "csv": "68b63c142360a919f96007937fa121a8",
        "nc": "3c2cdf3c2fa9ac2ea4cb16030dc81b53",
    },
}


def source(cache, instrument, extension, expected):
    """Fetch a versioned archive file atomically and check its digest."""
    name = f"PACE.2015.{instrument}.{extension}"
    path = cache / name
    if not path.exists():
        with NamedTemporaryFile(dir=cache, delete=False) as tmp:
            partial = Path(tmp.name)
        try:
            # All URLs use the fixed FMI HTTPS record above.
            url = ARCHIVE + name + "/content"
            with (
                urlopen(url, timeout=90) as remote,  # noqa: S310
                partial.open("wb") as out,
            ):
                shutil.copyfileobj(remote, out)
            with partial.open("rb") as f:
                if hashlib.file_digest(f, "md5").hexdigest() != expected:
                    raise ValueError(
                        f"Downloaded file failed archive checksum: {name}"
                    )
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    with path.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != expected:
            raise ValueError(f"Archive checksum mismatch: {path}")
    return path


def read_probe(cache, name):
    """Read time-aligned spectral counts and screen by ambient conditions."""
    files = {
        ext: source(cache, name, ext, digest)
        for ext, digest in CHECKSUMS[name].items()
    }
    obs = pd.read_csv(files["csv"])
    # CSV integer dates avoid rounding the single-precision NetCDF Unix times.
    dates = obs.rename(columns={"min": "minute"})[
        ["year", "month", "day", "hour", "minute"]
    ]
    obs.index = pd.to_datetime(dates, utc=True)
    if not obs.index.is_unique:
        raise ValueError(f"Duplicate archive timestamps for {name}")
    spectra = obs.filter(regex="^dN/dlogDp bin ")
    with netcdf_file(files["nc"], mmap=False) as nc:
        lower = np.round(
            nc.variables["size_bin_lower_limit"][:].astype(float), 3
        )
        upper = np.round(
            nc.variables["size_bin_upper_limit"][:].astype(float), 3
        )
        np.testing.assert_allclose(
            spectra,
            nc.variables["number_concentration_of_cloud_droplets"][:],
            rtol=1e-6,
            atol=1e-4,
        )
    np.testing.assert_array_equal(lower[1:], upper[:-1])
    good = (
        (spectra >= 0).all(axis=1)
        & obs["Nc (1/ccm)"].between(1, 300)
        & (obs["Temp 570m (C)"] >= 0.5)
        & obs["Humidity 570m (%)"].between(95, 105)
        & obs["FD12P visibility (m)"].between(1, 999)
        & (obs["Wind speed (m/s)"] >= 2)
        & obs["Wind dir (deg)"].between(200, 235)
    )
    counts = spectra.loc[good] * (60 * np.log10(upper / lower))
    if name == "CAS":
        counts = counts.mul(
            obs.loc[good, "Nc (1/ccm)"] / counts.sum(axis=1), axis=0
        )
    return np.r_[lower, upper[-1]], counts


def median_diameter(counts, edges):
    """Estimate the number median by log interpolation within its bin."""
    cumulative = np.r_[0, np.cumsum(counts)]
    mid = np.searchsorted(cumulative, cumulative[-1] * 0.5)
    fraction = (cumulative[-1] * 0.5 - cumulative[mid - 1]) / counts[mid - 1]
    return np.exp(
        np.log(edges[mid - 1]) + fraction * np.log(edges[mid] / edges[mid - 1])
    )


def shift_sizes(counts, edges, shift):
    """Shift the log-interpolated diameter CDF onto the native bin edges."""
    cdf = np.r_[0, np.cumsum(counts)]
    prior_edges = np.clip(edges - shift, edges[0], edges[-1])
    return np.diff(np.interp(np.log(prior_edges), np.log(edges), cdf))


def run(cache, output):
    """Select the longest paired liquid event and test imposed size shifts."""
    cache.mkdir(parents=True, exist_ok=True)
    probes = {name: read_probe(cache, name) for name in CHECKSUMS}
    common = (
        probes["FSSP"][1]
        .index.intersection(probes["CAS"][1].index)
        .sort_values()
    )
    times = pd.Series(common, index=common)
    sequence = times.diff().ne(pd.Timedelta(minutes=1)).cumsum()
    event = times[sequence == times.groupby(sequence).size().idxmax()].index
    if len(event) != 99 or event[0] != pd.Timestamp("2015-10-25T03:46:00Z"):
        raise ValueError("Unexpected event selected from the PaCE archive.")
    summary, recovery = [], []
    for name, (edges, all_counts) in probes.items():
        native = all_counts.loc[event].to_numpy()
        common_bins = (edges[:-1] >= 3.5) & (edges[1:] <= 40)
        cut_edges = edges[(edges >= 3.5) & (edges <= 40)]
        spectra = native[:, common_bins]
        for window, bins, bounds in [
            ("native", native, edges),
            ("3.5–40 µm", spectra, cut_edges),
        ]:
            pooled = bins.mean(axis=0)
            summary.append(
                dict(
                    instrument=name,
                    window=window,
                    minutes=len(event),
                    start_utc=event[0],
                    end_utc=event[-1],
                    number_cm3=pooled.sum(),
                    median_um=median_diameter(pooled, bounds),
                )
            )
        for shift in [0.3, 1.2]:
            for pair in range(len(event) // 10):
                early, late = np.split(spectra[pair * 10 : (pair + 1) * 10], 2)
                early, late = early.mean(axis=0), late.mean(axis=0)
                first = median_diameter(
                    shift_sizes(early, cut_edges, shift), cut_edges
                ) - median_diameter(late, cut_edges)
                second = median_diameter(
                    shift_sizes(late, cut_edges, shift), cut_edges
                ) - median_diameter(early, cut_edges)
                recovery.append(
                    dict(
                        instrument=name,
                        pair=pair + 1,
                        shift_um=shift,
                        treated_early_um=first,
                        treated_late_um=second,
                        positive_both=bool(first > 0 and second > 0),
                    )
                )
    output.mkdir(parents=True, exist_ok=True)
    summary, recovery = pd.DataFrame(summary), pd.DataFrame(recovery)
    summary.to_csv(output / "pallas-event.csv", index=False)
    recovery.to_csv(output / "pallas-blocks.csv", index=False)
    print(summary.to_string(index=False))
    print(recovery.groupby(["instrument", "shift_um"]).positive_both.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache", type=Path, help="Directory for the four original archive files"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.cache, args.output)
