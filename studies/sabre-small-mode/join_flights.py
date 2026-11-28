"""Join final 2023 aircraft products on the AMP sampling window."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PRODUCTS = {
    "amp": ("SABRE-AMP-SDAerosol-UCATStime", "R3"),
    "gas": ("SABRE-UCATS-GC", "R0"),
    "soap": ("SABRE-SOAP", "R0"),
    "mms": ("SABRE-MMS-1Hz", "R0"),
}


def read_icartt(path):
    """Read FFI 1001, mask per-variable fill flags, then scale."""
    with path.open(encoding="latin1") as infile:
        length, ffi = [int(x) for x in infile.readline().split(",")]
        if ffi != 1001:
            raise ValueError(f"Unexpected ICARTT format in {path.name}")
        header = [""] + [infile.readline().strip() for _ in range(length - 1)]
        values = pd.read_csv(infile, header=None, dtype=float).to_numpy()
    labels = [item.strip() for item in header[-1].split(",")]
    scales = np.fromstring(header[10], sep=",")
    fills = np.fromstring(header[11], sep=",")
    if values.shape[1] != len(labels) or len(labels) != len(fills) + 1:
        raise ValueError(f"Unexpected variable count in {path.name}")
    values[:, 1:] = np.where(values[:, 1:] == fills, np.nan, values[:, 1:])
    for line in header:
        if line.startswith(("ULOD_FLAG:", "LLOD_FLAG:")):
            flag = float(line.split(":")[1].strip())
            values[:, 1:][values[:, 1:] == flag] = np.nan
    values[:, 1:] *= scales
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(
            f"Non-monotone or repeated aircraft time in {path.name}"
        )
    return pd.DataFrame(values, columns=labels), header


def size_grid(header):
    """Read author-provided channel centers and edges in nanometers."""

    def diameters(phrase):
        line = next(
            line
            for line in header
            if line.startswith("OTHER_COMMENTS:") and phrase in line
        )
        return np.fromstring(line.rsplit(":", maxsplit=1)[1], sep=",")

    return diameters("Mid-point"), diameters("boundaries")


def windows(data, starts, stops, columns, minimum):
    """Average valid data centered at the reported instants in 75 s windows."""
    times = data.iloc[:, 0].to_numpy()
    values = data[columns].to_numpy()
    means, counts = [], []
    for start, stop in zip(starts, stops, strict=True):
        first = np.searchsorted(times, start - 0.5)
        last = np.searchsorted(times, stop + 0.5)
        subset = values[first:last]
        valid = np.isfinite(subset).all(axis=1)
        n = int(valid.sum())
        means.append(
            subset[valid].mean(axis=0)
            if n >= minimum
            else np.full(len(columns), np.nan)
        )
        counts.append(n)
    return np.array(means), np.array(counts)


def join(raw, date):
    """Match one flight’s gas, spectra, SOAP and navigation records."""
    frames = {}
    for key, (product, revision) in PRODUCTS.items():
        frames[key], header = read_icartt(
            raw / f"{product}_WB57_{date}_{revision}.ict"
        )
        if key == "amp":
            centers, edges = size_grid(header)
    amp, gas = frames["amp"], frames["gas"]
    start, stop, midpoint = amp.iloc[:, :3].to_numpy().T
    if not np.allclose(stop - start, 74) or not np.allclose(
        midpoint, (start + stop) / 2
    ):
        raise ValueError(f"Unexpected AMP window definition for {date}")
    # UCATS sample duration can vary. R3 AMP is already centered on these times.
    gas_mid = (gas.Time_Start + gas.Time_Stop) / 2
    if not gas_mid.is_unique:
        raise ValueError(f"Repeated UCATS midpoint in {date}")
    gas_at_amp = gas.set_index(gas_mid).reindex(midpoint)
    soap_mean, soap_n = windows(
        frames["soap"], start, stop, ["Extinction_SOAP"], 57
    )
    mms_fields = ["G_LAT_MMS", "G_ALT_MMS", "P_MMS", "T_MMS"]
    mms_mean, _ = windows(frames["mms"], start, stop, mms_fields, 57)
    ozone_file = raw / f"SABRE-UASO3_WB57_{date}_R0.ict"
    if ozone_file.exists():
        ozone, _ = read_icartt(ozone_file)
        ozone_mean, _ = windows(ozone, start, stop, ["O3_ppb"], 113)
    else:
        ozone_mean = np.full((len(start), 1), np.nan)
    spectra = amp.filter(regex=r"^dNdlogDp_dry_\d+_AMP$").to_numpy()
    valid = np.isfinite(spectra).all(axis=1) & (spectra >= 0).all(axis=1)
    valid &= (gas_at_amp.N2O_UCATS > 0).to_numpy() & (
        gas_at_amp.N2Oe_UCATS >= 0
    ).to_numpy()
    valid &= np.isfinite(soap_mean[:, 0]) & np.isfinite(mms_mean).all(axis=1)
    arrays = {
        "flight": np.full(len(start), int(date)),
        "start_s": start,
        "stop_s": stop,
        "n2o_ppbv": gas_at_amp.N2O_UCATS.to_numpy(),
        "n2o_err_ppbv": gas_at_amp.N2Oe_UCATS.to_numpy(),
        "latitude_deg": mms_mean[:, 0],
        "altitude_m": mms_mean[:, 1],
        "pressure_hpa": mms_mean[:, 2],
        "temperature_k": mms_mean[:, 3],
        "o3_ppbv": ozone_mean[:, 0],
        "soap_n": soap_n,
        "soap_Mm1": soap_mean[:, 0],
        "dndlog10d_cm3": spectra,
    }
    return {k: v[valid] for k, v in arrays.items()}, centers, edges


def main():
    """Build joined arrays for local final paired flight files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "raw")
    args = parser.parse_args()
    dates = sorted(
        f.stem.split("_")[2]
        for f in args.raw.glob("SABRE-SOAP_WB57_2023*_R0.ict")
    )
    chunks = []
    c0, e0 = None, None
    for date in dates:
        arrays, centers, edges = join(args.raw, date)
        if chunks and (
            not np.array_equal(c0, centers) or not np.array_equal(e0, edges)
        ):
            raise ValueError("Size grid changed between flights.")
        c0, e0 = centers, edges
        chunks.append(arrays)
        print(date, len(arrays["flight"]))
    if not chunks:
        raise ValueError("No final overlapping flights found.")
    output = {k: np.concatenate([a[k] for a in chunks]) for k in chunks[0]}
    np.savez_compressed(
        args.raw / "joined.npz", **output, diameter_nm=c0, edges_nm=e0
    )


if __name__ == "__main__":
    main()
