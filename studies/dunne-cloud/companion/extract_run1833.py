"""Extract observed run 1833 concentrations from Donahue et al. (2026)."""

import csv
import hashlib
import io
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "donahue2026-figure-data.zip"


def curves(archive, panel):
    data = archive.read(f"archive/fig{panel}.dat").decode()
    parts = re.split(r"^% xy for ", data, flags=re.MULTILINE)[1:]
    return [np.loadtxt(io.StringIO("\n".join(part.splitlines()[1:])), comments="%")
            for part in parts]


def spectrum(archive):
    data = archive.read("archive/fig06a.dat").decode()
    rows = [np.fromstring(line, sep=" ") for line in data.splitlines()
            if line.strip() and not line.startswith("%")]
    times, logd = rows[:2]
    log_density = np.array(rows[2:])
    if log_density.shape != (len(logd), len(times)):
        raise ValueError("Unexpected archive matrix orientation")
    # Figure 11a publishes the linear ordinates of one spectrum. Z = 0 can
    # represent either a small ordinate (<= 1) or no detections in the plot.
    density = np.where(log_density > 0, np.power(10.0, log_density), 0.0)
    edge_mid = (logd[1:] + logd[:-1]) / 2
    edges = np.r_[2 * logd[0] - edge_mid[0], edge_mid, 2 * logd[-1] - edge_mid[-1]]
    return times, np.power(10.0, logd), edges, density


def integrate(density, edges, cut):
    widths = np.maximum(0, edges[1:] - np.maximum(edges[:-1], np.log10(cut)))
    return widths @ density


def median_samples(trace, times):
    # Partition on midpoints, with a half-bin window at either end.
    edges = np.r_[times[0] - 1 / 60, (times[1:] + times[:-1]) / 2, times[-1] + 1 / 60]
    index = np.searchsorted(trace[:, 0], edges, side="left")
    result = []
    for start, stop in zip(index[:-1], index[1:]):
        values = trace[start:stop, 1]
        values = values[np.isfinite(values)]
        result.append(float(np.median(values)) if len(values) else np.nan)
    return np.array(result)


def write_csv(name, fields):
    with (HERE / name).open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(fields)
        for row in zip(*fields.values(), strict=True):
            out.writerow([format(value, ".10g") if np.isfinite(value) else "" for value in row])


def main():
    with ARCHIVE.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != "b2d281aa2d86868cf5c2fb4fdc85ac1a":
            raise ValueError("Unexpected Zenodo data archive")
    with ZipFile(ARCHIVE) as archive:
        time_full, diameters, edges, density_full = spectrum(archive)
        mask = time_full - time_full[0] <= 6.00002
        time = time_full[mask]
        density = density_full[:, mask]
        relative = time - time[0]
        cpc, author_number = curves(archive, "06b")
        if not (np.array_equal(cpc[:, 0], time_full) and np.array_equal(author_number[:, 0], time_full)):
            raise ValueError("Published CPC and size-spectrum grids differ")
        uhel, ufra = curves(archive, "02b")
        ozone, = curves(archive, "04a")
        sulfur_dioxide, = curves(archive, "04b")
        _, _, hglamp = curves(archive, "02a")
        n22 = integrate(density, edges, 2.2)
        n29 = integrate(density, edges, 2.9)
        n97 = integrate(density, edges, 9.7)
        reference, = curves(archive, "11a")
        col = np.argmin(abs(time_full + 1.483056))
        reconstructed_slice = density_full[:, col]
        if not np.allclose(reference[:, 0], diameters, rtol=2e-6):
            raise ValueError("Figure 11a and 6a diameters disagree")
        # A single exported spectrum recurs in independently stored linear units.
        resolved = reference[:, 1] > 1
        if not np.allclose(reference[resolved, 1], reconstructed_slice[resolved], rtol=1e-6):
            raise ValueError("Spectrum conversion disagrees with linear Figure 11a")
        if not np.allclose(reference[~resolved, 1], reconstructed_slice[~resolved], atol=1):
            raise ValueError("Unexpected floor behavior")
        index29 = np.flatnonzero(np.isclose(diameters, 2.9)).item()
        write_csv("donahue1833_observations.csv", {
            "archive_time_h": time, "elapsed_from_first_spectrum_h": relative,
            "H2SO4_UFRA_cm3": median_samples(ufra, time),
            "H2SO4_UHEL_cm3": median_samples(uhel, time),
            "O3_ppbv": median_samples(ozone, time), "SO2_ppbv": median_samples(sulfur_dioxide, time),
            "HgXe_lamp_au": median_samples(hglamp, time),
            "CPC2p5_filtered_cm3": cpc[mask, 1],
            "author_N_ge2p2phys_cm3": author_number[mask, 1],
            "derived_N_ge2p2phys_cm3": n22, "derived_N_ge2p9phys_cm3": n29,
            "derived_N_ge9p7phys_cm3": n97, "dN_dlog10dp_at2p9phys_cm3": density[index29],
        })
        write_csv("donahue1833_initial_spectrum.csv", {
            "physical_diameter_nm": diameters, "lower_log10_bin_edge_nm": edges[:-1],
            "upper_log10_bin_edge_nm": edges[1:], "dN_dlog10dp_cm3": density[:, 0],
            "integrated_bin_N_cm3": density[:, 0] * np.diff(edges),
        })
        print("rows, elapsed h, spectrum support nm:", len(time), relative[-1], 10**edges[[0, -1]])
        print("coverage UFRA, UHEL, O3, SO2, lamp:", [np.isfinite(median_samples(t, time)).sum()
              for t in (ufra, uhel, ozone, sulfur_dioxide, hglamp)])
        print("Derived N2.2 / author N2.2 p05, median, p95:",
              np.quantile(n22 / author_number[mask, 1], [.05, .5, .95]))
        print("Starting / peak / ending N3.2mob, N10mob, CPC2.5:",
              [(v[0], max(v), v[-1]) for v in (n29, n97, cpc[mask, 1])])
        print("Figure 11a low bins where log export floors to zero:", reference[~resolved,1])


if __name__ == "__main__":
    main()
