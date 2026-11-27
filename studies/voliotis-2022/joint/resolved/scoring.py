"""Compare measured bulk time series, number spectra and selected-particle κ."""

import importlib.util

import numpy as np
from particula.particles import (
    get_number_per_decade,
    get_rebinned_number_spectrum,
)
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from simulate import GRID, HERE, JOINT


def companion_records():
    """Load the archived water records with their original checksum checks."""
    spec = importlib.util.spec_from_file_location(
        "manchester_companion_water", JOINT / "companions/compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.water_records()


def read_water():
    """Attach seed references and selected dry diameters."""
    full = companion_records()
    groups = []
    for (run, ch), g in full.groupby(["run", "channel"]):
        anchor_ch = ch if ch != "HTDMA_150nm" else "HTDMA_100nm"
        k = (
            full[
                (full.run == run)
                & (full.channel == anchor_ch)
                & (full.seconds < 0)
            ]
            .sort_values("seconds")
            .iloc[-1]
            .observed
        )
        a = g[g.seconds >= 0].copy()
        a["seed_k"] = k
        a["diameter"] = (
            a.activation_nm
            if ch == "CCN"
            else float(ch.split("_")[1].removesuffix("nm"))
        )
        groups.append(a)
    return pd.concat(groups, ignore_index=True)


def spectra(hist, air, real):
    """Rebin count per dlog10D by overlap of the log-diameter intervals."""
    if real.empty:
        return np.empty(0, dtype=np.float64)
    source = 2e-9 * 1.12 ** np.arange(61)
    bounds = real[["low_nm", "high_nm"]].to_numpy() * 1e-9
    target, indices = np.unique(bounds, return_inverse=True)
    left, right = indices.reshape(bounds.shape).T
    time = real.time_s.to_numpy().astype(int) // 300
    # Rows may repeat a diameter interval at different times or omit a channel.
    counts = get_rebinned_number_spectrum(hist[:, :60] / air, source, target)
    if np.all(right == left + 1):
        return (
            get_number_per_decade(counts.bin_number_density, target)[time, left]
            * 1e-6
        )
    # A requested interval may span a boundary from another row.
    return (
        np.array(
            [
                get_number_per_decade(
                    [counts.bin_number_density[t, lo:hi].sum()],
                    target[[lo, hi]],
                )[0]
                for t, lo, hi in zip(time, left, right, strict=True)
            ]
        )
        * 1e-6
    )


class Targets:
    """Match particle outputs to archived observation operators."""

    def __init__(self, c, *, surface_window=False):
        """Read the run-matched and digitized size measurements."""
        self.surface_window = surface_window
        self.chamber = c
        self.water = read_water()
        self.psd = pd.read_csv(HERE / "size_data/run24_psd.csv")
        self.psd = (
            self.psd.dropna(subset=["dndlog10d_cm3"])
            .query("valid_fraction>=0.5")
            .copy()
        )
        if self.psd.empty:
            raise ValueError("No supported measured spectrum cells")
        self.modes = pd.read_csv(HERE / "size_data/dmps_modes.csv").query(
            "0<=time_s<=18000"
        )
        self.surface = pd.read_csv(HERE / "size_data/dmps_surface.csv").query(
            "0<=time_s<=18000"
        )
        self.time = np.arange(61) * 300

    def evaluate(self, values, settings, ids, details=False):  # noqa: C901
        """Score six blocks and optionally return individual records."""
        h, dist, events, _, air = values
        out = []
        records = []
        for b, config in enumerate(settings):
            mass = []
            count = []
            water = []
            table = []
            modes = []
            surface = None
            psd = None
            for j, run in enumerate(ids):
                i = self.chamber.ids.index(run)
                row = b * len(ids) + j
                sim = h[b, j]
                _, _, mt, measured, _ = self.chamber.runs[i]
                mass.append(
                    np.sqrt(
                        np.mean(
                            (np.interp(mt, self.time, sim[:, 0]) - measured)
                            ** 2
                        )
                    )
                    / max(1.0, measured.max())
                )
                obs = self.chamber.counter_data[i]
                keep = obs[:, 2] >= 0
                t = (np.flatnonzero(keep) + obs[keep, 2]) * 60
                num = sim[:, 3] / (air[row] * self.chamber.initial[i, 1])
                count.append(
                    np.sqrt(
                        np.mean(
                            (np.interp(t, self.time, num) - obs[keep, 3]) ** 2
                        )
                    )
                )
                mg = self.modes[self.modes.run == run].copy()
                if len(mg):
                    centers = 2 * 1.12 ** (np.arange(60) + 0.5)
                    ok = (centers >= 40) & (centers <= 600)
                    smooth = gaussian_filter1d(dist[row, :, :60], 1.5, axis=1)
                    modal = centers[ok][np.argmax(smooth[:, ok], axis=1)]
                    mg["predicted_nm"] = np.interp(mg.time_s, self.time, modal)
                    error = np.log(mg.predicted_nm / mg.mode_nm)
                    modes.append(
                        np.mean(
                            np.maximum(0, np.abs(error) - np.log(1.12)) ** 2
                        )
                    )
                    if details:
                        records.append(("modes", mg))
                if run == 18:
                    sg = self.surface.copy()
                    sg["prediction"] = np.interp(
                        sg.time_s,
                        self.time,
                        sim[:, 41 if self.surface_window else 10],
                    )
                    surface = np.sqrt(
                        np.mean(
                            (
                                (sg.prediction - sg.surface_nm2_cm3)
                                / sg.surface_nm2_cm3.max()
                            )
                            ** 2
                        )
                    )
                    if details:
                        records.append(("surface", sg))
                if run == 24:
                    predicted = spectra(dist[row], air[row], self.psd)
                    ref = self.psd.dndlog10d_cm3.to_numpy()
                    # Reading interval from the colour contours. Only excess
                    # counts as misfit.
                    deviation = np.maximum(
                        0,
                        np.abs(predicted - ref)
                        - self.psd.reading_band_cm3.to_numpy(),
                    )
                    psd = np.sqrt(np.mean(deviation**2)) / 40000.0
                    if details:
                        records.append(
                            ("psd", self.psd.assign(simulated=predicted))
                        )
                if run in (6, 15):
                    g = self.water[self.water.run == run].copy()
                    predicted = []
                    supports = []
                    for v in g.itertuples():
                        idx = np.arange(60)
                        centers = 2 * 1.12 ** (idx + 0.5)
                        if v.channel == "CCN":
                            bins = np.abs(np.log(centers / v.diameter)) < 0.12
                            n = dist[row, :, :60][:, bins].sum(axis=1)
                            frac = dist[row, :, 60:][:, bins].sum(axis=1)
                        else:
                            pos = int(np.flatnonzero(GRID == v.diameter)[0])
                            n = sim[:, 4 + pos]
                            frac = sim[:, 7 + pos]
                        nn = np.interp(v.seconds, self.time, n)
                        fv = np.interp(v.seconds, self.time, frac)
                        phi = fv / nn if nn >= 3 else np.nan
                        predicted.append(phi)
                        supports.append(nn)
                    g["organic_volume_fraction"] = predicted
                    g["sampled_particles"] = supports
                    k = np.where(g.channel == "CCN", config[-1], config[-2])
                    g["prediction"] = (
                        g.seed_k * (1 - g.organic_volume_fraction)
                        + k * g.organic_volume_fraction
                    )
                    # A missing size channel incurs a fixed 0.20 κ error.
                    g["error"] = (g.prediction - g.observed).fillna(0.20)
                    water.extend(
                        g.groupby("channel")
                        .error.apply(lambda x: np.mean(x**2))
                        .tolist()
                    )
                    if details:
                        records.append(("water", g))
                table.append(
                    dict(
                        run=run,
                        mass=mass[-1],
                        number=count[-1],
                        births=events[row, 5],
                        deposited=events[row, 2],
                        merged=events[row, 3],
                    )
                )
            if details:
                records.append(("runs", pd.DataFrame(table)))
            result = dict(
                mass=np.mean(mass),
                number=np.mean(count),
                worst_mass=max(mass),
                worst_number=max(count),
                size=float(psd),
                water=np.sqrt(np.mean(water)),
            )
            result["mode"] = np.sqrt(np.mean(modes))
            result["surface"] = float(surface)
            result["loss"] = (
                result["mass"]
                + result["number"]
                + result["size"]
                + result["mode"]
                + result["surface"]
                + result["water"] / 0.5
            ) / 6
            out.append(result)
        return pd.DataFrame(out), records
