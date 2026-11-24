"""Recover size-selected optical bin counts, with logged overflow and flow."""

# %%
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from acquisition import voltage
from extract_field import field

RAW_SHA = "9327375ae24a99f3e1abe6b1fa640ef48fa991894e40c4bfb9397a122f942edf"
PROC_SHA = "9f94d7f3e1e66bbb7d5149fd3b9e3f2eea776dd9f898f3af6d185ceca9c53800"


def verify(path, expected):
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise ValueError(f"Unexpected archive {path}")


def processed(zpath, accepted, output):
    verify(zpath, PROC_SHA)
    dct = defaultdict(list)
    coords = set()
    with TemporaryDirectory() as tmp, ZipFile(zpath) as z:
        scratch = Path(tmp)
        for member in z.namelist():
            if not member.endswith(".cdf") or not (
                ".202305" in member or ".202306" in member
            ):
                continue
            p = scratch / "day.cdf"
            p.write_bytes(z.read(member))
            t = field(p, "time", scratch, "<i4")
            flow = field(p, "sample_flow_ccn", scratch)
            coords.update(field(p, "bin_ccn_opc", scratch).tolist())
            hist = field(p, "droplet_size_distribution_ccn", scratch).reshape(
                len(t), 20
            )
            for root, ss, phase in accepted[
                (accepted[:, 0] >= t[0]) & (accepted[:, 0] < t[-1])
            ]:
                idx = np.searchsorted(t, root + 300 * phase + np.array([0, 600]))
                if idx[-1] >= t.size or not np.all(
                    t[idx] == root + 300 * phase + np.array([0, 600])
                ):
                    continue
                if not (
                    np.isfinite(hist[idx]).all()
                    and (hist[idx] >= 0).all()
                    and np.all((flow[idx] > 30) & (flow[idx] < 65))
                ):
                    continue
                dct[root // 21600, ss].append(hist[idx] * 60 / flow[idx, None])
    keys = sorted(dct)
    np.savez_compressed(
        output / "processed.npz",
        keys=keys,
        spectra=[np.mean(dct[k], axis=0) for k in keys],
        n=[len(dct[k]) for k in keys],
    )
    return sorted(coords)


def select_raw(zpath, accepted, extension, out):
    verify(zpath, RAW_SHA)
    bases, n = np.unique(accepted[:, 0], return_counts=True)
    both = set(bases[n == 2])
    pairs = {int(t): int(s) for t, s, _ in accepted if t in both and s in (4, 8)}
    groups = defaultdict(list)
    mobility = defaultdict(list)
    mobility_bounds = np.array([[35, 50], [50, 70], [70, 120], [120, 250]])
    acquisition = np.load(extension / "acquisition.npz")
    with np.errstate(invalid="ignore"):
        template = np.nanmedian(
            (acquisition["sums"] / acquisition["counts"])[:, :, :, 2],
            axis=(0, 1),
        )
    lo, hi = voltage([120, 250])
    size = (template >= lo) & (template <= hi)
    masks = []
    for low_nm, high_nm in mobility_bounds:
        low_v, high_v = voltage([low_nm, high_nm])
        masks.append((template >= low_v) & (template < high_v))
    delays = json.loads((extension / "lag.json").read_text())
    totals = np.zeros(5)
    settings = set()
    with ZipFile(zpath) as z:
        for member in sorted(z.namelist()):
            if not ("rack_202305" in member or "rack_202306" in member):
                continue
            day = Path(member).stem[-8:]
            d = int(np.datetime64(day[:4] + "-" + day[4:6] + "-" + day[6:]).astype(int))
            midnight = d * 86400
            need = {
                int((t - midnight) // 600) + b: (t, b)
                for t in pairs
                if t // 86400 == d
                for b in (0, 1)
            }
            if not need:
                continue
            a = np.full((144, 600, 26), np.nan)
            with z.open(member) as raw:
                for line in io.BufferedReader(raw, buffer_size=262144):
                    try:
                        sec = (
                            int(line[11:13]) * 3600
                            + int(line[14:16]) * 60
                            + int(line[17:19])
                        )
                        slot = sec // 600
                        if slot not in need:
                            continue
                        prefix, cpc, _, ccn, lab = line.split(b";", 4)
                        t, b = need[slot]
                        if prefix.rsplit(b",", 1)[-1] != (b"true" if b else b"false"):
                            continue
                        r = cpc.strip().split(b",")
                        h = ccn.strip().split(b",")
                        v = lab.strip().split(b",")
                        if not (
                            r[0] == b"RALL"
                            and r[3] == b"0000"
                            and len(h) == 42
                            and h[0] == b"H"
                            and h[18] == b"C"
                            and v[0] == b"LABJCACK"
                        ):
                            continue
                        tick = int(v[1]) - 1
                        if not 0 <= tick < 600 or abs(tick - sec % 600) > 6:
                            continue
                        c = float(r[2])
                        flow = float(h[11])
                        number = float(h[20])
                        u = float(v[2])
                        hist = np.fromiter(map(float, h[22:42]), float, count=20)
                        over = float(h[21])
                        setting = float(h[19])
                        if not (
                            c >= 0
                            and number >= 0
                            and 25 < u < 6500
                            and 35 < flow < 60
                            and np.min(hist) >= 0
                            and over >= 0
                        ):
                            continue
                        settings.add(setting)
                        a[slot, tick] = [
                            c,
                            number,
                            flow,
                            u,
                            over,
                            setting,
                            *hist,
                        ]
                    except (IndexError, ValueError):
                        continue
            for slot, (t, b) in need.items():
                if b:
                    continue
                y = a[[slot, slot + 1]]
                valid = np.isfinite(y).all(axis=(0, 2))
                if valid.sum() < 560:
                    continue
                y[:, ~valid] = np.nan
                # Explore serial timing on adjacent valid ticks, never wrap a cycle.
                counted = np.sum(y[:, :, 6:], axis=2) + y[:, :, 4]
                measured = y[:, :, 1] * y[:, :, 2] / 60
                good = (
                    np.isfinite(measured[:, 1:])
                    & np.isfinite(counted[:, :-1])
                    & np.isfinite(counted[:, 1:])
                )
                totals[:3] += [
                    np.sum(abs(measured[:, 1:][good] - counted[:, 1:][good])),
                    np.sum(abs(measured[:, 1:][good] - counted[:, :-1][good])),
                    good.sum(),
                ]
                totals[3:] += [np.nansum(y[:, :, 4]), np.nansum(counted)]
                values = []
                pattern = []
                for b, name in enumerate(("bypass", "denuded")):
                    cpc_mask = np.roll(size, delays[name + "_cpc_s"])
                    det_mask = np.roll(size, delays[name + "_ccn_s"])
                    hmask = np.roll(det_mask, -1)
                    cpc = np.nanmean(y[b, cpc_mask, 0])
                    ccn = np.nanmean(y[b, det_mask, 1])
                    # Per-second counts / sampled volume. The one-second time shift
                    # matches the concentration register; flow changes little.
                    bins = np.nanmean(
                        y[b, hmask, 6:] * 60 / y[b, hmask, 2, None], axis=0
                    )
                    overflow = np.nanmean(y[b, hmask, 4] * 60 / y[b, hmask, 2])
                    values.append([cpc, ccn, overflow, *bins])
                    row = []
                    for mask in masks:
                        cmask = np.roll(mask, delays[name + "_cpc_s"])
                        omask = np.roll(mask, delays[name + "_ccn_s"] - 1)
                        cgood = cmask & valid
                        hgood = omask & valid
                        row.append(
                            [
                                np.nansum(y[b, cgood, 0]),
                                np.sum(cgood),
                                np.nansum(y[b, hgood, 2]) / 60.0,
                                np.nansum(y[b, hgood, 4]),
                                *np.nansum(y[b, hgood, 6:], axis=0),
                            ]
                        )
                    pattern.append(row)
                mobility[t // 21600, pairs[t]].append(pattern)
                groups[t // 21600, pairs[t]].append(values)
    if settings != {0.0}:
        raise ValueError(f"Mixed CCN bin cutoffs {settings}")
    keys = sorted(groups)
    census = np.stack([np.sum(mobility[k], axis=0) for k in keys])
    np.savez_compressed(
        out / "mobility-outlet.npz",
        keys=keys,
        n=[len(groups[k]) for k in keys],
        bounds_nm=mobility_bounds,
        upper_um=np.r_[0.75, 1.0, np.arange(1.5, 10.01, 0.5)],
        cpc_integral=census[:, :, :, 0],
        cpc_seconds=census[:, :, :, 1],
        volume_cm3=census[:, :, :, 2],
        overflow=census[:, :, :, 3].astype(np.int64),
        bins=census[:, :, :, 4:].astype(np.int64),
    )
    np.savez_compressed(
        out / "large.npz",
        keys=keys,
        n=[len(groups[k]) for k in keys],
        observations=[np.mean(groups[k], axis=0) for k in keys],
    )
    return dict(
        archive_bins=20,
        logged_cutoff=list(settings),
        same_tick_mean_abs_counts=totals[0] / totals[2],
        next_tick_mean_abs_counts=totals[1] / totals[2],
        timing_comparisons=int(totals[2]),
        overflow_counts=int(totals[3]),
        all_detected_counts=int(totals[4]),
    )


if __name__ == "__main__":
    processed_path, raw_path, extension, out = map(Path, sys.argv[1:])
    accepted = np.load(extension / "blocks-final.npz")["accepted"]
    audit = {"processed_bin_coordinates": processed(processed_path, accepted, out)}
    audit.update(select_raw(raw_path, accepted, extension, out))
    (out / "acquisition-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
