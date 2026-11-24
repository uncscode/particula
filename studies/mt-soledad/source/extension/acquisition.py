"""Independent voltage-sweep check using original one-second instrument messages."""

# %%
import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from particula.particles.properties.diffusion_coefficient import (
    get_diffusion_coefficient_via_system_state,
)

from edges import EPOCH


def scan_logs(archive, blocks, out):
    with archive.open("rb") as src:
        if (
            hashlib.file_digest(src, "sha256").hexdigest()
            != "9327375ae24a99f3e1abe6b1fa640ef48fa991894e40c4bfb9397a122f942edf"
        ):
            raise ValueError("Unexpected original logger ZIP")
    with np.load(blocks) as data:
        accepted = data["accepted"]
    bases, n = np.unique(accepted[:, 0], return_counts=True)
    both = set(bases[n == 2])
    pairs = {int(t): int(s) for t, s, _ in accepted if t in both}
    stats = Counter()
    results = defaultdict(list)
    with ZipFile(archive) as z:
        for member in sorted(z.namelist()):
            if not member.endswith(".txt") or not (
                "rack_202305" in member or "rack_202306" in member
            ):
                continue
            ds = Path(member).stem.split("_")[-1]
            day = int(np.datetime64(ds[:4] + "-" + ds[4:6] + "-" + ds[6:8]).astype(int))
            midnight = day * 86400
            needed = {
                int((t - midnight) // 600) + b: (t, b)
                for t, s in pairs.items()
                if t // 86400 == day and s in (4, 8)
                for b in (0, 1)
            }
            if not needed:
                continue
            arr = np.full((144, 600, 4), np.nan)
            with z.open(member) as raw:
                for line in io.BufferedReader(raw, buffer_size=262144):
                    try:
                        second = (
                            int(line[11:13]) * 3600
                            + int(line[14:16]) * 60
                            + int(line[17:19])
                        )
                        slot = second // 600
                        if slot not in needed:
                            continue
                        prefix, cpc, _, ccn, lab = line.split(b";", 4)
                        root, b = needed[slot]
                        if prefix.rsplit(b",", 1)[-1] != (b"true" if b else b"false"):
                            continue
                        rd = cpc.strip().split(b",")
                        hd = ccn.strip().split(b",")
                        step = lab.strip().split(b",")
                        if not (
                            rd[0] == b"RALL"
                            and rd[3] == b"0000"
                            and hd[0] == b"H"
                            and hd[18] == b"C"
                            and step[0] == b"LABJCACK"
                        ):
                            continue
                        tick = int(step[1]) - 1
                        if not 0 <= tick < 600 or abs(tick - second % 600) > 6:
                            continue
                        cv = float(rd[2])
                        hv = float(hd[20])
                        volt = float(step[2])
                        flow = float(hd[11])
                        if (
                            cv < 0
                            or hv < 0
                            or not 25 < volt < 6500
                            or not 35 < flow < 60
                        ):
                            continue
                        arr[slot, tick] = [cv, hv, volt, flow]
                    except (ValueError, IndexError):
                        stats["malformed"] += 1
            for slot, (t, b) in needed.items():
                if b:
                    continue
                y = arr[[slot, slot + 1]]
                valid = np.isfinite(y).all(axis=(0, 2))
                if valid.sum() < 560:
                    continue
                y[:, :, 0:2] = np.where(valid[None, :, None], y[:, :, 0:2], np.nan)
                results[(t // 21600, pairs[t])].append(y)
                stats["raw_cycles_paired"] += 1
    keys = sorted(results)
    if not keys:
        raise ValueError("No original acquisition blocks passed")
    sums = []
    counts = []
    num = []
    for key in keys:
        a = np.stack(results[key])
        sums.append(np.nansum(a, axis=0))
        counts.append(np.isfinite(a).sum(axis=0))
        num.append(len(a))
    np.savez_compressed(out, keys=keys, sums=sums, counts=counts, n=num)
    print(dict(stats), len(keys))


def voltage(d_nm):
    """Nominal central voltage of a singly charged particle from the cylindrical DMA."""
    T = 295.0
    P = 99400.0
    D = get_diffusion_coefficient_via_system_state(np.asarray(d_nm) * 0.5e-9, T, P)
    mobility = D * 1.602176634e-19 / (1.380649e-23 * T)
    return 8.33333e-5 * np.log(0.01961 / 0.00937) / (2 * np.pi * 0.44369 * mobility)


def evaluate(inp, out):
    with np.load(inp) as a:
        keys, sums, counts, n = [a[k] for k in ("keys", "sums", "counts", "n")]
    t = np.arange(600)
    with np.errstate(invalid="ignore"):
        means = sums / counts
    # A common voltage template checks alignment to the logger tick on both paths.
    v = np.nanmedian(means[:, :, :, 2], axis=(0, 1))
    if not np.isfinite(v).all():
        raise ValueError("missing voltage template")
    low, high = voltage([120, 250])
    print("voltage range", low, high)
    bins = np.geomspace(*voltage([55, 260]), 12)
    h = (keys[:, 0] // 4 - EPOCH) % 3 == 0
    tr = (~h) & (keys[:, 1] == 4) & (n >= 2)
    test = h & (keys[:, 1] == 8) & (n >= 2)
    cost = []
    delays = {}
    for b in (0, 1):
        for ch in (0, 1):
            denom = np.sum(counts[tr, b, :, ch], axis=0)
            signal = np.divide(
                np.sum(sums[tr, b, :, ch], axis=0),
                denom,
                out=np.full(600, np.nan),
                where=denom > 0,
            )
            scores = []
            for lag in range(0, 61):
                aligned = np.roll(signal, -lag)
                diffs = []
                for lower, upper in zip(bins[:-1], bins[1:], strict=True):
                    x = (v >= lower) & (v < upper)
                    # Avoid turn-around and hold. Voltage trajectory is monotone on these limbs.
                    up = x & (t >= 100) & (t < 340)
                    down = x & (t >= 340) & (t < 580)
                    if not (up.any() and down.any()):
                        continue
                    y1, y2 = np.mean(aligned[up]), np.mean(aligned[down])
                    if min(y1, y2) > 0:
                        diffs.append(np.log(y1 / y2) ** 2)
                if len(diffs) < 7:
                    raise ValueError("too few sweep bins for lag estimate")
                scores.append(np.mean(diffs))
                cost.append([b, ch, lag, scores[-1]])
            delays[b, ch] = int(np.argmin(scores))
    print("CPC and CCN lag in each branch", delays)
    rec = []
    for i in np.flatnonzero(test):
        for method in ("fitted", "earlier", "later", "whole", "unshifted"):
            vals = []
            for b in (0, 1):
                channels = []
                for ch in (0, 1):
                    lag = delays[b, ch]
                    if ch == 1 and b == 1:
                        lag += (
                            -5 if method == "earlier" else 5 if method == "later" else 0
                        )
                    if method == "unshifted":
                        lag = 0
                    mask = np.roll((v >= low) & (v <= high), lag)
                    if method == "whole":
                        mask = np.ones(600, dtype=bool)
                    channels.append(
                        np.sum(sums[i, b, mask, ch]) / np.sum(counts[i, b, mask, ch])
                    )
                vals.append(channels[1] / channels[0])
            rec.append(
                [
                    keys[i, 0] * 21600,
                    keys[i, 0] // 4,
                    n[i],
                    method,
                    *vals,
                    vals[1] / vals[0],
                ]
            )
    f = pd.DataFrame(rec, columns="utc day cycles method bypass denuded factor".split())
    f.to_csv(out / "acquisition-blocks.csv", index=False)
    pd.DataFrame(
        cost, columns=["branch", "channel", "lag_s", "mse_log_up_down"]
    ).to_csv(out / "lag-fit.csv", index=False)
    summ = []
    for method, g in f.groupby("method"):
        d = g.groupby("day").factor.mean().values
        rng = np.random.default_rng(933)
        mu = d[rng.integers(len(d), size=(4096, len(d)))].mean(axis=1)
        summ.append(
            [
                method,
                len(d),
                len(g),
                g.cycles.sum(),
                d.mean(),
                *np.percentile(mu, [2.5, 97.5]),
            ]
        )
    pd.DataFrame(
        summ, columns="method days blocks cycles factor low high".split()
    ).to_csv(out / "acquisition-summary.csv", index=False)
    (out / "lag.json").write_text(
        json.dumps(
            {
                "voltage_V": [low, high],
                "training_days": len(set(keys[tr, 0] // 4)),
                "training_cycles": int(n[tr].sum()),
                "bypass_cpc_s": delays[0, 0],
                "bypass_ccn_s": delays[0, 1],
                "denuded_cpc_s": delays[1, 0],
                "denuded_ccn_s": delays[1, 1],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("blocks", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    scan_logs(a.source, a.blocks, a.out / "acquisition.npz")
    evaluate(a.out / "acquisition.npz", a.out)
