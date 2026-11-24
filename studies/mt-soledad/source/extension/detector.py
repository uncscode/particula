"""Pair the two CCN paths and aggregate raw and inverted spectra on UTC blocks."""

# %%
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from extract import read_field, verify_archive

DIAM = np.geomspace(30, 310, 84)
FIELDS = [
    "response_function_cpc_count",
    "response_function_ccn",
    "dN_dlog_Dp_DMA_count_0",
    "dN_dlog_Dp_CCN_0",
]


def extract_blocks(archive, output, hours=6):
    verify_archive(archive)
    counts = Counter()
    groups = defaultdict(list)
    accepted = []
    with TemporaryDirectory() as temp, ZipFile(archive) as z:
        tmp = Path(temp)
        src = tmp / "day.cdf"
        for n in sorted(z.namelist()):
            if not any(m in n for m in (".202305", ".202306")) or not n.endswith(
                ".cdf"
            ):
                continue
            counts["dates"] += 1
            src.write_bytes(z.read(n))

            def field(name, dt="<f8"):
                return read_field(src, name, tmp, dt)

            t = field("time", "<i4").astype(int)
            branch = field("denuder_flag", "<i4")
            cvi = field("cvi_flag")
            qc = field("qc_cpc_flag", "<i4")
            s = field("ccn_supersaturation")
            rh = field("sample_relative_humidity")
            T = field("sample_temperature") + 273.15
            p = field("sample_pressure")
            grad = field("ccn_gradient")
            sample_flow = field("sample_flow_ccn")
            sheath_flow = field("sheath_flow_ccn")
            serial = field("response_function_cpc_serial").reshape(len(t), -1)
            bounds = field("diameter_mobility_bounds").reshape(-1, 2)
            dm = np.sqrt(bounds.prod(axis=1))
            w = np.maximum(
                0,
                np.log(np.minimum(360, bounds[:, 1]) / np.maximum(30, bounds[:, 0])),
            )
            spectra = np.array([field(k).reshape(len(t), -1) for k in FIELDS])
            for i in range(len(t) - 3):
                block = slice(i, i + 4)
                if not (
                    np.array_equal(branch[block], [0, 0, 1, 1])
                    and np.all(np.diff(t[block]) == 300)
                    and np.all(s[block] == s[i])
                    and s[i] in (0.4, 0.6, 0.8, 1.0)
                ):
                    continue
                for j in (0, 1):
                    pair = [i + j, i + j + 2]
                    counts["candidate"] += 1
                    if not (
                        np.all(cvi[pair] == 0)
                        and np.all(qc[pair] == 0)
                        and np.all((rh[pair] > 0) & (rh[pair] < 100))
                        and np.all((T[pair] > 263) & (T[pair] < 323))
                        and np.all(abs(0.0777 * grad[pair] - 0.14895 - s[i]) < 0.025)
                    ):
                        continue
                    cpc = spectra[2, pair][:, w > 0]
                    if not np.isfinite(cpc).all() or min(cpc @ w[w > 0]) <= 0:
                        continue
                    diff = abs(cpc[0] - cpc[1]) @ w[w > 0] / np.mean(cpc @ w[w > 0])
                    if diff > 0.30:
                        continue
                    rows = np.stack(
                        [
                            [
                                np.interp(np.log(DIAM), np.log(dm[::-1]), f[row, ::-1])
                                for f in spectra
                            ]
                            for row in pair
                        ]
                    )
                    # Use identical valid pairs per diameter for both branches and both signals.
                    ok = np.isfinite(rows).all(axis=(0, 1)) & (rows >= 0).all(
                        axis=(0, 1)
                    )
                    if ok.sum() < 0.8 * len(DIAM):
                        continue
                    rows[:, :, ~ok] = np.nan
                    large = (DIAM >= 120) & (DIAM <= 250)
                    serial_pair = np.stack(
                        [
                            np.interp(
                                np.log(DIAM),
                                np.log(dm[::-1]),
                                serial[row, ::-1],
                            )
                            for row in pair
                        ]
                    )
                    full = large & np.isfinite(serial_pair).all(axis=0) & ok
                    large_stats = np.full((2, 5), np.nan)
                    if full.sum() >= 5:
                        for b in (0, 1):
                            large_stats[b] = [
                                np.mean(rows[b, 0, full]),
                                np.mean(rows[b, 1, full]),
                                np.mean(serial_pair[b, full]),
                                sample_flow[pair[b]],
                                sheath_flow[pair[b]],
                            ]
                    groups[(t[i] // (3600 * hours), int(round(s[i] * 10)), j)].append(
                        (
                            rows,
                            np.array(
                                [
                                    rh[pair].mean(),
                                    T[pair].mean(),
                                    np.nanmean(p[pair]),
                                    (0.0777 * grad[pair] - 0.14895).mean(),
                                ]
                            ),
                            large_stats,
                        )
                    )
                    counts["retained"] += 1
                    accepted.append((t[i], int(round(s[i] * 10)), j))
    keys = sorted(groups)
    sums = []
    valid = []
    env = []
    numbers = []
    extras = []
    for key in keys:
        data, forcing, diagnostic = zip(*groups[key], strict=True)
        extras.append(np.nanmean(diagnostic, axis=0))
        sums.append(np.nansum(data, axis=0))
        valid.append(np.isfinite(np.asarray(data)[:, 0, 0]).sum(axis=0))
        env.append(np.mean(forcing, axis=0))
        numbers.append(len(data))
    np.savez_compressed(
        output,
        keys=keys,
        sums=sums,
        valid=valid,
        env=env,
        n=numbers,
        diam=DIAM,
        diagnostic=extras,
        hours=hours,
        accepted=accepted,
    )
    print(counts, len(keys))


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("archive", type=Path)
    arg.add_argument("out", type=Path)
    arg.add_argument("--hours", type=int, choices=(3, 6), default=6)
    args = arg.parse_args()
    extract_blocks(args.archive, args.out, args.hours)
