"""Extract restricted-size matched pairs from the UC San Diego daily archive."""

# %%
import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np

SHA = "9f94d7f3e1e66bbb7d5149fd3b9e3f2eea776dd9f898f3af6d185ceca9c53800"


def read_field(source, name, directory, dtype="<f8"):
    """Read one numeric HDF5 field and convert archive fills to NaN."""
    raw = directory / "field.bin"
    subprocess.run(  # noqa: S603
        [
            "h5dump",
            "-d",
            name,
            "-b",
            "LE",
            "-o",
            str(raw),
            str(source),
        ],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    out = np.fromfile(raw, dtype=dtype).astype(float)
    out[np.abs(out) > 1e30] = np.nan
    return out


# %%
def verify_archive(archive):
    """Verify the identity of the ZIP before reading its members."""
    with archive.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SHA:
            raise ValueError("Author archive checksum mismatch")


def extract(archive, output):
    """Build compact paired forcing and observations for both inversions."""
    verify_archive(archive)
    output.mkdir(parents=True, exist_ok=True)
    rows, counts = [], Counter()
    with tempfile.TemporaryDirectory() as scratch, ZipFile(archive) as z:
        directory = Path(scratch)
        source = directory / "day.cdf"
        for member in sorted(n for n in z.namelist() if n.endswith(".cdf")):
            counts["daily_files"] += 1
            source.write_bytes(z.read(member))

            def get(name, dtype="<f8"):
                return read_field(source, name, directory, dtype)

            time = get("time", "<i4")
            branch, qc = get("denuder_flag", "<i4"), get("qc_cpc_flag", "<i4")
            cvi, rh, temp = [
                get(s)
                for s in (
                    "cvi_flag",
                    "sample_relative_humidity",
                    "sample_temperature",
                )
            ]
            ss, gradient = get("ccn_supersaturation"), get("ccn_gradient")
            lower, upper = get("diameter_mobility_bounds").reshape(-1, 2).T
            keep = (upper > 30) & (lower < 360)
            edges = np.column_stack(
                [np.maximum(30, lower[keep]), np.minimum(360, upper[keep])]
            )
            widths = np.log(edges[:, 1] / edges[:, 0])
            fields = np.array(
                [
                    [
                        get(f"dN_dlog_Dp_{channel}_{order}").reshape(time.size, -1)[
                            :, keep
                        ]
                        for channel in ("DMA_count", "CCN")
                    ]
                    for order in (0, 2)
                ]
            )
            for i in range(len(time) - 3):
                block = slice(i, i + 4)
                if not (
                    np.array_equal(branch[block], [0, 0, 1, 1])
                    and np.all(np.diff(time[block]) == 300)
                    and np.all(ss[block] == ss[i])
                    and min(abs(ss[i] - 0.4), abs(ss[i] - 0.6)) < 0.025
                ):
                    continue
                for j in (i, i + 1):
                    idx = [j, j + 2]
                    counts["raw_pairs"] += 1
                    if not (
                        np.all(qc[idx] == 0)
                        and np.all(cvi[idx] == 0)
                        and np.all((rh[idx] > 0) & (rh[idx] < 100))
                        and np.all((temp[idx] > -10) & (temp[idx] < 50))
                        and np.all(
                            abs(0.0777 * gradient[idx] - 0.14895 - ss[idx]) < 0.025
                        )
                    ):
                        counts["flag_or_forcing"] += 1
                        continue
                    density = fields[:, :, idx, :]
                    if not np.isfinite(density).all():
                        counts["missing_inversion"] += 1
                        continue
                    # Nonnegative population is required as model forcing. The
                    # signed CCN inversion retains its original integral.
                    cpc = np.maximum(density[:, 0], 0)
                    correction = abs((cpc - density[:, 0]) @ widths)
                    total = cpc @ widths
                    obs = density[:, 1] @ widths
                    if (
                        np.any(total <= 0)
                        or np.any(obs <= 0)
                        or np.any(correction > 0.02 * total)
                    ):
                        counts["invalid_population"] += 1
                        continue
                    mismatch = abs(cpc[0, 1] - cpc[0, 0]) @ widths / np.mean(total[0])
                    if mismatch > 0.30:
                        counts["cpc_mismatch"] += 1
                        continue
                    counts["pairs_with_negative_cpc"] += int(np.any(density[:, 0] < 0))
                    counts["pairs_with_negative_ccn"] += int(np.any(density[:, 1] < 0))
                    band_widths = np.array(
                        [
                            np.maximum(
                                0,
                                np.log(
                                    np.minimum(edges[:, 1], high)
                                    / np.maximum(edges[:, 0], low)
                                ),
                            )
                            for low, high in ((40, 90), (120, 250))
                        ]
                    )
                    band_obs = density[:, 1] @ band_widths.T
                    rows.append(
                        (
                            time[idx],
                            ss[idx],
                            0.0777 * gradient[idx] - 0.14895,
                            rh[idx],
                            temp[idx] + 273.15,
                            edges.copy(),
                            cpc,
                            obs,
                            mismatch,
                            band_obs,
                        )
                    )
    keys = (
        "time",
        "setpoint",
        "s",
        "rh",
        "T",
        "bounds",
        "cpc",
        "obs",
        "mismatch",
        "band_obs",
    )
    # Some instrument days use different grids. Store bounds per pair.
    data = {
        key: np.stack(col)
        for key, col in zip(keys, zip(*rows, strict=True), strict=True)
    }
    np.savez_compressed(output / "pairs.npz", **data)
    counts["retained_pairs"] = len(rows)
    (output / "extraction.json").write_text(json.dumps(dict(counts), indent=2) + "\n")
    print(counts, data["cpc"].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.archive, args.output)
