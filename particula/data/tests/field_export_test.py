"""Opt-in verification against complete, checksum-pinned beta exports."""

import csv
import os
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")
from particula.data import import_to_parquet, read_measurement  # noqa: E402
from polars.testing import assert_frame_equal  # noqa: E402

FILES = [
    (
        "CPC_3010_data/CPC_3010_data_20220709_Jul.csv",
        33267,
        0,
        "783a3fca68677b2ece216fc15614872c84f0269a176b0e42d29585eae6870650",
    ),
    (
        "CPC_3010_data/CPC_3010_data_20220710_Jul.csv",
        35297,
        0,
        "8c4dd3d899e73e6ee3f60b8f83c5afbc5c51f02935554a089c2dd24e685a019c",
    ),
    (
        "SMPS_data/2022-07-07_095151_SMPS.csv",
        2854,
        203,
        "144e5239097d1219deff7a0eeb465b9a4d67e2ef0b3001d64e98a50819b432bc",
    ),
    (
        "SMPS_data/2022-07-10_094659_SMPS.csv",
        1015,
        203,
        "3f351750f59587793ab559d0729386267266c1441d7dbd47915b185e22a6972a",
    ),
    (
        "chamber_data/2023-09-25_160155_SMPS.csv",
        336,
        216,
        "a280a657729ae214fd694f29c8922d6c476bc444fee0574eeb7e33bfc19531a0",
    ),
]


def _counter_reference(path):
    lines = iter(enumerate(path.read_text().splitlines(), 1))
    for first, text in lines:
        last = first
        if len(text.split(",")) < 5:
            last, continuation = next(lines)
            text += continuation
        epoch, rate, temperature, _, _ = text.split(",")
        yield first, last, int(epoch), float(rate), float(temperature) + 273.15


@pytest.mark.parametrize(("filename", "count", "width", "sha"), FILES)
def test_original_export(tmp_path, filename, count, width, sha):
    """Compare every source row with independently decoded observations."""
    root = os.environ.get("PARTICULA_BETA_FIXTURES")
    if not root:
        pytest.skip(
            "Set PARTICULA_BETA_FIXTURES to the pinned example_data directory"
        )
    path = Path(root) / filename
    profile = "smps-aim" if width else "cpc-3010"
    out = import_to_parquet(
        path, tmp_path / "dataset", profile=profile, batch_scans=541
    )
    assert out.manifest["source_sha256"] == sha
    frame = out.frame
    assert frame.height == count
    if not width:
        reference = list(_counter_reference(path))
        assert (
            frame.select(
                "scan_id",
                "source_line_end",
                pl.col("time_utc").dt.epoch("s"),
                "count_rate",
                "temperature",
            ).rows()
            == reference
        )
        assert frame["repaired"].sum() == (13 if count == 33267 else 0)
        for chunk, batch in ((1, 97), (7, 777), (64, 2500), (1024, 4000)):
            other = read_measurement(
                path,
                profile=profile,
                input_chunk_bytes=chunk,
                batch_scans=batch,
            )
            assert_frame_equal(frame, other.frame)
            assert_frame_equal(out.issues, other.issues)
        return
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = csv.reader(stream)
        for _ in range(24):
            next(records)
        header = next(records)
        a = header.index("Diameter Midpoint (nm)") + 1
        b = header.index("Scan Time (s)")
        source = np.array(
            [[float(x) for x in record[a:b]] for record in records]
        )
    assert source.shape == (count, width)
    np.testing.assert_allclose(
        frame["number_spectrum"].to_numpy(), source * 1e6, rtol=2e-15
    )
    if count == 2854:
        scan_totals = (
            out.integrated_number().collect()["integrated_number"].to_numpy()
        )
        assert (
            np.max(np.abs(scan_totals / frame["total_number"].to_numpy() - 1))
            < 5e-6
        )
        flags = out.flag_summary().collect()
        assert dict(flags.select("vendor_flags", "scans").iter_rows()) == {
            "Incomplete Scan": 2854,
            "Detector aerosol flow rate error": 2854,
            "Sheath flow error": 111,
        }
        policy = out.quality_summary(
            "total_number", exclude_flags=["Sheath flow error"]
        ).collect()
        assert policy["retained"][0] == 2743
        none = out.quality_summary(
            "total_number", exclude_flags=["Incomplete Scan"]
        ).collect()
        assert none.select("valid_samples", "mean").row(0) == (0, None)
