"""Check a full campaign hour against independent readings of both exports."""

import csv
import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("pint")
pytest.importorskip("jsonschema")
from particula.data import (  # noqa: E402
    MeasurementCollection,
    MobilityNumberSpectrum,
    ParticleCountRate,
    import_to_parquet,
    open_measurement,
)

START = datetime(2022, 7, 10, 12, tzinfo=UTC)
STOP = START + timedelta(hours=1)
COUNTER = "CPC_3010_data/CPC_3010_data_20220710_Jul.csv"
SIZER = "SMPS_data/2022-07-10_094659_SMPS.csv"
FILES = [
    (
        "counter",
        COUNTER,
        "cpc-3010",
        "8c4dd3d899e73e6ee3f60b8f83c5afbc5c51f02935554a089c2dd24e685a019c",
    ),
    (
        "sizer",
        SIZER,
        "smps-aim",
        "3f351750f59587793ab559d0729386267266c1441d7dbd47915b185e22a6972a",
    ),
]


def _counter_samples(path):
    """Read the original Unix clock and detector counts/s on July 10."""
    with path.open(newline="") as stream:
        for epoch, rate, _, _, _ in csv.reader(stream):
            yield int(epoch), float(rate)


def _sizer_samples(path):
    """Interpret the local clock as UTC per the pinned donor settings."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(
            row for row in csv.reader(stream) if row[:1] == ["Sample #"]
        )
        for row in csv.DictReader(stream, fieldnames=header):
            stamp = datetime.strptime(
                f"{row['Date']} {row['Start Time']}", "%m/%d/%Y %H:%M:%S"
            ).replace(tzinfo=UTC)
            # The vendor header reports concentration in cm^-3.
            yield (
                int(stamp.timestamp()),
                float(row["Total Conc. (#/cm³)"]) * 1e6,
            )


def _reference_by_minute(samples):
    """Compute the independent sample counts and means in 60 half-open bins."""
    start, stop = int(START.timestamp()), int(STOP.timestamp())
    minute, values = [], []
    for timestamp, value in samples:
        if start <= timestamp < stop:
            minute.append((timestamp - start) // 60)
            values.append(value)
    count = np.bincount(minute, minlength=60)
    total = np.bincount(minute, weights=values, minlength=60)
    mean = np.divide(total, count, out=np.full(60, np.nan), where=count > 0)
    return count, mean


def test_july_campaign_hour(tmp_path):
    """Check every minute and both denominators on reopened Parquet members."""
    raw = os.environ.get("PARTICULA_BETA_FIXTURES")
    if not raw:
        pytest.skip("Set PARTICULA_BETA_FIXTURES to the pinned example_data")
    root = Path(raw)
    saved = os.environ.get("PARTICULA_CAMPAIGN_DATASETS")
    datasets = Path(saved) if saved else tmp_path
    members = {}
    for alias, filename, profile, sha256 in FILES:
        with (root / filename).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == sha256
        if not saved:
            import_to_parquet(
                root / filename,
                datasets / alias,
                profile=profile,
                batch_scans=541,
            )
        member = open_measurement(datasets / alias)
        assert member.manifest["source_sha256"] == sha256
        members[alias] = member
    assert isinstance(members["counter"], ParticleCountRate)
    assert isinstance(members["sizer"], MobilityNumberSpectrum)
    hour = MeasurementCollection(members).select_time(START, STOP)
    fields = {"counter": ["count_rate"], "sizer": ["total_number"]}
    minute = hour.average(
        every="1m", origin=START, weighting="sample", fields=fields
    )
    frame = minute.align_windows(how="grid").collect()
    assert frame["window_start"].to_list() == [
        START + timedelta(minutes=index) for index in range(60)
    ]
    for column, samples, total, occupied in (
        ("counter.count_rate", _counter_samples(root / COUNTER), 1203, 60),
        ("sizer.total_number", _sizer_samples(root / SIZER), 40, 40),
    ):
        count, mean = _reference_by_minute(samples)
        assert count.sum() == total
        assert np.count_nonzero(count) == occupied
        np.testing.assert_array_equal(frame[f"{column}.n"].to_numpy(), count)
        np.testing.assert_array_equal(
            frame[column].is_null().to_numpy(), count == 0
        )
        np.testing.assert_allclose(frame[column].to_numpy(), mean, rtol=1e-12)
    excluded = (
        hour.average(
            every="1m",
            origin=START,
            weighting="sample",
            fields=fields,
            exclude_flags={"sizer": ["Incomplete Scan"]},
        )
        .align_windows(how="grid")
        .collect()
    )
    assert excluded["sizer.total_number.n"].sum() == 0
    assert excluded["sizer.total_number"].null_count() == 60
    assert excluded["counter.count_rate"].equals(frame["counter.count_rate"])
