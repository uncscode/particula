"""Scientific and persistence checks for multi-instrument comparisons."""

import csv
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("pint")
pytest.importorskip("jsonschema")
from particula.data import (  # noqa: E402
    CampaignEvent,
    MeasurementCollection,
    MobilityNumberSpectrum,
    ParticleCountRate,
    describe_profile,
    read_measurement,
)
from particula_docs import documentation_path  # noqa: E402

START = datetime(2022, 7, 10, 12, tzinfo=UTC)
MOBILITY = documentation_path() / "data/measurement-import/mobility.csv"


def _counter(
    tmp_path, alias, readings, *, count=False, rate_field="count_rate"
):
    """Read (timestamp offset in s, count/s, verified live time in s)."""
    profile = describe_profile("cpc-3010")
    profile["fields"][0]["name"] = rate_field
    profile["on_bad_record"] = "error"
    profile["fields"].append(
        {
            "name": "live_seconds",
            "column": 3,
            "description": "Verified live time",
            "quantity": "duration",
            "source_unit": "s",
            "output_unit": "s",
            "dtype": "float64",
        }
    )
    if count:
        profile["fields"][0].update(
            name="pulses",
            quantity="count",
            source_unit="count",
            output_unit="count",
        )
    raw = tmp_path / f"{alias}.csv"
    epoch = int(START.timestamp())
    raw.write_text(
        "".join(
            f"{epoch + offset},{rate},17,{seconds},0\n"
            for offset, rate, seconds in readings
        )
    )
    return read_measurement(raw, profile=profile)


def _minute_grid(campaign, fields, *, start=START, stop=None, **policy):
    selected = campaign.select_time(start, stop or start + timedelta(minutes=2))
    return (
        selected.average(every="1m", origin=start, fields=fields, **policy)
        .align_windows(how="grid")
        .collect()
    )


def test_sample_scopes_and_time_boundaries(tmp_path):
    """Equal clocks on distinct records or deployments preserve each reading."""
    inlet = _counter(
        tmp_path, "inlet", [(0, 10, 1), (0, 20, 1), (60, 0, 1), (120, 1, 1)]
    )
    outlet = _counter(tmp_path, "outlet", [(0, 90, 1), (60, ".", 1)])
    event = CampaignEvent(
        "zero_check",
        START,
        START + timedelta(minutes=1),
        "inlet",
        field="count_rate",
    )
    campaign = MeasurementCollection(
        {"inlet": inlet, "outlet": outlet}, events=[event]
    )
    fields = {"inlet": ["count_rate", "temperature"], "outlet": ["count_rate"]}
    unmasked = _minute_grid(campaign, fields, weighting="sample")
    masked = _minute_grid(
        campaign, fields, weighting="sample", exclude_events=["zero_check"]
    )
    assert unmasked["inlet.count_rate"].to_list() == [15.0, 0.0]
    assert unmasked["inlet.count_rate.n"].to_list() == [2, 1]
    assert masked["inlet.count_rate"].to_list() == [None, 0.0]
    assert masked["inlet.count_rate.n"].to_list() == [0, 1]
    assert masked["inlet.temperature.n"].to_list() == [2, 1]
    assert masked["outlet.count_rate"].to_list() == [90.0, None]
    assert masked["outlet.count_rate.n"].to_list() == [1, 0]
    assert inlet.frame.height == 4


def test_verified_exposure_across_window_and_selection(tmp_path):
    """Use the length of measured intersection with each analysis minute."""
    inlet = _counter(tmp_path, "inlet", [(-10, 20, 20), (45, 100, 75)])
    campaign = MeasurementCollection({"inlet": inlet})
    fields = {"inlet": ["count_rate"]}
    exposed = _minute_grid(
        campaign,
        fields,
        weighting="exposure",
        exposure={"inlet": "live_seconds"},
    )
    # First minute has 10 s at 20 counts/s and 15 s at 100 counts/s.
    np.testing.assert_allclose(
        exposed["inlet.count_rate"].to_numpy(), [68.0, 100.0], rtol=2e-15
    )
    assert exposed["inlet.count_rate.n"].to_list() == [2, 1]
    np.testing.assert_allclose(
        exposed["inlet.count_rate.exposure_s"].to_numpy(),
        [25.0, 60.0],
        rtol=2e-15,
    )
    assert exposed["inlet.count_rate.coverage"].max() <= 1.0
    np.testing.assert_allclose(
        exposed["inlet.count_rate.coverage"].to_numpy(), [25 / 60, 1.0]
    )
    sampled = _minute_grid(campaign, fields, weighting="sample")
    assert sampled["inlet.count_rate"].to_list() == [100.0, None]


def test_unverified_and_overlapping_support(tmp_path):
    """Reject duration ambiguity before claiming a measured coverage."""
    inlet = _counter(tmp_path, "inlet", [(0, 25, 40), (30, 15, 10)])
    campaign = MeasurementCollection({"inlet": inlet})
    fields = {"inlet": ["count_rate"]}
    with pytest.raises(ValueError, match="(?i)exposure|duration"):
        _minute_grid(campaign, fields, weighting="exposure")
    with pytest.raises(ValueError, match="(?i)overlap"):
        _minute_grid(
            campaign,
            fields,
            weighting="exposure",
            exposure={"inlet": "live_seconds"},
        )
    with pytest.raises(ValueError, match="(?i)unit|second|duration"):
        _minute_grid(
            campaign,
            fields,
            weighting="exposure",
            exposure={"inlet": "temperature"},
        )


def test_exposure_validates_only_retained_intervals(tmp_path):
    """Clip the crossing scan and ignore invalid support outside the analysis."""
    readings = [
        (-100, 5, "."),
        (-10, 20, 20),
        (5, 999, "."),
        (20, 40, 5),
        (60, 7, "."),
        (70, 4, 20),
        (75, 8, 20),
    ]
    raw = _counter(tmp_path, "inlet", readings)
    profile = raw.manifest["profile"]
    profile["vendor_flags"] = {"column": 1, "separator": ";"}
    product = read_measurement(tmp_path / "inlet.csv", profile=profile)
    selected = MeasurementCollection({"inlet": product}).select_time(
        START, START + timedelta(minutes=1)
    )
    fields = {"inlet": ["count_rate"]}
    exposure = {"inlet": "live_seconds"}
    flags = {"inlet": ["999"]}
    average = selected.average(
        every="1m",
        origin=START,
        fields=fields,
        weighting="exposure",
        exposure=exposure,
        exclude_flags=flags,
    )
    with pytest.raises(ValueError, match="Invalid or overlapping"):
        selected.average(
            every="1m",
            origin=START,
            fields=fields,
            weighting="exposure",
            exposure=exposure,
        ).align_windows().collect()
    # The 10 s before the crossing scan ends and the later 5 s both count.
    row = average.align_windows().collect().row(0, named=True)
    assert row["inlet.count_rate"] == pytest.approx((20 * 10 + 40 * 5) / 15)
    assert row["inlet.count_rate.exposure_s"] == 15
    assert row["inlet.count_rate.n"] == 2

    fields["inlet"].append("temperature")
    flags["inlet"].append("40")
    exposure["inlet"] = "temperature"
    average.save(tmp_path / "windows")
    restored = type(average).open(tmp_path / "windows")
    assert restored.manifest["fields"] == {"inlet": ["count_rate"]}
    assert restored.manifest["exposure"] == {"inlet": "live_seconds"}
    assert restored.manifest["exclude_flags"] == {"inlet": ["999"]}
    assert restored.align_windows().collect().row(0, named=True) == row


def test_save_retry_rejects_incomplete_payloads(tmp_path):
    """An intact root manifest cannot stand in for a missing nested dataset."""
    product = _counter(tmp_path, "meter", [(0, 10, 1)])
    collection = MeasurementCollection({"meter": product}).select_time(
        START, START + timedelta(minutes=1)
    )
    windows = collection.average(
        every="1m",
        origin=START,
        fields={"meter": ["count_rate"]},
        weighting="sample",
    )
    for result, name, payload in [
        (collection, "campaign", "member-000000"),
        (windows, "comparison", "windows.parquet"),
        (collection, "nested", "member-000000/issues.parquet"),
    ]:
        destination = tmp_path / name
        result.save(destination)
        result.save(destination)
        (destination / payload).rename(tmp_path / f"removed-{name}")
        with pytest.raises(FileExistsError, match="incomplete or invalid"):
            result.save(destination)
        assert not (destination / payload).exists()


def test_time_bounds_reject_precision_loss(tmp_path):
    """Preserve nanosecond readings and reject unrepresentable UTC bounds."""
    source = tmp_path / "counter.csv"
    source.write_text("1700000040.000000123,12,17,1,0\n")
    campaign = MeasurementCollection(
        {"meter": read_measurement(source, profile="cpc-3010")}
    )
    low, high = "2023-11-14T22:14:00", "2023-11-14T22:14:01Z"
    for fraction in (".0000002Z", ",00000020+00:00", ".000000200Z"):
        with pytest.raises(ValueError, match="microsecond"):
            campaign.select_time(low + fraction, high)
    reading = campaign.select_time(low + ".000000000Z", high)["meter"]
    assert (
        reading.frame["time_utc"].cast(pl.Int64).item() == 1700000040000000123
    )
    assert campaign.select_time(low + ".000001000Z", high)[
        "meter"
    ].frame.is_empty()


def test_window_start_as_measurement_field(tmp_path):
    """Preserve a physical channel named like the public UTC window key."""
    inlet = _counter(
        tmp_path, "inlet", [(0, 10, 1), (1, 30, 3)], rate_field="window_start"
    )
    for weighting, expected in [("sample", 20), ("exposure", 25)]:
        summary = _minute_grid(
            MeasurementCollection({"inlet": inlet}),
            {"inlet": ["window_start"]},
            weighting=weighting,
            exposure={"inlet": "live_seconds"},
        )
        assert summary["window_start"][0] == START
        assert summary["inlet.window_start"].to_list() == [expected, None]


def test_ambiguous_clocks_and_duplicate_scan_ids(tmp_path):
    """Reject a naive interval bound and repeated source-scoped identifiers."""
    inlet = _counter(tmp_path, "inlet", [(0, 25, 1)])
    with pytest.raises(ValueError, match="(?i)clock|aware|zone|UTC"):
        MeasurementCollection({"inlet": inlet}).select_time(
            START.replace(tzinfo=None), START + timedelta(minutes=1)
        )
    campaign = MeasurementCollection({"inlet": inlet}).select_time(
        START, START + timedelta(minutes=2)
    )
    with pytest.raises(ValueError, match="(?i)clock|aware|zone|UTC"):
        campaign.average(
            every="1m",
            origin=START.replace(tzinfo=None),
            fields={"inlet": ["count_rate"]},
            weighting="sample",
        )
    duplicate = type(inlet)(
        pl.concat([inlet.lazy_frame] * 2), inlet.lazy_issues, inlet.manifest
    )
    with pytest.raises(ValueError, match="(?i)duplicate|unique"):
        _minute_grid(
            MeasurementCollection({"inlet": duplicate}),
            {"inlet": ["count_rate"]},
            weighting="sample",
        )


def test_relocated_collection(tmp_path):
    """Open moved products by their campaign names, retaining types and events."""
    inlet = _counter(tmp_path, "one", [(0, 15, 1)])
    outlet = _counter(tmp_path, "two", [(0, 18, 1)])
    sizer = read_measurement(MOBILITY, profile="smps-aim")
    event = CampaignEvent(
        "zero_check", START, START + timedelta(seconds=1), "inlet"
    )
    before = MeasurementCollection(
        {"inlet": inlet, "outlet": outlet, "sizer": sizer}, events=[event]
    )
    path = tmp_path / "campaign"
    before.save(path)
    moved = path.rename(tmp_path / "moved")
    after = MeasurementCollection.open(moved)
    assert isinstance(after["inlet"], ParticleCountRate)
    assert isinstance(after["outlet"], ParticleCountRate)
    assert isinstance(after["sizer"], MobilityNumberSpectrum)
    for alias in before:
        assert (
            after[alias].manifest["source_sha256"]
            == before[alias].manifest["source_sha256"]
        )
        assert after[alias].product_id == before[alias].product_id
        assert after[alias].grids == before[alias].grids
        assert after[alias].frame.equals(before[alias].frame)
        assert after[alias].issues.equals(before[alias].issues)
    fields = {"inlet": ["count_rate"], "outlet": ["count_rate"]}
    ambient = _minute_grid(
        after, fields, weighting="sample", exclude_events=["zero_check"]
    )
    assert ambient["inlet.count_rate.n"].sum() == 0
    assert ambient["outlet.count_rate.n"].sum() == 1
    member_manifest = next(
        path
        for path in moved.rglob("manifest.json")
        if json.loads(path.read_bytes()).get("product_id") == inlet.product_id
    )
    member_manifest.rename(member_manifest.with_suffix(".missing"))
    with pytest.raises((OSError, ValueError)):
        MeasurementCollection.open(moved)


def test_channel_means_require_compatible_grids(tmp_path):
    """Count each observed channel and reject unequal coordinates of equal width."""
    with MOBILITY.open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.reader(stream))
    header, scan = source[-2:]
    first = header.index("Diameter Midpoint (nm)") + 1
    last = header.index("Scan Time (s)")
    changed = scan.copy()
    changed[first] = "."
    changed[first + 1] = str(3 * float(scan[first + 1]))
    raw = tmp_path / "spectra.csv"
    with raw.open("w", newline="") as stream:
        csv.writer(stream).writerows([*source, changed])
    stamp = datetime.strptime(
        f"{scan[1]} {scan[2]}", "%m/%d/%Y %H:%M:%S"
    ).replace(tzinfo=UTC)
    begin = stamp.replace(second=0)
    fields = {"sizer": ["number_spectrum"]}
    member = read_measurement(raw, profile="smps-aim")
    frame = _minute_grid(
        MeasurementCollection({"sizer": member}),
        fields,
        start=begin,
        weighting="sample",
    )
    count = frame["sizer.number_spectrum.n"].to_list()[0]
    means = frame["sizer.number_spectrum"].to_list()[0]
    assert count == [1, *([2] * (last - first - 1))]
    assert means[0] == pytest.approx(float(scan[first]) * 1e6)
    assert means[1] == pytest.approx(float(scan[first + 1]) * 2e6)
    second_grid = header.copy()
    second_grid[first:last] = [
        str(2 * float(diameter)) for diameter in header[first:last]
    ]
    with raw.open("w", newline="") as stream:
        csv.writer(stream).writerows(
            [*source, *source[:-2], second_grid, changed]
        )
    member = read_measurement(raw, profile="smps-aim")
    assert len(member.grids) == 2
    with pytest.raises(ValueError, match="(?i)grid"):
        _minute_grid(
            MeasurementCollection({"sizer": member}),
            fields,
            start=begin,
            weighting="sample",
        )


def test_nearest_matches_within_tolerance(tmp_path):
    """Retain signed clock offsets and a null when the match is too distant."""
    left = _counter(tmp_path, "left", [(3, 10, 1), (5, 12, 1), (30, 20, 1)])
    right = _counter(tmp_path, "right", [(2, 15, 1), (7, 40, 1)])
    campaign = MeasurementCollection({"left": left, "right": right})
    matches = campaign.match_nearest("left", "right", tolerance="2s").collect()
    assert matches["left.count_rate"].to_list() == [10, 12, 20]
    assert matches["right.count_rate"].to_list() == [15, 40, None]
    assert matches["right.time_utc"].to_list() == [
        START + timedelta(seconds=2),
        START + timedelta(seconds=7),
        None,
    ]
    assert matches["time_offset_ns"].to_list() == [-(10**9), 2 * 10**9, None]


def test_event_clips_exposure_across_windows(tmp_path):
    """Subtract the union of scoped exclusions from the observed support."""
    inlet = _counter(tmp_path, "one", [(45, 20, 75)])
    events = [
        CampaignEvent(
            "zero_check",
            START + timedelta(seconds=50),
            START + timedelta(seconds=70),
            "inlet",
        ),
        CampaignEvent(
            "zero_check",
            START + timedelta(seconds=65),
            START + timedelta(seconds=75),
            "inlet",
            field="count_rate",
        ),
    ]
    masked = _minute_grid(
        MeasurementCollection({"inlet": inlet}, events=events),
        {"inlet": ["count_rate", "temperature"]},
        weighting="exposure",
        exposure={"inlet": "live_seconds"},
        exclude_events=("zero_check",),
    )
    assert masked["inlet.count_rate.exposure_s"].to_list() == [5.0, 45.0]
    assert masked["inlet.temperature.exposure_s"].to_list() == [5.0, 50.0]
    assert masked["inlet.count_rate"].to_list() == [20.0, 20.0]


def test_actual_detector_counts_add_within_window(tmp_path):
    """Preserve pulse totals, a measured zero, and the empty window."""
    observation = _counter(
        tmp_path, "pulses", [(1, 2, 1), (10, 6, 1), (61, 0, 1)], count=True
    )
    campaign = MeasurementCollection({"detector": observation})
    fields = {"detector": ["pulses"]}
    sampled = _minute_grid(
        campaign, fields, stop=START + timedelta(minutes=3), weighting="sample"
    )
    assert sampled["detector.pulses.n"].to_list() == [2, 1, 0]
    assert sampled["detector.pulses"].to_list() == [8, 0, None]
    with pytest.raises(ValueError, match="(?i)exposure|interval|count"):
        _minute_grid(
            campaign,
            fields,
            weighting="exposure",
            exposure={"detector": "live_seconds"},
        )


def test_offset_grid_round_trip_with_verified_coverage(tmp_path):
    """Preserve a shifted window origin, clipped coverage and scalar units."""
    from particula.data import WindowComparison

    product = _counter(tmp_path, "meter", [(0, 10, 10), (30, 50, 30)])
    field = {"meter": ["count_rate"]}
    selected = MeasurementCollection({"meter": product}).select_time(
        START, START + timedelta(minutes=1)
    )
    origin = START + timedelta(seconds=15)
    windows = selected.average(
        every="1m",
        origin=origin,
        weighting="sample",
        fields=field,
        exposure={"meter": "live_seconds"},
    )
    data = windows["meter"].collect()
    assert data["meter.count_rate"].to_list() == [10.0, 50.0]
    np.testing.assert_allclose(
        data["meter.count_rate.coverage"], [10 / 15, 30 / 45]
    )
    windows.save(tmp_path / "means")
    restored = WindowComparison.open(tmp_path / "means")
    assert restored.manifest["parents"] == {"meter": product.product_id}
    assert restored.manifest["units"] == {"meter.count_rate": "count/s"}
    assert restored.align_windows().collect().equals(data)
    with pytest.raises(ValueError, match="zone"):
        selected.average(
            every="1m",
            origin=origin.replace(tzinfo=None),
            weighting="sample",
            fields=field,
        )
