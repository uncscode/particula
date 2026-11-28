"""Behavioral checks for profiles, source spans, grids and Parquet publication."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

pl = pytest.importorskip("polars")
pytest.importorskip("pint")
Draft202012Validator = pytest.importorskip("jsonschema").Draft202012Validator
from particula.data import (  # noqa: E402
    Measurement,
    MeasurementReader,
    MobilityNumberSpectrum,
    ParticleCountRate,
    ProfileError,
    describe_profile,
    import_to_parquet,
    migrate_settings,
    open_measurement,
    read_measurement,
    validate_profile,
)
from particula.instruments import (  # noqa: E402
    Instrument,
    read_mobility_sizer,
    read_particle_counter,
)
from polars.testing import assert_frame_equal  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[3] / "docs/data/measurement-import"


def test_portable_profiles(tmp_path):
    """YAML/JSON normalize to identical source interpretations and digests."""
    Draft202012Validator.check_schema(describe_profile())
    profile = describe_profile("cpc-3010")
    installed = files("particula.data").joinpath("profiles")
    assert (
        yaml.safe_load(installed.joinpath("cpc-3010.yaml").read_text())
        == profile
    )
    source = FIXTURES / "counter.csv"
    for name, text in (
        ("custom.yaml", yaml.safe_dump(profile)),
        ("custom.json", json.dumps(profile)),
    ):
        path = tmp_path / name
        path.write_text(text)
        validate_profile(path, source=source).raise_for_errors()
        result = read_measurement(source, profile=path)
        builtin = read_particle_counter(source)
        assert result.manifest == builtin.manifest
        assert_frame_equal(result.frame, builtin.frame)
        assert result.frame["temperature"][0] == pytest.approx(290.25)


@pytest.mark.parametrize(
    ("suffix", "text"),
    [
        ("yaml", "x: 1\nx: 2"),
        ("json", '{"x":1,"x":2}'),
        ("yaml", "x: &v [1]\ny: *v"),
        ("yaml", "x: !custom 22"),
        ("yaml", "x: .nan"),
    ],
)
def test_invalid_profile_serialization(tmp_path, suffix, text):
    """Reject constructs which cannot represent an unambiguous portable tree."""
    path = tmp_path / f"profile.{suffix}"
    path.write_text(text)
    with pytest.raises(ProfileError):
        validate_profile(path).raise_for_errors()


def test_profile_diagnostics(tmp_path):
    """Return field-level diagnostics for invalid units, names and headers."""
    p = describe_profile("cpc-3010")
    p["fields"][0]["output_unit"] = "1/m^3"
    assert validate_profile(p).errors[0].path == "/fields/0/output_unit"
    p = describe_profile("cpc-3010")
    p["fields"][0]["name"] = "time_utc"
    assert validate_profile(p).errors[0].code == "name_conflict"
    p["schema_version"] = "4"
    assert validate_profile(p).errors[0].path == "/schema_version"
    p = describe_profile("cpc-3010")
    p["on_bad_record"] = "error"
    p["input"]["header_record"] = 0
    p["fields"][0]["column"] = "rate"
    path = tmp_path / "ambiguous.csv"
    path.write_text("date,rate,temp,rate,aux\n")
    assert validate_profile(p, source=path).errors[0].code == "source_column"


@pytest.mark.parametrize(
    ("chunk", "flush"), [(1, 1), (2, 3), (7, 11), (1024, 100)]
)
def test_repair_spans_and_reopen(tmp_path, chunk, flush):
    """Recover adjacent splits and isolate invalid or ambiguous continuations."""
    path = tmp_path / "sample.csv"
    path.write_bytes(
        b"1657343115,24951,\r\n17.0,1,0.017\r\n1657343118,24627,\n17.0,1,0.017\n"
        b"1657343121,20,\n1657343122,30,21.0,1,0.017\n1657343123,4,"
    )
    result = import_to_parquet(
        path,
        tmp_path / "out",
        profile="cpc-3010",
        input_chunk_bytes=chunk,
        batch_scans=flush,
    )
    assert isinstance(result, ParticleCountRate)
    assert result.frame.select(
        "scan_id", "source_line_end", "repaired", "count_rate"
    ).rows() == [(1, 2, True, 24951), (3, 4, True, 24627), (6, 6, False, 30)]
    assert result.frame["time_utc"].dt.epoch("s").to_list() == [
        1657343115,
        1657343118,
        1657343122,
    ]
    assert (
        result.manifest["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert result.manifest["issue_counts"] == {
        "split_record_repaired": 2,
        "unresolved_fragment": 2,
    }
    assert result.issues["source_line_start"].to_list() == [1, 3, 5, 7]
    assert result.available_operations()["number_concentration"] == (
        "sample_flow",
    )
    assert result.filter(pl.col("count_rate") < 100).frame[
        "scan_id"
    ].to_list() == [6]
    with pytest.raises(ValueError, match="count_rate"):
        ParticleCountRate(
            result.lazy_frame.drop("count_rate"),
            result.lazy_issues,
            result.manifest,
        )
    with pytest.raises(ValueError, match="mobility"):
        MobilityNumberSpectrum(
            result.lazy_frame, result.lazy_issues, result.manifest
        )
    # A separate source never resumes the preceding file's trailing fragment.
    second = tmp_path / "second.csv"
    second.write_text("17.0,1,0.017\n")
    assert read_particle_counter(second).frame.is_empty()


def test_atomic_retry_changed_input_and_strict_policy(tmp_path, monkeypatch):
    """A late write error exposes no dataset and a conflicting retry is safe."""
    src = tmp_path / "counter.csv"
    src.write_bytes((FIXTURES / "counter.csv").read_bytes())
    dest = tmp_path / "saved"
    strict = describe_profile("cpc-3010")
    strict["on_bad_record"] = "error"
    with pytest.raises(ProfileError, match="Lines 1"):
        import_to_parquet(src, dest, profile=strict)
    writer = pl.DataFrame.write_parquet

    def fail(self, path, *args, **kwargs):
        if Path(path).name == "part-000000001.parquet":
            raise OSError("injected failure")
        return writer(self, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pl.DataFrame, "write_parquet", fail)
        with pytest.raises(OSError, match="injected failure"):
            import_to_parquet(src, dest, profile="cpc-3010", batch_scans=1)
    assert not dest.exists()
    assert not list(tmp_path.glob(".measurement-*"))
    saved = import_to_parquet(src, dest, profile="cpc-3010", batch_scans=2)
    retry = import_to_parquet(src, dest, profile="cpc-3010")
    assert_frame_equal(saved.frame, retry.frame)
    with src.open("ab") as f:
        f.write(b"1657343200,10,21.0,1,0.017\n")
    with pytest.raises(FileExistsError):
        import_to_parquet(src, dest, profile="cpc-3010")
    assert open_measurement(dest).frame.height == 20


def test_canonical_fast_path_and_fallback(tmp_path):
    """Compare vectorized, byte-at-a-time and independent minute controls."""
    src = tmp_path / "pattern.csv"
    raw = "".join(
        f"{1700000040 + i},{100 + i % 60},17.0,1,0.017\n" for i in range(600)
    )
    src.write_text(raw + "1700000640,222,\n17.0,1,0.017\n")
    slow = read_particle_counter(src, input_chunk_bytes=1, batch_scans=53)
    fast = read_particle_counter(src, input_chunk_bytes=2048, batch_scans=61)
    assert_frame_equal(fast.frame, slow.frame)
    assert_frame_equal(fast.issues, slow.issues)
    assert slow.manifest == fast.manifest
    frame = fast.frame.head(600)
    assert frame["count_rate"].to_list() == list(range(100, 160)) * 10
    assert frame["time_utc"].dt.epoch("s").to_list() == list(
        range(1700000040, 1700000640)
    )
    assert fast.frame.row(-1, named=True)["source_line_end"] == 602


def test_quote_encoding_and_clocks(tmp_path):
    """Carry quoted physical lines, UTF-8, time precision and validity masks."""
    p = describe_profile("cpc-3010")
    p["on_bad_record"] = "error"
    p["input"].update(header_record=0, encoding="utf-8-sig")
    p["time"] = {
        "kind": "calendar",
        "format": "%Y-%m-%d %H:%M:%S",
        "timezone": "America/Denver",
        "columns": [0],
    }
    src = tmp_path / "trace.csv"
    src.write_text(
        'clock,rate,temp,unused,unused2\n2022-07-07 09:00:00,.,17.1,"µ,\nline",x\n',
        encoding="utf-8-sig",
    )
    out = Instrument(p).read(src, input_chunk_bytes=1)
    assert out.frame["time_utc"].dt.hour()[0] == 15
    assert out.frame["count_rate"][0] is None
    assert out.frame["source_line_end"][0] == 3
    assert out.issues["code"][0] == "missing_token"
    for t in ("2022-11-06 01:30:00", "2022-03-13 02:15:00"):
        src.write_text(f"clock,rate,temp,x,y\n{t},5,17,0,0\n")
        with pytest.raises(ProfileError, match="Ambiguous or nonexistent"):
            Instrument(p).read(src)
    with pytest.raises(ProfileError, match="byte limit"):
        Instrument(p).read(src, max_record_bytes=7)
    p = describe_profile("cpc-3010")
    p["on_bad_record"] = "error"
    src.write_text("1700000040.000000123,12,17.1,0,0")
    assert (
        Instrument(p).read(src).frame["time_utc"].cast(pl.Int64)[0]
        == 1700000040000000123
    )


def test_arrays_units_masks_and_grid_identity(tmp_path):
    """Round-trip a masked 2×2 signal; coordinates distinguish equal shapes."""
    p = describe_profile("cpc-3010")
    p["on_bad_record"] = "error"
    field = dict(p.pop("fields")[1])
    field.pop("column")
    field.update(
        name="matrix",
        columns=[1, 2, 3, 4],
        axes=[
            {"name": "x", "unit": "m", "values": [0, 1]},
            {"name": "y", "unit": "m", "values": [0, 2]},
        ],
    )
    p["arrays"] = [field]
    src = tmp_path / "matrix.csv"
    src.write_text("1700000040,0,.,20,30\n")
    result = import_to_parquet(src, tmp_path / "array", profile=p)
    assert type(result) is Measurement
    assert result.frame["matrix"].dtype == pl.Array(pl.Float64, (2, 2))
    assert result.frame["matrix"].to_list() == [
        [[273.15, None], [293.15, 303.15]]
    ]
    assert result.issues["code"].to_list() == ["missing_token"]
    field["axes"][1]["values"] = [0, 3]
    changed = read_measurement(src, profile=p)
    assert list(result.grids) != list(changed.grids)
    field["columns"].pop()
    assert validate_profile(p).errors[0].code == "array_shape"


def test_mobility_spectrum_and_grid_validation(tmp_path):
    """Preserve AIM coordinates, validity, timing, units and raw warnings."""
    raw = FIXTURES / "mobility.csv"
    result = read_mobility_sizer(raw, input_chunk_bytes=1)
    row = result.frame.row(0, named=True)
    grid = result.grids[row["grid_id"]]
    assert len(grid["coordinates"]["diameter"]) == 203
    np.testing.assert_allclose(
        np.array(grid["coordinates"]["diameter"])[[0, -1]],
        [20.72e-9, 784.39e-9],
    )
    assert row["vendor_flags"] == row["vendor_status"].split(";")
    assert row["scan_duration_s"] == 75 and row["retrace_duration_s"] == 4
    assert result.integrated_number().collect()["integrated_number"][
        0
    ] == pytest.approx(2168.9981640625e6)
    p = tmp_path / "missing.csv"
    p.write_bytes(raw.read_bytes().replace(b",6103.186,", b",.,"))
    masked = read_mobility_sizer(p)
    assert masked.integrated_number().collect()["integrated_number"][0] is None
    p.write_bytes(raw.read_bytes().replace(b"Weight,Number", b"Weight,Mass"))
    assert validate_profile("smps-aim", source=p).errors[0].code == "preamble"


def test_legacy_migration(tmp_path):
    """Require an explicit filter policy and definitions for unknown units."""
    settings = {
        "data_loading_function": "general_1d_load",
        "time_column": [0],
        "time_format": "epoch",
        "timezone_identifier": "UTC",
        "data_column": [1, 2],
        "data_header": ["CPC_count[#/sec]", "Temperature[degC]"],
        "data_checks": {"char_counts": {",": 4}},
        "time_shift_seconds": 2,
    }
    with pytest.raises(ProfileError, match="policy"):
        migrate_settings(settings).resolved_profile()
    p = migrate_settings(
        settings, on_bad_record="join_3010_logger_fragments"
    ).resolved_profile()
    result = read_measurement(FIXTURES / "counter.csv", profile=p)
    assert result.frame["time_utc"].dt.epoch("s")[0] == 1657342803
    assert result.manifest["profile"]["legacy_settings"] == settings
    partial = migrate_settings(settings)
    assert (
        validate_profile(partial.profile).errors[0].code == "legacy_unresolved"
    )
    settings["data_header"][1] = "unknown"
    assert (
        migrate_settings(settings, on_bad_record="error").errors[0].path
        == "/data_header/1"
    )


def test_record_limits_empty_and_all_invalid(tmp_path):
    """Empty imports retain schema and issue-only batches can be reopened."""
    src = tmp_path / "empty"
    src.write_bytes(b"")
    assert import_to_parquet(
        src, tmp_path / "out", profile="cpc-3010"
    ).frame.is_empty()
    src.write_text("garbage\n" * 5)
    result = import_to_parquet(
        src, tmp_path / "errors", profile="cpc-3010", batch_scans=1
    )
    assert result.frame.is_empty() and result.issues.height == 5
    with pytest.raises(ValueError, match="positive"):
        MeasurementReader(src, "cpc-3010", input_chunk_bytes=0)


def test_repeated_aim_header_starts_a_new_grid_segment(tmp_path):
    """A new midpoint grid with the same width stays distinguishable on disk."""
    lines = (FIXTURES / "mobility.csv").read_bytes().splitlines(keepends=True)
    # Retain logarithmic spacing while changing every coordinate by a decade.
    import csv

    h = next(csv.reader([lines[24].decode()]))
    a, b = h.index("Diameter Midpoint (nm)") + 1, h.index("Scan Time (s)")
    h[a:b] = [str(float(n) * 10) for n in h[a:b]]
    p = tmp_path / "both.csv"
    p.write_bytes(b"".join(lines) + (",".join(h) + "\n").encode() + lines[25])
    dataset = import_to_parquet(p, tmp_path / "out", profile="smps-aim")
    assert dataset.frame.height == 2
    assert dataset.frame["grid_id"].n_unique() == 2
    assert dataset.frame["scan_id"].to_list() == [26, 28]
    assert dataset.frame["number_spectrum"].dtype == pl.Array(pl.Float64, 203)
    # A third complete export has one fewer channel and its own preamble.
    h.pop(b - 1)
    row = next(csv.reader([lines[25].decode()]))
    row.pop(b - 1)
    with p.open("a", newline="") as output:
        output.writelines(x.decode() for x in lines[:24])
        csv.writer(output).writerows([h, row])
    mixed = import_to_parquet(p, tmp_path / "mixed", profile="smps-aim")
    assert mixed.frame["number_spectrum"].list.len().to_list() == [
        203,
        203,
        202,
    ]
    assert mixed.integrated_number().collect().height == 3
    assert_frame_equal(mixed.frame, read_mobility_sizer(p).frame)


def test_source_edit_during_import_and_missing_part(tmp_path):
    """Interrupt a changing source and reject a partially copied dataset."""
    path = tmp_path / "changing.csv"
    path.write_bytes((FIXTURES / "counter.csv").read_bytes())
    reader = MeasurementReader(path, "cpc-3010", batch_scans=2)
    batches = reader.batches()
    next(batches)
    with path.open("ab") as out:
        out.write(b"1657343200,1,20.0,1,0.017\n")
    with pytest.raises(ValueError, match="changed during import"):
        list(batches)
    assert reader.manifest is None
    dest = tmp_path / "complete"
    import_to_parquet(path, dest, profile="cpc-3010", batch_scans=3)
    next(dest.glob("part-*.parquet")).unlink()
    with pytest.raises(ValueError, match="Incomplete Parquet"):
        open_measurement(dest)


def test_quality_policy_and_empty_mean(tmp_path):
    """Record analyst exclusions separately from the instrument's raw status."""
    scan = read_mobility_sizer(FIXTURES / "mobility.csv")
    q = (
        scan.quality_summary("total_number", exclude_flags=["Incomplete Scan"])
        .collect()
        .row(0, named=True)
    )
    assert (q["scans"], q["retained"], q["valid_samples"], q["mean"]) == (
        1,
        0,
        0,
        None,
    )
    assert q["excluded_flags"] == ["Incomplete Scan"]
    assert q["first_excluded"] == q["last_excluded"]
    assert scan.flag_summary().collect()["scans"].to_list() == [1, 1]
    repeated = tmp_path / "repeated.csv"
    repeated.write_bytes(
        (FIXTURES / "mobility.csv")
        .read_bytes()
        .replace(b"Incomplete Scan", b"Incomplete Scan;Incomplete Scan")
    )
    duplicate = read_mobility_sizer(repeated)
    assert (
        duplicate.frame["vendor_flags"].to_list()[0].count("Incomplete Scan")
        == 2
    )
    assert duplicate.flag_summary().collect()["scans"].to_list() == [1, 1]
    assert (
        duplicate.quality_summary(
            "total_number", exclude_flags="Incomplete Scan"
        ).collect()["retained"][0]
        == 0
    )
    assert "flag_summary" in duplicate.available_operations()
    counter = read_particle_counter(FIXTURES / "counter.csv")
    q = (
        counter.quality_summary("count_rate", exclude_repaired=True)
        .collect()
        .row(0, named=True)
    )
    assert (q["scans"], q["retained"], q["valid_samples"]) == (20, 19, 19)
    assert counter.quality_summary("temperature").collect()["retained"][0] == 20


def test_concurrent_publication_has_one_complete_product(tmp_path):
    """Concurrent identical imports expose the same rows exactly once."""
    from concurrent.futures import ThreadPoolExecutor

    path = FIXTURES / "counter.csv"
    dest = tmp_path / "out"

    def publish(batch_size):
        return import_to_parquet(
            path, dest, profile="cpc-3010", batch_scans=batch_size
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(publish, (3, 8))
    assert_frame_equal(first.frame, second.frame)
    assert first.frame.height == first.frame["scan_id"].n_unique() == 20
    assert not list(tmp_path.glob(".measurement-*"))
