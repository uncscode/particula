"""Collect reader batches in memory or publish a complete Parquet dataset."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from particula.data.measurement import Measurement, measurement
from particula.data.profile import canonical, digest, load_profile
from particula.data.reader import MeasurementReader


def read_measurement(source, *, profile, **limits) -> Measurement:
    """Import one raw source into memory, applying the supplied profile."""
    reader = MeasurementReader(source, profile, **limits)
    scans, issues = zip(*reader.batches(), strict=True)
    return measurement(
        pl.concat(scans, how="vertical_relaxed").lazy(),
        pl.concat(issues).lazy(),
        reader.manifest,
    )


def open_measurement(destination) -> Measurement:
    """Open the completed scans and diagnostics with their units."""
    root = Path(destination)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest.get("format_version") != 1:
        raise ValueError("Unrecognized measurement manifest version")
    profile = load_profile(manifest["profile"])
    if digest(profile) != manifest["profile_sha256"]:
        raise ValueError("Measurement profile digest mismatch")
    if any(digest(grid) != key for key, grid in manifest["grids"].items()):
        raise ValueError("Measurement grid digest mismatch")

    def parts(prefix):
        count = manifest["part_counts"][prefix]
        paths = sorted(root.glob(prefix + "-*.parquet"))
        if (
            not paths
            or len(paths) != count
            or any(
                p.name != f"{prefix}-{i:09d}.parquet"
                for i, p in enumerate(paths)
            )
        ):
            raise ValueError(f"Incomplete Parquet parts for {prefix}")
        return pl.scan_parquet(paths, glob=False)

    sections = [
        parts("part" if i == 0 else f"segment{i}")
        for i in range(len(manifest["segments"]))
    ]
    return measurement(
        pl.concat(sections, how="vertical_relaxed"), parts("issues"), manifest
    )


def _sync(path):
    if os.name == "nt" and path.is_dir():
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def import_to_parquet(source, destination, *, profile, **limits) -> Measurement:
    """Write bounded Parquet parts and atomically publish a completed import.

    Repeating an import with identical source bytes and profile reopens the
    existing result. A conflicting destination raises FileExistsError.
    """
    reader = MeasurementReader(source, profile, **limits)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".measurement-") as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        counts = {"part": 0, "issues": 0}
        for scans, issues in reader.batches():
            section = f"segment{reader.segment}" if reader.segment else "part"
            counts.setdefault(section, 0)
            for prefix, frame in ((section, scans), ("issues", issues)):
                if frame.height or not counts[prefix]:
                    path = stage / f"{prefix}-{counts[prefix]:09d}.parquet"
                    frame.write_parquet(path)
                    _sync(path)
                    counts[prefix] += 1
        manifest = reader.manifest
        manifest["part_counts"] = counts
        meta = stage / "manifest.json"
        meta.write_bytes(canonical(manifest))
        _sync(meta)
        _sync(stage)
        if not dest.exists():
            try:
                stage.rename(dest)
            except OSError:
                if not dest.is_dir():
                    raise
            else:
                _sync(dest.parent)
                return open_measurement(dest)
        try:
            existing = open_measurement(dest)
        except (
            OSError,
            ValueError,
            KeyError,
            pl.exceptions.PolarsError,
        ) as exc:
            raise FileExistsError(
                f"Destination already exists: {dest}"
            ) from exc
        if any(
            manifest[k] != existing.manifest[k]
            for k in ("source_sha256", "profile_sha256", "parser_version")
        ):
            raise FileExistsError(
                f"Destination contains a different measurement: {dest}"
            )
        return existing
