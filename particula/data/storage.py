"""Collect reader batches in memory or publish a complete Parquet dataset."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import polars as pl

from particula.data.measurement import Measurement, measurement
from particula.data.profile import (
    ProfileSource,
    canonical,
    digest,
    load_profile,
)
from particula.data.reader import BatchReader, MeasurementReader


def read_measurement(
    source: str | os.PathLike[str],
    *,
    profile: ProfileSource,
    reader_factory: Callable[..., BatchReader] = MeasurementReader,
    **limits: int,
) -> Measurement:
    """Read a single instrument export into memory.

    Args:
        source: Path to the delimited export.
        profile: Built-in ID, mapping, or path to a JSON or YAML profile.
        reader_factory: Trusted strategy constructing a new batch reader.
        **limits: Optional MeasurementReader chunk, record and batch limits.

    Returns:
        Measurement selected from the profile's quantities and coordinate
        basis. Scans use the declared output units and include UTC timestamps.

    Raises:
        ProfileError: The profile or source violates the input rules.
        ValueError: A reader limit is invalid or the source changes during
            import.
    """
    reader = reader_factory(source, profile, **limits)
    scans, issues = zip(*reader.batches(), strict=True)
    return measurement(
        pl.concat(scans, how="vertical_relaxed").lazy(),
        pl.concat(issues).lazy(),
        cast(dict, reader.manifest),  # Finalized by exhausting batches().
    )


def open_measurement(destination: str | os.PathLike[str]) -> Measurement:
    """Open a completed Parquet measurement for lazy analysis.

    Args:
        destination: Dataset directory created by import_to_parquet or
            save_measurement.

    Returns:
        Measurement with lazy scans and diagnostics, validated profile units
        and coordinates from the stored manifest.

    Raises:
        OSError: The manifest cannot be opened or read.
        ValueError: The format, profile or grid digests, part inventory, or
            scan layout fails validation.
    """
    root = Path(destination)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if "snapshot_version" in manifest:
        from particula.data.snapshot import open_snapshot

        return open_snapshot(root, manifest)
    return _open_import(root, manifest)


def _open_import(root: Path, manifest: dict) -> Measurement:
    """Validate and reopen the complete raw import at the given path."""
    if manifest.get("format_version") != 1:
        raise ValueError("Unrecognized measurement manifest version")
    profile = load_profile(manifest["profile"])
    if digest(profile) != manifest["profile_sha256"]:
        raise ValueError("Measurement profile digest mismatch")
    if any(digest(grid) != key for key, grid in manifest["grids"].items()):
        raise ValueError("Measurement grid digest mismatch")

    def parts(prefix: str) -> pl.LazyFrame:
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


def _sync(path: Path) -> None:
    if os.name == "nt" and path.is_dir():
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def import_to_parquet(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    profile: ProfileSource,
    reader_factory: Callable[..., BatchReader] = MeasurementReader,
    **limits: int,
) -> Measurement:
    """Write bounded Parquet parts and atomically publish a completed import.

    Args:
        source: Path to a delimited instrument export.
        destination: Directory to publish after all parts have been written.
        profile: Built-in ID, mapping, or path to a JSON or YAML profile.
        reader_factory: Trusted strategy constructing a new batch reader.
        **limits: Optional MeasurementReader chunk, record and batch limits.

    Returns:
        Disk-backed measurement with lazy scans and source diagnostics.
        An existing raw import is reused when the source digest, profile
        digest and parser version match.

    Raises:
        FileExistsError: The destination holds a saved snapshot, an incomplete
            import or a different dataset.
        ProfileError: The profile or source violates the input rules.
        ValueError: A reader limit is invalid or the source changes during
            import.
    """
    reader = reader_factory(source, profile, **limits)
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
        manifest = cast(dict, reader.manifest)  # All batches are written.
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
            existing = _open_import(
                dest, json.loads((dest / "manifest.json").read_bytes())
            )
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
