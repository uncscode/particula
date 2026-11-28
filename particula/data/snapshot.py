"""Stage portable Parquet snapshots of selected raw or derived measurements."""

import json
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from particula.data.measurement import Measurement, measurement
from particula.data.product import DerivedMeasurement
from particula.data.profile import canonical, digest


def publish_directory(
    destination: Path,
    write: Callable[[Path], None],
    reopen: Callable[[Path], object],
) -> None:
    """Publish a directory or verify a matching existing product on retry."""
    from particula.data.storage import _sync

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=destination.parent, prefix=".measurement-"
    ) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        write(stage)
        for path in stage.iterdir():
            _sync(path)
        _sync(stage)
        if not destination.exists():
            try:
                stage.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
            else:
                _sync(destination.parent)
                return
        existing = destination / "manifest.json"
        if (
            not existing.is_file()
            or existing.read_bytes() != (stage / "manifest.json").read_bytes()
        ):
            raise FileExistsError(
                f"Destination contains a different product: {destination}"
            )
        try:
            reopen(destination)
        except (
            OSError,
            ValueError,
            KeyError,
            pl.exceptions.PolarsError,
        ) as exc:
            raise FileExistsError(
                f"Existing product is incomplete or invalid: {destination}"
            ) from exc


def save_measurement(
    product: Measurement,
    destination: str | PathLike[str],
    *,
    batch_scans: int = 65536,
) -> Measurement:
    """Save scans, diagnostics and provenance together and reopen them lazily.

    Args:
        product: Imported or derived product, including any scan selection.
        destination: Portable directory to publish once both tables succeed.
        batch_scans: Maximum Parquet row group size, in observations.

    Returns:
        Reopened measurement retaining the selected product's type and ID.
    """
    from particula.data.storage import open_measurement

    if type(batch_scans) is not int or batch_scans <= 0:
        raise ValueError("batch_scans must be a positive integer")

    def write(stage: Path) -> None:
        product.lazy_frame.sink_parquet(
            stage / "scans.parquet", row_group_size=batch_scans
        )
        product.lazy_issues.sink_parquet(
            stage / "issues.parquet", row_group_size=batch_scans
        )
        metadata = dict(product.manifest)
        metadata.pop("part_counts", None)
        envelope = {
            "snapshot_version": 1,
            "product_id": product.product_id,
            "metadata": metadata,
            "metadata_sha256": digest(metadata),
        }
        (stage / "manifest.json").write_bytes(canonical(envelope))
        if "recipe" in metadata:
            (stage / "recipe.json").write_bytes(canonical(metadata["recipe"]))

    root = Path(destination)
    publish_directory(root, write, open_measurement)
    return open_measurement(root)


def open_snapshot(root: Path, envelope: dict) -> Measurement:
    """Validate a snapshot envelope and restore its public measurement type."""
    metadata = envelope["metadata"]
    if (
        envelope["snapshot_version"] != 1
        or digest(metadata) != envelope["metadata_sha256"]
    ):
        raise ValueError(
            "Invalid measurement snapshot version or metadata digest"
        )
    if (
        "recipe" in metadata
        and json.loads((root / "recipe.json").read_bytes())
        != metadata["recipe"]
    ):
        raise ValueError("Saved recipe disagrees with product manifest")
    for name in ("scans.parquet", "issues.parquet"):
        if not (root / name).is_file():
            raise ValueError(f"Incomplete measurement snapshot: missing {name}")
    scans = pl.scan_parquet(root / "scans.parquet")
    issues = pl.scan_parquet(root / "issues.parquet")
    product = (
        DerivedMeasurement(scans, issues, metadata)
        if "lineage" in metadata
        else measurement(scans, issues, metadata)
    )
    if product.product_id != envelope["product_id"]:
        raise ValueError("Snapshot product identity mismatch")
    return product
