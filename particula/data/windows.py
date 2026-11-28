"""Aggregate independent UTC measurement tables on a common half-open grid."""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from os import PathLike
from pathlib import Path

import polars as pl

from particula.data.deployment import utc_time
from particula.data.measurement import Measurement
from particula.data.profile import canonical, digest
from particula.data.snapshot import publish_directory


def duration_ns(value: str | timedelta) -> int:
    """Return a positive fixed duration in integer nanoseconds."""
    scales = {
        "ns": 1,
        "us": 1000,
        "ms": 10**6,
        "s": 10**9,
        "m": 60 * 10**9,
        "h": 3600 * 10**9,
        "d": 86400 * 10**9,
    }
    if isinstance(value, timedelta):
        result = value // timedelta(microseconds=1) * 1000
    else:
        match = re.fullmatch(r"([0-9]+)(ns|us|ms|s|m|h|d)", value)
        if not match:
            raise ValueError("Expected a fixed window duration such as 1m")
        result = int(match[1]) * scales[match[2]]
    if result <= 0:
        raise ValueError("Duration must be positive")
    return result


def instant_ns(value: datetime | str) -> int:
    """Return UTC nanoseconds for an aware timestamp."""
    return (
        (utc_time(value) - datetime(1970, 1, 1, tzinfo=UTC))
        // timedelta(microseconds=1)
        * 1000
    )


@dataclass(frozen=True)
class CampaignEvent:
    """A half-open annotation on one deployment and optionally one channel."""

    kind: str
    start: datetime | str
    stop: datetime | str
    member: str
    field: str | None = None

    def record(self) -> dict:
        """Return portable bounds in UTC and the scope of this annotation."""
        start, stop = utc_time(self.start), utc_time(self.stop)
        if start >= stop or not self.kind:
            raise ValueError(
                "Campaign event requires a kind and increasing interval"
            )
        return {
            "kind": self.kind,
            "start": start.isoformat(),
            "stop": stop.isoformat(),
            "member": self.member,
            "field": self.field,
        }


def checked_rows(
    frame: pl.LazyFrame, invalid: pl.Expr, message: str
) -> pl.LazyFrame:
    """Check one aggregate invariant when the Polars query executes."""

    def verify(failed: bool) -> bool:
        if failed:
            raise ValueError(message)
        return True

    valid = frame.select(
        invalid.map_elements(verify, return_dtype=pl.Boolean).alias("_valid")
    )
    return frame.join(valid.filter("_valid"), how="cross").drop("_valid")


def checked_clock(product: Measurement) -> pl.LazyFrame:
    """Require a UTC clock and unique source-scoped scan IDs at evaluation."""
    frame = product.lazy_frame
    if frame.collect_schema()["time_utc"] != pl.Datetime("ns", "UTC"):
        raise ValueError("Expected a common UTC nanosecond clock")
    invalid = (
        pl.col("scan_id").is_duplicated().any()
        | pl.col("scan_id").is_null().any()
        | pl.col("time_utc").is_null().any()
    )
    return checked_rows(
        frame, invalid, "Missing time or duplicate/missing scan ID"
    )


class WindowComparison:
    """Member summaries on one common UTC grid, with aggregation provenance."""

    def __init__(self, frame: pl.LazyFrame, metadata: dict) -> None:
        """Bind the aligned query to its units and analysis policy."""
        self._frame = frame
        self.manifest = metadata

    def align_windows(self, how: str = "grid") -> pl.LazyFrame:
        """Return the full grid, including empty windows, for lazy analysis."""
        if how != "grid":
            raise ValueError("Supported window alignment is 'grid'")
        return self._frame

    def __getitem__(self, member: str) -> pl.LazyFrame:
        """Select the window key and all summaries for one campaign member."""
        if member not in self.manifest["fields"]:
            raise KeyError(member)
        return self._frame.select(
            "window_start", pl.selectors.starts_with(member + ".")
        )

    def save(self, destination: str | PathLike[str]) -> None:
        """Write the comparison with its units, parents and window policy."""

        def write(stage: Path) -> None:
            self._frame.sink_parquet(stage / "windows.parquet")
            envelope = {
                "window_version": 1,
                "metadata": self.manifest,
                "product_id": digest(self.manifest),
            }
            (stage / "manifest.json").write_bytes(canonical(envelope))

        publish_directory(Path(destination), write)

    @classmethod
    def open(cls, destination: str | PathLike[str]) -> "WindowComparison":
        """Restore a window comparison and verify its aggregation provenance."""
        root = Path(destination)
        envelope = json.loads((root / "manifest.json").read_bytes())
        metadata = envelope["metadata"]
        if (
            envelope.get("window_version") != 1
            or digest(metadata) != envelope["product_id"]
        ):
            raise ValueError("Invalid window comparison manifest")
        frame = pl.scan_parquet(root / "windows.parquet")
        schema = frame.collect_schema()
        if schema.get("window_start") != pl.Datetime("ns", "UTC") or not set(
            metadata["units"]
        ).issubset(schema):
            raise ValueError("Comparison columns disagree with manifest")
        return cls(frame, metadata)


def _event_intervals(
    events: list[CampaignEvent], field: str
) -> list[tuple[int, int]]:
    spans = sorted(
        (instant_ns(e.start), instant_ns(e.stop))
        for e in events
        if e.field is None or e.field == field
    )
    merged: list[tuple[int, int]] = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _field_summary(
    frame: pl.LazyFrame,
    grid: pl.LazyFrame,
    field: str,
    label: str,
    quantity: str,
    events: list[CampaignEvent],
    weighting: str,
    width: int | None,
) -> pl.LazyFrame:
    duration = pl.col("_end") - pl.col("_begin")
    keep = pl.lit(True)
    for start, stop in _event_intervals(events, field):
        keep &= ~pl.col("_time").is_between(start, stop, closed="left")
        duration -= (
            pl.min_horizontal("_end", pl.lit(stop))
            - pl.max_horizontal("_begin", pl.lit(start))
        ).clip(lower_bound=0)
    value = pl.col(field)
    if weighting == "exposure":
        if width is not None or quantity == "count":
            raise ValueError(
                "Exposure weighting requires an interval-mean scalar"
            )
        weight = pl.when(value.is_not_null()).then(duration)
        summary = (
            frame.filter(duration > 0)
            .group_by("window_start")
            .agg(
                (value * weight).sum().alias("_integral"),
                weight.sum().alias("_weight_ns"),
                value.count().alias(label + ".n"),
            )
        )
        result = grid.join(summary, on="window_start", how="left").with_columns(
            pl.col(label + ".n").fill_null(0),
            pl.col("_weight_ns").fill_null(0),
        )
        # Split whole seconds before float conversion to keep exact whole
        # exposures in both the in-memory and streaming Polars engines.
        weight_ns = pl.col("_weight_ns")
        seconds = weight_ns // 10**9 + (weight_ns % 10**9) / 10**9
        return result.select(
            "window_start",
            label + ".n",
            seconds.alias(label + ".exposure_s"),
            pl.when(pl.col("_weight_ns") > 0)
            .then(pl.col("_integral") / pl.col("_weight_ns"))
            .alias(label),
            (pl.col("_weight_ns") / (pl.col("_hi") - pl.col("_lo"))).alias(
                label + ".coverage"
            ),
        )
    data = frame.filter(keep)
    if width is not None:
        channels = [
            value.cast(pl.List(pl.Float64)).list.get(i) for i in range(width)
        ]
        aggregations = [
            pl.concat_list([channel.mean() for channel in channels]).alias(
                label
            ),
            pl.concat_list([channel.count() for channel in channels]).alias(
                label + ".n"
            ),
        ]
        zero = pl.lit([0] * width, dtype=pl.List(pl.UInt32))
    else:
        reduction = (
            pl.when(value.count() > 0).then(value.sum())
            if quantity == "count"
            else value.mean()
        )
        aggregations = [
            reduction.alias(label),
            value.count().alias(label + ".n"),
        ]
        zero = pl.lit(0)
    summary = data.group_by("window_start").agg(*aggregations)
    return (
        grid.select("window_start")
        .join(summary, on="window_start", how="left")
        .with_columns(pl.col(label + ".n").fill_null(zero))
    )


def _assign_windows(
    frame: pl.LazyFrame,
    definitions: dict,
    support: str | None,
    weighting: str,
    anchor: int,
    step: int,
    low: int,
    high: int,
) -> pl.LazyFrame:
    """Assign samples or clipped verified intervals to fixed window keys."""
    if weighting == "exposure":
        if (
            not support
            or definitions.get(support, {}).get("quantity") != "duration"
            or definitions[support]["output_unit"] != "s"
        ):
            raise ValueError(
                "Exposure requires a verified duration field in seconds"
            )
        length = pl.col(support)
        frame = frame.sort("_time").with_columns(
            (pl.col("_time") + (length * 1e9).round().cast(pl.Int64)).alias(
                "_stop"
            )
        )
        invalid = (
            length.is_null()
            | ~length.is_finite()
            | (pl.col("_stop") <= pl.col("_time"))
            | (pl.col("_time") < pl.col("_stop").cum_max().shift(1))
        ).any()
        frame = checked_rows(
            frame, invalid, "Invalid or overlapping verified exposure intervals"
        )
        frame = frame.filter((pl.col("_stop") > low) & (pl.col("_time") < high))
        begin = pl.max_horizontal("_time", pl.lit(low))
        end = pl.min_horizontal("_stop", pl.lit(high))
        bins = (begin - anchor) // step * step + anchor
        frame = (
            frame.with_columns(
                pl.int_ranges(bins, end, step=step).alias("window_start")
            )
            .explode("window_start")
            .with_columns(
                pl.max_horizontal(begin, "window_start").alias("_begin"),
                pl.min_horizontal(end, pl.col("window_start") + step).alias(
                    "_end"
                ),
            )
        )
    else:
        frame = frame.filter(
            pl.col("_time").is_between(low, high, closed="left")
        ).with_columns(
            ((pl.col("_time") - anchor) // step * step + anchor).alias(
                "window_start"
            )
        )
    return frame


def _window_field(
    product: Measurement, name: str, definitions: dict, spectrum: dict
) -> tuple[dict, int | None]:
    """Determine the reduction and validate any spectrum coordinates."""
    width = None
    if spectrum.get("name") == name:
        if len(product.grids) != 1:
            raise ValueError(
                "Spectrum averaging requires a common diameter grid"
            )
        width = len(
            next(iter(product.grids.values()))["coordinates"]["diameter"]
        )
        definition = spectrum
    elif name in definitions:
        definition = definitions[name]
    else:
        raise ValueError(f"Unsupported aggregation field: {name}")
    return definition, width


def _quality_filter(
    frame: pl.LazyFrame,
    product: Measurement,
    alias: str,
    exclude_repaired: bool,
    flags_to_exclude: list[str],
) -> pl.LazyFrame:
    """Retain the analyst's selected records for one instrument."""
    if exclude_repaired:
        frame = frame.filter(~pl.col("repaired"))
    if flags_to_exclude:
        flags = (
            "quality_flags" if "lineage" in product.manifest else "vendor_flags"
        )
        if flags not in frame.collect_schema():
            raise ValueError(f"{alias} has no status flags")
        frame = frame.filter(
            ~pl.any_horizontal(
                [pl.col(flags).list.contains(flag) for flag in flags_to_exclude]
            )
        )
    return frame


def average_members(
    members: dict[str, Measurement],
    events: tuple[CampaignEvent, ...],
    start: datetime,
    stop: datetime,
    *,
    every: str | timedelta,
    origin: datetime,
    weighting: str,
    fields: dict[str, list[str]],
    exposure: dict[str, str] | None = None,
    exclude_events: tuple[str, ...] = (),
    exclude_flags: dict[str, list[str]] | None = None,
    exclude_repaired: bool = False,
) -> WindowComparison:
    """Compute independent member summaries before joining their window keys."""
    if weighting not in ("sample", "exposure") or not fields:
        raise ValueError(
            "Choose sample or exposure weighting and at least one field"
        )
    step = duration_ns(every)
    anchor, low, high = instant_ns(origin), instant_ns(start), instant_ns(stop)
    first = anchor + (low - anchor) // step * step
    grid = (
        pl.LazyFrame()
        .select(pl.int_range(first, high, step=step).alias("window_start"))
        .with_columns(
            pl.max_horizontal("window_start", pl.lit(low)).alias("_lo"),
            pl.min_horizontal(
                pl.col("window_start") + step, pl.lit(high)
            ).alias("_hi"),
        )
    )
    result = grid.select("window_start")
    units = {}
    exposure = exposure or {}
    exclude_flags = exclude_flags or {}
    if (set(exclude_flags) | set(exposure)).difference(fields):
        raise ValueError("Policy names an unselected member")
    for alias, selected in fields.items():
        product = members[alias]
        frame = checked_clock(product).with_columns(
            pl.col("time_utc").cast(pl.Int64).alias("_time")
        )
        profile = product.manifest["profile"]
        definitions = {d["name"]: d for d in profile.get("fields", [])}
        spectrum = profile.get("spectrum", {})
        support = exposure.get(alias)
        modes = (
            ("sample", "exposure")
            if weighting == "sample" and support
            else (weighting,)
        )
        frames = {
            mode: _quality_filter(
                _assign_windows(
                    frame, definitions, support, mode, anchor, step, low, high
                ),
                product,
                alias,
                exclude_repaired,
                exclude_flags.get(alias, []),
            )
            for mode in modes
        }
        frame = frames[weighting]
        exposure_frame = (
            frames.get("exposure") if weighting == "sample" else None
        )
        annotations = [
            e for e in events if e.member == alias and e.kind in exclude_events
        ]
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Expected distinct selected field names")
        for name in selected:
            definition, width = _window_field(
                product, name, definitions, spectrum
            )
            label = f"{alias}.{name}"
            units[label] = definition["output_unit"]
            summary = _field_summary(
                frame,
                grid,
                name,
                label,
                definition["quantity"],
                annotations,
                weighting,
                width,
            )
            if exposure_frame is not None:
                durations = _field_summary(
                    exposure_frame,
                    grid,
                    name,
                    label,
                    definition["quantity"],
                    annotations,
                    "exposure",
                    width,
                )
                summary = summary.join(
                    durations.select(
                        "window_start",
                        label + ".exposure_s",
                        label + ".coverage",
                    ),
                    on="window_start",
                    how="left",
                )
            result = result.join(summary, on="window_start", how="left")
    return WindowComparison(
        result.sort("window_start").with_columns(
            pl.col("window_start").cast(pl.Datetime("ns", "UTC"))
        ),
        {
            "weighting": weighting,
            "every_ns": step,
            "origin": utc_time(origin).isoformat(),
            "start": start.isoformat(),
            "stop": stop.isoformat(),
            "units": units,
            "fields": fields,
            "grids": {k: members[k].grids for k in fields if members[k].grids},
            "parents": {k: members[k].product_id for k in fields},
            "events": [e.record() for e in events],
            "exclude_events": list(exclude_events),
            "exclude_flags": exclude_flags,
            "exclude_repaired": exclude_repaired,
            "exposure": exposure,
        },
    )
