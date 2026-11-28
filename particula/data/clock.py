"""Apply traceable instrument-clock corrections to observation times."""

from datetime import datetime
from decimal import Decimal

import polars as pl

from particula.data.deployment import utc_time
from particula.data.measurement import Measurement


def correct_clock(
    product: Measurement,
    *,
    offset_seconds: float,
    start: datetime | str,
    stop: datetime | str,
    source: str,
) -> Measurement:
    """Add a signed clock offset to reported timestamps in a half-open interval.

    Args:
        product: Source measurements whose clock was compared to a reference.
        offset_seconds: Seconds to add to the reported time to obtain UTC.
        start: Inclusive lower bound on the originally reported clock.
        stop: Exclusive upper bound on the originally reported clock.
        source: Reference observation or clock-comparison record. An inlet
            residence time is a separate physical quantity.

    Returns:
        A typed product with corrected time_utc and original reported_time_utc.
        Scan identity and acquisition durations remain tied to the source.

    Raises:
        ValueError: The interval overlaps an existing clock correction or the
            offset cannot be expressed as whole nanoseconds.
    """
    low, high = utc_time(start), utc_time(stop)
    ticks = Decimal(str(offset_seconds)) * 10**9
    if (
        not source
        or low >= high
        or not ticks.is_finite()
        or ticks != ticks.to_integral_value()
        or abs(ticks) >= 2**63
    ):
        raise ValueError(
            "Expected a reference, valid interval and finite offset in ns"
        )
    records = product.manifest.get("clock_corrections", [])
    if any(
        utc_time(item["stop"]) > low and utc_time(item["start"]) < high
        for item in records
    ):
        raise ValueError("Overlapping clock corrections")
    record = {
        "start": low.isoformat(),
        "stop": high.isoformat(),
        "offset_ns": int(ticks),
        "source": source,
    }
    frame = product.lazy_frame
    if not records:
        if "reported_time_utc" in frame.collect_schema():
            raise ValueError("The reported_time_utc column already exists")
        frame = frame.with_columns(
            pl.col("time_utc").alias("reported_time_utc")
        )
    reported = pl.col("reported_time_utc")
    shifted = reported.cast(pl.Int64).cast(pl.Int128) + int(ticks)
    frame = frame.with_columns(
        pl.when(reported.is_between(low, high, closed="left"))
        .then(shifted)
        .otherwise(pl.col("time_utc").cast(pl.Int64))
        .cast(pl.Int64)
        .cast(pl.Datetime("ns", "UTC"))
        .alias("time_utc")
    )
    manifest = {**product.manifest, "clock_corrections": [*records, record]}
    return type(product)(frame, product.lazy_issues, manifest)
