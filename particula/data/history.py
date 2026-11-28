"""Carry contributor identities and quality records across causal windows."""

import polars as pl


def add_history(
    frame: pl.LazyFrame, roles: tuple[str, ...], window: int, gap_seconds: float
) -> pl.LazyFrame:
    """Accumulate row lineage over trailing samples within each clock run."""
    clock_gap = (
        pl.col("time_utc").diff().dt.total_nanoseconds() > gap_seconds * 1e9
    )
    frame = frame.with_columns(
        clock_gap.fill_null(False).cum_sum().alias("_run")
    )
    expressions = []
    for role in roles:
        link = f"parent_{role}_scan_id"
        history = [
            pl.col(link).shift(lag).over("_run")
            for lag in reversed(range(window))
        ]
        expressions.append(
            pl.concat_list(history).list.drop_nulls().alias(link + "s")
        )
    for name in ("quality_flags", "calibration_refs"):
        labels = [
            pl.col(name)
            .shift(lag)
            .over("_run")
            .fill_null(pl.lit([], dtype=pl.List(pl.String)))
            for lag in range(window)
        ]
        expressions.append(pl.concat_list(labels).alias(name))
    expressions.append(
        pl.any_horizontal(
            [
                pl.col("repaired").shift(lag).over("_run").fill_null(False)
                for lag in range(window)
            ]
        ).alias("repaired")
    )
    return frame.with_columns(*expressions)
