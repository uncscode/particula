"""Keep heterogeneous instrument products and campaign annotations together."""

import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import Self

import polars as pl

from particula.data.deployment import utc_time
from particula.data.measurement import Measurement
from particula.data.profile import canonical
from particula.data.snapshot import publish_directory, save_measurement
from particula.data.storage import open_measurement
from particula.data.windows import (
    CampaignEvent,
    WindowComparison,
    average_members,
    checked_clock,
    checked_rows,
    duration_ns,
)


class MeasurementCollection(Mapping[str, Measurement]):
    """Named products from a campaign with independent clocks, grids and QC.

    Args:
        members: Unique logical instrument names mapped to measurements.
        events: Time-bounded annotations scoped to a member or one field.
    """

    def __init__(
        self,
        members: Mapping[str, Measurement],
        *,
        events: Sequence[CampaignEvent] = (),
    ) -> None:
        """Validate member names and event scopes without joining records."""
        self._members = dict(members)
        if any(
            not isinstance(k, str)
            or not k
            or "." in k
            or not isinstance(v, Measurement)
            for k, v in self._members.items()
        ):
            raise ValueError(
                "Expected measurement products and dot-free member names"
            )
        self.events = tuple(events)
        for event in events:
            event.record()
            if event.member not in members:
                raise ValueError(f"Unknown event member {event.member!r}")
            if (
                event.field
                and event.field
                not in members[event.member].lazy_frame.collect_schema()
            ):
                raise ValueError(f"Unknown event field {event.field!r}")
        self._bounds: tuple[datetime, datetime] | None = None

    def __len__(self) -> int:
        """Return the number of named products."""
        return len(self._members)

    def __iter__(self) -> Iterator[str]:
        """Iterate over member names in insertion order."""
        return iter(self._members)

    def __getitem__(self, key: str) -> Measurement:
        """Return one typed member within the collection selection."""
        product = self._members[key]
        if self._bounds is None:
            return product
        start, stop = self._bounds
        return product.filter(
            pl.col("time_utc").is_between(start, stop, closed="left")
        )

    def select_time(self, start: datetime | str, stop: datetime | str) -> Self:
        """Select a half-open interval using timezone-aware bounds."""
        low, high = utc_time(start), utc_time(stop)
        if self._bounds:
            low, high = max(low, self._bounds[0]), min(high, self._bounds[1])
        if low >= high:
            raise ValueError("Select a non-empty UTC interval")
        result = type(self)(self._members, events=self.events)
        result._bounds = low, high
        return result

    def average(
        self,
        *,
        every: str | timedelta,
        origin: datetime,
        weighting: str,
        fields: dict[str, list[str]],
        exposure: dict[str, str] | None = None,
        exclude_events: Sequence[str] = (),
        exclude_flags: dict[str, list[str]] | None = None,
        exclude_repaired: bool = False,
    ) -> WindowComparison:
        """Aggregate each selected field on one origin-anchored UTC grid.

        Args:
            every: Fixed window width, e.g. '1m'.
            origin: Offset-aware instant anchoring window boundaries.
            weighting: 'sample' for equal-weight means or 'exposure' for
                verified interval overlap. Actual counts sum under 'sample'.
            fields: Per-member scalar or common-grid spectrum field names.
            exposure: Verified duration in seconds per member, starting at the
                scan timestamp. Nominal scan or retrace times are insufficient.
            exclude_events: Event kinds to omit within their field/member scope.
            exclude_flags: Vendor or derived flags to omit by member.
            exclude_repaired: Omit reconstructed records if true.

        Returns:
            A comparison with aggregation provenance and a lazy aligned table.
            Empty scalar windows have zero contributors and a null value.
        """
        if self._bounds is None:
            raise ValueError(
                "Call select_time with explicit bounds before averaging"
            )
        return average_members(
            self._members,
            self.events,
            *self._bounds,
            every=every,
            origin=origin,
            weighting=weighting,
            fields=fields,
            exposure=exposure,
            exclude_events=tuple(exclude_events),
            exclude_flags=exclude_flags,
            exclude_repaired=exclude_repaired,
        )

    def match_nearest(
        self, left: str, right: str, *, tolerance: str | timedelta
    ) -> pl.LazyFrame:
        """Match to the nearest right timestamp within an explicit tolerance.

        Returns:
            All left observations with qualified columns, nullable right
            observations and signed right-minus-left offset in nanoseconds.
        """
        if left == right:
            raise ValueError(
                "Nearest matching requires two distinct member names"
            )
        frames = []
        for key in (left, right):
            product = self[key]
            frame = checked_clock(product)
            if key == right:
                frame = checked_rows(
                    frame,
                    pl.col("time_utc").is_duplicated().any(),
                    "Duplicate times in the right member are ambiguous",
                )
            frames.append(
                frame.rename(
                    {name: f"{key}.{name}" for name in frame.collect_schema()}
                ).sort(f"{key}.time_utc")
            )
        ltime, rtime = f"{left}.time_utc", f"{right}.time_utc"
        return (
            frames[0]
            .join_asof(
                frames[1],
                left_on=ltime,
                right_on=rtime,
                strategy="nearest",
                tolerance=f"{duration_ns(tolerance)}ns",
                coalesce=False,
            )
            .with_columns(
                (pl.col(rtime) - pl.col(ltime))
                .dt.total_nanoseconds()
                .alias("time_offset_ns")
            )
        )

    def save(self, destination: str | PathLike[str]) -> None:
        """Save the campaign and its members atomically under one root."""

        def write(stage: Path) -> None:
            entries = []
            for index, (alias, product) in enumerate(self._members.items()):
                path = f"member-{index:06d}"
                save_measurement(product, stage / path)
                entries.append(
                    {
                        "name": alias,
                        "path": path,
                        "product_id": product.product_id,
                    }
                )
            manifest = {
                "collection_version": 1,
                "members": entries,
                "events": [e.record() for e in self.events],
                "bounds": [d.isoformat() for d in self._bounds]
                if self._bounds
                else None,
            }
            (stage / "manifest.json").write_bytes(canonical(manifest))

        publish_directory(Path(destination), write, self.open)

    @classmethod
    def open(cls, destination: str | PathLike[str]) -> Self:
        """Reopen a relocated campaign and verify each referenced member ID."""
        root = Path(destination)
        manifest = json.loads((root / "manifest.json").read_bytes())
        if manifest.get("collection_version") != 1:
            raise ValueError("Unsupported measurement collection version")
        members = {}
        for index, entry in enumerate(manifest["members"]):
            if (
                entry["path"] != f"member-{index:06d}"
                or entry["name"] in members
            ):
                raise ValueError(
                    "Invalid collection member path or duplicate name"
                )
            product = open_measurement(root / entry["path"])
            if product.product_id != entry["product_id"]:
                raise ValueError(
                    f"Collection member identity mismatch: {entry['name']}"
                )
            members[entry["name"]] = product
        result = cls(
            members, events=[CampaignEvent(**e) for e in manifest["events"]]
        )
        return (
            result.select_time(*manifest["bounds"])
            if manifest["bounds"]
            else result
        )
