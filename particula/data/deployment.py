"""Deployment and calibration records for time-resolved instrument products."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite


def utc_time(value: datetime | str) -> datetime:
    """Normalize an offset-aware campaign time to UTC."""
    instant = datetime.fromisoformat(value) if isinstance(value, str) else value
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("An explicit time zone or UTC offset is required")
    return instant.astimezone(UTC)


@dataclass(frozen=True)
class Calibration:
    """Traceable coefficients valid over a half-open UTC interval.

    Args:
        reference: Unique certificate or revision identifier.
        operation: Qualified operation consuming these coefficients.
        start: Inclusive effective time with explicit timezone.
        stop: Exclusive effective time.
        coefficients: Finite numbers keyed by the operation's coefficient names.
        units: Unit of each coefficient.
        source: Traceable source of the calibration.
        group: Optional shared name for consecutive revisions.
    """

    reference: str
    operation: str
    start: datetime | str
    stop: datetime | str
    coefficients: dict[str, float]
    units: dict[str, str]
    source: str
    group: str | None = None

    def record(self) -> dict:
        """Validate the certificate and return JSON-compatible UTC metadata."""
        start, stop = utc_time(self.start), utc_time(self.stop)
        if not self.reference or not self.operation or not self.source:
            raise ValueError(
                "Calibration requires reference, operation and source"
            )
        if start >= stop or not self.coefficients:
            raise ValueError(
                "Calibration requires an interval and coefficients"
            )
        if self.coefficients.keys() != self.units.keys() or not all(
            isfinite(value) and self.units[name]
            for name, value in self.coefficients.items()
        ):
            raise ValueError("Each finite coefficient requires units")
        return {
            **asdict(self),
            "start": start.isoformat(),
            "stop": stop.isoformat(),
        }


@dataclass(frozen=True)
class Deployment:
    """Identify an instrument instance, its inlet and sample-volume basis.

    Args:
        name: Run-specific deployment name.
        inlet: Sampling location or flow path.
        volume_basis: Reference conditions for sample volume, such as ambient.
        serial: Instrument serial number, when known.
        calibrations: Effective-dated records available to this deployment.
        start: Optional inclusive start of deployment.
        stop: Optional exclusive end of deployment.
        site: Optional site identifier.
    """

    name: str
    inlet: str | None = None
    volume_basis: str | None = None
    serial: str | None = None
    calibrations: tuple[Calibration, ...] = ()
    start: datetime | str | None = None
    stop: datetime | str | None = None
    site: str | None = None

    def record(self) -> dict:
        """Normalize times and verify unique calibration references."""
        record = asdict(self)
        if not self.name:
            raise ValueError("Deployment requires a name")
        for key in ("start", "stop"):
            if record[key] is not None:
                record[key] = utc_time(record[key]).isoformat()
        if (
            self.start
            and self.stop
            and utc_time(self.start) >= utc_time(self.stop)
        ):
            raise ValueError("Deployment stop must follow its start")
        records = [cal.record() for cal in self.calibrations]
        if len({cal["reference"] for cal in records}) != len(records):
            raise ValueError("Duplicate calibration reference in deployment")
        record["calibrations"] = records
        return record
