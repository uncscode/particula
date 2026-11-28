"""Validated materialized measurement products and portable field contracts."""

from dataclasses import dataclass
from functools import cache

import polars as pl

from particula.data._binding import _affine
from particula.data.measurement import Measurement
from particula.data.profile import describe_profile, digest


def product_fields(product: Measurement) -> list[dict]:
    """Describe scalar and array inputs with their coordinate grids."""
    profile = product.manifest["profile"]
    arrays = list(profile.get("arrays", []))
    if "spectrum" in profile:
        arrays.append(profile["spectrum"])
    return [
        *profile.get("fields", []),
        *[{**definition, "grids": product.grids} for definition in arrays],
    ]


@cache
def _reference_units() -> dict[str, str]:
    """Read physical dimensions from the installed measurement schema."""
    definitions = describe_profile()["$defs"]
    rules = (
        definitions["numeric_field"]["allOf"] + definitions["spectrum"]["allOf"]
    )
    return {
        rule["if"]["properties"]["quantity"]["const"]: rule["then"][
            "properties"
        ]["output_unit"]["enum"][0]
        for rule in rules
    }


@dataclass(frozen=True)
class FieldContract:
    """Select a field by quantity, unit and optional coordinate requirements.

    Args:
        quantity: Measured quantity, e.g. voltage or number_per_log10_diameter.
        unit: Unit expected by the implementation.
        field: Optional explicit column if several channels match.
        axes: Ordered axis names for an array or spectrum. Empty for a scalar.
        basis: Required diameter convention, e.g. electrical_mobility.
    """

    quantity: str
    unit: str
    field: str | None = None
    axes: tuple[str, ...] = ()
    basis: str | None = None

    def __post_init__(self) -> None:
        """Require enough information to interpret the measured values."""
        if not self.quantity or not self.unit:
            raise ValueError("A physical quantity and unit are required")
        reference = _reference_units().get(self.quantity)
        if reference:
            try:
                _affine(self.unit, reference)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Incompatible unit {self.unit} for {self.quantity}"
                ) from exc

    def resolve(self, fields: list[dict]) -> dict:
        """Select exactly one compatible field by quantity and optional name."""
        matches = [
            field
            for field in fields
            if field["quantity"] == self.quantity
            and (self.field is None or self.field == field["name"])
            and self.axes
            == tuple(
                a["name"]
                for a in field.get(
                    "axes", [field["axis"]] if "axis" in field else []
                )
            )
            and (
                self.basis is None
                or field.get("axis", {}).get("basis") == self.basis
            )
            and (not self.axes or len(field["grids"]) == 1)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {self.quantity} field; compatible fields: "
                f"{[field['name'] for field in matches]}"
            )
        return matches[0]


class DerivedMeasurement(Measurement):
    """A scalar product with per-scan quality, parents and calibration links."""

    def __init__(
        self, scans: pl.LazyFrame, issues: pl.LazyFrame, manifest: dict
    ) -> None:
        """Verify the declared operation, result field and row-level lineage."""
        lineage = manifest["lineage"]
        if digest(lineage) != manifest["source_sha256"]:
            raise ValueError("Derived measurement lineage digest mismatch")
        field = lineage["output"]
        if manifest["profile"]["fields"] != [field]:
            raise ValueError("Derived product field disagrees with lineage")
        required = {
            field["name"]: pl.Float64,
            "time_utc": pl.Datetime("ns", "UTC"),
            "scan_id": pl.UInt64,
            "quality_flags": pl.List(pl.String),
            "calibration_refs": pl.List(pl.String),
            "repaired": pl.Boolean,
            **{
                f"parent_{role}_scan_id": pl.UInt64
                for role in lineage["parents"]
            },
        }
        schema = scans.collect_schema()
        if any(schema.get(name) != dtype for name, dtype in required.items()):
            raise ValueError(
                "Derived product schema disagrees with its contract"
            )
        self.lazy_frame = scans.with_columns(
            pl.lit(manifest["source_sha256"]).alias("source_sha256")
        )
        self.lazy_issues = issues
        self.manifest = manifest
        self.grids = {}
