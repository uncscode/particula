"""Execute versioned scalar measurement operations on aligned Polars queries."""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

import polars as pl
from jsonschema import Draft202012Validator

from particula.data._binding import _affine
from particula.data.deployment import utc_time
from particula.data.history import add_history
from particula.data.measurement import Measurement
from particula.data.product import (
    DerivedMeasurement,
    FieldContract,
    product_fields,
)
from particula.data.profile import _RESERVED, canonical, digest
from particula.data.reader import ISSUE_SCHEMA
from particula.data.windows import checked_clock

if TYPE_CHECKING:
    from particula.data.recipe import Recipe

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _empty_labels() -> pl.Expr:
    return pl.lit([], dtype=pl.List(pl.String))


def _tag(condition: pl.Expr, reason: str) -> pl.Expr:
    return pl.when(condition).then(pl.lit(reason)).otherwise(None)


def _unit_transform(source: str, target: str) -> tuple[float, float]:
    try:
        return _affine(source, target)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Incompatible units: {source!r} and {target!r}"
        ) from exc


def _applied_corrections(product: Measurement) -> set[str]:
    """Carry forward declared process and vendor-enabled correction records."""
    applied = set(product.manifest.get("corrections", []))
    profile = product.manifest["profile"]
    preserved = profile.get("spectrum", {}).get("preserve_preamble", [])
    for segment in product.manifest.get("segments", []):
        for key in preserved:
            if "correction" in key.lower() and str(
                segment["preamble"].get(key)
            ).lower() in ("yes", "true", "on", "1"):
                applied.add(key)
    return applied


class InstrumentProducts(Protocol):
    """A deployment capable of resolving an operation role to one product."""

    def compatible_product(self, contract: FieldContract) -> Measurement:
        """Return the sole product compatible with the physical contract."""
        ...


def _project_role(
    product: Measurement, role: str, contract: FieldContract
) -> tuple[pl.LazyFrame, dict]:
    """Normalize one input and qualify its columns and upstream scan IDs."""
    definition = contract.resolve(product_fields(product))
    scale, offset = _unit_transform(definition["output_unit"], contract.unit)
    schema = product.lazy_frame.collect_schema()
    if schema["time_utc"] != pl.Datetime("ns", "UTC"):
        raise ValueError("Process inputs require compatible UTC clocks")
    link = f"parent_{role}_scan_id"
    value = pl.col(definition["name"]) * scale + offset
    flags = (
        pl.col("quality_flags")
        if "lineage" in product.manifest
        else (
            pl.col("vendor_flags")
            if "vendor_flags" in schema
            else _empty_labels()
        )
    )
    local_refs = (
        pl.col("calibration_refs")
        if "lineage" in product.manifest
        else _empty_labels()
    )

    projection = checked_clock(product).select(
        "time_utc",
        pl.col("scan_id").alias(link),
        value.alias(f"value_{role}"),
        flags.alias(f"flags_{role}"),
        local_refs.alias(f"refs_{role}"),
        pl.col("repaired").alias(f"repaired_{role}"),
        *[
            pl.col(name).alias(f"parent_{role}__{name[7:]}")
            for name in schema
            if "lineage" in product.manifest
            and name.startswith("parent_")
            and name.endswith(("_scan_id", "_scan_ids"))
        ],
    )
    return projection, definition


@dataclass(frozen=True)
class MeasurementProcess:
    """A versioned Polars operation over physically declared inputs.

    Args:
        operation: Package-qualified operation ID.
        version: Implementation version used to replay recipes.
        inputs: Ordered roles with scalar quantity and required unit. The
            first role supplies the output clock for exact-time alignment.
        output: Result quantity and unit.
        output_field: Column name for the computed values.
        implementation: Trusted callable receiving named input expressions,
            calibrated coefficient expressions and validated parameters.
        coefficient_units: Required units of effective-dated calibration
            coefficients on the first role's deployment.
        parameter_schema: JSON Schema for ordinary operation parameters.
        mode: Elementwise, ordered (e.g. rolling), or whole-record evaluation.
        positive_inputs: Roles requiring values strictly greater than zero.
        same_volume_basis: Require a common inlet and declared volume basis.
        history_parameters: Names of the trailing sample count and maximum
            clock gap (seconds) for an ordered process.
        correction: Optional correction identifier, also checked against
            vendor preamble records before applying it.
    """

    operation: str
    version: str
    inputs: dict[str, FieldContract]
    output: FieldContract
    output_field: str
    implementation: Callable[..., pl.Expr]
    coefficient_units: dict[str, str] = field(default_factory=dict)
    parameter_schema: dict = field(
        default_factory=lambda: {
            "type": "object",
            "additionalProperties": False,
        }
    )
    mode: str = "elementwise"
    positive_inputs: tuple[str, ...] = ()
    same_volume_basis: bool = False
    history_parameters: tuple[str, str] | None = None
    correction: str | None = None
    parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject incomplete contracts before they can enter a registry."""
        if (
            not re.fullmatch(
                r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", self.operation
            )
            or not self.version
        ):
            raise ValueError("Expected a qualified operation ID and version")
        if not self.inputs or not all(
            _NAME.fullmatch(key) for key in self.inputs
        ):
            raise ValueError("Expected named input roles")

        if (
            not _NAME.fullmatch(self.output_field)
            or self.output_field
            in (
                _RESERVED
                | {"quality_flags", "calibration_refs", "reported_time_utc"}
            )
            or self.output_field.startswith("parent_")
        ):
            raise ValueError("Invalid or reserved output field")
        if self.output.axes or self.output.basis:
            raise ValueError("A scalar output contract is required")
        if set(self.positive_inputs).difference(self.inputs):
            raise ValueError("Unknown positive input role")
        if self.mode not in ("elementwise", "ordered", "whole_record"):
            raise ValueError("Unknown process execution mode")
        if self.inputs.keys() & self.coefficient_units.keys():
            raise ValueError(
                "Input roles and coefficient names must be distinct"
            )
        Draft202012Validator.check_schema(self.parameter_schema)

    def describe(self) -> dict:
        """Return a portable description for discovery and recipe authors."""
        return {
            "operation": self.operation,
            "version": self.version,
            "inputs": {
                role: asdict(contract) for role, contract in self.inputs.items()
            },
            "output": asdict(self.output),
            "output_field": self.output_field,
            "coefficient_units": self.coefficient_units,
            "parameters": self.parameter_schema,
            "mode": self.mode,
            "positive_inputs": list(self.positive_inputs),
            "history_parameters": list(self.history_parameters)
            if self.history_parameters
            else None,
            "correction": self.correction,
            "same_volume_basis": self.same_volume_basis,
        }

    def configure(self, **parameters: object) -> "MeasurementProcess":
        """Validate parameters and select a calibration reference or group."""
        configured = replace(self, parameters={**self.parameters, **parameters})
        configured._arguments()
        return configured

    def _arguments(self) -> tuple[dict, str | None]:
        arguments = dict(self.parameters)
        selector = (
            arguments.pop("calibration", None)
            if self.coefficient_units
            else None
        )
        if self.coefficient_units and (
            not isinstance(selector, str) or not selector
        ):
            raise ValueError(
                f"{self.operation} requires a calibration reference"
            )
        if arguments.keys() & (
            self.inputs.keys() | self.coefficient_units.keys()
        ):
            raise ValueError("A parameter shadows an input role or coefficient")
        error = next(
            Draft202012Validator(self.parameter_schema).iter_errors(arguments),
            None,
        )
        if error:
            raise ValueError(f"{self.operation} parameters: {error.message}")
        canonical(arguments)
        return arguments, selector

    def bind(self, *, output: str, **inputs: str) -> "Recipe":
        """Bind role and result aliases for ordered Python composition."""
        from particula.data.recipe import Recipe

        return Recipe.from_process(self, inputs, output)

    def execute(
        self, **inputs: Measurement | InstrumentProducts
    ) -> DerivedMeasurement:
        """Compute a value-added product on the first input's UTC clock.

        Args:
            **inputs: Measurements keyed by the operation's named roles.

        Returns:
            Lazy product retaining unmatched rows, QC, calibration references
            and scan identifiers of the contributing measurements.

        Raises:
            ValueError: Roles, units, deployment, clock or calibration fail
                the operation contract.
        """
        if set(inputs) != set(self.inputs):
            raise ValueError(
                f"Expected input roles {list(self.inputs)}, got {list(inputs)}"
            )
        arguments, selector = self._arguments()
        bound = {}
        role_frames = []
        reasons = []
        refs = []
        repairs = []
        parents = {}
        products = {
            role: value
            if isinstance(value, Measurement)
            else value.compatible_product(self.inputs[role])
            for role, value in inputs.items()
        }
        for role, contract in self.inputs.items():
            product = products[role]
            projection, definition = _project_role(product, role, contract)
            role_frames.append(projection)
            link = f"parent_{role}_scan_id"
            value = pl.col(f"value_{role}")
            reasons.extend(
                [
                    _tag(pl.col(link).is_null(), f"unmatched:{role}"),
                    _tag(
                        pl.col(link).is_not_null() & value.is_null(),
                        f"missing:{role}",
                    ),
                ]
            )
            if role in self.positive_inputs:
                reasons.append(_tag(value <= 0, f"nonpositive:{role}"))
                value = pl.when(value > 0).then(value)
            bound[role] = value
            refs.append(pl.col(f"refs_{role}").fill_null(_empty_labels()))
            repairs.append(pl.col(f"repaired_{role}").fill_null(False))
            parents[role] = {
                "product_id": product.product_id,
                "source_sha256": product.manifest["source_sha256"],
                "clock_corrections": product.manifest.get(
                    "clock_corrections", []
                ),
                "field": definition,
                "deployment": product.manifest.get("deployment"),
                "lineage": product.manifest.get("lineage"),
                "corrections": sorted(_applied_corrections(product)),
            }
        primary = next(iter(self.inputs))
        deployment = products[primary].manifest.get("deployment") or {}
        self._check_deployments(products)
        frame = role_frames[0]
        for other in role_frames[1:]:
            frame = frame.join(other, on="time_utc", how="left", validate="1:1")
        frame = frame.sort("time_utc", f"parent_{primary}_scan_id")
        if self.history_parameters:
            window, gap = (arguments[name] for name in self.history_parameters)
            frame = add_history(frame, tuple(self.inputs), window, gap)
            reasons.append(
                _tag(
                    pl.col(f"parent_{primary}_scan_ids").list.len() < window,
                    "incomplete_window",
                )
            )
        certificates: list[dict] = []
        if selector:
            coefficients, reference, certificates = self._calibrations(
                selector, deployment
            )
            if frame.select(reference.is_null().any()).collect().item():
                raise ValueError(f"Missing or expired calibration {selector!r}")
            bound.update(coefficients)
            refs.append(pl.concat_list(reference))
        output = self.implementation(**bound, **arguments)
        if not isinstance(output, pl.Expr):
            raise TypeError(
                "A measurement operation must return a Polars expression"
            )
        frame = frame.with_columns(output.cast(pl.Float64).alias("_output"))
        finite = pl.col("_output").is_finite()
        valid = finite & pl.all_horizontal(
            [
                pl.col(f"parent_{role}_scan_id").is_not_null()
                for role in self.inputs
            ]
        )
        reasons.append(
            _tag(pl.col("_output").is_not_null() & ~finite, "nonfinite_output")
        )
        flags = [
            pl.col(f"flags_{role}").fill_null(_empty_labels())
            for role in self.inputs
        ]
        flags.append(pl.concat_list(*reasons).list.drop_nulls())
        result = frame.select(
            "time_utc",
            pl.col(f"parent_{primary}_scan_id").alias("scan_id"),
            pl.selectors.starts_with("parent_"),
            pl.any_horizontal(repairs).alias("repaired"),
            pl.concat_list(flags)
            .list.unique()
            .list.sort()
            .alias("quality_flags"),
            pl.concat_list(refs)
            .list.unique()
            .list.sort()
            .alias("calibration_refs"),
            pl.when(valid).then(pl.col("_output")).alias(self.output_field),
        )
        definition = {
            "name": self.output_field,
            "quantity": self.output.quantity,
            "output_unit": self.output.unit,
        }
        lineage = {
            "operation": self.operation,
            "version": self.version,
            "contract": self.describe(),
            "parameters": self.parameters,
            "calibrations": certificates,
            "parents": parents,
            "alignment": {"policy": "exact", "anchor": primary},
            "quality_policy": "propagate",
            "output": definition,
        }
        lineage = json.loads(canonical(lineage))
        manifest = {
            "format_version": 1,
            "source_sha256": digest(lineage),
            "profile": {"fields": [definition]},
            "deployment": deployment,
            "grids": {},
            "corrections": sorted(
                set().union(
                    *(_applied_corrections(p) for p in products.values())
                )
                | ({self.correction} if self.correction else set())
            ),
            "lineage": lineage,
        }
        return DerivedMeasurement(
            result, pl.LazyFrame(schema=ISSUE_SCHEMA), manifest
        )

    def _check_deployments(self, products: dict[str, Measurement]) -> None:
        """Check reference conditions and previously applied corrections."""
        grids = {
            tuple(product.grids)
            for role, product in products.items()
            if self.inputs[role].axes
        }
        if len(grids) > 1:
            raise ValueError("Array roles require a common coordinate grid")
        if self.same_volume_basis:
            for key in ("volume_basis", "inlet"):
                values = {
                    (p.manifest.get("deployment") or {}).get(key)
                    for p in products.values()
                }
                if None in values or "" in values or len(values) != 1:
                    raise ValueError(f"Inputs require a common declared {key}")
        correction = self.correction
        if not correction:
            return
        for product in products.values():
            if correction in _applied_corrections(product):
                raise ValueError(f"Correction already applied: {correction}")

    def _calibrations(
        self, selector: str, deployment: dict
    ) -> tuple[dict, pl.Expr, list]:
        records = sorted(
            (
                r
                for r in deployment.get("calibrations", [])
                if r["operation"] == self.operation
                and selector in (r["reference"], r["group"])
            ),
            key=lambda r: r["start"],
        )
        if not records:
            raise ValueError(
                f"No calibration {selector!r} for {self.operation}"
            )
        values = {
            name: pl.lit(None, dtype=pl.Float64)
            for name in self.coefficient_units
        }
        ref = pl.lit(None, dtype=pl.String)
        stop = None
        for record in records:
            start = utc_time(record["start"])
            if stop is not None and start < stop:
                raise ValueError(
                    f"Overlapping calibration intervals for {selector}"
                )
            stop = utc_time(record["stop"])
            if record["coefficients"].keys() != self.coefficient_units.keys():
                raise ValueError(
                    "Calibration coefficients disagree with operation"
                )
            active = (pl.col("time_utc") >= start) & (pl.col("time_utc") < stop)
            for key, unit in self.coefficient_units.items():
                scale, offset = _unit_transform(record["units"][key], unit)
                value = record["coefficients"][key] * scale + offset
                values[key] = pl.when(active).then(value).otherwise(values[key])
            ref = (
                pl.when(active).then(pl.lit(record["reference"])).otherwise(ref)
            )
        return values, ref, records
