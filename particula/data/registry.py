"""Register trusted measurement algorithms and custom readers."""

from collections.abc import Callable
from functools import partial

import polars as pl

from particula.data._binding import _affine
from particula.data.measurement import Measurement
from particula.data.process import MeasurementProcess
from particula.data.product import FieldContract, product_fields
from particula.data.reader import BatchReader, MeasurementReader


def _concentration(rate: pl.Expr, sample_flow: pl.Expr) -> pl.Expr:
    """Divide counts/s by sampled cm³/s to obtain particles/cm³."""
    return rate / sample_flow


def _moving_average(
    signal: pl.Expr, *, window: int, max_gap_seconds: float
) -> pl.Expr:
    """Compute complete trailing sample means, resetting after a clock gap."""
    gap = (
        pl.col("time_utc").diff().dt.total_nanoseconds() > max_gap_seconds * 1e9
    )
    return signal.rolling_mean(window_size=window, min_samples=window).over(
        gap.fill_null(False).cum_sum()
    )


class OperationRegistry:
    """Select versioned process strategies and trusted user readers by name."""

    def __init__(self) -> None:
        """Register the particle concentration and causal voltage mean."""
        self._processes: dict[tuple[str, str], MeasurementProcess] = {}
        self._readers: dict[str, Callable] = {}
        self._batch_readers: dict[str, Callable[..., BatchReader]] = {
            "delimited": MeasurementReader
        }
        self.register(
            MeasurementProcess(
                "particula.particle_counter.number_concentration",
                "1",
                inputs={
                    "rate": FieldContract("count_rate", "count/s"),
                    "sample_flow": FieldContract("volumetric_flow", "cm^3/s"),
                },
                output=FieldContract("number_concentration", "1/cm^3"),
                output_field="number_concentration",
                implementation=_concentration,
                positive_inputs=("sample_flow",),
                same_volume_basis=True,
            )
        )
        self.register(
            MeasurementProcess(
                "particula.signal.voltage_mean",
                "1",
                inputs={"signal": FieldContract("voltage", "V")},
                output=FieldContract("voltage", "V"),
                output_field="mean_voltage",
                implementation=_moving_average,
                mode="ordered",
                history_parameters=("window", "max_gap_seconds"),
                parameter_schema={
                    "type": "object",
                    "properties": {
                        "window": {"type": "integer", "minimum": 1},
                        "max_gap_seconds": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                        },
                    },
                    "required": ["window", "max_gap_seconds"],
                    "additionalProperties": False,
                },
            )
        )

    def register(self, process: MeasurementProcess) -> None:
        """Register one implementation for a qualified ID and version."""
        key = process.operation, process.version
        if key in self._processes:
            raise ValueError(f"Duplicate process registration: {key}")
        self._processes[key] = process

    def resolve(self, operation: str, version: str) -> MeasurementProcess:
        """Resolve a versioned operation or report the registered choices."""
        try:
            return self._processes[operation, version]
        except KeyError as exc:
            raise ValueError(
                f"Unknown operation/version {(operation, version)}; "
                f"registered: {list(self._processes)}"
            ) from exc

    def describe(self) -> list[dict]:
        """Describe registered operations, units and parameter schemas."""
        return [process.describe() for process in self._processes.values()]

    def available_operations(self, product: Measurement) -> dict[str, dict]:
        """List compatible and required roles for each operation."""
        available = {}
        for (name, version), process in self._processes.items():
            matches = []
            for role, contract in process.inputs.items():
                try:
                    field = contract.resolve(product_fields(product))
                    _affine(field["output_unit"], contract.unit)
                except (ValueError, TypeError):
                    continue
                matches.append(role)
            if matches:
                available[f"{name}@{version}"] = {
                    "compatible_roles": matches,
                    **process.describe(),
                }
        return available

    def register_reader(self, name: str, reader: Callable) -> None:
        """Register a callable accepting source, profile and reader limits."""
        if not name or name == "delimited" or name in self._readers:
            raise ValueError(
                f"Duplicate or reserved reader registration: {name}"
            )
        self._readers[name] = reader

    def register_batch_reader(
        self, name: str, factory: Callable[..., BatchReader]
    ) -> None:
        """Register a batch strategy for both memory and disk imports."""
        from particula.data.storage import read_measurement

        self.register_reader(
            name, partial(read_measurement, reader_factory=factory)
        )
        self._batch_readers[name] = factory

    def batch_reader(self, name: str) -> Callable[..., BatchReader]:
        """Return the batch strategy for use with import_to_parquet."""
        if name not in self._batch_readers:
            raise ValueError(f"Unknown batch reader {name!r}")
        return self._batch_readers[name]

    def reader(self, name: str) -> Callable:
        """Resolve a built-in or caller-installed reader strategy."""
        from particula.data.storage import read_measurement

        if name == "delimited":
            return read_measurement
        if name not in self._readers:
            raise ValueError(
                f"Unknown reader {name!r}; "
                f"choices: {['delimited', *self._readers]}"
            )
        return self._readers[name]
