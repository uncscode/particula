"""Read measurements from an instrument with a declared import profile."""

from dataclasses import dataclass, field
from os import PathLike

import polars as pl

from particula.data._binding import _affine
from particula.data.deployment import Deployment, utc_time
from particula.data.measurement import Measurement
from particula.data.process import MeasurementProcess
from particula.data.product import FieldContract, product_fields
from particula.data.profile import ProfileSource, describe_profile
from particula.data.registry import OperationRegistry


@dataclass
class Instrument:
    """Read delimited exports that share one profile.

    Args:
        profile: Built-in profile ID, JSON or YAML file path, or profile
            mapping. Construction validates and stores a copy of the profile.
        deployment: Instance, inlet, volume basis and calibration records.
        registry: Trusted operations and reader implementations.
        reader: Registered reader name, defaulting to delimited text.

    Raises:
        ProfileError: The profile cannot be loaded or fails validation.
    """

    profile: ProfileSource

    deployment: Deployment | None = field(default=None, kw_only=True)
    registry: OperationRegistry = field(
        default_factory=OperationRegistry, kw_only=True
    )
    reader: str = field(default="delimited", kw_only=True)
    _products: dict[str, Measurement] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Resolve the supplied profile before reading any sources."""
        self.profile = describe_profile(self.profile)
        self.registry.reader(self.reader)
        if self.deployment is not None:
            self.deployment.record()

    def read(
        self,
        source: str | PathLike[str],
        *,
        as_product: str | None = None,
        **limits: int,
    ) -> Measurement:
        """Read one export into memory using this instrument's profile.

        Args:
            source: Path to the delimited instrument export.
            as_product: Optional unique name for later role binding.
            **limits: Optional chunk, record and batch limits accepted by
                MeasurementReader.

        Returns:
            Measurement with scans in the profile's output units, source
            diagnostics and the import manifest.
        """
        if as_product is not None and (
            not as_product or as_product in self._products
        ):
            raise ValueError(
                f"Invalid or duplicate product name: {as_product!r}"
            )
        product = self.registry.reader(self.reader)(
            source, profile=self.profile, **limits
        )
        if not isinstance(product, Measurement):
            raise TypeError("Reader must return a validated Measurement")
        if self.deployment is not None:
            deployment = self.deployment.record()
            outside = pl.lit(False)
            if deployment["start"]:
                outside |= pl.col("time_utc") < utc_time(deployment["start"])
            if deployment["stop"]:
                outside |= pl.col("time_utc") >= utc_time(deployment["stop"])
            if (
                deployment["start"] or deployment["stop"]
            ) and product.lazy_frame.filter(outside).limit(1).collect().height:
                raise ValueError("Observation outside the deployment interval")
            product = type(product)(
                product.lazy_frame,
                product.lazy_issues,
                {**product.manifest, "deployment": deployment},
            )
        if as_product is not None:
            self._products[as_product] = product
        return product

    def product(self, name: str) -> Measurement:
        """Return the named product previously read for this instrument."""
        return self._products[name]

    def compatible_product(self, contract: FieldContract) -> Measurement:
        """Choose a unique product that can fill a physical input role."""
        matches = []
        for name, product in self._products.items():
            try:
                definition = contract.resolve(product_fields(product))

                _affine(definition["output_unit"], contract.unit)
            except (TypeError, ValueError):
                continue
            matches.append(name)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one compatible product; choices: {matches}"
            )
        return self.product(matches[0])

    def available_operations(self, as_product: str) -> dict[str, dict]:
        """Describe registered operations compatible with a bound product."""
        return self.registry.available_operations(self.product(as_product))

    def register_process(self, process: MeasurementProcess) -> None:
        """Expose a trusted versioned process on this instrument's registry."""
        self.registry.register(process)

    def process(self, operation: str, version: str) -> MeasurementProcess:
        """Look up a named measurement operation for explicit execution."""
        return self.registry.resolve(operation, version)
