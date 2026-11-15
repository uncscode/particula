"""Explicit upload/download of fixed-schema registered field storage."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from particula.extension_fields import ExtensionFields, FieldSpec


class _DeviceArray(Protocol):
    """The explicit observation surface required from retained device arrays."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Expose immutable dimensions without reading device values."""
        ...

    def reshape(self, shape: tuple[int, ...]) -> "_DeviceArray":
        """Make a setup-only view sharing the same device allocation."""
        ...

    def numpy(self) -> NDArray[np.generic]:
        """Download a host array at an explicit observation boundary."""
        ...


@dataclass(frozen=True)
class FieldBuffers:
    """Own descriptors and device arrays without CPU simulation state."""

    specs: tuple[FieldSpec, ...]
    dimensions: tuple[tuple[str, int], ...]
    arrays: dict[str, _DeviceArray]
    ledgers: dict[str, _DeviceArray]

    @classmethod
    def upload(cls, fields: ExtensionFields, dimensions, device: str):
        """Copy each supported dtype once on the caller-selected stream."""
        import warp as wp

        def upload_array(array):
            return wp.array(
                array, dtype=getattr(wp, array.dtype.name), device=device
            )

        return cls(
            tuple(fields.specs.values()),
            tuple(dimensions.items()),
            {name: upload_array(array) for name, array in fields.items()},
            {
                name: upload_array(array)
                for name, array in fields.transaction_arrays()
            },
        )

    def download(self) -> ExtensionFields:
        """Download detached CPU values at an explicit transfer boundary."""
        result = ExtensionFields(dict(self.dimensions))
        for spec in self.specs:
            result.add(spec, self.arrays[spec.name].numpy().copy())
        for name, target in result.transaction_arrays():
            target[:] = self.ledgers[name].numpy()
        return result
