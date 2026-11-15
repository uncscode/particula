"""Explicit descriptors and storage for persistent user simulation fields."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

_LAYOUTS = {
    ("box",),
    ("box", "particle"),
    ("box", "gas_species"),
    ("box", "particle_species"),
    ("box", "particle", "particle_species"),
}
_DTYPES = {np.dtype("float64"), np.dtype("int32"), np.dtype("int64")}


@dataclass(frozen=True)
class FieldSpec:
    """Describe persistent state; units are metadata, not a conversion rule.

    Particle policies are passive or additive per physical particle. Births
    use initial_value, while retired slots use zero. Bounds apply to active
    state; finite values are always required.
    """

    name: str
    axes: tuple[str, ...]
    units: str
    dtype: str = "float64"
    initial_value: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    particle_policy: str = "passive"
    version: int = 1

    def __post_init__(self):  # noqa: C901
        """Reject ambiguous schemas before allocating storage."""
        if not isinstance(self.name, str) or "." not in self.name:
            raise ValueError("field name must be namespaced")
        if self.name.split(".")[0] in {
            "particles",
            "gas",
            "environment",
            "__field_ledger__",
        }:
            raise ValueError("field namespace conflicts with native state")
        if any(not part.isidentifier() for part in self.name.split(".")):
            raise ValueError("field name must contain identifier components")
        if type(self.axes) is not tuple or self.axes not in _LAYOUTS:
            raise ValueError("unsupported field axes")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("field units must be explicit")
        dtype = np.dtype(self.dtype)
        if dtype not in _DTYPES:
            raise ValueError("unsupported field dtype")
        object.__setattr__(self, "dtype", dtype.name)
        if self.particle_policy not in ("passive", "additive"):
            raise ValueError("unsupported particle field policy")
        if self.particle_policy == "additive" and (
            "particle" not in self.axes or dtype != np.dtype("float64")
        ):
            raise ValueError("additive fields require float64 particle values")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported field schema version")
        for value in (self.initial_value, self.minimum, self.maximum):
            if value is not None and not np.isfinite(value):
                raise ValueError("field defaults and bounds must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("field bounds are reversed")
        self.validate(np.asarray(self.initial_value))
        if dtype.kind == "i":
            limits = np.iinfo(dtype)
            if (
                self.initial_value != int(self.initial_value)
                or not limits.min <= self.initial_value <= limits.max
            ):
                raise ValueError("initial value is not representable")

    def validate(self, values):
        """Check finite values and declared physical bounds."""
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{self.name} contains nonfinite values")
        if self.minimum is not None and np.any(values < self.minimum):
            raise ValueError(f"{self.name} is below its minimum")
        if self.maximum is not None and np.any(values > self.maximum):
            raise ValueError(f"{self.name} exceeds its maximum")


class ExtensionFields:
    """Own named numeric arrays with explicit shape and dtype admission."""

    def __init__(self, dimensions):
        """Bind named axis sizes from an owning aerosol schema."""
        self._dimensions = dict(dimensions)
        self._specs: dict[str, FieldSpec] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._ledgers: dict[str, np.ndarray] = {}

    @property
    def specs(self):
        """Return a read-only view of registered descriptors."""
        return MappingProxyType(self._specs)

    def __getitem__(self, name):
        """Return a held numeric field by identity."""
        return self._arrays[name]

    def add(self, spec, values=None):
        """Register owned storage without replacing an existing field."""
        if type(spec) is not FieldSpec:
            raise TypeError("spec must be FieldSpec")
        if spec.name in self._specs:
            raise ValueError("field already registered")
        shape = tuple(self._dimensions[axis] for axis in spec.axes)
        if values is None:
            array = np.full(shape, spec.initial_value, dtype=spec.dtype)
        else:
            array = np.asarray(values)
            if array.shape != shape or array.dtype != np.dtype(spec.dtype):
                raise ValueError("field shape or dtype differs from schema")
            array = array.copy()
        if "particle" not in spec.axes:
            spec.validate(array)
        elif not np.all(np.isfinite(array)):
            raise ValueError("particle field values must be finite")
        self._specs[spec.name] = spec
        self._arrays[spec.name] = array
        if "particle" in spec.axes and spec.particle_policy == "additive":
            ledger_shape = (shape[0],) + shape[2:]
            for event in ("source", "removal"):
                self._ledgers[f"__field_ledger__.{event}.{spec.name}"] = (
                    np.zeros(ledger_shape, dtype=np.float64)
                )

    def copy(self):
        """Detach every field while retaining immutable descriptors."""
        result = ExtensionFields(self._dimensions)
        for name, spec in self._specs.items():
            result.add(spec, self._arrays[name])
        result._ledgers = {k: v.copy() for k, v in self._ledgers.items()}
        return result

    def items(self):
        """Iterate registered names and arrays in registration order."""
        return self._arrays.items()

    def box_view(self, box):
        """Retain writable slices of staged storage for one box."""
        dimensions = dict(self._dimensions, box=1)
        result = ExtensionFields(dimensions)
        result._specs = self._specs.copy()
        result._arrays = {
            name: array[box : box + 1] for name, array in self._arrays.items()
        }
        result._ledgers = {
            name: array[box : box + 1] for name, array in self._ledgers.items()
        }
        return result

    def validate_layout(self, dimensions, population, active):
        """Validate shape, population semantics and active-field values."""
        if self._specs and dimensions != self._dimensions:
            raise ValueError("extension field layout differs from aerosol")
        if any(not np.all(np.isfinite(v)) for v in self._ledgers.values()):
            raise ValueError("field inventory ledger must be finite")
        for name, spec in self._specs.items():
            array = self._arrays[name]
            shape = tuple(dimensions[axis] for axis in spec.axes)
            if array.shape != shape or array.dtype != np.dtype(spec.dtype):
                raise ValueError("extension field shape or dtype changed")
            if "particle" in spec.axes:
                if population != "particle_resolved":
                    raise ValueError("particle fields require resolved storage")
                if np.any(array[~active] != 0):
                    raise ValueError("inactive particle fields must be zero")
                spec.validate(array[active])
            else:
                spec.validate(array)

    def particle_names(self):
        """Return names whose values follow resolved particle slots."""
        return tuple(
            name
            for name, spec in self._specs.items()
            if "particle" in spec.axes
        )

    def require_additive(self):
        """Reject undefined merger rules before selecting collisions."""
        for name in self.particle_names():
            if self._specs[name].particle_policy != "additive":
                raise ValueError("population merging requires additive fields")

    def merge_particles(self, donor, recipient):
        """Apply admitted single-box resolved pairs to additive quantities."""
        self.require_additive()
        candidates = {}
        for name in self.particle_names():
            values = self._arrays[name].copy()
            with np.errstate(over="ignore", invalid="ignore"):
                np.add.at(values[0], recipient, values[0, donor].copy())
            values[0, donor] = 0
            if not np.all(np.isfinite(values)):
                raise ValueError("particle tracer merger overflow")
            candidates[name] = values
        for name, values in candidates.items():
            np.copyto(self._arrays[name], values)

    def _resample_particles(self, plan):
        """Transport tracers using the native remap's exact count overlaps.

        Called on private source-transaction state after native plan validation.
        Changing representation adds no source or removal inventory.
        """
        self.require_additive()
        for box, remap in enumerate(plan.box_plans):
            if not remap.retained_indices:
                continue
            for name in self.particle_names():
                values = self._arrays[name]
                source = values[box].copy()
                values[box] = 0
                for target, donor, overlap in remap.transport:
                    values[box, target] += overlap * source[donor]
                retained = np.asarray(remap.retained_indices)
                counts = np.asarray(remap.replacement_concentrations)
                divisor = counts if values.ndim == 2 else counts[:, None]
                values[box, retained] /= divisor
                if not np.all(np.isfinite(values[box])):
                    raise ValueError("particle tracer resampling overflow")

    def _activate_particles(self, before_counts, after_counts):
        """Initialize newly active slots and record their additive inventory."""
        born = (before_counts == 0) & (after_counts > 0)
        for name in self.particle_names():
            spec = self._specs[name]
            values = self._arrays[name]
            values[born] = spec.initial_value
            if spec.particle_policy == "additive":
                counts = np.where(born, after_counts, 0)
                weights = counts if values.ndim == 2 else counts[..., None]
                self._ledgers[f"__field_ledger__.source.{name}"] += np.sum(
                    weights * values, axis=1
                )

    def transaction_arrays(self):
        """Expose internal inventory arrays to transactional publication."""
        return self._ledgers.items()

    def inventory_changes(self, name):
        """Return detached cumulative source and removal inventories per box."""
        return {
            event: self._ledgers[f"__field_ledger__.{event}.{name}"].copy()
            for event in ("source", "removal")
        }

    def retire_particles(self, before_counts, after_counts):
        """Account count-only sinks, then clear retired slot values."""
        if np.any(after_counts > before_counts):
            raise ValueError("sink unexpectedly increased represented counts")
        removed = before_counts - after_counts
        for name in self.particle_names():
            values = self._arrays[name]
            if self._specs[name].particle_policy == "additive":
                weights = removed if values.ndim == 2 else removed[..., None]
                self._ledgers[f"__field_ledger__.removal.{name}"] += np.sum(
                    weights * values, axis=1
                )
            values[after_counts == 0] = 0


def field_dimensions(particles, gas):
    """Resolve independent particle-material and gas-species widths."""
    return {
        "box": particles.n_boxes,
        "particle": particles.n_particles,
        "particle_species": particles.n_species,
        "gas_species": gas.concentration.shape[1],
    }
