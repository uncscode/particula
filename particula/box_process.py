"""Public, staged single-box Python process extensions."""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from particula import _cpu_batch
from particula.execution.cpu_context import CPUExecutionContext
from particula.runnable import RunnableABC

_WRITABLE = frozenset(
    {
        "particles.masses",
        "particles.concentration",
        "particles.charge",
        "particles.volume",
        "gas.concentration",
        "environment.temperature",
        "environment.pressure",
    }
)


def validate_time(time):
    """Validate absolute simulation time in seconds without advancing it."""
    if isinstance(time, (bool, np.bool_)) or not isinstance(time, Real):
        raise TypeError("time must be a real non-boolean scalar")
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    return float(time)


@dataclass(frozen=True)
class BoxContext:
    """Logical box identity and optional staged random generator."""

    box_id: str
    rng: np.random.Generator | None = None


class BoxProcess(RunnableABC):
    """Implement validate and advance to compose a custom Python process.

    Advances receive ordinary native containers with a leading box axis.
    Only declared writes are published after the complete batch succeeds.
    External side effects and hidden callback state cannot be rolled back.
    """

    def __init__(  # noqa: C901
        self,
        *,
        name,
        reads=(),
        writes=(),
        stochastic=False,
        representations=("discrete", "continuous_pdf", "particle_resolved"),
        extension_id=None,
        version="1",
        parameters=(),
    ):
        """Declare instance identity, field access and scientific support."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("process name must be nonempty")
        if type(stochastic) is not bool:
            raise TypeError("stochastic must be a bool")
        for label, value in (
            ("extension_id", extension_id),
            ("version", version),
        ):
            if (label == "version" or value is not None) and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{label} must be a nonempty string")
        for label, values in (
            ("reads", reads),
            ("writes", writes),
            ("representations", representations),
        ):
            if isinstance(values, str) or not isinstance(values, (tuple, list)):
                raise TypeError(f"{label} must be a tuple or list")
            if any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"{label} must contain nonempty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {label}")
        if not representations or not set(representations) <= {
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        }:
            raise ValueError("unknown or empty particle representations")
        self.name = name
        self.reads = tuple(reads)
        self.writes = tuple(writes)
        self.stochastic = stochastic
        self.representations = tuple(representations)
        self.extension_id = extension_id or name
        from particula.execution.process_extensions import _immutable

        if type(parameters) is not tuple or not _immutable(parameters):
            raise ValueError("parameters must be immutable scalar tuples")
        self.parameters = parameters
        self.version = version

    def validate(self, aerosol):
        """Check process requirements without modifying the supplied state."""

    def advance(self, aerosol, *, time, time_step, context):
        """Modify declared arrays in one box over the supplied interval [s]."""
        raise NotImplementedError

    def execute(  # noqa: C901
        self, aerosol, time_step, sub_steps=1, *, context=None, time=0.0
    ):
        """Publish a successful process substep atomically across boxes."""
        start = validate_time(time)
        dt = _cpu_batch.duration(time_step, sub_steps)
        validate_time(start + float(time_step))
        _cpu_batch.validate(aerosol)
        fields = _cpu_batch.arrays(aerosol)
        if not set(self.writes) <= _WRITABLE | aerosol.fields.specs.keys():
            raise ValueError("unsupported writer field")
        if not set(self.reads) <= fields.keys():
            raise ValueError("unknown reader field")
        if aerosol.particles.distribution_type not in self.representations:
            raise ValueError("unsupported particle representation")
        if context is not None and not isinstance(context, CPUExecutionContext):
            raise TypeError("context must be CPUExecutionContext")
        if self.stochastic and context is None:
            raise ValueError("stochastic processes require CPUExecutionContext")
        boxes = aerosol.particles.n_boxes
        if context is not None:
            context._begin(boxes)
        _cpu_batch.validate(aerosol, self.writes)
        validation = _cpu_batch.detached(aerosol)
        baseline = _cpu_batch.detached(validation)
        retained = _cpu_batch.arrays(validation)
        containers = (
            validation.particles,
            validation.gas,
            validation.environment,
        )
        result = self.validate(validation)
        if result is not None:
            raise ValueError("validate must return None")
        _check_candidate(validation, baseline, retained, (), containers)
        if dt == 0:
            return aerosol
        for substep in range(sub_steps):
            candidate = _cpu_batch.detached(aerosol)
            transaction = None if context is None else context._begin(boxes)
            for box in range(boxes):
                view = _cpu_batch.box_view(candidate, box)
                before = _cpu_batch.detached(view)
                retained = _cpu_batch.arrays(view)
                containers = (view.particles, view.gas, view.environment)
                for name, array in retained.items():
                    if name not in self.writes:
                        array.flags.writeable = False
                box_context = BoxContext(
                    str(box) if context is None else context.box_ids[box],
                    transaction.generator(
                        f"extension:{self.extension_id}:{self.name}", box
                    )
                    if self.stochastic
                    else None,
                )
                result = self.advance(
                    view,
                    time=start + substep * dt,
                    time_step=dt,
                    context=box_context,
                )
                if result is not None:
                    raise ValueError("advance must return None")
                _check_candidate(
                    view, before, retained, self.writes, containers
                )
            _cpu_batch.publish(aerosol, candidate, self.writes)
            if transaction is not None:
                transaction.commit()
        return aerosol


def _check_candidate(  # noqa: C901
    candidate, before, retained, writes, containers
):
    """Reject replacement, undeclared writes and invalid native state."""
    if any(
        new is not old
        for new, old in zip(
            (candidate.particles, candidate.gas, candidate.environment),
            containers,
            strict=True,
        )
    ):
        raise ValueError("process replaced a native container")
    if candidate.fields.specs != before.fields.specs:
        raise ValueError("process changed field schema")
    if any("particle" in spec.axes for spec in before.fields.specs.values()):
        if not np.array_equal(
            candidate.particles.concentration > 0,
            before.particles.concentration > 0,
        ):
            raise ValueError("custom processes must preserve particle activity")
    from particula.extension_fields import field_dimensions

    candidate.fields.validate_layout(
        field_dimensions(candidate.particles, candidate.gas),
        candidate.particles.distribution_type,
        candidate.particles.concentration > 0,
    )
    actual = _cpu_batch.arrays(candidate)
    original = _cpu_batch.arrays(before)
    if actual.keys() != retained.keys():
        raise ValueError("process changed array structure")
    if (
        candidate.gas.name != before.gas.name
        or candidate.particles.distribution_type
        != before.particles.distribution_type
    ):
        raise ValueError("process changed species or representation")
    for name, array in actual.items():
        if (
            array is not retained[name]
            or array.shape != original[name].shape
            or array.dtype != original[name].dtype
        ):
            raise ValueError(f"process replaced or reshaped {name}")
        if name not in writes and not np.array_equal(array, original[name]):
            raise ValueError(f"undeclared change to {name}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"nonfinite values in {name}")
    candidate.particles.validate_representation()
    candidate.environment.__post_init__()
    if np.any(candidate.gas.concentration < 0):
        raise ValueError("gas concentrations must be nonnegative")


class FunctionProcess(BoxProcess):
    """Adapt a Python function to the public box-process contract."""

    def __init__(self, function, *, validate=None, **kwargs):
        """Retain advancement and optional validation callables."""
        super().__init__(**kwargs)
        if not callable(function) or (
            validate is not None and not callable(validate)
        ):
            raise TypeError("process callbacks must be callable")
        self.function = function
        self.validator = validate

    def validate(self, aerosol):
        """Invoke the optional requirement validator."""
        if self.validator is not None:
            return self.validator(aerosol)

    def advance(self, aerosol, *, time, time_step, context):
        """Delegate one box update without hiding callback exceptions."""
        return self.function(
            aerosol, time=time, time_step=time_step, context=context
        )
