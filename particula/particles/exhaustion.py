"""Particle-capacity exhaustion planning and concrete apply helpers.

P1 consumes fixed-shape capacity sidecars and creates an immutable, later
commit-boundary plan.  It validates every box before resolving any box, writes
no caller state, and can therefore be retried safely with corrected sidecars.
The exact requested/admitted and required-release counts, policy code, and
activation-prefix identity are diagnostics.  Release tuples are empty and
``float("nan")`` is the no-scale sentinel in P1; ``-1`` is the unused-index
sentinel.

Weighted inventory is calculated in float64 as ``sum(weights)``,
``sum(weights[..., None] * masses, axis=particle)``, and
``sum(weights * charge)`` (without materializing the weighted-mass product).
Intensive values divide these quantities by volume.  A later unscaled commit
must equal ``pre_state + source`` and a scaling commit must equal
``scale * pre_state + source``.  No radius-cubed or other moment is exact in
P1; any later preservation claim must specify its moment, domain, and
tolerance before a commit can rely on it.

P2 adds a CPU-only deterministic equal-weight resampling reference. It plans
detached fixed-capacity remaps without mutation, then commits one validated
all-box plan. P2 neither provides GPU parity nor implements activation,
resizing, or compaction. P4 is an opt-in, concrete-module-only direct commit:
after all-box preflight, it scales selected rows' box volume, particle
concentration, and provisional demand by a resolved factor. It consumes
caller-owned sidecars, reports the resolved factor, and neither consumes a P1
plan nor resolves capacity policy, transfers data, resizes capacity, or adds a
runnable.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from particula.particles.particle_data import ParticleData

POLICY_ACTIVATE = 0
POLICY_RESAMPLE_DEFERRED = 1
POLICY_SCALE_DEFERRED = 2


@dataclass(frozen=True)
class ExhaustionControls:
    """Control which deferred policies may represent exhausted capacity.

    Attributes:
        resampling: Whether a sufficiently releasable box may use the deferred
            resampling policy.
        representative_volume_scaling: Whether a box not represented by
            resampling may use the deferred scaling policy.

    Raises:
        TypeError: If either control is not an exact Python ``bool``.
    """

    resampling: bool = True
    representative_volume_scaling: bool = False

    def __post_init__(self) -> None:
        """Reject bool-like values so policy controls remain unambiguous."""
        for name, value in (
            ("resampling", self.resampling),
            (
                "representative_volume_scaling",
                self.representative_volume_scaling,
            ),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be an exact Python bool")


@dataclass(frozen=True)
class ExhaustionInputs:
    """Hold fixed-shape capacity sidecars for exhaustion planning.

    Attributes:
        requested_count: Requested activation count per box, as int32 values
            with shape ``(B,)``.
        free_count: Available free-slot count per box, as int32 values with
            shape ``(B,)``.
        resampling_releasable_count: Count per box that deferred resampling
            could release, as int32 values with shape ``(B,)``.
        free_indices: Ascending free-slot prefixes followed by ``-1`` unused
            sentinels, as int32 values with shape ``(B, N)``.

    Validation occurs when :func:`resolve_exhaustion` consumes this record;
    record construction intentionally performs no coercion or mutation.
    """

    requested_count: NDArray[np.int32]
    free_count: NDArray[np.int32]
    resampling_releasable_count: NDArray[np.int32]
    free_indices: NDArray[np.int32]


@dataclass(frozen=True)
class ExhaustionBoxPlan:
    """Describe one immutable capacity decision at the later commit boundary.

    Attributes:
        requested_count: Requested activation count for the box.
        admitted_count: Count admitted by this plan; equals ``requested_count``
            in P1.
        required_release_count: Additional slots needed after free capacity is
            exhausted.
        releasable_count: Capacity reported as releasable by resampling.
        policy_code: ``POLICY_ACTIVATE``, ``POLICY_RESAMPLE_DEFERRED``, or
            ``POLICY_SCALE_DEFERRED``.
        activation_indices: Exact free-index prefix for activation plans; empty
            for deferred plans.
        release_indices: Empty in P1 because release selection is deferred.
        scale_factor: ``float("nan")`` in P1 because scaling is deferred.
    """

    requested_count: int
    admitted_count: int
    required_release_count: int
    releasable_count: int
    policy_code: int
    activation_indices: tuple[int, ...]
    release_indices: tuple[int, ...]
    scale_factor: float


@dataclass(frozen=True)
class ExhaustionPlan:
    """Hold immutable per-box decisions for one later commit boundary.

    Attributes:
        box_plans: One plan per validated input box, in input-box order.
    """

    box_plans: tuple[ExhaustionBoxPlan, ...]


@dataclass(frozen=True)
class WeightedInventory:
    """Hold tuple-backed float64 extensive and intensive inventories.

    Attributes:
        number: Weighted particle number, ``sum(weights)``, per box.
        mass: Weighted mass, ``sum(weights * masses)``, per box and species.
        charge: Weighted charge, ``sum(weights * charge)``, per box.
        number_per_volume: Weighted number divided by box volume.
        mass_per_volume: Weighted species mass divided by box volume.
        charge_per_volume: Weighted charge divided by box volume.
    """

    number: tuple[np.float64, ...]
    mass: tuple[tuple[np.float64, ...], ...]
    charge: tuple[np.float64, ...]
    number_per_volume: tuple[np.float64, ...]
    mass_per_volume: tuple[tuple[np.float64, ...], ...]
    charge_per_volume: tuple[np.float64, ...]


@dataclass(frozen=True)
class _ResamplingBoxPlan:
    """Hold one detached fixed-capacity equal-weight remap.

    P2 is CPU-only and deliberately fixed-capacity: retained slots receive
    replacement particles and released slots are cleared during the later,
    all-box commit. It does not provide GPU parity, activation, or scaling.
    """

    retained_indices: tuple[int, ...]
    released_indices: tuple[int, ...]
    replacement_masses: tuple[tuple[np.float64, ...], ...]
    replacement_concentrations: tuple[np.float64, ...]
    replacement_charges: tuple[np.float64, ...]
    source_concentrations: tuple[np.float64, ...] = ()
    source_masses: tuple[tuple[np.float64, ...], ...] = ()
    source_charges: tuple[np.float64, ...] = ()


@dataclass(frozen=True)
class _ResamplingPlan:
    """Hold detached P2 remaps in particle-box order for one commit.

    Empty per-box remaps preserve boxes that do not resample, including
    zero-release resampling boxes. The tuple-backed records retain no views
    into the P1 plan or particle storage.
    """

    box_plans: tuple[_ResamplingBoxPlan, ...]


@dataclass(frozen=True)
class _ResamplingBounds:
    """Hold finite nonnegative diagnostic tolerances for P2 remapping."""

    radius_cubed_relative_error: float
    mean_radius_relative_error: float
    surface_relative_error: float
    diversity_absolute_error: float


@dataclass(frozen=True)
class _CachedBoxState:
    """Hold one preflighted box's active values for planning only."""

    active_indices: NDArray[np.intp]
    concentrations: NDArray[np.float64]
    masses: NDArray[np.float64]
    charges: NDArray[np.float64]
    radii: NDArray[np.float64]


def _validate_sidecar_schema(inputs: ExhaustionInputs) -> None:
    """Validate sidecar array types, dtypes, and ranks without coercion."""
    arrays = (
        ("requested_count", inputs.requested_count, 1),
        ("free_count", inputs.free_count, 1),
        (
            "resampling_releasable_count",
            inputs.resampling_releasable_count,
            1,
        ),
        ("free_indices", inputs.free_indices, 2),
    )
    for name, values, rank in arrays:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if values.dtype != np.dtype(np.int32):
            raise TypeError(f"{name} must have dtype int32")
        if values.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")


def _validate_sidecar_shapes(inputs: ExhaustionInputs) -> tuple[int, int]:
    """Validate cross-box sidecar shapes and return box and particle counts."""
    box_count = inputs.requested_count.shape[0]
    if (
        inputs.free_count.shape != (box_count,)
        or inputs.resampling_releasable_count.shape != (box_count,)
        or inputs.free_indices.shape[0] != box_count
    ):
        raise ValueError("sidecars must have matching box dimensions")

    particle_count = inputs.free_indices.shape[1]
    return box_count, particle_count


def _validate_capacity_counts(
    inputs: ExhaustionInputs,
    particle_count: int,
    *,
    allow_oversized_requests: bool = False,
) -> None:
    """Validate count ranges against fixed particle capacity."""
    for name, values in (
        ("requested_count", inputs.requested_count),
        ("free_count", inputs.free_count),
        ("resampling_releasable_count", inputs.resampling_releasable_count),
    ):
        if np.any(values < 0):
            raise ValueError(f"{name} must be nonnegative")
        if name == "requested_count" and allow_oversized_requests:
            continue
        if np.any(values > particle_count):
            raise ValueError(f"{name} exceeds particle capacity")
    maximum_releasable = np.maximum(particle_count - inputs.free_count - 1, 0)
    if np.any(inputs.resampling_releasable_count > maximum_releasable):
        raise ValueError(
            "resampling_releasable_count exceeds active capacity retaining "
            "one slot"
        )


def _validate_free_indices(
    inputs: ExhaustionInputs,
    particle_count: int,
) -> None:
    """Validate free-index prefixes and unused suffix sentinels per box."""
    for box_index, free_count in enumerate(inputs.free_count):
        count = int(free_count)
        prefix = inputs.free_indices[box_index, :count]
        suffix = inputs.free_indices[box_index, count:]
        if np.any(prefix < 0) or np.any(prefix >= particle_count):
            raise ValueError(
                "free_indices prefix contains an out-of-range index"
            )
        if count > 1 and np.any(prefix[1:] <= prefix[:-1]):
            raise ValueError("free_indices prefix must be strictly ascending")
        if np.any(suffix != -1):
            raise ValueError("free_indices unused suffix must contain -1")


def _validate_sidecars(
    inputs: ExhaustionInputs,
    *,
    allow_oversized_requests: bool = False,
) -> tuple[int, int]:
    """Validate all sidecars without coercion, sorting, copying, or mutation."""
    _validate_sidecar_schema(inputs)
    box_count, particle_count = _validate_sidecar_shapes(inputs)
    _validate_capacity_counts(
        inputs,
        particle_count,
        allow_oversized_requests=allow_oversized_requests,
    )
    _validate_free_indices(inputs, particle_count)
    return box_count, particle_count


def resolve_exhaustion(
    inputs: ExhaustionInputs,
    controls: ExhaustionControls,
    *,
    allow_oversized_requests: bool = False,
) -> ExhaustionPlan:
    """Resolve immutable capacity decisions after complete sidecar validation.

    The resolver validates every input box before creating a returned plan.
    Activation uses the exact requested free-index prefix. Exhausted capacity
    uses sufficiently releasable resampling before enabled scaling; both remain
    deferred and therefore have empty release indices and a ``nan`` scale.

    Args:
        inputs: Fixed-shape, int32 capacity sidecars.
        controls: Strict policy enablement controls.
        allow_oversized_requests: Permit provisional requests above physical
            capacity so a later scaling policy can reduce them. The default
            preserves strict fixed-capacity validation.

    Returns:
        A tuple-backed plan with activation-prefix identity or deferred policy.

    Raises:
        TypeError: If either public record has the wrong type.
        ValueError: If sidecars are invalid or no enabled policy can represent
            exhausted capacity.
    """
    if not isinstance(controls, ExhaustionControls):
        raise TypeError("controls must be an ExhaustionControls")
    if not isinstance(inputs, ExhaustionInputs):
        raise TypeError("inputs must be an ExhaustionInputs")

    box_count, _ = _validate_sidecars(
        inputs,
        allow_oversized_requests=allow_oversized_requests,
    )
    box_plans: list[ExhaustionBoxPlan] = []
    for box_index in range(box_count):
        requested = int(inputs.requested_count[box_index])
        free = int(inputs.free_count[box_index])
        releasable = int(inputs.resampling_releasable_count[box_index])
        if requested <= free:
            activation_indices = tuple(
                int(index)
                for index in inputs.free_indices[box_index, :requested]
            )
            box_plans.append(
                ExhaustionBoxPlan(
                    requested_count=requested,
                    admitted_count=requested,
                    required_release_count=0,
                    releasable_count=releasable,
                    policy_code=POLICY_ACTIVATE,
                    activation_indices=activation_indices,
                    release_indices=(),
                    scale_factor=float("nan"),
                )
            )
            continue

        required_release = requested - free
        if controls.resampling and releasable >= required_release:
            policy_code = POLICY_RESAMPLE_DEFERRED
        elif controls.representative_volume_scaling:
            policy_code = POLICY_SCALE_DEFERRED
        else:
            raise ValueError(
                "exhaustion policy cannot represent requested capacity"
            )
        box_plans.append(
            ExhaustionBoxPlan(
                requested_count=requested,
                admitted_count=requested,
                required_release_count=required_release,
                releasable_count=releasable,
                policy_code=policy_code,
                activation_indices=(),
                release_indices=(),
                scale_factor=float("nan"),
            )
        )
    return ExhaustionPlan(box_plans=tuple(box_plans))


def get_weighted_inventory(  # noqa: C901
    masses: object,
    weights: object,
    charge: object,
    volume: object,
) -> WeightedInventory:
    """Calculate float64 weighted inventories without retaining caller arrays.

    Array-like inputs are converted to float64. The mass reduction avoids a
    full weighted-mass broadcast product. Returned tuples do not retain caller
    array storage.

    Args:
        masses: Array-like particle masses with shape ``(B, N, S)``.
        weights: Array-like particle weights with shape ``(B, N)``.
        charge: Array-like particle charge with shape ``(B, N)``.
        volume: Array-like box volumes with shape ``(B,)``.

    Returns:
        Tuple-backed extensive and intensive float64 inventories.

    Raises:
        ValueError: If converted inputs have invalid shape or physical values.
    """
    mass_values = np.asarray(masses, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    charge_values = np.asarray(charge, dtype=np.float64)
    volume_values = np.asarray(volume, dtype=np.float64)
    if mass_values.ndim != 3:
        raise ValueError("masses must have shape (B, N, S)")
    box_count, particle_count, _ = mass_values.shape
    if weight_values.shape != (box_count, particle_count):
        raise ValueError("weights must have shape (B, N)")
    if charge_values.shape != (box_count, particle_count):
        raise ValueError("charge must have shape (B, N)")
    if volume_values.shape != (box_count,):
        raise ValueError("volume must have shape (B,)")
    for name, values in (
        ("masses", mass_values),
        ("weights", weight_values),
        ("charge", charge_values),
        ("volume", volume_values),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
    if np.any(volume_values <= 0):
        raise ValueError("volume must be positive")
    if np.any(mass_values < 0.0):
        raise ValueError("masses must be nonnegative")
    if np.any(weight_values < 0.0):
        raise ValueError("weights must be nonnegative")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        number = np.sum(weight_values, axis=1, dtype=np.float64)
        mass = np.einsum(
            "bn,bns->bs",
            weight_values,
            mass_values,
            dtype=np.float64,
            optimize=True,
        )
        total_charge = np.sum(
            weight_values * charge_values, axis=1, dtype=np.float64
        )
        number_per_volume = number / volume_values
        mass_per_volume = mass / volume_values[:, None]
        charge_per_volume = total_charge / volume_values
    if not all(
        np.all(np.isfinite(values))
        for values in (
            number,
            mass,
            total_charge,
            number_per_volume,
            mass_per_volume,
            charge_per_volume,
        )
    ):
        raise ValueError("weighted inventory reductions must be finite")
    return WeightedInventory(
        number=tuple(np.float64(value) for value in number),
        mass=tuple(tuple(np.float64(value) for value in row) for row in mass),
        charge=tuple(np.float64(value) for value in total_charge),
        number_per_volume=tuple(
            np.float64(value) for value in number_per_volume
        ),
        mass_per_volume=tuple(
            tuple(np.float64(value) for value in row) for row in mass_per_volume
        ),
        charge_per_volume=tuple(
            np.float64(value) for value in charge_per_volume
        ),
    )


def _validate_exact_float(name: str, value: object) -> float:
    """Return an exact finite nonnegative Python float diagnostic bound."""
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact Python float")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _validate_resampling_bounds(
    radius_cubed_relative_error: object,
    mean_radius_relative_error: object,
    surface_relative_error: object,
    diversity_absolute_error: object,
) -> _ResamplingBounds:
    """Validate and detach P2 diagnostic bounds before particle inspection."""
    return _ResamplingBounds(
        radius_cubed_relative_error=_validate_exact_float(
            "radius_cubed_relative_error", radius_cubed_relative_error
        ),
        mean_radius_relative_error=_validate_exact_float(
            "mean_radius_relative_error", mean_radius_relative_error
        ),
        surface_relative_error=_validate_exact_float(
            "surface_relative_error", surface_relative_error
        ),
        diversity_absolute_error=_validate_exact_float(
            "diversity_absolute_error", diversity_absolute_error
        ),
    )


def _validate_particle_schema(  # noqa: C901
    particles: object,
) -> ParticleData:
    """Validate P2's exact writable CPU particle storage schema."""
    if not isinstance(particles, ParticleData):
        raise TypeError("particles must be a ParticleData")
    arrays = (
        ("masses", particles.masses, 3),
        ("concentration", particles.concentration, 2),
        ("charge", particles.charge, 2),
        ("density", particles.density, 1),
        ("volume", particles.volume, 1),
    )
    for name, values, rank in arrays:
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if values.dtype != np.dtype(np.float64):
            raise TypeError(f"{name} must have dtype float64")
        if values.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")
        if not values.flags.writeable:
            raise ValueError(f"{name} must be writable")
    box_count, particle_count, species_count = particles.masses.shape
    if particle_count == 0:
        raise ValueError("particle capacity must be positive")
    if species_count == 0:
        raise ValueError("species capacity must be positive")
    if particles.concentration.shape != (box_count, particle_count):
        raise ValueError("concentration must have shape (B, N)")
    if particles.charge.shape != (box_count, particle_count):
        raise ValueError("charge must have shape (B, N)")
    if particles.density.shape != (species_count,):
        raise ValueError("density must have shape (S,)")
    if particles.volume.shape != (box_count,):
        raise ValueError("volume must have shape (B,)")
    for index, (name, values, _) in enumerate(arrays):
        for other_name, other_values, _ in arrays[index + 1 :]:
            if np.shares_memory(values, other_values):
                raise ValueError(
                    f"{name} and {other_name} must not share writable memory"
                )
    return particles


def _cache_particle_state(  # noqa: C901
    particles: ParticleData,
) -> tuple[_CachedBoxState, ...]:
    """Fully validate particle state once and cache active values by box."""
    if not np.all(np.isfinite(particles.masses)) or np.any(
        particles.masses < 0
    ):
        raise ValueError("masses must be finite and nonnegative")
    if not np.all(np.isfinite(particles.concentration)) or np.any(
        particles.concentration < 0
    ):
        raise ValueError("concentration must be finite and nonnegative")
    if not np.all(np.isfinite(particles.charge)):
        raise ValueError("charge must be finite")
    if not np.all(np.isfinite(particles.density)) or np.any(
        particles.density <= 0
    ):
        raise ValueError("density must be finite and positive")
    if not np.all(np.isfinite(particles.volume)) or np.any(
        particles.volume <= 0
    ):
        raise ValueError("volume must be finite and positive")

    cached: list[_CachedBoxState] = []
    for box_index in range(particles.n_boxes):
        active = particles.concentration[box_index] > 0.0
        inactive = ~active
        if np.any(particles.masses[box_index, inactive] != 0.0) or np.any(
            particles.charge[box_index, inactive] != 0.0
        ):
            raise ValueError("inactive slots must contain literal zero state")
        indices = np.flatnonzero(active)
        # Integer indexing already detaches compact active storage; avoid a
        # second full source copy while preserving the immutable plan boundary.
        masses = particles.masses[box_index, indices]
        material_volume = np.sum(
            masses / particles.density, axis=1, dtype=np.float64
        )
        if np.any(~np.isfinite(material_volume)) or np.any(
            material_volume <= 0.0
        ):
            raise ValueError("active slots must have positive material volume")
        radii = np.cbrt(3.0 * material_volume / (4.0 * np.pi))
        cached.append(
            _CachedBoxState(
                active_indices=indices.astype(np.intp, copy=False),
                concentrations=particles.concentration[box_index, indices],
                masses=masses,
                charges=particles.charge[box_index, indices],
                radii=radii,
            )
        )
    return tuple(cached)


def _validate_p1_plan(  # noqa: C901
    exhaustion_plan: object,
    box_count: int,
    particle_count: int,
) -> ExhaustionPlan:
    """Validate P1 records without resolving policy or retaining their data."""
    if not isinstance(exhaustion_plan, ExhaustionPlan):
        raise TypeError("exhaustion_plan must be an ExhaustionPlan")
    if len(exhaustion_plan.box_plans) != box_count:
        raise ValueError("exhaustion plan must have one box plan per box")
    for box_plan in exhaustion_plan.box_plans:
        if not isinstance(box_plan, ExhaustionBoxPlan):
            raise TypeError("exhaustion plan contains an invalid box plan")
        integer_values = (
            box_plan.requested_count,
            box_plan.admitted_count,
            box_plan.required_release_count,
            box_plan.releasable_count,
            box_plan.policy_code,
        )
        if any(type(value) is not int for value in integer_values):
            raise TypeError("exhaustion box fields must be exact Python ints")
        if any(
            value < 0 or value > particle_count for value in integer_values[:4]
        ):
            raise ValueError("exhaustion counts exceed particle capacity")
        if box_plan.requested_count != box_plan.admitted_count:
            raise ValueError("requested and admitted counts must match")
        if box_plan.required_release_count > box_plan.requested_count:
            raise ValueError("required release count exceeds requested count")
        if box_plan.policy_code not in (
            POLICY_ACTIVATE,
            POLICY_RESAMPLE_DEFERRED,
            POLICY_SCALE_DEFERRED,
        ):
            raise ValueError("exhaustion plan has an unknown policy code")
        if (
            type(box_plan.activation_indices) is not tuple
            or type(box_plan.release_indices) is not tuple
        ):
            raise TypeError("exhaustion indices must be tuples")
        for name, indices in (
            ("activation", box_plan.activation_indices),
            ("release", box_plan.release_indices),
        ):
            if any(type(index) is not int for index in indices):
                raise TypeError(f"{name} indices must be exact Python ints")
            if any(index < 0 or index >= particle_count for index in indices):
                raise ValueError(f"{name} indices are out of range")
            if any(
                right <= left
                for left, right in zip(indices, indices[1:], strict=False)
            ):
                raise ValueError(f"{name} indices must be strictly ascending")
        if box_plan.policy_code == POLICY_RESAMPLE_DEFERRED:
            if (
                box_plan.required_release_count > box_plan.releasable_count
                or box_plan.activation_indices
                or box_plan.release_indices
                or not (
                    type(box_plan.scale_factor) is float
                    and np.isnan(box_plan.scale_factor)
                )
            ):
                raise ValueError("invalid deferred resampling P1 sentinel")
        elif box_plan.policy_code == POLICY_SCALE_DEFERRED:
            if box_plan.release_indices or not (
                type(box_plan.scale_factor) is float
                and np.isnan(box_plan.scale_factor)
            ):
                raise ValueError("invalid deferred scaling P1 sentinel")
        elif (
            box_plan.required_release_count != 0
            or box_plan.release_indices
            or not (
                type(box_plan.scale_factor) is float
                and np.isnan(box_plan.scale_factor)
            )
        ):
            raise ValueError("invalid non-resampling P1 sentinel")
        if box_plan.policy_code == POLICY_ACTIVATE and (
            len(box_plan.activation_indices) != box_plan.requested_count
        ):
            raise ValueError("activation indices must match requested count")
        if (
            box_plan.policy_code != POLICY_ACTIVATE
            and box_plan.activation_indices
        ):
            raise ValueError("deferred P1 plans cannot activate slots")
    return exhaustion_plan


def _relative_error(before: np.float64, after: np.float64) -> float:
    """Calculate a finite relative diagnostic error, including zero values."""
    if before == 0.0:
        return 0.0 if after == 0.0 else float("inf")
    return float(abs(after - before) / abs(before))


def _riemer_diversity(
    masses: NDArray[np.float64],
    concentrations: NDArray[np.float64],
) -> np.float64:
    """Calculate particle mixing-state Riemer diversity using natural logs."""
    particle_mass = np.sum(masses, axis=1, dtype=np.float64)
    represented_mass = concentrations * particle_mass
    total = np.sum(represented_mass, dtype=np.float64)
    if not np.isfinite(total) or total <= 0.0:
        return np.float64(0.0)
    particle_fractions = represented_mass / total
    species_fractions = np.divide(
        masses,
        particle_mass[:, None],
        out=np.zeros_like(masses),
        where=particle_mass[:, None] > 0.0,
    )
    positive = species_fractions > 0.0
    entropy = -np.sum(
        particle_fractions[:, None]
        * np.where(
            positive,
            species_fractions
            * np.log(np.where(positive, species_fractions, 1.0)),
            0.0,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(entropy):
        return np.float64(0.0)
    return np.float64(np.exp(entropy))


def _validate_remap_diagnostics(
    source: _CachedBoxState,
    replacement_masses: NDArray[np.float64],
    replacement_concentrations: NDArray[np.float64],
    replacement_charges: NDArray[np.float64],
    bounds: _ResamplingBounds | None,
) -> None:
    """Check exact inventories and bounded scalar P2 diagnostics."""
    source_number = np.sum(source.concentrations, dtype=np.float64)
    source_mass = np.sum(
        source.concentrations[:, None] * source.masses, axis=0, dtype=np.float64
    )
    source_charge = np.sum(
        source.concentrations * source.charges, dtype=np.float64
    )
    number = np.sum(replacement_concentrations, dtype=np.float64)
    mass = np.sum(
        replacement_concentrations[:, None] * replacement_masses,
        axis=0,
        dtype=np.float64,
    )
    charge = np.sum(
        replacement_concentrations * replacement_charges, dtype=np.float64
    )
    if not (
        np.allclose(number, source_number, rtol=1e-12, atol=1e-30)
        and np.allclose(mass, source_mass, rtol=1e-12, atol=1e-30)
        and np.allclose(charge, source_charge, rtol=1e-12, atol=1e-30)
    ):
        raise ValueError("resampling remap does not conserve inventory")
    if bounds is None:
        return

    # Diversity is density-independent; radius diagnostics use density in the
    # planner immediately after this exact-inventory check.
    diversity_error = abs(
        _riemer_diversity(source.masses, source.concentrations)
        - _riemer_diversity(replacement_masses, replacement_concentrations)
    )
    if (
        not np.isfinite(diversity_error)
        or diversity_error > bounds.diversity_absolute_error
    ):
        raise ValueError("resampling diversity diagnostic exceeds bound")


def _build_box_remap(  # noqa: C901
    source: _CachedBoxState,
    required_release_count: int,
    density: NDArray[np.float64],
    bounds: _ResamplingBounds,
) -> _ResamplingBoxPlan:
    """Build a stable, linear-sweep equal-weight remap from cached state."""
    active_count = source.active_indices.size
    if required_release_count > active_count:
        raise ValueError("required release count exceeds active slots")
    retained_count = active_count - required_release_count
    if retained_count == 0:
        raise ValueError("resampling must retain at least one active slot")
    if required_release_count == 0:
        return _ResamplingBoxPlan((), (), (), (), ())
    retained = tuple(
        int(index) for index in source.active_indices[:retained_count]
    )
    released = tuple(
        int(index) for index in source.active_indices[retained_count:]
    )

    totals = np.sum(source.masses, axis=1, dtype=np.float64)
    fractions = source.masses / totals[:, None]
    order = np.array(
        sorted(
            range(active_count),
            key=lambda index: (
                float(source.radii[index]),
                *(float(value) for value in fractions[index]),
                float(source.charges[index]),
                int(source.active_indices[index]),
            ),
        ),
        dtype=np.intp,
    )
    concentrations = source.concentrations[order]
    masses = source.masses[order]
    charges = source.charges[order]
    total_number = np.sum(concentrations, dtype=np.float64)
    equal_concentration = np.float64(total_number / retained_count)
    replacement_masses = np.zeros(
        (retained_count, masses.shape[1]), dtype=np.float64
    )
    replacement_charges = np.zeros(retained_count, dtype=np.float64)
    source_index = 0
    source_start = np.float64(0.0)
    source_end = concentrations[0]
    for target_index in range(retained_count):
        target_start = np.float64(target_index) * equal_concentration
        target_end = np.float64(target_index + 1) * equal_concentration
        while source_index < active_count and source_end <= target_start:
            source_index += 1
            source_start = source_end
            if source_index < active_count:
                source_end = source_start + concentrations[source_index]
        cursor = target_start
        while cursor < target_end and source_index < active_count:
            overlap_end = min(target_end, source_end)
            overlap = overlap_end - cursor
            replacement_masses[target_index] += overlap * masses[source_index]
            replacement_charges[target_index] += overlap * charges[source_index]
            cursor = overlap_end
            if cursor >= source_end:
                source_index += 1
                source_start = source_end
                if source_index < active_count:
                    source_end = source_start + concentrations[source_index]
    replacement_concentrations = np.full(
        retained_count, equal_concentration, dtype=np.float64
    )
    replacement_masses /= equal_concentration
    replacement_charges /= equal_concentration
    _validate_remap_diagnostics(
        source,
        replacement_masses,
        replacement_concentrations,
        replacement_charges,
        bounds,
    )
    source_moment = np.sum(
        source.concentrations * source.radii**3, dtype=np.float64
    )
    source_mean = (
        np.sum(source.concentrations * source.radii, dtype=np.float64)
        / total_number
    )
    source_surface = np.sum(
        source.concentrations * 4.0 * np.pi * source.radii**2, dtype=np.float64
    )
    radii = np.cbrt(
        3.0
        * np.sum(replacement_masses / density, axis=1, dtype=np.float64)
        / (4.0 * np.pi)
    )
    replacement_moment = np.sum(
        replacement_concentrations * radii**3, dtype=np.float64
    )
    replacement_mean = (
        np.sum(replacement_concentrations * radii, dtype=np.float64)
        / total_number
    )
    replacement_surface = np.sum(
        replacement_concentrations * 4.0 * np.pi * radii**2, dtype=np.float64
    )
    errors = (
        (
            _relative_error(source_moment, replacement_moment),
            bounds.radius_cubed_relative_error,
        ),
        (
            _relative_error(source_mean, replacement_mean),
            bounds.mean_radius_relative_error,
        ),
        (
            _relative_error(source_surface, replacement_surface),
            bounds.surface_relative_error,
        ),
    )
    if any(not np.isfinite(error) or error > bound for error, bound in errors):
        raise ValueError("resampling moment diagnostic exceeds bound")
    return _ResamplingBoxPlan(
        retained_indices=retained,
        released_indices=released,
        replacement_masses=tuple(
            tuple(np.float64(value) for value in row)
            for row in replacement_masses
        ),
        replacement_concentrations=tuple(
            np.float64(value) for value in replacement_concentrations
        ),
        replacement_charges=tuple(
            np.float64(value) for value in replacement_charges
        ),
        source_concentrations=tuple(
            np.float64(value) for value in source.concentrations
        ),
        source_masses=tuple(
            tuple(np.float64(value) for value in row) for row in source.masses
        ),
        source_charges=tuple(np.float64(value) for value in source.charges),
    )


def plan_resampling(
    particles: ParticleData,
    exhaustion_plan: ExhaustionPlan,
    *,
    radius_cubed_relative_error: float = 1.0,
    mean_radius_relative_error: float = 1.0,
    surface_relative_error: float = 1.0,
    diversity_absolute_error: float = 1.0,
) -> _ResamplingPlan:
    """Create an immutable, deterministic CPU fixed-capacity resampling plan.

    The function validates all boxes and only plans; it does not mutate P1
    records or particle state. The detached plan retains existing active
    capacity only, so it neither activates, scales, resizes, nor compacts
    particles, and it provides no GPU-parity path. Pass it to
    :func:`apply_resampling` for the all-box commit boundary.

    Args:
        particles: Writable, fixed-shape CPU particle data.
        exhaustion_plan: Immutable P1 capacity decisions to consume unchanged.
        radius_cubed_relative_error: Exact Python float, finite nonnegative
            bound for the weighted radius-cubed diagnostic.
        mean_radius_relative_error: Exact Python float, finite nonnegative
            bound for the number-weighted-radius diagnostic.
        surface_relative_error: Exact Python float, finite nonnegative bound
            for the weighted-surface-area diagnostic.
        diversity_absolute_error: Exact Python float, finite nonnegative bound
            for the Riemer bulk-diversity diagnostic.

    Returns:
        A detached immutable plan, one remap record per particle box.

    Raises:
        TypeError: If particle storage, P1 records, or diagnostic bounds have
            an unsupported type or dtype.
        ValueError: If the fixed-capacity state or P1 decisions are invalid,
            resampling is infeasible, or a diagnostic exceeds its bound.
    """
    bounds = _validate_resampling_bounds(
        radius_cubed_relative_error,
        mean_radius_relative_error,
        surface_relative_error,
        diversity_absolute_error,
    )
    particle_data = _validate_particle_schema(particles)
    p1_plan = _validate_p1_plan(
        exhaustion_plan, particle_data.n_boxes, particle_data.n_particles
    )
    cached_state = _cache_particle_state(particle_data)
    box_plans: list[_ResamplingBoxPlan] = []
    for box_index, p1_box_plan in enumerate(p1_plan.box_plans):
        if p1_box_plan.policy_code != POLICY_RESAMPLE_DEFERRED:
            box_plans.append(_ResamplingBoxPlan((), (), (), (), ()))
            continue
        if cached_state[box_index].active_indices.size == 0:
            raise ValueError("resampling box must have active slots")
        box_plans.append(
            _build_box_remap(
                cached_state[box_index],
                p1_box_plan.required_release_count,
                particle_data.density,
                bounds,
            )
        )
    return _ResamplingPlan(box_plans=tuple(box_plans))


def _validate_resampling_plan(  # noqa: C901
    plan: object,
    particles: ParticleData,
) -> _ResamplingPlan:
    """Validate detached remaps before the apply boundary writes anything."""
    if not isinstance(plan, _ResamplingPlan):
        raise TypeError("plan must be a _ResamplingPlan")
    if len(plan.box_plans) != particles.n_boxes:
        raise ValueError("resampling plan must have one box plan per box")
    for box_plan in plan.box_plans:
        if not isinstance(box_plan, _ResamplingBoxPlan):
            raise TypeError("resampling plan contains an invalid box plan")
        tuples = (
            box_plan.retained_indices,
            box_plan.released_indices,
            box_plan.replacement_masses,
            box_plan.replacement_concentrations,
            box_plan.replacement_charges,
            box_plan.source_concentrations,
            box_plan.source_masses,
            box_plan.source_charges,
        )
        if any(type(value) is not tuple for value in tuples):
            raise TypeError("resampling plan fields must be tuples")
        retained = box_plan.retained_indices
        released = box_plan.released_indices
        if (
            len(retained) != len(box_plan.replacement_masses)
            or len(retained) != len(box_plan.replacement_concentrations)
            or len(retained) != len(box_plan.replacement_charges)
        ):
            raise ValueError(
                "replacement tuple lengths must match retained indices"
            )
        for indices in (retained, released):
            if any(type(index) is not int for index in indices):
                raise TypeError("resampling indices must be exact Python ints")
            if any(
                index < 0 or index >= particles.n_particles for index in indices
            ):
                raise ValueError("resampling index is out of range")
            if any(
                right <= left
                for left, right in zip(indices, indices[1:], strict=False)
            ):
                raise ValueError(
                    "resampling indices must be strictly ascending"
                )
        if set(retained).intersection(released):
            raise ValueError("retained and released indices must be disjoint")
        for masses in box_plan.replacement_masses:
            if type(masses) is not tuple or len(masses) != particles.n_species:
                raise ValueError(
                    "replacement mass rows must have species length"
                )
            if any(type(value) is not np.float64 for value in masses):
                raise TypeError("replacement masses must be numpy float64")
            if not all(np.isfinite(value) and value >= 0.0 for value in masses):
                raise ValueError(
                    "replacement masses must be finite and nonnegative"
                )
        for name, values in (
            ("replacement concentrations", box_plan.replacement_concentrations),
            ("replacement charges", box_plan.replacement_charges),
        ):
            if any(type(value) is not np.float64 for value in values):
                raise TypeError(f"{name} must be numpy float64")
            if not all(np.isfinite(value) for value in values):
                raise ValueError(f"{name} must be finite")
        if any(value <= 0.0 for value in box_plan.replacement_concentrations):
            raise ValueError("replacement concentrations must be positive")
        if retained or released:
            source_count = len(retained) + len(released)
            if (
                len(box_plan.source_concentrations) != source_count
                or len(box_plan.source_masses) != source_count
                or len(box_plan.source_charges) != source_count
            ):
                raise ValueError("resampling plan is missing source binding")
            for masses in box_plan.source_masses:
                if type(masses) is not tuple or len(masses) != (
                    particles.n_species
                ):
                    raise ValueError(
                        "source mass rows must have species length"
                    )
                if any(type(value) is not np.float64 for value in masses):
                    raise TypeError("source masses must be numpy float64")
            for name, values in (
                ("source concentrations", box_plan.source_concentrations),
                ("source charges", box_plan.source_charges),
            ):
                if any(type(value) is not np.float64 for value in values):
                    raise TypeError(f"{name} must be numpy float64")
    return plan


def apply_resampling(
    particles: ParticleData,
    plan: _ResamplingPlan,
) -> ParticleData:
    """Atomically apply a previously planned CPU fixed-capacity remap.

    Every plan and target-state validation runs before the first assignment, so
    ordinary validation errors cannot partially mutate an earlier box. This
    commit performs no policy resolution, selection, activation, scaling,
    resizing, or compaction, and has no GPU-parity path.

    Args:
        particles: Writable fixed-shape CPU particle storage to update in
            place.
        plan: Detached plan returned by :func:`plan_resampling`.

    Returns:
        The identical ``particles`` container after the commit.

    Raises:
        TypeError: If the target storage or detached plan has an unsupported
            type or dtype.
        ValueError: If the target state or plan is invalid, or the plan no
            longer exactly covers the active slots in a resampling box.
    """
    particle_data = _validate_particle_schema(particles)
    cached_state = _cache_particle_state(particle_data)
    resampling_plan = _validate_resampling_plan(plan, particle_data)
    for box_index, box_plan in enumerate(resampling_plan.box_plans):
        if not box_plan.retained_indices and not box_plan.released_indices:
            continue
        active_indices = tuple(
            int(index) for index in cached_state[box_index].active_indices
        )
        planned_indices = box_plan.retained_indices + box_plan.released_indices
        if active_indices != planned_indices:
            raise ValueError(
                "resampling plan must exactly cover current active slots"
            )
        source = cached_state[box_index]
        source_masses = tuple(
            tuple(np.float64(value) for value in row) for row in source.masses
        )
        if (
            tuple(np.float64(value) for value in source.concentrations)
            != box_plan.source_concentrations
            or source_masses != box_plan.source_masses
            or tuple(np.float64(value) for value in source.charges)
            != box_plan.source_charges
        ):
            raise ValueError("resampling plan source state is stale")
        _validate_remap_diagnostics(
            source,
            np.asarray(box_plan.replacement_masses, dtype=np.float64),
            np.asarray(box_plan.replacement_concentrations, dtype=np.float64),
            np.asarray(box_plan.replacement_charges, dtype=np.float64),
            None,
        )
    for box_index, box_plan in enumerate(resampling_plan.box_plans):
        if not box_plan.retained_indices and not box_plan.released_indices:
            continue
        if box_plan.retained_indices:
            retained = np.asarray(box_plan.retained_indices, dtype=np.intp)
            particle_data.masses[box_index, retained] = np.asarray(
                box_plan.replacement_masses, dtype=np.float64
            )
            particle_data.concentration[box_index, retained] = np.asarray(
                box_plan.replacement_concentrations, dtype=np.float64
            )
            particle_data.charge[box_index, retained] = np.asarray(
                box_plan.replacement_charges, dtype=np.float64
            )
        if box_plan.released_indices:
            released = np.asarray(box_plan.released_indices, dtype=np.intp)
            particle_data.masses[box_index, released] = np.float64(0.0)
            particle_data.concentration[box_index, released] = np.float64(0.0)
            particle_data.charge[box_index, released] = np.float64(0.0)
    return particle_data


def _validate_p4_sidecar(
    name: str,
    values: object,
    dtype: np.dtype[np.generic],
    box_count: int,
) -> NDArray[Any]:
    """Validate one writable contiguous P4 per-box sidecar."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if values.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if values.ndim != 1:
        raise ValueError(f"{name} must have rank 1")
    if values.shape != (box_count,):
        raise ValueError(f"{name} must have shape (B,)")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous")
    if not values.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return values


def _validate_p4_non_aliasing(
    particles: ParticleData,
    sidecars: tuple[NDArray[Any], ...],
) -> None:
    """Reject P4 sidecar and writable particle-storage overlap before writes."""
    arrays = (
        particles.masses,
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        *sidecars,
    )
    for index, left in enumerate(arrays):
        if any(np.shares_memory(left, right) for right in arrays[index + 1 :]):
            raise ValueError("representative scaling arrays must not overlap")


def apply_representative_volume_scaling(  # noqa: C901, PLR0913
    particles: ParticleData,
    provisional_source_demand: NDArray[np.float64],
    scaling_required: NDArray[np.bool_],
    requested_scale: NDArray[np.float64],
    minimum_scale: NDArray[np.float64],
    minimum_volume: NDArray[np.float64],
    resolved_scale: NDArray[np.float64],
) -> tuple[ParticleData, NDArray[np.float64], NDArray[np.float64]]:
    """Apply P4 scaling to selected CPU particle-data rows in place.

    A row is selected exactly when ``scaling_required`` is true and its
    provisional demand is positive. Each selected row scales its box volume,
    every particle concentration, and provisional demand by its requested
    factor. Masses, charge, density, all configuration sidecars, and
    unselected rows remain unchanged. Complete all-box preflight precedes the
    diagnostic write and commit, so validation rejection leaves every
    caller-owned array unchanged. ``resolved_scale`` receives the requested
    factor for selected rows and ``1.0`` otherwise.

    Args:
        particles: Writable fixed-shape CPU particle data to update.
        provisional_source_demand: Writable contiguous ``float64`` demand per
            box, shaped ``(B,)``.
        scaling_required: Writable contiguous ``bool`` selection flags per
            box, shaped ``(B,)``.
        requested_scale: Writable contiguous ``float64`` requested factors per
            box, shaped ``(B,)``.
        minimum_scale: Writable contiguous ``float64`` lower factor bounds per
            box, shaped ``(B,)``.
        minimum_volume: Writable contiguous ``float64`` post-scaling box-volume
            bounds in m³, shaped ``(B,)``.
        resolved_scale: Writable contiguous ``float64`` diagnostic output per
            box, shaped ``(B,)``.

    Returns:
        The identical particle container, demand sidecar, and resolved-scale
        sidecar after a successful in-place diagnostic write and selected-row
        commit.

    Raises:
        TypeError: If particle storage or a sidecar has an unsupported type or
            dtype.
        ValueError: If schemas, physical values, factor bounds, or selected
            scaled-volume bounds are invalid. Rejection is atomic.
    """
    particle_data = _validate_particle_schema(particles)
    box_count = particle_data.n_boxes
    demand = cast(
        NDArray[np.float64],
        _validate_p4_sidecar(
            "provisional_source_demand",
            provisional_source_demand,
            np.dtype(np.float64),
            box_count,
        ),
    )
    flags = cast(
        NDArray[np.bool_],
        _validate_p4_sidecar(
            "scaling_required", scaling_required, np.dtype(np.bool_), box_count
        ),
    )
    requested = cast(
        NDArray[np.float64],
        _validate_p4_sidecar(
            "requested_scale", requested_scale, np.dtype(np.float64), box_count
        ),
    )
    minimum = cast(
        NDArray[np.float64],
        _validate_p4_sidecar(
            "minimum_scale", minimum_scale, np.dtype(np.float64), box_count
        ),
    )
    minimum_box_volume = cast(
        NDArray[np.float64],
        _validate_p4_sidecar(
            "minimum_volume", minimum_volume, np.dtype(np.float64), box_count
        ),
    )
    resolved = cast(
        NDArray[np.float64],
        _validate_p4_sidecar(
            "resolved_scale", resolved_scale, np.dtype(np.float64), box_count
        ),
    )

    _validate_p4_non_aliasing(
        particle_data,
        (demand, flags, requested, minimum, minimum_box_volume, resolved),
    )

    _cache_particle_state(particle_data)
    if not np.all(np.isfinite(demand)) or np.any(demand < 0.0):
        raise ValueError(
            "provisional_source_demand must be finite and nonnegative"
        )
    if (
        not np.all(np.isfinite(requested))
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(minimum_box_volume))
    ):
        raise ValueError("scaling factors and minimum_volume must be finite")
    if (
        np.any(minimum <= 0.0)
        or np.any(requested < minimum)
        or np.any(requested > 1.0)
    ):
        raise ValueError(
            "scaling factors must satisfy 0 < minimum <= requested <= 1"
        )
    if np.any(minimum_box_volume <= 0.0):
        raise ValueError("minimum_volume must be finite and positive")

    selected = flags & (demand > 0.0)
    if np.any(
        particle_data.volume[selected] * requested[selected]
        < minimum_box_volume[selected]
    ):
        raise ValueError("selected scaled volume must meet minimum_volume")
    factors = np.where(selected, requested, 1.0)
    scaled_concentration = particle_data.concentration * factors[:, None]
    active = particle_data.concentration > 0.0
    if np.any(active & (~np.isfinite(scaled_concentration))) or np.any(
        active & (scaled_concentration <= 0.0)
    ):
        raise ValueError("selected scale must preserve active concentrations")
    resolved[:] = factors
    if np.any(selected):
        particle_data.volume[selected] *= factors[selected]
        particle_data.concentration[selected, :] *= factors[selected, None]
        demand[selected] *= factors[selected]
    return particle_data, demand, resolved
