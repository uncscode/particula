"""Translate supported native runnables into immutable resident process plans.

Host validation finishes before any state upload. Compilers then allocate only
the selected process resources and retain the existing direct-kernel enqueue
delegates. User-defined strategies never receive an inferred GPU equivalent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from particula.aerosol import Aerosol
from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
    ProcessRegistry,
)
from particula.execution.preparation_context import BuiltinPreparationContext


@dataclass(frozen=True)
class _ProcessPlan:
    """Own normalized scalar/tuple configuration independent of CPU objects."""

    family: str
    parameters: tuple[tuple[str, Any], ...] = ()

    @property
    def values(self) -> dict[str, Any]:
        """Make setup-only keyword arguments from the immutable plan."""
        return dict(self.parameters)


def configure_coagulation_work(plans, config=None):
    """Freeze resource policy and sampler version in built-in process plans."""
    from dataclasses import replace

    from particula.gpu.kernels._coagulation_budget import CoagulationWorkConfig

    config = CoagulationWorkConfig() if config is None else config
    if not isinstance(config, CoagulationWorkConfig):
        raise ValueError(
            "coagulation_work_config must be a CoagulationWorkConfig"
        )
    result = []
    for plan in plans:
        if type(plan) is _ProcessPlan and plan.family == "coagulation":
            values = plan.values
            values.update(
                coagulation_sampler_version=1,
                max_trials_per_box=int(config.max_trials_per_box),
                max_collision_buffer_bytes=int(
                    config.max_collision_buffer_bytes
                ),
            )
            plan = replace(plan, parameters=tuple(values.items()))
        result.append(plan)
    return tuple(result)


def _plan_process(process: Any, aerosol: Aerosol) -> _ProcessPlan:
    """Reject unknown runnable/strategy semantics before any GPU allocation."""
    from particula.cpu_transfer import CPUTransfer, _CPUTransferAdapter

    if type(process) is CPUTransfer:
        from particula.execution.prepared_extensions import plan_extension

        return plan_extension(_CPUTransferAdapter(), process, aerosol)
    registry = _builtin_registry()
    if not registry.has_type(type(process)):
        raise ValueError(
            f"{type(process).__name__} has no prepared GPU translation"
        )
    return registry.resolve(process).plan(process, aerosol)


@dataclass(frozen=True)
class _BuiltinAdapter:
    """Pair admitted native planning with resource-only preparation."""

    planner: Callable[[Any, Aerosol], _ProcessPlan]
    compiler: Callable[
        [_ProcessPlan, BuiltinPreparationContext], Callable[[], object]
    ]

    def plan(self, process: Any, aerosol: Aerosol) -> _ProcessPlan:
        """Keep exact native validation ahead of all GPU allocation."""
        process._validate_batch(aerosol)
        return self.planner(process, aerosol)

    def prepare(
        self, plan: _ProcessPlan, context: BuiltinPreparationContext
    ) -> PreparedOperation:
        """Retain the launch handler and fixed native resources explicitly."""
        handler = self.compiler(plan, context)
        return PreparedOperation(
            name=plan.family,
            handler=handler,
            resources=(
                tuple(context.bindings.values()),
                tuple(context.auxiliary.values()),
            ),
            capabilities=ProcessCapabilities(
                resident=True, capture=True, checkpoint=True
            ),
        )


def _builtin_adapters_by_family() -> dict[str, _BuiltinAdapter]:
    """Resolve normalized families without changing scientific admission."""
    from particula.execution.prepared_cohort import compile_cohort, plan_cohort
    from particula.execution.prepared_condensation import (
        compile_condensation,
        plan_condensation,
    )
    from particula.execution.prepared_ions import (
        compile_ion_cleanup,
        compile_ion_exclusion,
        compile_ion_source,
        plan_ion_cleanup,
        plan_ion_exclusion,
        plan_ion_source,
    )
    from particula.execution.prepared_nucleation import (
        compile_nucleation,
        plan_nucleation,
    )

    return {
        "dilution": _BuiltinAdapter(_plan_dilution, _compile_dilution),
        "coagulation": _BuiltinAdapter(_plan_coagulation, _compile_coagulation),
        "wall_loss": _BuiltinAdapter(_plan_wall_loss, _compile_wall_loss),
        "condensation": _BuiltinAdapter(
            plan_condensation, compile_condensation
        ),
        "nucleation": _BuiltinAdapter(plan_nucleation, compile_nucleation),
        "cohort": _BuiltinAdapter(plan_cohort, compile_cohort),
        "ion_source": _BuiltinAdapter(plan_ion_source, compile_ion_source),
        "ion_cleanup": _BuiltinAdapter(plan_ion_cleanup, compile_ion_cleanup),
        "ion_condensation": _BuiltinAdapter(
            plan_ion_exclusion, compile_ion_exclusion
        ),
    }


def _builtin_registry() -> ProcessRegistry[_BuiltinAdapter]:
    """Build exact-type native adapters using the common registry contract."""
    from particula.dynamics.ions.cleanup import IonNeutralCleanup
    from particula.dynamics.ions.coagulation import AtmosphericCoagulation
    from particula.dynamics.ions.condensation import IonCondensationExclusion
    from particula.dynamics.ions.source import IonPairSource
    from particula.dynamics.particle_process import (
        Coagulation,
        Dilution,
        MassCondensation,
        Nucleation,
        WallLoss,
    )
    from particula.execution.cohort import SeedCohort
    from particula.execution.prepared_atmospheric import plan_atmospheric
    from particula.execution.prepared_cohort import plan_cohort
    from particula.execution.prepared_condensation import plan_condensation
    from particula.execution.prepared_ions import (
        plan_ion_cleanup,
        plan_ion_exclusion,
        plan_ion_source,
    )
    from particula.execution.prepared_nucleation import plan_nucleation

    planners: dict[type, Callable[[Any, Aerosol], _ProcessPlan]] = {
        AtmosphericCoagulation: plan_atmospheric,
        IonCondensationExclusion: plan_ion_exclusion,
        SeedCohort: plan_cohort,
        IonNeutralCleanup: plan_ion_cleanup,
        IonPairSource: plan_ion_source,
        Dilution: _plan_dilution,
        Coagulation: _plan_coagulation,
        WallLoss: _plan_wall_loss,
        MassCondensation: plan_condensation,
        Nucleation: plan_nucleation,
    }
    compilers = _builtin_adapters_by_family()
    families = {
        AtmosphericCoagulation: "coagulation",
        IonCondensationExclusion: "ion_condensation",
        SeedCohort: "cohort",
        IonNeutralCleanup: "ion_cleanup",
        IonPairSource: "ion_source",
        Dilution: "dilution",
        Coagulation: "coagulation",
        WallLoss: "wall_loss",
        MassCondensation: "condensation",
        Nucleation: "nucleation",
    }
    registry: ProcessRegistry[_BuiltinAdapter] = ProcessRegistry()
    for process_type, planner in planners.items():
        registry.register(
            process_type,
            _BuiltinAdapter(
                planner, compilers[families[process_type]].compiler
            ),
        )
    return registry.snapshot()


def _plan_dilution(process: Any, _aerosol: Aerosol) -> _ProcessPlan:
    """Freeze the exact exponential dilution coefficient in inverse seconds."""
    from particula.dynamics.dilution import (
        DilutionStrategy,
        _validate_nonnegative_scalar,
    )

    strategy = process.dilution_strategy
    if type(strategy) is not DilutionStrategy:
        raise ValueError("prepared dilution requires exact DilutionStrategy")
    coefficient = _validate_nonnegative_scalar(
        strategy.coefficient, "coefficient"
    )
    return _ProcessPlan("dilution", (("coefficient", float(coefficient)),))


def _plan_coagulation(process: Any, aerosol: Aerosol) -> _ProcessPlan:  # noqa: C901
    """Freeze supported collision laws and equal represented counts."""
    from particula.dynamics import (
        BrownianCoagulationStrategy,
        ChargedCoagulationStrategy,
        CombineCoagulationStrategy,
        SedimentationCoagulationStrategy,
        TurbulentShearCoagulationStrategy,
    )
    from particula.dynamics.coagulation import charged_kernel_strategy as models

    supported = {
        models.HardSphereKernelStrategy: "hard_sphere",
        models.CoulombDyachkov2007KernelStrategy: "dyachkov2007",
        models.CoulombGatti2008KernelStrategy: "gatti2008",
        models.CoulombGopalakrishnan2012KernelStrategy: "gopalakrishnan2012",
        models.CoulumbChahl2019KernelStrategy: "chahl2019",
    }
    strategy = process.coagulation_strategy
    components = (
        tuple(strategy.strategies)
        if type(strategy) is CombineCoagulationStrategy
        else (strategy,)
    )
    if len({type(child) for child in components}) != len(components):
        raise ValueError("prepared coagulation requires unique mechanisms")
    values: dict[str, Any] = {}
    mechanisms = []
    for child in components:
        if type(child) is BrownianCoagulationStrategy:
            mechanisms.append("brownian")
        elif type(child) is ChargedCoagulationStrategy:
            model = supported.get(type(child.kernel_strategy))
            if model is None:
                raise ValueError(
                    "charged kernel has no prepared GPU translation"
                )
            values["charged_model"] = model
            mechanisms.append("charged_hard_sphere")
        elif type(child) is SedimentationCoagulationStrategy:
            mechanisms.append("sedimentation_sp2016")
            values["mean_sedimentation_density"] = True
        elif type(child) is TurbulentShearCoagulationStrategy:
            mechanisms.append("turbulent_shear_st1956")
            values["turbulent_dissipation"] = float(child.turbulent_dissipation)
            values["fluid_density"] = float(child.fluid_density)
        else:
            raise ValueError(
                "coagulation strategy has no prepared GPU translation"
            )
    if aerosol.particles.distribution_type != "particle_resolved":
        raise ValueError("prepared coagulation requires particle_resolved")
    for child in (strategy, *components):
        if (
            child.particle_resolved_radius is not None
            or child.particle_resolved_bins_number is not None
            or child.particle_resolved_bins_per_decade != 10
        ):
            raise ValueError(
                "CPU kernel interpolation controls have no GPU equivalent"
            )
    for row in aerosol.particles.concentration:
        active = row[row > 0]
        if active.size and not np.all(active == active[0]):
            raise ValueError(
                "Brownian GPU coagulation requires equal active counts"
            )
    # Preserve the existing Brownian/charged checkpoint signatures.
    if mechanisms not in (["brownian"], ["charged_hard_sphere"]):
        values["mechanisms"] = tuple(sorted(mechanisms))
    return _ProcessPlan("coagulation", tuple(values.items()))


def _plan_wall_loss(process: Any, aerosol: Aerosol) -> _ProcessPlan:
    """Freeze neutral/charged chamber geometry and explicit electric fields."""
    from particula.dynamics.wall_loss._native import _controls
    from particula.dynamics.wall_loss.wall_loss_strategies import (
        ChargedWallLossStrategy,
        RectangularWallLossStrategy,
        SphericalWallLossStrategy,
    )

    strategy = process.wall_loss_strategy
    if type(strategy) not in (
        SphericalWallLossStrategy,
        RectangularWallLossStrategy,
        ChargedWallLossStrategy,
    ):
        raise ValueError("wall-loss strategy has no prepared GPU translation")
    if aerosol.particles.distribution_type != "particle_resolved":
        raise ValueError("prepared wall loss requires particle_resolved")
    _controls(strategy)
    charged = type(strategy) is ChargedWallLossStrategy
    geometry = (
        strategy.chamber_geometry
        if charged
        else "spherical"
        if type(strategy) is SphericalWallLossStrategy
        else "rectangular"
    )
    field: Any = strategy.wall_electric_field if charged else 0.0
    if geometry == "spherical":
        if np.ndim(field):
            raise ValueError("spherical GPU wall loss requires a scalar field")
        field = float(field)
    elif charged:
        if np.ndim(field) == 0:
            # CPU scalar fields have a signed magnitude; a vector norm cannot
            # represent a negative scalar without changing the drift model.
            if field < 0:
                raise ValueError(
                    "rectangular GPU wall loss requires a vector field"
                )
            field = (float(field), 0.0, 0.0)
        else:
            field = tuple(float(value) for value in field)
    return _ProcessPlan(
        "wall_loss",
        tuple(
            dict(
                geometry=geometry,
                wall_eddy_diffusivity=float(strategy.wall_eddy_diffusivity),
                chamber_radius=float(strategy.chamber_radius)
                if geometry == "spherical"
                else None,
                chamber_dimensions=tuple(
                    float(x) for x in strategy.chamber_dimensions
                )
                if geometry == "rectangular"
                else None,
                mode="charged" if charged else "neutral",
                wall_potential=float(strategy.wall_potential)
                if charged
                else 0.0,
                wall_electric_field=field,
            ).items()
        ),
    )


def _compile_process(plan: _ProcessPlan, context: BuiltinPreparationContext):
    """Prepare a built-in through its explicit adapter contract."""
    adapter = _builtin_adapters_by_family().get(plan.family)
    if adapter is None:
        raise ValueError(f"unsupported prepared process family: {plan.family}")
    return adapter.prepare(plan, context)


def _compile_dilution(plan: _ProcessPlan, context: BuiltinPreparationContext):
    """Bind the native exponential dilution operation."""
    session, registry = context.session, context.registry
    from particula.execution.process_adapters import (
        ResidentDilutionAdapter,
        ResidentDilutionRequest,
    )

    binding = ResidentDilutionAdapter().prepare(
        ResidentDilutionRequest(
            session,
            registry,
            plan.values["coefficient"],
            context.time_step,
        )
    )
    binding = replace(
        binding,
        prepared=replace(binding.prepared, failure=context.failure),
    )
    context.bindings[plan.family] = binding
    return binding.execute


def _compile_wall_loss(plan: _ProcessPlan, context: BuiltinPreparationContext):
    """Bind the admitted neutral or charged wall-loss operation."""
    session, registry = context.session, context.registry
    from particula.execution.process_adapters import (
        ResidentWallLossAdapter,
        ResidentWallLossRequest,
    )
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    values = plan.values
    if values["geometry"] == "rectangular" and values["mode"] == "charged":
        values["wall_electric_field"] = context.wp.array(
            values["wall_electric_field"],
            dtype=context.wp.float64,
            device=context.device.native,
        )
    binding = ResidentWallLossAdapter().prepare(
        ResidentWallLossRequest(
            session,
            registry,
            registry.acquire_wall_loss(),
            NeutralWallLossConfig(**values),
            context.time_step,
        )
    )
    from particula.gpu.kernels import wall_loss as kernels

    prepared = binding.prepared
    if prepared.launch_kernel is not None:
        prepared = replace(
            prepared,
            launch_kernel=getattr(
                kernels, prepared.launch_kernel.key + "_guarded"
            ),
            launch_inputs=(*prepared.launch_inputs, context.failure),
        )
    binding = replace(binding, prepared=prepared)
    context.bindings[plan.family] = binding
    return binding.execute


def _compile_coagulation(plan, context: BuiltinPreparationContext):
    """Retain the common collision sampler with optional ion rate work."""
    session, registry = context.session, context.registry
    from particula.gpu.kernels.coagulation import (
        CoagulationMechanismConfig,
        _enqueue_prepared_coagulation_call,
        _prepare_coagulation_step_gpu,
    )

    model = plan.values.get("charged_model")
    mechanisms = plan.values.get("mechanisms")
    if model is not None and mechanisms is None:
        mechanisms = ("charged_hard_sphere",)
    mechanism = CoagulationMechanismConfig(
        mechanisms=mechanisms, charged_model=model or "hard_sphere"
    )
    from particula.gpu.kernels._coagulation_budget import CoagulationWorkConfig

    work_config = CoagulationWorkConfig(
        max_trials_per_box=plan.values["max_trials_per_box"],
        max_collision_buffer_bytes=plan.values["max_collision_buffer_bytes"],
    )
    from particula.gpu.kernels.coagulation import _resolve_collision_capacity

    capacity = _resolve_collision_capacity(
        max(1, session.dimensions.n_particles // 2),
        session.dimensions.n_boxes,
        session.dimensions.n_particles,
        work_config.max_collision_buffer_bytes,
    )
    resources = registry.acquire_coagulation(capacity)
    prepared = _prepare_coagulation_step_gpu(
        context.particles,
        None,
        None,
        context.time_step,
        max_collisions=capacity,
        collision_pairs=resources.collision_pairs,
        n_collisions=resources.n_collisions,
        rng_states=resources.rng_states,
        initialize_rng=False,
        environment=context.environment,
        mechanism_config=mechanism,
        work_config=work_config,
        turbulent_dissipation=plan.values.get("turbulent_dissipation"),
        fluid_density=plan.values.get("fluid_density"),
    )
    if plan.values.get("mean_sedimentation_density"):
        work = prepared.brownian_parallel
        prepared = replace(
            prepared,
            brownian_parallel=replace(
                work,
                mean_density=context.wp.empty(
                    session.dimensions.n_boxes,
                    dtype=context.wp.float64,
                    device=context.device.native,
                ),
            ),
        )
    from particula.execution.prepared_guards import equal_active_counts

    prepared = replace(prepared, failure=context.failure)
    if plan.values.get("atmospheric"):
        from particula.execution.prepared_atmospheric import (
            configure_atmospheric,
        )

        prepared = configure_atmospheric(plan, context, prepared)
    context.bindings[plan.family] = prepared
    counts, failure = context.particles.concentration, context.failure
    wp = context.wp

    def enqueue():
        wp.launch(
            equal_active_counts,
            dim=counts.shape[0],
            inputs=[counts, failure],
            device=prepared.device,
        )
        _enqueue_prepared_coagulation_call(prepared)

    return enqueue
