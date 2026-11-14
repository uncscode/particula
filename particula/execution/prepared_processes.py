"""Translate supported native runnables into immutable resident process plans.

Host validation finishes before any state upload. Compilers then allocate only
the selected process resources and retain the existing direct-kernel enqueue
delegates. User-defined strategies never receive an inferred GPU equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from particula.aerosol import Aerosol


@dataclass(frozen=True)
class _ProcessPlan:
    """Own normalized scalar/tuple configuration independent of CPU objects."""

    family: str
    parameters: tuple[tuple[str, Any], ...] = ()

    @property
    def values(self) -> dict[str, Any]:
        """Make setup-only keyword arguments from the immutable plan."""
        return dict(self.parameters)


def _plan_process(process: Any, aerosol: Aerosol) -> _ProcessPlan:
    """Reject unknown runnable/strategy semantics before any GPU allocation."""
    from particula.dynamics.ions.source import IonPairSource
    from particula.dynamics.particle_process import (
        Coagulation,
        Dilution,
        MassCondensation,
        Nucleation,
        WallLoss,
    )
    from particula.execution.prepared_condensation import plan_condensation
    from particula.execution.prepared_ions import plan_ion_source
    from particula.execution.prepared_nucleation import plan_nucleation

    planners = {
        IonPairSource: plan_ion_source,
        Dilution: _plan_dilution,
        Coagulation: _plan_coagulation,
        WallLoss: _plan_wall_loss,
        MassCondensation: plan_condensation,
        Nucleation: plan_nucleation,
    }
    planner = planners.get(type(process))
    if planner is None:
        raise ValueError(
            f"{type(process).__name__} has no prepared GPU translation"
        )
    process._validate_batch(aerosol)
    return planner(process, aerosol)


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


def _plan_coagulation(process: Any, aerosol: Aerosol) -> _ProcessPlan:
    """Admit direct Brownian physics and equal active represented counts."""
    from particula.dynamics import (
        BrownianCoagulationStrategy,
        ChargedCoagulationStrategy,
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
    charged_model = None
    if type(strategy) is ChargedCoagulationStrategy:
        charged_model = supported.get(type(strategy.kernel_strategy))
        if charged_model is None:
            raise ValueError("charged kernel has no prepared GPU translation")
    elif type(strategy) is not BrownianCoagulationStrategy:
        raise ValueError(
            "prepared coagulation requires Brownian or charged strategy"
        )
    if aerosol.particles.distribution_type != "particle_resolved":
        raise ValueError("prepared coagulation requires particle_resolved")
    if (
        strategy.particle_resolved_radius is not None
        or strategy.particle_resolved_bins_number is not None
        or strategy.particle_resolved_bins_per_decade != 10
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
    if charged_model is None:
        return _ProcessPlan("coagulation")
    return _ProcessPlan("coagulation", (("charged_model", charged_model),))


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


def _compile_process(plan: _ProcessPlan, owner: Any):
    """Retain one existing prepared direct-kernel call and its resources."""
    session, registry = owner._session, owner._registry
    if plan.family == "ion_source":
        from particula.execution.prepared_ions import compile_ion_source

        return compile_ion_source(plan, owner)
    if plan.family == "nucleation":
        from particula.execution.prepared_nucleation import compile_nucleation

        return compile_nucleation(plan, owner)
    if plan.family == "condensation":
        from particula.execution.prepared_condensation import (
            compile_condensation,
        )

        return compile_condensation(plan, owner)
    if plan.family == "dilution":
        from particula.execution.process_adapters import (
            ResidentDilutionAdapter,
            ResidentDilutionRequest,
        )

        binding = ResidentDilutionAdapter().prepare(
            ResidentDilutionRequest(
                session,
                registry,
                plan.values["coefficient"],
                owner.time_step,
            )
        )
        binding = replace(
            binding,
            prepared=replace(binding.prepared, failure=owner._device_failure),
        )
        owner._bindings[plan.family] = binding
        return binding.execute
    if plan.family == "wall_loss":
        from particula.execution.process_adapters import (
            ResidentWallLossAdapter,
            ResidentWallLossRequest,
        )
        from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

        values = plan.values
        if values["geometry"] == "rectangular" and values["mode"] == "charged":
            values["wall_electric_field"] = owner._wp.array(
                values["wall_electric_field"],
                dtype=owner._wp.float64,
                device=owner.device.native,
            )
        binding = ResidentWallLossAdapter().prepare(
            ResidentWallLossRequest(
                session,
                registry,
                registry.acquire_wall_loss(),
                NeutralWallLossConfig(**values),
                owner.time_step,
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
                launch_inputs=(*prepared.launch_inputs, owner._device_failure),
            )
        binding = replace(binding, prepared=prepared)
        owner._bindings[plan.family] = binding
        return binding.execute
    if plan.family == "coagulation":
        from particula.gpu.kernels.coagulation import (
            CoagulationMechanismConfig,
            _enqueue_prepared_coagulation_call,
            _prepare_coagulation_step_gpu,
        )

        model = plan.values.get("charged_model")
        mechanism = (
            CoagulationMechanismConfig(
                mechanisms=("charged_hard_sphere",), charged_model=model
            )
            if model is not None
            else CoagulationMechanismConfig()
        )
        resources = registry.acquire_coagulation(session.dimensions.n_particles)
        prepared = _prepare_coagulation_step_gpu(
            session.particles,
            None,
            None,
            owner.time_step,
            collision_pairs=resources.collision_pairs,
            n_collisions=resources.n_collisions,
            rng_states=resources.rng_states,
            initialize_rng=False,
            environment=session.environment,
            mechanism_config=mechanism,
        )
        from particula.execution.prepared_guards import equal_active_counts

        prepared = replace(prepared, failure=owner._device_failure)
        owner._bindings[plan.family] = prepared
        counts, failure = session.particles.concentration, owner._device_failure
        wp = owner._wp

        def enqueue():
            wp.launch(
                equal_active_counts,
                dim=counts.shape[0],
                inputs=[counts, failure],
                device=prepared.device,
            )
            _enqueue_prepared_coagulation_call(prepared)

        return enqueue
    raise ValueError(f"unsupported prepared process family: {plan.family}")
