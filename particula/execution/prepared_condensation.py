"""Host-side condensation capability checks and retained GPU compilation.

The direct solver has four equal substeps, one activity-corrected material,
and a mixture Kelvin density. Admit only CPU configuration that can be
represented by those controls; do not discard unsupported strategy choices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from particula.execution.prepared_processes import _ProcessPlan


def _vector(value, width, name, *, positive=False):
    """Normalize finite nonnegative scalar/vector metadata without aliasing."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim > 1:
        raise ValueError(f"{name} must be a real scalar or species vector")
    if raw.ndim and raw.shape != (width,):
        raise ValueError(f"{name} must have {width} entries")
    values = np.broadcast_to(raw, (width,)).astype(np.float64).copy()
    if not np.all(np.isfinite(values)) or np.any(
        values <= 0 if positive else values < 0
    ):
        raise ValueError(f"{name} must have finite physical values")
    return values


def _vapor_plan(strategy, width):
    """Freeze constant/Buck definitions in complete gas order."""
    from particula.gas.vapor_pressure_strategies import (
        ConstantVaporPressureStrategy,
        WaterBuckStrategy,
    )

    if isinstance(strategy, Sequence):
        if len(strategy) != width:
            raise ValueError("vapor pressure strategies must follow gas order")
        values = []
        for item in strategy:
            if type(item) is WaterBuckStrategy:
                values.append((1, 0.0))
            elif type(item) is ConstantVaporPressureStrategy:
                value = _vector(
                    item.vapor_pressure, 1, "vapor_pressure", positive=True
                )[0]
                values.append((0, float(value)))
            else:
                raise ValueError(
                    "GPU vapor pressure supports constant/Buck only"
                )
        return tuple(values)
    if type(strategy) is WaterBuckStrategy:
        return ((1, 0.0),) * width
    if type(strategy) is ConstantVaporPressureStrategy:
        return tuple(
            (0, float(value))
            for value in _vector(
                strategy.vapor_pressure, width, "vapor_pressure", positive=True
            )
        )
    raise ValueError("GPU vapor pressure supports constant/Buck only")


def _surface_plan(surface, particles):
    """Require an exact static or single-phase volume-weighted surface law."""
    from particula.particles.surface_strategies import (
        SurfaceStrategyMass,
        SurfaceStrategyMolar,
        SurfaceStrategyVolume,
    )

    if type(surface) not in (
        SurfaceStrategyMass,
        SurfaceStrategyMolar,
        SurfaceStrategyVolume,
    ):
        raise ValueError("surface strategy has no prepared GPU translation")
    if (
        surface.surface_tension_table is not None
        or surface.temperature_table is not None
    ):
        raise ValueError("GPU surface tension tables are unsupported")
    width = particles.n_species
    tensions = _vector(
        surface.surface_tension, width, "surface_tension", positive=True
    )
    density = _vector(surface.density, width, "surface density", positive=True)
    # The direct kernel uses the live mixture density for Kelvin curvature.
    # A fixed CPU density can match this for arbitrary future composition only
    # when every material has that same density.
    if not np.all(density == particles.density[0]) or not np.all(
        particles.density == particles.density[0]
    ):
        raise ValueError(
            "GPU Kelvin density requires uniform material/surface density"
        )
    mode = 0
    if surface.phase_index is not None:
        phases = np.asarray(surface.phase_index)
        if (
            type(surface) is not SurfaceStrategyVolume
            or phases.shape != (width,)
            or not np.all(phases == phases[0])
        ):
            raise ValueError("GPU mixing supports one volume-weighted phase")
        mode = 1
    return tuple(float(x) for x in tensions), mode


def _activity_plan(activity, particles, gas, pairs, vapor):  # noqa: C901
    """Require one participating volatile material lane."""
    from particula.particles.activity_strategies import (
        ActivityIdealMass,
        ActivityIdealMolar,
        ActivityIdealVolume,
        ActivityKappaParameter,
    )

    if type(activity) not in (
        ActivityIdealMass,
        ActivityIdealMolar,
        ActivityIdealVolume,
        ActivityKappaParameter,
    ):
        raise ValueError("activity strategy has no prepared GPU translation")
    width = particles.n_species
    volatile = [
        target
        for source, target in pairs
        if gas.partitioning[source]
        and (vapor[source][0] or vapor[source][1] > 0)
    ]
    if len(volatile) > 1:
        raise ValueError(
            "GPU activity supports only one volatile material lane"
        )
    water = volatile[0] if volatile else 0
    mode = 0
    kappas = np.zeros(width)
    references = np.ones(width)
    for source, target in pairs:
        references[target] = gas.molar_mass[source]
    if type(activity) is ActivityIdealMolar:
        references = _vector(
            activity.molar_mass, width, "activity molar_mass", positive=True
        )
    elif type(activity) is ActivityKappaParameter:
        references = _vector(
            activity.molar_mass, width, "activity molar_mass", positive=True
        )
        kappas = _vector(activity.kappa, width, "kappa")
        densities = _vector(
            activity.density, width, "activity density", positive=True
        )
        if not np.array_equal(densities, particles.density):
            raise ValueError(
                "kappa densities must match particle material densities"
            )
        if isinstance(activity.water_index, (bool, np.bool_)) or not isinstance(
            activity.water_index, (int, np.integer)
        ):
            raise ValueError("water_index must be an integer material index")
        water = int(activity.water_index)
        if not 0 <= water < width or (volatile and volatile != [water]):
            raise ValueError(
                "only the configured water material may be volatile"
            )
        mode = 1
    elif volatile and width > 1:
        raise ValueError(
            "volatile mixtures require ideal-molar or kappa GPU activity"
        )
    if any(
        references[target] != gas.molar_mass[source] for source, target in pairs
    ):
        raise ValueError(
            "material molar masses must match mapped full-gas values"
        )
    return (
        mode,
        water,
        tuple(float(x) for x in kappas),
        tuple(float(x) for x in references),
    )


def plan_condensation(process, aerosol) -> _ProcessPlan:
    """Freeze representable physical parameters before any upload."""
    from particula._cpu_batch import box_view
    from particula.dynamics.condensation.condensation_strategies import (
        CondensationIsothermal,
        CondensationLatentHeat,
    )
    from particula.gas.latent_heat_strategies import ConstantLatentHeat

    strategy = process.condensation_strategy
    if type(strategy) not in (CondensationIsothermal, CondensationLatentHeat):
        raise ValueError(
            "GPU preparation supports equal-step condensation only"
        )
    if strategy.update_gases is not True:
        raise ValueError("prepared condensation requires coupled gas updates")
    parameters = None
    for box in range(aerosol.particles.n_boxes):
        view = box_view(aerosol, box)
        parameters = strategy._native_parameters(
            view.particles,
            view.gas,
            float(view.environment.temperature[0]),
            float(view.environment.pressure[0]),
        )
    if parameters is None:
        raise ValueError("prepared condensation requires at least one box")
    sources, targets, molar, diffusion, accommodation = parameters
    pairs = tuple(
        (int(s), int(t)) for s, t in zip(sources, targets, strict=True)
    )
    vapor = _vapor_plan(strategy.vapor_pressure_strategy, len(aerosol.gas.name))
    tensions, surface_mode = _surface_plan(
        strategy.surface_strategy, aerosol.particles
    )
    activity_mode, water, kappas, references = _activity_plan(
        strategy.activity_strategy, aerosol.particles, aerosol.gas, pairs, vapor
    )
    latent = None
    if type(strategy) is CondensationLatentHeat:
        heat = strategy._latent_heat_strategy
        if heat is not None:
            if type(heat) is not ConstantLatentHeat:
                raise ValueError("prepared latent heat must be constant")
            latent = tuple(
                float(x)
                for x in _vector(
                    heat.latent_heat_ref,
                    len(molar),
                    "latent_heat",
                    positive=True,
                )
            )
    if latent is not None and any(value > 1e9 for value in latent):
        raise ValueError("GPU latent heat must not exceed 1e9 J/kg")
    return _ProcessPlan(
        "condensation",
        tuple(
            dict(
                gas_names=tuple(aerosol.gas.name),
                pairs=pairs,
                molar_mass=tuple(float(x) for x in molar),
                vapor=vapor,
                diffusion=tuple(float(x) for x in diffusion),
                accommodation=tuple(float(x) for x in accommodation),
                tensions=tensions,
                surface_mode=surface_mode,
                activity_mode=activity_mode,
                water=water,
                kappas=kappas,
                references=references,
                latent=latent,
            ).items()
        ),
    )


def compile_condensation(plan: _ProcessPlan, owner: Any):
    """Allocate retained thermodynamics, species maps and transport sidecars."""
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.execution import (
        CondensationActivityMode,
        CondensationConfiguration,
        CondensationExecutionMode,
        CondensationSurfaceMode,
    )
    from particula.execution.adapters.condensation import (
        CondensationExecutionConfig,
        WarpCondensationExecutionAdapter,
        WarpCondensationExecutionState,
        WarpCondensationState,
    )
    from particula.gpu.kernels.condensation import (
        CondensationActivitySurfaceConfig,
    )
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
    from particula.gpu.species_mapping import prepare_condensation_species_map

    wp, session = owner._wp, owner._session
    config = plan.values

    def array(values, dtype=None):
        return wp.array(
            values, dtype=dtype or wp.float64, device=owner.device.native
        )

    pressure_parameters = np.zeros((len(config["vapor"]), 4))
    pressure_parameters[:, 0] = [value for _, value in config["vapor"]]
    thermodynamics = ThermodynamicsConfig(
        array([mode for mode, _ in config["vapor"]], wp.int32),
        array(pressure_parameters),
        array(config["molar_mass"]),
    )
    activity = CondensationActivitySurfaceConfig(
        config["activity_mode"],
        config["surface_mode"],
        config["water"],
        array(config["kappas"]),
        array(config["references"]),
    )
    resources = owner._registry.acquire_condensation()
    species_binding = prepare_condensation_species_map(
        session.particles,
        session.gas,
        gas_names=session.metadata.gas_names,
        species_map=CondensationSpeciesMap(
            config["gas_names"], config["pairs"]
        ),
    )
    profile = CondensationConfiguration(
        CondensationExecutionMode.EQUAL_STEP,
        config["latent"] is not None,
        CondensationActivityMode.KAPPA
        if config["activity_mode"]
        else CondensationActivityMode.IDEAL,
        CondensationSurfaceMode.COMPOSITION_WEIGHTED
        if config["surface_mode"]
        else CondensationSurfaceMode.STATIC,
    )
    state = WarpCondensationState(
        CondensationExecutionConfig(profile),
        session.particles,
        session.gas,
        session.environment,
        thermodynamics,
        activity_surface=activity,
        scratch_buffers=resources.scratch_buffers,
        species_binding=species_binding,
        surface_tension=array(config["tensions"]),
        mass_accommodation=array(config["accommodation"]),
        diffusion_coefficient_vapor=array(config["diffusion"]),
        energy_transfer=wp.zeros(
            session.gas.concentration.shape,
            dtype=wp.float64,
            device=owner.device.native,
        )
        if config["latent"] is not None
        else None,
        latent_heat=array(config["latent"])
        if config["latent"] is not None
        else None,
    )
    binding = WarpCondensationExecutionAdapter().prepare(
        WarpCondensationExecutionState(state, owner.time_step)
    )
    binding = replace(
        binding,
        prepared=replace(binding.prepared, failure=owner._device_failure),
    )
    owner._bindings[plan.family] = binding
    if state.energy_transfer is not None:
        owner._auxiliary["latent_heat_energy"] = state.energy_transfer
    return binding.execute
