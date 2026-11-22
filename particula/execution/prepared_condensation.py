"""Host-side condensation capability checks and retained GPU compilation.

The direct solver has four equal substeps, material activity corrections,
and a mixture Kelvin density. Admit only CPU configuration that can be
represented by those controls; do not discard unsupported strategy choices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from particula.execution._bat_table import _binary_activity_table
from particula.execution.preparation_context import BuiltinPreparationContext
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


def _vapor_plan(strategy, width, *, allow_zero=False):
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
                    item.vapor_pressure,
                    1,
                    "vapor_pressure",
                    positive=not allow_zero,
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
                strategy.vapor_pressure,
                width,
                "vapor_pressure",
                positive=not allow_zero,
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
    # A constant CPU surface density is distinct from physical material density.
    # Retain the legacy mixture path only where the two are always identical.
    if not np.all(density == density[0]):
        raise ValueError("GPU Kelvin density requires uniform surface density")
    kelvin_density = (
        None if np.all(particles.density == density[0]) else float(density[0])
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
        if not np.all(particles.density == particles.density[0]):
            raise ValueError(
                "GPU surface weighting requires uniform material densities"
            )
        mode = 1
    return tuple(float(x) for x in tensions), mode, kelvin_density


def _activity_plan(activity, particles, gas, pairs, vapor):  # noqa: C901
    """Translate material activities to the direct solver configuration."""
    from particula.particles.activity_strategies import (
        ActivityBAT,
        ActivityIdealMass,
        ActivityIdealMolar,
        ActivityIdealVolume,
        ActivityKappaParameter,
    )
    from particula.particles.prepared_activity import provider_identity

    if provider_identity(activity) is not None:
        activity.validate_material_layout(
            particles.n_species, tuple(t for _, t in pairs)
        )
        references = np.ones(particles.n_species)
        for source, target in pairs:
            references[target] = gas.molar_mass[source]
        return 3, 0, (0.0,) * particles.n_species, tuple(references), None
    if type(activity) is ActivityBAT:
        if particles.n_species != 2:
            raise ValueError("BAT requires [water, organic] material order")
        return (
            2,
            0,
            (0.0, 0.0),
            (0.01801528, activity.molar_mass_kg),
            tuple(_binary_activity_table(activity).tolist()),
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
        None,
    )


def plan_condensation(
    process, aerosol, *, allow_zero_vapor=False
) -> _ProcessPlan:
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
    prepared_activity = (
        getattr(strategy.activity_strategy, "prepared_provider_version", None)
        == 1
    )
    vapor = _vapor_plan(
        strategy.vapor_pressure_strategy,
        len(aerosol.gas.name),
        allow_zero=allow_zero_vapor or prepared_activity,
    )
    tensions, surface_mode, kelvin_density = _surface_plan(
        strategy.surface_strategy, aerosol.particles
    )
    activity_mode, water, kappas, references, bat_table = _activity_plan(
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
                **(
                    {"kelvin_density": kelvin_density}
                    if kelvin_density is not None
                    else {}
                ),
                activity_mode=activity_mode,
                **(
                    {"provider": strategy.activity_strategy.prepared_identity()}
                    if activity_mode == 3
                    else {}
                ),
                water=water,
                kappas=kappas,
                **({"bat_table": bat_table} if bat_table is not None else {}),
                references=references,
                latent=latent,
            ).items()
        ),
    )


def compile_condensation(
    plan: _ProcessPlan, context: BuiltinPreparationContext
):
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

    wp, session = context.wp, context.session
    config = plan.values

    def array(values, dtype=None):
        return wp.array(
            values, dtype=dtype or wp.float64, device=context.device.native
        )

    pressure_parameters = np.zeros((len(config["vapor"]), 4))
    pressure_parameters[:, 0] = [value for _, value in config["vapor"]]
    thermodynamics = ThermodynamicsConfig(
        array([mode for mode, _ in config["vapor"]], wp.int32),
        array(pressure_parameters),
        array(config["molar_mass"]),
        allow_zero_pressure="provider" in config,
    )
    provider = None
    if "provider" in config:
        from particula.particles.prepared_activity import restore_provider

        provider = restore_provider(config["provider"]).bind_device(
            context.particles.masses.shape,
            tuple(t for _, t in config["pairs"]),
            context.device.native,
        )
    activity = CondensationActivitySurfaceConfig(
        config["activity_mode"],
        config["surface_mode"],
        config["water"],
        array(config["kappas"]),
        array(config["references"]),
        array(config["bat_table"]) if "bat_table" in config else None,
        provider,
    )
    resources = context.registry.acquire_condensation()
    species_binding = prepare_condensation_species_map(
        context.particles,
        context.gas,
        gas_names=session.metadata.gas_names,
        species_map=CondensationSpeciesMap(
            config["gas_names"], config["pairs"]
        ),
    )
    profile = CondensationConfiguration(
        CondensationExecutionMode.EQUAL_STEP,
        config["latent"] is not None,
        (
            CondensationActivityMode.IDEAL,
            CondensationActivityMode.KAPPA,
            CondensationActivityMode.BAT,
            CondensationActivityMode.PREPARED,
        )[config["activity_mode"]],
        CondensationSurfaceMode.COMPOSITION_WEIGHTED
        if config["surface_mode"]
        else CondensationSurfaceMode.STATIC,
    )
    state = WarpCondensationState(
        CondensationExecutionConfig(profile),
        context.particles,
        context.gas,
        context.environment,
        thermodynamics,
        activity_surface=activity,
        scratch_buffers=resources.scratch_buffers,
        species_binding=species_binding,
        surface_tension=array(config["tensions"]),
        mass_accommodation=array(config["accommodation"]),
        diffusion_coefficient_vapor=array(config["diffusion"]),
        energy_transfer=wp.zeros(
            context.gas.concentration.shape,
            dtype=wp.float64,
            device=context.device.native,
        )
        if config["latent"] is not None
        else None,
        latent_heat=array(config["latent"])
        if config["latent"] is not None
        else None,
    )
    binding = WarpCondensationExecutionAdapter().prepare(
        WarpCondensationExecutionState(state, context.time_step)
    )
    exclusion = next(
        (p for p in context.plans if p.family == "ion_condensation"), None
    )
    ion_materials = (
        (
            exclusion.values["positive_species"],
            exclusion.values["negative_species"],
        )
        if exclusion is not None
        else None
    )
    binding = replace(
        binding,
        prepared=replace(
            binding.prepared,
            failure=context.failure,
            ion_material_indices=ion_materials,
            kelvin_density=config.get("kelvin_density", 0.0),
        ),
    )
    context.bindings[plan.family] = binding
    if state.energy_transfer is not None:
        context.auxiliary["latent_heat_energy"] = state.energy_transfer
    return binding.execute
