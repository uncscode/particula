"""Translation of the existing unit-count, neutral GPU nucleation model."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from particula.execution.prepared_processes import _ProcessPlan

_BOUND_NAMES = (
    "radius_cubed_relative_error",
    "mean_radius_relative_error",
    "surface_relative_error",
    "diversity_absolute_error",
)


def plan_nucleation(process, aerosol) -> _ProcessPlan:
    """Freeze full-gas provenance, source physics and exhaustion controls."""
    from particula._cpu_batch import box_view
    from particula.dynamics.nucleation.nucleation_strategies import (
        ActivationNucleationStrategy,
        KineticNucleationStrategy,
    )
    from particula.gpu.kernels.nucleation import NucleationConfig

    source, commit = process.source_config, process.commit_config
    strategy = source.strategy
    if type(strategy) not in (
        ActivationNucleationStrategy,
        KineticNucleationStrategy,
    ):
        raise ValueError("prepared nucleation supports activation/kinetic only")
    if aerosol.particles.distribution_type != "particle_resolved":
        raise ValueError("prepared nucleation requires particle_resolved")
    if commit.maximum_slot_weight != 1 or commit.source_charge != 0:
        raise ValueError(
            "GPU nucleation requires maximum_slot_weight=1 and source_charge=0"
        )
    for box in range(aerosol.particles.n_boxes):
        process._state(box_view(aerosol, box))
    domain = strategy.validity_domain
    config = dict(
        rate_law="activation"
        if type(strategy) is ActivationNucleationStrategy
        else "kinetic",
        coefficient=float(strategy.coefficient),
        survival_factor=float(strategy.survival_factor),
        precursor_index=int(source.precursor_index),
        molecule_counts=tuple(strategy.injection_composition.molecule_counts),
        formation_diameter=float(
            strategy.formation_metadata.formation_diameter
        ),
        precursor_number_concentration_lower=domain.precursor_number_concentration.lower,
        precursor_number_concentration_upper=domain.precursor_number_concentration.upper,
        temperature_lower=domain.temperature.lower,
        temperature_upper=domain.temperature.upper,
        saturation_lower=domain.saturation.lower if domain.saturation else None,
        saturation_upper=domain.saturation.upper if domain.saturation else None,
    )
    NucleationConfig(**config)
    controls = {}
    for name in ("requested_scale", "minimum_scale", "minimum_volume"):
        values = np.asarray(getattr(commit, name))
        if values.shape != (aerosol.particles.n_boxes,):
            raise ValueError(f"{name} must contain one value per box")
        controls[name] = tuple(float(x) for x in values)
    requested = np.asarray(controls["requested_scale"])
    minimum = np.asarray(controls["minimum_scale"])
    volume = np.asarray(controls["minimum_volume"])
    if not (
        np.all(np.isfinite(requested))
        and np.all(np.isfinite(minimum))
        and np.all(np.isfinite(volume))
        and np.all(minimum > 0)
        and np.all(requested >= minimum)
        and np.all(requested <= 1)
        and np.all(volume > 0)
    ):
        raise ValueError(
            "nucleation scaling controls are outside physical bounds"
        )
    return _ProcessPlan(
        "nucleation",
        tuple(
            dict(
                config=tuple(config.items()),
                gas_names=tuple(aerosol.gas.name),
                pairs=tuple(source.species_map.gas_to_particle),
                resampling=commit.exhaustion_controls.resampling,
                representative_volume_scaling=commit.exhaustion_controls.representative_volume_scaling,
                bounds=tuple(getattr(commit, name) for name in _BOUND_NAMES),
                **controls,
            ).items()
        ),
    )


def compile_nucleation(plan: _ProcessPlan, owner: Any):
    """Retain all source/exhaustion buffers and explicit status observation."""
    from functools import partial

    from particula.dynamics.nucleation.species_mapping import (
        NucleationSpeciesMap,
    )
    from particula.execution.prepared_nucleation_kernels import (
        _releasable_slots,
    )
    from particula.gpu.kernels.nucleation import (
        NucleationConfig,
        NucleationExhaustionControls,
        _enqueue_prepared_nucleation_call,
        _observe_prepared_nucleation_call,
        _prepare_nucleation_step_gpu,
    )
    from particula.gpu.nucleation_species_mapping import (
        prepare_nucleation_species_map,
    )

    wp, session = owner._wp, owner._session
    values = plan.values
    resources = owner._registry.acquire_nucleation()
    for name in ("requested_scale", "minimum_scale", "minimum_volume"):
        getattr(resources.exhaustion, name).assign(np.asarray(values[name]))
    binding = prepare_nucleation_species_map(
        session.particles,
        session.gas,
        gas_names=session.metadata.gas_names,
        species_map=NucleationSpeciesMap(values["gas_names"], values["pairs"]),
    )
    prepared = _prepare_nucleation_step_gpu(
        session.particles,
        session.gas,
        NucleationConfig(**dict(values["config"])),
        owner.time_step,
        scratch=resources.scratch,
        finalized_demand=resources.finalized_demand,
        diagnostics=resources.diagnostics,
        exhaustion_buffers=resources.exhaustion,
        exhaustion_controls=NucleationExhaustionControls(
            values["resampling"], values["representative_volume_scaling"]
        ),
        environment=session.environment,
        species_binding=binding,
    )
    if prepared.resampling is not None:
        # The CPU record already validates exact nonnegative finite bounds.
        # Retain the same prepared buffers and replace only immutable controls.
        prepared = replace(
            prepared,
            resampling=replace(prepared.resampling, bounds=values["bounds"]),
        )
    owner._observers.append(
        partial(_observe_prepared_nucleation_call, prepared)
    )
    counts = session.particles.concentration
    budget = resources.exhaustion.resampling_releasable_counts

    def enqueue():
        wp.launch(
            _releasable_slots,
            dim=counts.shape[0],
            inputs=[counts, budget],
            device=owner.device.native,
        )
        _enqueue_prepared_nucleation_call(prepared)

    return enqueue
