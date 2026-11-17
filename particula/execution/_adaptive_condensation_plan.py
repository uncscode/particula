"""Freeze native material laws for adaptive isothermal device rates.

This profile is separate from the inventory-limited, four-substep prepared
process. It retains every mapped material and the CPU surface phase law.
"""

import numpy as np

from particula import _cpu_batch
from particula.execution.prepared_condensation import _vapor_plan, _vector
from particula.execution.prepared_processes import _ProcessPlan
from particula.particles.activity_strategies import (
    ActivityIdealMass,
    ActivityIdealMolar,
    ActivityIdealVolume,
    ActivityKappaParameter,
)
from particula.particles.surface_strategies import (
    SurfaceStrategyMass,
    SurfaceStrategyMolar,
    SurfaceStrategyVolume,
)


def plan_adaptive_condensation(strategy, aerosol):  # noqa: C901
    """Validate and freeze constant/Buck, ideal/kappa and phased surfaces."""
    if strategy.update_gases is not True:
        raise ValueError("adaptive condensation requires coupled gas updates")
    parameters = None
    for box in range(aerosol.particles.n_boxes):
        view = _cpu_batch.box_view(aerosol, box)
        parameters = strategy._native_parameters(
            view.particles,
            view.gas,
            float(view.environment.temperature[0]),
            float(view.environment.pressure[0]),
        )
    if parameters is None:
        raise ValueError("adaptive condensation requires at least one box")
    sources, targets, molar, diffusion, accommodation = parameters
    width = aerosol.particles.n_species
    activity = strategy.activity_strategy
    references = np.ones(width)
    activity_density = np.ones(width)
    kappas = np.zeros(width)
    water = 0
    mode = 0
    if type(activity) in (ActivityIdealMolar, ActivityKappaParameter):
        references = _vector(
            activity.molar_mass, width, "activity molar mass", positive=True
        )
    elif type(activity) is ActivityIdealVolume:
        references = _vector(
            activity.density, width, "activity density", positive=True
        )
    elif type(activity) is not ActivityIdealMass:
        raise ValueError("adaptive GPU activity supports ideal/kappa only")
    if type(activity) is ActivityKappaParameter:
        mode = 1
        activity_density = _vector(
            activity.density, width, "activity density", positive=True
        )
        kappas = _vector(activity.kappa, width, "kappa")
        water = activity.water_index
        if (
            isinstance(water, (bool, np.bool_))
            or not isinstance(water, (int, np.integer))
            or not 0 <= water < width
        ):
            raise ValueError("water_index must be an integer material index")
    surface = strategy.surface_strategy
    if type(surface) not in (
        SurfaceStrategyMass,
        SurfaceStrategyMolar,
        SurfaceStrategyVolume,
    ):
        raise ValueError("GPU surface supports mass/molar/volume laws")
    if (
        surface.surface_tension_table is not None
        or surface.temperature_table is not None
    ):
        raise ValueError("GPU surface tension tables are unsupported")
    tensions = _vector(
        surface.surface_tension, width, "surface tension", positive=True
    )
    surface_density = _vector(
        surface.density, width, "surface density", positive=True
    )
    surface_reference = np.ones(width)
    if type(surface) is SurfaceStrategyVolume:
        surface_reference = surface_density
    elif type(surface) is SurfaceStrategyMolar:
        surface_reference = _vector(
            surface.molar_mass, width, "surface molar mass", positive=True
        )
    phases = np.arange(width, dtype=np.int32)
    if surface.phase_index is not None:
        raw = np.asarray(surface.phase_index)
        if raw.shape != (width,) or raw.dtype.kind not in "iu":
            raise ValueError("phase_index must follow material order")
        # Compact equality labels avoid integer overflow.
        _, phases = np.unique(raw, return_inverse=True)
    values = dict(
        pairs=tuple(
            (int(s), int(t)) for s, t in zip(sources, targets, strict=True)
        ),
        vapor=_vapor_plan(
            strategy.vapor_pressure_strategy, len(molar), allow_zero=True
        ),
        diffusion=diffusion,
        accommodation=accommodation,
        tensions=tensions,
        surface_density=surface_density,
        surface_reference=surface_reference,
        phases=phases,
        activity_mode=mode,
        water=int(water),
        kappas=kappas,
        references=references,
        activity_density=activity_density,
    )
    return _ProcessPlan(
        "condensation",
        tuple(
            (
                name,
                tuple(value.tolist())
                if isinstance(value, np.ndarray)
                else value,
            )
            for name, value in values.items()
        ),
    )
