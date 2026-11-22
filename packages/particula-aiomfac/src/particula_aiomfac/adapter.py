"""Optional Particula v1 provider for native CPU condensation."""

from operator import index
from types import MappingProxyType

import numpy as np
from particula.particles.activity_strategies import ActivityStrategy

from . import AIOMFACModel, ConvergenceError


class AIOMFACActivity(ActivityStrategy):
    """Bind native material indices to dissolved components and explicit inert lanes."""

    provider_interface_version = 1
    capabilities = frozenset({"direct_cpu", "native_isothermal"})

    def __init__(
        self, mixture, material_mapping, inert_indices=(), execution="cpu"
    ):
        if execution != "cpu":
            raise ValueError("native condensation requires CPU execution")
        self.prepared = AIOMFACModel.prepare(mixture, execution=execution)
        names = tuple(mixture.components)
        inert = tuple(inert_indices)
        lanes = (*material_mapping, *inert)
        if any(isinstance(k, (bool, np.bool_)) for k in lanes):
            raise ValueError("material indices must be integers")
        indices = sorted(index(k) for k in lanes)
        if indices != list(range(len(indices))):
            raise ValueError(
                "material indices must be unique and cover the entire layout"
            )
        if sorted(material_mapping.values()) != sorted(names):
            raise ValueError(
                "material mapping must cover each dissolved component once"
            )
        self.width = len(indices)
        by_name = {name: lane for lane, name in material_mapping.items()}
        self.material_order = tuple(by_name[k] for k in names)
        self.neutral_lanes = MappingProxyType(
            {
                by_name[k]: i
                for i, k in enumerate(self.prepared.neutrals)
                if k in by_name and mixture.components[k] != {173: 1}
            }
        )

    def activity(self, mass_concentration):
        raise TypeError("AIOMFAC requires temperature and neutral target lanes")

    def validate_material_layout(self, width, transfer_lanes):
        """Admit the declared gas map, including temporarily disabled transfers."""
        if width != self.width:
            raise ValueError(
                "particle material width differs from prepared AIOMFAC layout"
            )
        for lane in transfer_lanes:
            if lane not in self.neutral_lanes:
                raise ValueError(
                    f"material lane {lane} is not a volatile neutral target"
                )

    def neutral_activities(self, masses, temperature, targets, active):
        self.validate_material_layout(masses.shape[-1], targets)
        selected = [self.neutral_lanes[k] for k in targets]
        result = np.zeros((len(masses), len(targets)))
        if np.any(active):
            dissolved = masses[np.ix_(active, self.material_order)]
            try:
                evaluation = self.prepared.evaluate(dissolved, temperature)
            except ConvergenceError as error:
                slot = int(np.flatnonzero(active)[error.batch_index[0]])
                raise ConvergenceError(
                    error.reason, batch_index=(slot,)
                ) from error
            result[active] = evaluation.neutral_activities[:, selected]
        return result
