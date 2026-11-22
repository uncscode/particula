"""Particula activity provider for native CPU and prepared CUDA condensation."""

from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from operator import index
from types import MappingProxyType

import numpy as np
from particula.particles.activity_strategies import ActivityStrategy

from . import (
    UPSTREAM_REVISION,
    AIOMFACModel,
    MixtureSpec,
    __version__,
)

from ._equilibrium import ActivityDomainError


@lru_cache(maxsize=1)
def _parameter_digest():
    return sha256(
        files(__package__).joinpath("data/parameters.npz").read_bytes()
    ).hexdigest()


class AIOMFACActivity(ActivityStrategy):
    """Bind native material indices to dissolved components and explicit inert lanes."""

    prepared_provider_version = 1
    provider_interface_version = 1
    trial_failure_types = (ActivityDomainError,)
    capabilities = frozenset(
        {
            "direct_cpu",
            "native_isothermal",
            "prepared_cuda",
            "resident_condensation",
        }
    )

    def __init__(
        self, mixture, material_mapping, inert_indices=(), execution="cpu"
    ):
        if execution != "cpu":
            raise ValueError("native condensation requires CPU execution")
        self.mixture = mixture
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

    def prepared_identity(self):
        """Freeze chemistry, analytical order, reference conventions and solver."""
        components = tuple(
            (name, tuple(sorted(groups.items())))
            for name, groups in self.mixture.components.items()
        )
        return (
            "aiomfac",
            self.prepared_provider_version,
            __version__,
            UPSTREAM_REVISION,
            _parameter_digest(),
            "float64-x-neutral-molal-ions-cold-v1",
            components,
            self.material_order,
            self.width,
        )

    def __deepcopy__(self, memo):
        # Keep a private provider object without trying to pickle mapping proxies.
        return self.from_identity(self.prepared_identity())

    @classmethod
    def from_identity(cls, identity):
        _, _, _, _, _, _, components, order, width = identity
        mixture = MixtureSpec(
            {name: dict(groups) for name, groups in components}
        )
        mapping = dict(zip(order, mixture.components, strict=True))
        provider = cls(
            mixture, mapping, tuple(i for i in range(width) if i not in mapping)
        )
        if provider.prepared_identity() != identity:
            raise ValueError(
                "incompatible AIOMFAC parameter or provider identity"
            )
        return provider

    def bind_device(self, shape, targets, device):
        """Bind one layout; each owner has independent stream-compatible buffers."""
        self.validate_material_layout(shape[2], targets)
        from ._condensation import DeviceActivities

        return DeviceActivities(self, shape, targets, device)

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
                tk = np.broadcast_to(temperature, (len(masses),))[active]
                evaluation = self.prepared.evaluate(dissolved, tk)
            except ActivityDomainError as error:
                slot = int(np.flatnonzero(active)[error.batch_index[0]])
                raise type(error)(error.reason, batch_index=(slot,)) from error
            result[active] = evaluation.neutral_activities[:, selected]
        return result
