"""Particula activity provider for native CPU and prepared CUDA condensation."""

from copy import copy
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from operator import index
from pathlib import Path
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
            "hybrid_neutral_activities",
            "hybrid_ros3",
        }
    )

    def __init__(
        self,
        mixture,
        material_mapping,
        inert_indices=(),
        execution="cpu",
        evaluation="direct",
        table=None,
    ):
        if execution != "cpu":
            raise ValueError("native condensation requires CPU execution")
        self.mixture = mixture
        self.prepared = AIOMFACModel.prepare(mixture, execution=execution)
        if evaluation not in ("direct", "hybrid") or (
            evaluation == "hybrid"
        ) != (table is not None):
            raise ValueError(
                "hybrid evaluation requires a table and direct evaluation takes none"
            )
        self.table = None
        self.evaluation = evaluation
        if table is not None:
            from .sparse import ActivityTable

            path = Path(table)
            self.table_source = (
                str(path.resolve()) if path.is_file() else str(table)
            )
            self.table = ActivityTable.load(self.table_source)
            if tuple(self.table.recipe.mixture.components.items()) != tuple(
                mixture.components.items()
            ):
                raise ValueError("sparse table analytical mixture mismatch")
        self.counts = np.zeros(7, dtype=np.int64)
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
        ) + (
            ()
            if self.table is None
            else ((self.table_source, self.table.digest),)
        )

    def __deepcopy__(self, memo):
        # Keep a private provider object without trying to pickle mapping proxies.
        return self.from_identity(self.prepared_identity())

    @classmethod
    def from_identity(cls, identity):
        """Restore a provider after checking its layout, revision and artifact."""
        _, _, _, _, _, _, components, order, width, *artifact = identity
        mixture = MixtureSpec(
            {name: dict(groups) for name, groups in components}
        )
        mapping = dict(zip(order, mixture.components, strict=True))
        provider = cls(
            mixture,
            mapping,
            tuple(i for i in range(width) if i not in mapping),
            **(
                dict(evaluation="hybrid", table=artifact[0][0])
                if artifact
                else {}
            ),
        )
        if provider.prepared_identity() != identity:
            raise ValueError(
                "incompatible AIOMFAC parameter or provider identity"
            )
        return provider

    def direct_for_linearization(self):
        """Share the prepared CPU chemistry for physical rate derivatives."""
        direct = copy(self)
        direct.table = None
        direct.evaluation = "direct"
        return direct

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
        """Return dimensionless neutral activities in native material order."""
        return self.neutral_with_error(masses, temperature, targets, active)[0]

    def neutral_with_error(
        self, masses, temperature, targets, active, *, exact=False
    ):
        """Return neutral activities and empirical absolute log error by lane."""
        self.validate_material_layout(masses.shape[-1], targets)
        selected = [self.neutral_lanes[k] for k in targets]
        result = np.zeros((len(masses), len(targets)))
        error = np.zeros_like(result)
        rows = np.flatnonzero(active)
        if not len(rows):
            return result, error
        dissolved = masses[np.ix_(rows, self.material_order)]
        tk = np.broadcast_to(temperature, (len(masses),))[rows]
        names = [self.prepared.neutrals[i] for i in selected]
        if (
            self.table is not None
            and names
            and not exact
            and set(names) <= set(self.table.recipe.targets)
        ):
            values, estimates, reasons = self.table.lookup(dissolved, tk)
            lanes = [self.table.recipe.targets.index(k) for k in names]
            result[rows] = values[:, lanes]
            error[rows] = estimates[:, lanes]
            self.counts[:5] += np.bincount(reasons, minlength=5)
            misses = reasons != 0
            dissolved, tk, rows = dissolved[misses], tk[misses], rows[misses]
        if len(rows):
            try:
                values = self.prepared.evaluate(
                    dissolved, tk
                ).neutral_activities
            except ActivityDomainError as exc:
                self.counts[6] += 1
                slot = int(rows[exc.batch_index[0]])
                raise type(exc)(exc.reason, batch_index=(slot,)) from exc
            result[rows] = values[:, selected]
        return result, error

    def refine_neutrals(self, masses, temperature, targets, mask):
        """Recompute all requested neutrals of flux-ambiguous occupied rows."""
        self.counts[0] -= np.count_nonzero(mask)
        self.counts[5] += np.count_nonzero(mask)
        return self.neutral_with_error(
            masses, temperature, targets, mask, exact=True
        )[0]
