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
    """Map dissolved components and inert materials to native indices."""

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
        phase_mode="single_liquid",
        solids=None,
        carbonate_model="pinned",
    ):
        """Prepare a mixture for the declared native material layout."""
        if execution != "cpu":
            raise ValueError("native condensation requires CPU execution")
        self._validate_phase_options(
            phase_mode, solids, carbonate_model, evaluation, table
        )
        self.solids = MappingProxyType(dict(solids or {}))
        self.phase_mode = phase_mode
        self.carbonate_model = carbonate_model
        self.mixture = mixture
        self.prepared = AIOMFACModel.prepare(
            mixture, execution=execution, **self._carbonate_options()
        )
        if evaluation not in ("direct", "hybrid") or (
            evaluation == "hybrid"
        ) != (table is not None):
            raise ValueError(
                "hybrid requires a table and direct evaluation takes none"
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
                if k in by_name
                and (
                    mixture.components[k] != {173: 1}
                    or carbonate_model == "gibbs_consistent"
                )
            }
        )

    @staticmethod
    def _validate_phase_options(
        phase_mode, solids, carbonate_model, evaluation, table
    ):
        if phase_mode not in ("single_liquid", "stable"):
            raise ValueError("phase_mode must be single_liquid or stable")
        if phase_mode == "stable" and (
            table is not None or evaluation != "direct"
        ):
            raise ValueError("stable phases require direct activity evaluation")
        if solids and phase_mode != "stable":
            raise ValueError(
                "solid-aware condensation requires stable phase mode"
            )
        if carbonate_model not in ("pinned", "gibbs_consistent"):
            raise ValueError("unsupported carbonate activity model")
        if carbonate_model != "pinned" and (
            table is not None or evaluation != "direct"
        ):
            raise ValueError(
                "Gibbs-consistent carbonate requires direct evaluation"
            )

    def prepared_identity(self):
        """Freeze the chemistry, material order, references and solver."""
        components = tuple(
            (name, tuple(sorted(groups.items())))
            for name, groups in self.mixture.components.items()
        )
        return (
            (
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
            + (
                ()
                if self.table is None
                else ((self.table_source, self.table.digest),)
            )
            + (("stable",) if self.phase_mode == "stable" else ())
            + ((("solids", self._solid_identity()),) if self.solids else ())
            + (
                (("carbonate", self.carbonate_model),)
                if self.carbonate_model != "pinned"
                else ()
            )
        )

    def _carbonate_options(self):
        return (
            {}
            if self.carbonate_model == "pinned"
            else {"carbonate_model": self.carbonate_model}
        )

    def _solid_identity(self):
        from .solids import SolidSalt

        return tuple(
            (
                name,
                (spec.log10_ksp, spec.temperature_min_k, spec.temperature_max_k)
                if isinstance(spec, SolidSalt)
                else spec,
            )
            for name, spec in sorted(self.solids.items())
        )

    def __deepcopy__(self, memo):
        """Recreate a private provider from its checkpoint identity."""
        return self.from_identity(self.prepared_identity())

    @classmethod
    def from_identity(cls, identity):
        """Restore and verify the provider layout, revision and artifact."""
        _, _, _, _, _, _, components, order, width, *artifact = identity
        mixture = MixtureSpec(
            {name: dict(groups) for name, groups in components}
        )
        carbonate_model = "pinned"
        if (
            artifact
            and isinstance(artifact[-1], tuple)
            and artifact[-1][0] == "carbonate"
        ):
            carbonate_model = artifact.pop()[1]
        solids = {}
        if (
            artifact
            and isinstance(artifact[-1], tuple)
            and artifact[-1][0] == "solids"
        ):
            from .solids import SolidSalt

            for name, spec in artifact.pop()[1]:
                solids[name] = (
                    SolidSalt(*spec) if isinstance(spec, tuple) else spec
                )
        mode = "stable" if artifact == ["stable"] else "single_liquid"
        if mode == "stable":
            artifact = []
        mapping = dict(zip(order, mixture.components, strict=True))
        provider = cls(
            mixture,
            mapping,
            tuple(i for i in range(width) if i not in mapping),
            phase_mode=mode,
            solids=solids,
            carbonate_model=carbonate_model,
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
        """Bind one layout with independent stream-compatible buffers."""
        self.validate_material_layout(shape[2], targets)
        from ._condensation import DeviceActivities

        return DeviceActivities(self, shape, targets, device)

    def activity(self, mass_concentration):
        """Require the explicit-temperature activity entry point."""
        raise TypeError("AIOMFAC requires temperature and neutral target lanes")

    @property
    def water_index(self):
        """Analytical water component in mixture declaration order."""
        waters = [
            i
            for i, groups in enumerate(self.mixture.components.values())
            if groups == {16: 1}
        ]
        if len(waters) != 1:
            raise ValueError(
                "phase equilibrium requires one declared water component"
            )
        return waters[0]

    def phase_equilibrium(
        self, masses, temperature, *, execution="cpu", device=None
    ):
        """Resolve stable liquids assuming the supplied material is dissolved.

        For precipitation and residual dry ions, use solid_liquid_equilibrium.
        """
        mass = np.asarray(masses, dtype=float)
        if mass.shape != (self.width,):
            raise ValueError("expected one native material inventory")
        model = self.equilibrium_backend(execution, device)
        return model.phase_equilibrium(
            mass[list(self.material_order)], temperature, self.water_index
        )

    def equilibrium_backend(self, execution, device=None):
        """Prepare an explicitly selected backend for a standalone solve."""
        if execution == "cpu" and device in (None, "cpu"):
            return self.prepared
        return AIOMFACModel.prepare(
            self.mixture,
            execution=execution,
            device=device,
            **self._carbonate_options(),
        )

    def validate_material_names(self, material_names):
        """Verify that every dissolved component labels its native lane."""
        if any(
            material_names[i] != name
            for name, i in zip(
                self.mixture.components, self.material_order, strict=True
            )
        ):
            raise ValueError(
                "material_names disagree with the configured mixture"
            )

    def material_lanes(self, species):
        """Resolve named volatile neutral targets to native material indices."""
        lanes = {
            self.prepared.neutrals[col]: lane
            for lane, col in self.neutral_lanes.items()
        }
        try:
            return tuple(lanes[name] for name in species)
        except KeyError as exc:
            raise ValueError(
                f"unknown volatile neutral {exc.args[0]!r}"
            ) from exc

    def apply_species_equilibrium(self, aerosol, layout, **options):
        """Publish physical species amounts in mapped native reservoirs."""
        from .native_inventory import apply_species_equilibrium

        return apply_species_equilibrium(aerosol, self, layout, **options)

    def equilibrate_species(self, **kwargs):
        """Return dissolved, crystalline, interfacial and gas inventories."""
        from .reservoirs import equilibrate_species

        kwargs.setdefault("solids", self.solids)
        return equilibrate_species(self, **kwargs)

    def solid_liquid_equilibrium(self, masses, temperature, **kwargs):
        """Resolve ions, neutrals and crystals from native material totals."""
        from .solids import solve_solid_liquid

        mass = np.asarray(masses, dtype=float)
        if mass.shape != (self.width,):
            raise ValueError("expected one native material inventory")
        solids = kwargs.pop("solids", self.solids)
        return solve_solid_liquid(
            self, mass[list(self.material_order)], temperature, solids, **kwargs
        )

    def solve_partitioning(self, **kwargs):
        """Equilibrate the supplied material inventories with an ideal gas."""
        from .partitioning import gas_liquid_equilibrium

        return gas_liquid_equilibrium(self, **kwargs)

    def validate_material_layout(self, width, transfer_lanes):
        """Validate the gas map, including temporarily disabled transfers."""
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
        """Return neutral activities and absolute log error by lane."""
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
                if self.phase_mode == "stable":
                    values = []
                    water_only = names == [
                        tuple(self.mixture.components)[self.water_index]
                    ]
                    for j, (row, temperature) in enumerate(
                        zip(dissolved, tk, strict=True)
                    ):
                        try:
                            values.append(
                                self._stable_activities(
                                    row, temperature, water_only
                                )
                            )
                        except ActivityDomainError as exc:
                            raise type(exc)(
                                exc.reason, batch_index=(j,)
                            ) from exc
                    values = np.stack(values)
                else:
                    values = self.prepared.evaluate(
                        dissolved, tk
                    ).neutral_activities
            except ActivityDomainError as exc:
                self.counts[6] += 1
                slot = int(rows[exc.batch_index[0]])
                raise type(exc)(exc.reason, batch_index=(slot,)) from exc
            result[rows] = values[:, selected]
        return result, error

    def _stable_activities(self, row, temperature, water_only):
        """Resolve wet phases or the incipient brine on a dry mineral."""
        if not self.solids:
            phases = self.prepared.phase_equilibrium(
                row, temperature, self.water_index
            )
            return phases.alpha.neutral_activities

        from .solids import solve_solid_liquid

        if (
            not np.all(np.isfinite(row))
            or not np.any(row > 0)
            or np.any(row < 0)
        ):
            raise ActivityDomainError("invalid dissolved masses")
        # Intensive activities are independent of the particle's mass scale.
        feed = row / row.max()
        state = solve_solid_liquid(self, feed, temperature, self.solids)
        if state.liquids is None:
            if (
                not water_only
                or state.dry_ion_masses.any()
                or np.count_nonzero(state.solid_masses) != 1
            ):
                raise ActivityDomainError(
                    "dry surface requires a single mineral exchanging water"
                )
            feed[self.water_index] = 1e-4 * feed.sum()
            state = solve_solid_liquid(self, feed, temperature, self.solids)
            if not state.solid_masses.any() or state.liquids is None:
                raise ActivityDomainError(
                    "could not resolve the incipient saturated brine"
                )
        return state.liquids.alpha.neutral_activities

    def refine_neutrals(self, masses, temperature, targets, mask):
        """Recompute all requested neutrals of flux-ambiguous occupied rows."""
        self.counts[0] -= np.count_nonzero(mask)
        self.counts[5] += np.count_nonzero(mask)
        return self.neutral_with_error(
            masses, temperature, targets, mask, exact=True
        )[0]
