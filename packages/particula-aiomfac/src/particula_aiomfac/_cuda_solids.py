"""Prepare anhydrous mineral parameters for resident phase-aware transport."""

import numpy as np
import warp as wp

from ._cuda import _snippet


@wp.func_native(_snippet(phases=True, solids=True))
def _mineral_equilibrium(
    row: int,
    layout: wp.array(dtype=wp.int32),
    parameters: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    scratch: wp.array2d(dtype=wp.float64),
    formula_mw: wp.array(dtype=wp.float64),
    solver: wp.array2d(dtype=wp.float64),
    split: wp.array2d(dtype=wp.float64),
    ionic_basis: int,
    minerals: wp.array2d(dtype=wp.float64),
    mineral_work: wp.array2d(dtype=wp.float64),
): ...


@wp.kernel(enable_backward=False)
def _incipient_brine(
    source: wp.array2d(dtype=wp.float64),
    species: wp.array2d(dtype=wp.float64),
    mineral_index: wp.array(dtype=wp.int32),
    limit: wp.array(dtype=wp.int32),
    neutral_count: int,
    water: int,
):
    row = wp.tid()
    limit[row] = -1
    for j in range(neutral_count):
        if species[row, j] != wp.float64(0.0):
            return
    # At zero solvent an infinitesimal saturated film defines water fugacity.
    # Source and accepted native particle mass remain exactly dry.
    count = int(0)
    chosen = int(-1)
    salt_mass = wp.float64(0.0)
    for j in range(source.shape[1]):
        if source[row, j] > wp.float64(0.0):
            count += 1
            chosen = mineral_index[j]
            salt_mass += source[row, j]
    if count != 1 or chosen < 0 or water < 0:
        limit[row] = -2
        return
    species[row, water] = wp.float64(1.0e-4) * salt_mass
    limit[row] = chosen


@wp.kernel(enable_backward=False)
def _batch(
    layout: wp.array(dtype=wp.int32),
    parameters: wp.array(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    out: wp.array2d(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    scratch: wp.array2d(dtype=wp.float64),
    formula_mw: wp.array(dtype=wp.float64),
    solver: wp.array2d(dtype=wp.float64),
    split: wp.array2d(dtype=wp.float64),
    ionic_basis: int,
    active: wp.array(dtype=wp.int32),
    minerals: wp.array2d(dtype=wp.float64),
    mineral_work: wp.array2d(dtype=wp.float64),
    dry_mineral: wp.array(dtype=wp.int32),
):
    row = wp.tid()
    if active[row] == 0:
        status[row] = 0
        return
    dry = dry_mineral[row]
    if dry == -2:
        status[row] = -20
        return
    _mineral_equilibrium(
        row,
        layout,
        parameters,
        masses,
        temperature,
        out,
        status,
        scratch,
        formula_mw,
        solver,
        split,
        ionic_basis,
        minerals,
        mineral_work,
    )

    if dry >= 0 and status[row] >= 0:
        extent = mineral_work[row, 3 * masses.shape[1] + dry]
        if extent <= wp.float64(0.0):
            status[row] = -20


class DeviceSolids:
    """Prepared crystal reactions and liquid/solid solver workspace."""

    def __init__(self, provider, capacity, model, targets):
        from .solids import solid_parameters

        specs, stoich = solid_parameters(provider, provider.solids)
        p = model.prepared
        self.neutral_count = len(p.neutrals)
        src = provider.prepared
        water = np.flatnonzero(src.neutral_indices == provider.water_index)
        self.seed_water = -1
        if len(water) and tuple(provider.neutral_lanes[j] for j in targets) == (
            water[0],
        ):
            self.seed_water = int(water[0])
        dry = np.full(src.ncomponents, -1, dtype=np.int32)
        for lane, nu in zip(src.salt_indices, src.stoich, strict=True):
            if lane < src.ncomponents:
                found = np.flatnonzero(np.all(stoich == nu, axis=1))
                if found.size == 1:
                    dry[lane] = found[0]
        self.dry_index = wp.array(dry, device=model.device)
        self.dry_mineral = wp.empty(capacity, dtype=int, device=model.device)
        transfer = stoich.copy()
        if p.sulfate:
            hs, proton, sulfate = map(p.ions.index, (248, 205, 261))
            transfer[:, proton] += transfer[:, hs]
            transfer[:, sulfate] += transfer[:, hs]
            transfer[:, hs] = 0.0
        if p.carbonate and p.carbonate_model == "gibbs_consistent":
            hc, proton, carbonate = map(p.ions.index, (250, 205, 262))
            # Crystal extents transfer conservative carbon and solvent-water
            # equivalents. Reactive proton equivalents follow electroneutrality.
            neutral = np.zeros((len(specs), self.neutral_count))
            carbon = transfer[:, carbonate] + transfer[:, hc]
            hydroxide = p.ions.index(247)
            neutral[:, p.co2] = carbon
            neutral[:, p.water] = carbon + transfer[:, hydroxide]
            transfer[:, [proton, carbonate, hc, hydroxide]] = 0.0
        else:
            neutral = np.zeros((len(specs), self.neutral_count))
        transfer = np.column_stack((neutral, transfer))
        constants = np.array(
            [
                (v.log10_ksp, v.temperature_min_k, v.temperature_max_k)
                for v in specs
            ]
        )
        self.params = wp.array(
            np.column_stack((constants, stoich, transfer)), device=model.device
        )
        self.device = model.device
        width = 3 * p.ncomponents + len(specs)
        self.work = wp.empty(
            (capacity, width), dtype=wp.float64, device=model.device
        )

    def evaluate_into(self, inputs, rows, stream, source):
        """Enqueue the complementarity solve on the caller's stream."""
        wp.launch(
            _incipient_brine,
            rows,
            inputs=[
                source,
                inputs[2],
                self.dry_index,
                self.dry_mineral,
                self.neutral_count,
                self.seed_water,
            ],
            device=self.device,
            stream=stream,
        )
        wp.launch(
            _batch,
            rows,
            inputs=[*inputs, self.params, self.work, self.dry_mineral],
            device=self.device,
            stream=stream,
        )
