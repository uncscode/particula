"""Resident fixed-inventory phase stability search on a CUDA stream."""

import numpy as np
import warp as wp

from ._cuda import _snippet
from .phases import component_molar_masses


@wp.func_native(_snippet(phases=True))
def _phases(
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
): ...


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
):
    row = wp.tid()
    if active[row] == 0:
        status[row] = 0
        return
    _phases(
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
    )


@wp.kernel(enable_backward=False)
def _speciate(
    mass: wp.array2d(dtype=wp.float64),
    transform: wp.array2d(dtype=wp.float64),
    species: wp.array2d(dtype=wp.float64),
):
    row, component = wp.tid()
    value = wp.float64(0.0)
    for lane in range(mass.shape[1]):
        source = mass[row, lane]
        if not wp.isfinite(source) or source < wp.float64(0.0):
            value = wp.float64(-1.0)
            break
        value += source * transform[lane, component]
    species[row, component] = value


@wp.kernel(enable_backward=False)
def _hydrate_co2(
    species: wp.array2d(dtype=wp.float64),
    neutral_count: int,
    water: int,
    carbon: int,
    proton: int,
    carbonate: int,
    hydroxide: int,
    water_mw: wp.float64,
    carbon_mw: wp.float64,
    carbonate_mw: wp.float64,
    proton_share: wp.float64,
    carbon_share: wp.float64,
):
    row = wp.tid()
    ionic_mass = wp.float64(0.0)
    for k in range(neutral_count, species.shape[1]):
        ionic_mass += species[row, k]
    if ionic_mass <= wp.float64(0.0):
        return
    donor = carbon
    credit1 = proton
    credit2 = carbonate
    moles = species[row, carbon] / carbon_mw
    share = proton_share
    if moles == wp.float64(0.0):
        if species[row, proton] > wp.float64(0.0):
            return
        # Basic feeds retain independent carbon and counterion exchange:
        # CO3-- + water is equivalent to CO2 + 2 OH-.
        donor = carbonate
        credit1 = carbon
        credit2 = hydroxide
        moles = species[row, carbonate] / carbonate_mw
        share = carbon_share
    if moles > wp.float64(0.0):
        debit = moles * water_mw
        if species[row, water] <= debit:
            species[row, water] = wp.float64(-1.0)
            return
        transfer = species[row, donor] + debit
        species[row, water] -= debit
        species[row, donor] = wp.float64(0.0)
        species[row, credit1] += transfer * share
        species[row, credit2] += transfer * (wp.float64(1.0) - share)


class DevicePhases:
    """One thread owns all trial chemistry for a row."""

    def __init__(self, model, *, capacity=None, provider=None, targets=()):
        from ._cuda import CudaModel

        capacity = model.capacity if capacity is None else capacity
        source = model.prepared
        self.transform = None
        self.hydration = None
        self.source = source
        extension = (
            source.carbonate and source.carbonate_model == "gibbs_consistent"
        )
        self.solids = (
            provider.solids
            if provider is not None and getattr(provider, "solids", None)
            else None
        )
        if (
            self.solids
            or extension
            and np.any(source.salt_indices < source.ncomponents)
            or not source.carbonate
            and (
                source.sulfate
                or (
                    source.ncation > 1 and len(source.ions) - source.ncation > 1
                )
            )
        ):
            from .reciprocal import ion_basis

            basis, self.transform = ion_basis(source)
            self.device_transform = wp.array(
                self.transform, device=model.device
            )
            self.species = wp.empty(
                (capacity, basis.ncomponents),
                dtype=wp.float64,
                device=model.device,
            )
            if extension:
                n = len(basis.neutrals)
                h, c, oh = map(basis.ions.index, (205, 262, 247))
                carbon_mw = basis.neutral_mw[basis.co2]
                self.hydration = [
                    n,
                    basis.water,
                    basis.co2,
                    n + h,
                    n + c,
                    n + oh,
                    basis.neutral_mw[basis.water],
                    carbon_mw,
                    basis.salt_mw[c],
                    2
                    * basis.salt_mw[h]
                    / (2 * basis.salt_mw[h] + basis.salt_mw[c]),
                    carbon_mw / (carbon_mw + 2 * basis.salt_mw[oh]),
                ]
            model = CudaModel(basis, model.device, capacity=1)
        self.model = model
        if self.solids:
            from ._cuda_solids import DeviceSolids

            self.minerals = DeviceSolids(provider, capacity, model, targets)
        d = model.prepared.ncomponents
        self.mw = wp.array(
            component_molar_masses(model.prepared),
            dtype=wp.float64,
            device=model.device,
        )
        self.work = wp.empty(
            (capacity, d * d + 19 * d), dtype=wp.float64, device=model.device
        )
        # Per-component fractions in the first liquid, then minimum TPD and
        # largest interphase chemical potential residual, both divided by RT.
        self.split = wp.empty(
            (capacity, d + 2), dtype=wp.float64, device=model.device
        )

    def evaluate_into(
        self, mass, temperature, out, status, scratch, active, stream
    ):
        """Write stable neutral activities into the direct chemistry layout."""
        model = self.model
        rows = mass.shape[0]
        source_mass = mass
        if self.transform is not None:
            wp.launch(
                _speciate,
                dim=(rows, self.species.shape[1]),
                inputs=[mass, self.device_transform, self.species],
                stream=stream,
                device=model.device,
            )
            if self.hydration and not self.solids:
                wp.launch(
                    _hydrate_co2,
                    dim=rows,
                    inputs=[self.species, *self.hydration],
                    stream=stream,
                    device=model.device,
                )
            mass = self.species
        inputs = [
            model.layout,
            model.parameters,
            mass,
            temperature,
            out,
            status,
            scratch,
            self.mw,
            self.work,
            self.split,
            int(self.transform is not None),
            active,
        ]
        if self.solids:
            self.minerals.evaluate_into(inputs, rows, stream, source_mass)
        else:
            wp.launch(
                _batch,
                dim=rows,
                inputs=inputs,
                stream=stream,
                device=model.device,
            )


def phase_equilibrium(model, masses, temperature, water_index):
    """Solve one liquid equilibrium on CUDA and return a host result."""
    from ._equilibrium import ConvergenceError
    from .phases import LiquidEquilibrium

    mass = np.asarray(masses, dtype=np.float64)
    if mass.shape != (model.prepared.ncomponents,):
        raise ValueError("phase_equilibrium expects one analytical inventory")
    if not 0 <= water_index < mass.size:
        raise ValueError("water_index is outside the analytical inventory")
    with model.lock, wp.ScopedStream(model.stream):
        phases = model._standalone_phases
        if phases is None:
            phases = DevicePhases(model, capacity=1)
            model._standalone_phases = phases
        dev = model.device
        phases.evaluate_into(
            wp.array(mass[None, :], device=dev),
            wp.array([float(temperature)], dtype=wp.float64, device=dev),
            model.output,
            model.status,
            model.scratch,
            wp.ones(1, dtype=int, device=dev),
            model.stream,
        )
        status = int(model.status[:1].numpy()[0])
        if status < 0:
            reason = {
                -10: "stable liquid calculation requires a solid phase",
                -11: "liquid stability search unresolved",
                -12: "liquid coexistence could not be verified",
                -13: "two-liquid stability unresolved or additional phase",
                -14: "input cannot span independent ion exchanges",
                -15: "carbonate lacks Gibbs-consistent CO2 activities",
                -16: "input phase violates electroneutrality",
            }.get(status, "invalid chemistry or speciation failure")
            raise ConvergenceError(f"CUDA phase status {status}: {reason}")
        fractions = phases.split[:1].numpy()[0]
    fraction, tpd, error = fractions[:-2], fractions[-2], fractions[-1]
    if status != 2:
        return LiquidEquilibrium(
            mass.copy(),
            None,
            model.evaluate(mass, temperature, diagnostics=True),
            None,
            0.0,
            float(tpd),
        )
    if phases.transform is None:
        sources = fraction
        species_mass = mass
    else:
        transform = phases.transform
        if phases.hydration and np.any(
            (mass @ transform)[len(phases.model.prepared.neutrals) :] > 0
        ):
            from .reciprocal import hydrate_carbon

            transform = hydrate_carbon(
                phases.source, transform, mass, phases.model.prepared
            )
        sources = transform @ fraction
        species_mass = mass @ transform
    alpha, beta = mass * sources, mass * (1.0 - sources)
    alpha_species = species_mass * fraction
    beta_species = species_mass * (1.0 - fraction)
    if alpha[water_index] / alpha.sum() < beta[water_index] / beta.sum():
        alpha, beta = beta, alpha
        alpha_species, beta_species = beta_species, alpha_species
    return LiquidEquilibrium(
        alpha,
        beta,
        phases.model.evaluate(alpha_species, temperature, diagnostics=True),
        phases.model.evaluate(beta_species, temperature, diagnostics=True),
        float(error),
        float(tpd),
    )
