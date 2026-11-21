# Warp compiles native structs, arrays, and multidimensional indices.
# mypy: disable-error-code="index,valid-type,operator"
"""Resident optical heating using staged particle and box inventories."""

from dataclasses import dataclass
from json import loads
from math import pi
from typing import no_type_check

import numpy as np
import warp as wp
from scipy.constants import Stefan_Boltzmann

from particula.dynamics.optical_heating import _MAX_K, _MIN_K
from particula.execution.extension_registry import (
    PreparedOperation,
    ProcessCapabilities,
)
from particula.execution.process_extensions import GPUProcessPlan
from particula.gpu.kernels.thermodynamics import _buck_vapor_pressure
from particula.gpu.warp_types import WarpParticleData
from particula.optics.gpu.state import ParticleOpticalCases
from particula.optics.gpu.workspace import prepare_mie
from particula.optics.state import OpticalConfig
from particula.util.constants import GAS_CONSTANT

_PI = wp.constant(wp.float64(pi))
_MIN = wp.constant(wp.float64(_MIN_K))
_MAX = wp.constant(wp.float64(_MAX_K))
_SIGMA = wp.constant(wp.float64(Stefan_Boltzmann))
_R = wp.constant(wp.float64(GAS_CONSTANT))
_ZERO = wp.constant(wp.float64(0.0))
_TWO = wp.constant(wp.float64(2.0))
_FOUR = wp.constant(wp.float64(4.0))


@wp.struct
class _Settings:
    kappa: wp.float64
    conductivity: wp.float64
    diffusivity: wp.float64
    cp_water: wp.float64
    cp_solute: wp.float64
    latent: wp.float64
    emissivity: wp.float64
    radiation_temperature: wp.float64


@wp.struct
class _BoxEnergy:
    absorbed: wp.array(dtype=wp.float64)
    conduction: wp.array(dtype=wp.float64)
    radiation: wp.array(dtype=wp.float64)
    latent: wp.array(dtype=wp.float64)
    sensible: wp.array(dtype=wp.float64)


@wp.struct
class _Balance:
    water: wp.float64
    saturation_to_density: wp.float64
    gas_vapor: wp.float64
    transfer: wp.float64
    latent_heat: wp.float64
    heat_capacity: wp.float64
    conduction: wp.float64
    emission: wp.float64
    radiative_background: wp.float64
    old_temperature: wp.float64
    gas_temperature: wp.float64


@wp.func
def _fourth(value: wp.float64) -> wp.float64:
    square = value * value
    return square * square


@wp.func
def _terms(temperature: wp.float64, balance: _Balance):
    """Evaluate step energies and the evaporation proposal at temperature."""
    saturation = (
        balance.saturation_to_density
        * _buck_vapor_pressure(temperature)
        / temperature
    )
    proposed_loss = wp.min(
        balance.water,
        balance.transfer * wp.max(saturation - balance.gas_vapor, _ZERO),
    )
    loss = balance.water - (balance.water - proposed_loss)
    return loss, wp.vec4d(
        balance.conduction * (temperature - balance.gas_temperature),
        balance.emission
        * (_fourth(temperature) - balance.radiative_background),
        balance.latent_heat * loss,
        balance.heat_capacity * (temperature - balance.old_temperature),
    )


@wp.func
def _sum4(x: wp.vec4d) -> wp.float64:
    return x[0] + x[1] + x[2] + x[3]


@wp.func
def _absorbed_power(
    box: int,
    slot: int,
    n_slots: int,
    cross_area: wp.float64,
    irradiance: wp.array(dtype=wp.float64),
    efficiencies: wp.array2d(dtype=wp.float64),
    optical_status: wp.array(dtype=wp.int32),
) -> wp.float64:
    waves = irradiance.shape[0]
    power = wp.float64(0.0)
    for wave in range(waves):
        case = (box * waves + wave) * n_slots + slot
        if optical_status[case] != 0:
            return wp.float64(-1.0)
        power = power + irradiance[wave] * cross_area * efficiencies[case, 2]
    return power


@wp.func
def _temperature_root(balance: _Balance, absorbed: wp.float64) -> wp.float64:
    _, lo_terms = _terms(_MIN, balance)
    _, hi_terms = _terms(_MAX, balance)
    if _sum4(lo_terms) > absorbed or _sum4(hi_terms) < absorbed:
        return wp.float64(-1.0)
    low = wp.float64(_MIN)
    high = wp.float64(_MAX)
    for _iteration in range(54):
        middle = (low + high) / _TWO
        _, terms = _terms(middle, balance)
        if _sum4(terms) < absorbed:
            low = middle
        else:
            high = middle
    return (low + high) / _TWO


@no_type_check
@wp.kernel(enable_backward=False)
def _stage_particle(
    particles: WarpParticleData,
    gas: wp.array2d(dtype=wp.float64),
    molar_mass: wp.array(dtype=wp.float64),
    ambient: wp.array(dtype=wp.float64),
    temperature: wp.array2d(dtype=wp.float64),
    settings: _Settings,
    particle_water: int,
    gas_water: int,
    dt: wp.float64,
    irradiance: wp.array(dtype=wp.float64),
    radii: wp.array2d(dtype=wp.float64),
    efficiencies: wp.array2d(dtype=wp.float64),
    optical_status: wp.array(dtype=wp.int32),
    proposed: wp.array3d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Propose each active particle's temperature, transfer and energies."""
    if failure[0] != 0:
        return
    box, slot = wp.tid()
    if particles.concentration[box, slot] <= _ZERO:
        return
    liquid = particles.masses[box, slot, particle_water]
    solute = particles.masses[box, slot, 1 - particle_water]
    ta = ambient[box]
    tp = temperature[box, slot]
    if tp == _ZERO:
        tp = ta
    if (
        solute <= _ZERO
        or ta < _MIN
        or ta > _MAX
        or tp < _MIN
        or tp > _MAX
        or not wp.isfinite(tp)
    ):
        wp.atomic_max(failure, 0, 101)
        return
    radius = radii[box * irradiance.shape[0], slot]
    cross_area = _PI * radius * radius
    power = _absorbed_power(
        box,
        slot,
        radii.shape[1],
        cross_area,
        irradiance,
        efficiencies,
        optical_status,
    )
    if power < _ZERO or not wp.isfinite(power):
        wp.atomic_max(failure, 0, 102)
        return

    activity = wp.float64(0.0)
    if liquid > _ZERO:
        wv = liquid / particles.density[particle_water]
        sv = solute / particles.density[1 - particle_water]
        activity = wv / (wv + settings.kappa * sv)
    four_pi_radius = _FOUR * _PI * radius
    balance = _Balance()
    balance.water = liquid
    balance.saturation_to_density = activity * molar_mass[gas_water] / _R
    balance.gas_vapor = gas[box, gas_water]
    balance.transfer = dt * four_pi_radius * settings.diffusivity
    balance.latent_heat = settings.latent
    balance.heat_capacity = (
        liquid * settings.cp_water + solute * settings.cp_solute
    )
    balance.conduction = dt * four_pi_radius * settings.conductivity
    balance.emission = dt * _FOUR * cross_area * settings.emissivity * _SIGMA
    balance.radiative_background = _fourth(settings.radiation_temperature)
    balance.old_temperature = tp
    balance.gas_temperature = ta

    absorbed = power * dt
    new_temp = _temperature_root(balance, absorbed)
    if new_temp < _MIN:
        wp.atomic_max(failure, 0, 103)
        return
    loss, diagnostics = _terms(new_temp, balance)
    remaining = liquid - loss
    if not wp.isfinite(remaining) or not wp.isfinite(diagnostics):
        wp.atomic_max(failure, 0, 103)
        return
    proposed[box, slot, 0] = remaining
    proposed[box, slot, 1] = new_temp
    proposed[box, slot, 2] = power
    proposed[box, slot, 3] = absorbed
    for channel in range(4):
        proposed[box, slot, channel + 4] = diagnostics[channel]


@no_type_check
@wp.kernel(enable_backward=False)
def _stage_box(
    particles: WarpParticleData,
    gas: wp.array2d(dtype=wp.float64),
    ledger: _BoxEnergy,
    weights: wp.array2d(dtype=wp.float64),
    waves: int,
    particle_water: int,
    gas_water: int,
    proposed: wp.array3d(dtype=wp.float64),
    next_box: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Form extensive changes in a fixed order before publishing any box."""
    if failure[0] != 0:
        return
    box = wp.tid()
    gas_gain = wp.float64(0.0)
    absorbed = wp.float64(0.0)
    energy = wp.vec4d(_ZERO, _ZERO, _ZERO, _ZERO)
    for slot in range(particles.masses.shape[1]):
        if particles.concentration[box, slot] > _ZERO:
            count = weights[box * waves, slot]
            lost = (
                particles.masses[box, slot, particle_water]
                - proposed[box, slot, 0]
            )
            gas_gain = gas_gain + count * lost
            absorbed = absorbed + count * proposed[box, slot, 3]
            particle_energy = wp.vec4d(
                proposed[box, slot, 4],
                proposed[box, slot, 5],
                proposed[box, slot, 6],
                proposed[box, slot, 7],
            )
            energy = energy + count * particle_energy
    next_gas = gas[box, gas_water] + gas_gain
    next_absorbed = ledger.absorbed[box] + absorbed
    energy = energy + wp.vec4d(
        ledger.conduction[box],
        ledger.radiation[box],
        ledger.latent[box],
        ledger.sensible[box],
    )
    if (
        not wp.isfinite(next_gas)
        or not wp.isfinite(next_absorbed)
        or not wp.isfinite(energy)
    ):
        wp.atomic_max(failure, 0, 104)
        return
    next_box[box, 0] = next_gas
    next_box[box, 1] = next_absorbed
    for channel in range(4):
        next_box[box, channel + 2] = energy[channel]


@no_type_check
@wp.kernel(enable_backward=False)
def _publish(
    particles: WarpParticleData,
    gas: wp.array2d(dtype=wp.float64),
    temperature: wp.array2d(dtype=wp.float64),
    power: wp.array2d(dtype=wp.float64),
    ledger: _BoxEnergy,
    particle_water: int,
    gas_water: int,
    proposed: wp.array3d(dtype=wp.float64),
    next_box: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Commit both phases and the energy ledger after a successful batch."""
    if failure[0] != 0:
        return
    box, slot = wp.tid()
    if particles.concentration[box, slot] > _ZERO:
        particles.masses[box, slot, particle_water] = proposed[box, slot, 0]
        temperature[box, slot] = proposed[box, slot, 1]
        power[box, slot] = proposed[box, slot, 2]
    if slot == 0:
        gas[box, gas_water] = next_box[box, 0]
        ledger.absorbed[box] = next_box[box, 1]
        ledger.conduction[box] = next_box[box, 2]
        ledger.radiation[box] = next_box[box, 3]
        ledger.latent[box] = next_box[box, 4]
        ledger.sensible[box] = next_box[box, 5]


def _optical_config(metadata):
    data = loads(metadata)
    index = np.array(data["index_real"]) + 1j * np.array(data["index_imag"])
    option = (
        "species_index" if data["index_type"] == "species" else "particle_index"
    )
    return OpticalConfig(
        data["wavelength_m"],
        medium_index=data["medium_index"],
        **{option: index},
    )


@dataclass(frozen=True)
class OpticalHeatingAdapter:
    """Bind optical heating to Warp with bounded coefficient-series scratch."""

    coefficient_capacity: int = 512
    particle_tile: int = 128

    def plan(self, process, aerosol):
        """Validate process identity, native fields and workspace limits."""
        process.validate(aerosol)
        if (
            type(self.coefficient_capacity) is not int
            or not 1 <= self.coefficient_capacity <= 10090
            or type(self.particle_tile) is not int
            or self.particle_tile <= 0
            or aerosol.particles.n_particles == 0
        ):
            raise ValueError("Invalid resident optical heating workspace")
        if (
            aerosol.particles.n_boxes
            * aerosol.particles.n_particles
            * len(process.optics.wavelength)
            >= np.iinfo(np.int32).max
        ):
            raise ValueError("Optical heating exceeds device index limits")
        kind = (
            3
            if aerosol.particles.distribution_type == "particle_resolved"
            else 1
        )
        return GPUProcessPlan(
            name=process.name,
            extension_id=process.extension_id,
            version=process.version,
            reads=process.reads,
            writes=process.writes,
            parameters=(
                ("identity", process.configuration()),
                ("gas_lane", aerosol.gas.name.index(process.gas_water)),
                ("kind", kind),
                ("capacity", self.coefficient_capacity),
                ("tile", self.particle_tile),
            ),
            capabilities=ProcessCapabilities(resident=True, checkpoint=True),
        )

    def prepare(self, plan, context):
        """Retain an optical workspace, then stage and commit on its stream."""
        meta, flux, _, water, constants = plan.values["identity"]
        gas_water = plan.values["gas_lane"]
        device = context.device
        array = context.arrays
        particles = WarpParticleData()
        for key in ("masses", "concentration", "volume", "density"):
            setattr(particles, key, array["particles." + key])
        particles.distribution_kind = wp.int32(plan.values["kind"])
        boxes, slots = particles.concentration.shape
        optics = _optical_config(meta)
        waves = len(optics.wavelength)
        cases = ParticleOpticalCases(
            particles,
            optics,
            device=device,
            stream=context.stream,
        )
        mie = prepare_mie(
            cases.index.flatten(),
            cases.size.flatten(),
            coefficient_capacity=plan.values["capacity"],
            case_tile=plan.values["tile"],
            stream=context.stream,
        )
        irradiance = wp.array(flux, dtype=wp.float64, device=device)
        staged = wp.empty((boxes, slots, 8), dtype=wp.float64, device=device)
        box_stage = wp.empty((boxes, 6), dtype=wp.float64, device=device)
        settings = _Settings()
        for key, value in zip(
            (
                "kappa",
                "conductivity",
                "diffusivity",
                "cp_water",
                "cp_solute",
                "latent",
                "emissivity",
                "radiation_temperature",
            ),
            constants,
            strict=True,
        ):
            setattr(settings, key, wp.float64(value))
        ledger = _BoxEnergy()
        for key in (
            "absorbed",
            "conduction",
            "radiation",
            "latent",
            "sensible",
        ):
            setattr(ledger, key, array[f"{plan.name}.{key}_energy"])
        gas = array["gas.concentration"]
        temp = array[f"{plan.name}.temperature"]
        dt = wp.float64(context.time_step)
        fail = context.failure
        particle_inputs = (
            particles,
            gas,
            array["gas.molar_mass"],
            array["environment.temperature"],
            temp,
            settings,
            water,
            gas_water,
            dt,
            irradiance,
            cases.radius,
            mie.efficiencies,
            mie.status,
            staged,
            fail,
        )
        box_inputs = (
            particles,
            gas,
            ledger,
            cases.weight,
            waves,
            water,
            gas_water,
            staged,
            box_stage,
            fail,
        )
        publish_inputs = (
            particles,
            gas,
            temp,
            array[f"{plan.name}.absorbed_power"],
            ledger,
            water,
            gas_water,
            staged,
            box_stage,
            fail,
        )

        def enqueue():
            cases.enqueue()
            mie.enqueue()
            wp.launch(
                _stage_particle,
                (boxes, slots),
                inputs=particle_inputs,
                device=device,
                stream=context.stream,
            )
            wp.launch(
                _stage_box,
                boxes,
                inputs=box_inputs,
                device=device,
                stream=context.stream,
            )
            wp.launch(
                _publish,
                (boxes, slots),
                inputs=publish_inputs,
                device=device,
                stream=context.stream,
            )

        return PreparedOperation(
            plan.name,
            enqueue,
            (cases, mie, *particle_inputs, *box_inputs, *publish_inputs),
            plan.capabilities,
        )
