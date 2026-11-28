"""Optically thin heating and one-way water evaporation of wet spheres."""

from dataclasses import astuple, dataclass
from json import dumps

import numpy as np
from scipy.constants import Stefan_Boltzmann

from particula.box_process import BoxProcess
from particula.extension_fields import FieldSpec
from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.optics.state import OpticalConfig, particle_optics
from particula.particles._population import slot_density
from particula.util.constants import GAS_CONSTANT

_HEAT_CHANNELS = ("absorbed", "conduction", "radiation", "latent", "sensible")
_MIN_K = 273.15
_MAX_K = 373.15
_WATER = WaterBuckStrategy()


@dataclass(frozen=True)
class HeatingPhysics:
    """Prescribed continuum transport and liquid-solution properties.

    Conductivity is W/(m K), water diffusivity is m²/s, specific heat
    capacities are J/(kg K), latent heat is J/kg, and radiation_temperature
    is K. Kappa and grey infrared emissivity are dimensionless.
    """

    kappa: float = 0.3
    conductivity: float = 0.026
    diffusivity: float = 2.5e-5
    water_heat_capacity: float = 4180.0
    solute_heat_capacity: float = 1500.0
    latent_heat: float = 2.44e6
    emissivity: float = 0.95
    radiation_temperature: float = 298.15

    def __post_init__(self):
        """Validate physical constants before evaluating a temperature root."""
        if (
            not all(np.isfinite(v) for v in astuple(self))
            or min(
                self.conductivity,
                self.diffusivity,
                self.water_heat_capacity,
                self.solute_heat_capacity,
                self.latent_heat,
            )
            <= 0
            or self.kappa < 0
            or not 0 <= self.emissivity <= 1
            or not _MIN_K <= self.radiation_temperature <= _MAX_K
        ):
            raise ValueError(
                "Invalid optical heating material or transport data"
            )


def heating_step(
    radius,
    water,
    solute,
    activity,
    absorbed,
    old_temperature,
    ambient,
    vapor,
    water_molar_mass,
    dt,
    physics,
):
    """Solve one frozen-radius energy balance with conservative mass loss.

    Returns new temperature [K], water transferred to gas [kg per particle],
    and one energy amount [J per particle] for each heating ledger.
    """
    heat_capacity = (
        water * physics.water_heat_capacity
        + solute * physics.solute_heat_capacity
    )
    conductance = 4 * np.pi * radius * physics.conductivity
    emission_factor = (
        4 * np.pi * radius**2 * physics.emissivity * Stefan_Boltzmann
    )
    transfer_factor = 4 * np.pi * radius * physics.diffusivity * dt

    def terms(temperature):
        conduction = dt * conductance * (temperature - ambient)
        radiation = (
            dt
            * emission_factor
            * (temperature**4 - physics.radiation_temperature**4)
        )
        saturated = (
            _WATER.pure_vapor_pressure(temperature)
            * water_molar_mass
            / (GAS_CONSTANT * temperature)
        )
        loss = np.minimum(
            water, transfer_factor * np.maximum(activity * saturated - vapor, 0)
        )
        loss = water - (water - loss)
        latent = physics.latent_heat * loss
        sensible = heat_capacity * (temperature - old_temperature)
        return loss, conduction, radiation, latent, sensible

    incident = absorbed * dt

    def residual(temperature):
        return sum(terms(temperature)[1:]) - incident

    if np.any(residual(_MIN_K) > 0) or np.any(residual(_MAX_K) < 0):
        raise ValueError(
            "Optical heating temperature left the liquid model domain"
        )
    low = np.full_like(radius, _MIN_K)
    high = np.full_like(radius, _MAX_K)
    for _ in range(54):
        midpoint = (low + high) / 2
        cold = residual(midpoint) < 0
        low = np.where(cold, midpoint, low)
        high = np.where(cold, high, midpoint)
    new_temperature = (low + high) / 2
    loss, *energy_terms = terms(new_temperature)
    return new_temperature, loss, (incident, *energy_terms)


class OpticalHeating(BoxProcess):
    """Heat a particle-resolved water/solute population with irradiance.

    The irradiances are band-integrated W/m², one value per configured
    wavelength. Absorption is recomputed from live wet sizes and composition
    at the start of the process. Use attach_fields() before preparing the
    sequence. Each box exchanges heat with its fixed-temperature gas and a
    grey radiation bath. Evaporated water enters the named gas species.

    The liquid, spherical continuum model neglects curvature in the vapor
    balance. Its transport coefficients are prescribed. Complete optical
    heat exchange and mass loss are implicit in particle temperature.
    Geometry, irradiance, heat capacity and surrounding gas are evaluated
    at the beginning of each cycle.
    """

    _namespace = "optical_heating"

    def __init__(
        self,
        optics,
        irradiance,
        *,
        gas_water,
        water_material=0,
        physics=None,
        name="optical_heating",
    ):
        """Bind optical forcing, liquid material index and gas species name."""
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Optical heating needs an identifier-style name")
        self.optics = optics
        self.physics = HeatingPhysics() if physics is None else physics
        if not isinstance(optics, OpticalConfig) or not isinstance(
            self.physics, HeatingPhysics
        ):
            raise TypeError("Expected OpticalConfig and HeatingPhysics")
        if np.iscomplexobj(irradiance):
            raise ValueError("Irradiance must be real")
        flux = np.asarray(irradiance, dtype=np.float64)
        if flux.shape != (len(optics.wavelength),) or np.any(
            ~np.isfinite(flux) | (flux < 0)
        ):
            raise ValueError(
                "Provide one finite nonnegative irradiance per wavelength"
            )
        if type(water_material) is not int or water_material not in (0, 1):
            raise ValueError("Water material index must be 0 or 1")
        if not isinstance(gas_water, str) or not gas_water:
            raise ValueError("Provide the native gas species name for water")
        self.irradiance = tuple(float(value) for value in flux)
        self.gas_water = gas_water
        self.water_material = water_material
        self.temperature_field = f"{name}.temperature"
        self.power_field = f"{name}.absorbed_power"
        self.energy_fields = tuple(
            f"{name}.{key}_energy" for key in _HEAT_CHANNELS
        )
        super().__init__(
            name=name,
            extension_id="particula.optical_heating",
            reads=(
                "particles.masses",
                "particles.density",
                "particles.population",
                "particles.volume",
                "gas.concentration",
                "gas.molar_mass",
                "environment.temperature",
                self.temperature_field,
                *self.energy_fields,
            ),
            writes=(
                "particles.masses",
                "gas.concentration",
                self.temperature_field,
                self.power_field,
                *self.energy_fields,
            ),
            representations=("particle_resolved",),
            parameters=self.configuration(),
        )

    def configuration(self):
        """Return the parameters retained in exact process identity."""
        return (
            dumps(self.optics.metadata(), sort_keys=True),
            self.irradiance,
            self.gas_water,
            self.water_material,
            tuple(float(value) for value in astuple(self.physics)),
        )

    def _specs(self):
        return (
            FieldSpec(
                self.temperature_field, ("box", "particle"), "K", minimum=0
            ),
            FieldSpec(self.power_field, ("box", "particle"), "W", minimum=0),
            *(FieldSpec(key, ("box",), "J/m^3") for key in self.energy_fields),
        )

    def _validate_representation(self, aerosol):
        if aerosol.particles.distribution_type not in self.representations:
            raise ValueError(
                "Optical heating requires particle_resolved storage"
            )

    def attach_fields(self, aerosol):
        """Register diagnostic and thermal state on the initial aerosol."""
        self._validate_representation(aerosol)
        specs = self._specs()
        if any(spec.name in aerosol.fields.specs for spec in specs):
            raise ValueError("An optical heating field is already registered")
        for spec in specs:
            aerosol.fields.add(spec)

    def validate(self, aerosol):
        """Require resolved slots, two materials and mapped vapor."""
        self._validate_representation(aerosol)
        if (
            aerosol.particles.n_species != 2
            or self.gas_water not in aerosol.gas.name
        ):
            raise ValueError(
                "Require water, one nonvolatile solute and named gas water"
            )
        gas_index = aerosol.gas.name.index(self.gas_water)
        if not aerosol.gas.partitioning[gas_index]:
            raise ValueError(
                "Water vapor must be enabled for phase partitioning"
            )
        self.optics.validate_materials(2)
        for spec in self._specs():
            if aerosol.fields.specs.get(spec.name) != spec:
                raise ValueError(
                    "Attach optical heating fields before executing"
                )
        ambient = aerosol.environment.temperature
        if np.any((ambient < _MIN_K) | (ambient > _MAX_K)):
            raise ValueError("Gas temperature outside the liquid domain")
        temperature = aerosol.fields[self.temperature_field]
        active = aerosol.particles.population > 0
        outside = (temperature != 0) & (
            (temperature < _MIN_K) | (temperature > _MAX_K)
        )
        if np.any(active & outside):
            raise ValueError("Particle temperature outside the liquid domain")
        i = self.water_material
        if np.any(
            (aerosol.particles.population > 0)
            & (aerosol.particles.masses[..., 1 - i] <= 0)
        ):
            raise ValueError(
                "Each heated active slot needs the nonvolatile solute"
            )

    def advance(self, aerosol, *, time, time_step, context):
        """Advance one box and add represented water and energy inventories."""
        particles = aerosol.particles
        active = particles.population[0] > 0
        if not np.any(active):
            return
        _, cross = particle_optics(particles, self.optics)
        absorbed = np.asarray(self.irradiance) @ cross.absorption[0][:, active]
        water_index = self.water_material
        gas_index = aerosol.gas.name.index(self.gas_water)
        water = particles.masses[0, active, water_index]
        solute = particles.masses[0, active, 1 - water_index]
        water_volume = water / particles.density[water_index]
        solute_volume = solute / particles.density[1 - water_index]
        activity = np.where(
            water > 0,
            water_volume
            / np.where(
                water > 0,
                water_volume + self.physics.kappa * solute_volume,
                1,
            ),
            0,
        )
        env = aerosol.environment.temperature[0]
        temp = aerosol.fields[self.temperature_field]
        old = np.where(temp[0, active] > 0, temp[0, active], env)
        new, loss, energy = heating_step(
            particles.radii[0, active],
            water,
            solute,
            activity,
            absorbed,
            old,
            env,
            aerosol.gas.concentration[0, gas_index],
            aerosol.gas.molar_mass[gas_index],
            time_step,
            self.physics,
        )
        weight = slot_density(particles)[0, active]
        remaining = water - loss
        particles.masses[0, active, water_index] = remaining
        aerosol.gas.concentration[0, gas_index] += weight @ (water - remaining)
        temp[0, active] = new
        aerosol.fields[self.power_field][0, active] = absorbed
        for key, values in zip(self.energy_fields, energy, strict=True):
            aerosol.fields[key][0] += weight @ values
