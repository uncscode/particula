"""Native discrete and radius-PDF condensation regression contracts."""

import copy
import unittest

import numpy as np
import particula as par
import pytest
from particula.dynamics.condensation import (
    CondensationIsothermal,
    CondensationLatentHeat,
)
from particula.dynamics.condensation.mass_transfer import (
    get_latent_heat_energy_released,
    get_mass_transfer,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas.gas_data import GasData
from particula.gas.latent_heat_strategies import ConstantLatentHeat
from particula.particles.particle_data import ParticleData


@pytest.mark.usefixtures("forbid_legacy_facades")
class TestNativeBinnedCondensation(unittest.TestCase):
    """Retain binned science with explicit distribution meaning and physics."""

    def setUp(self):
        """Set the original shared environmental and transport parameters."""
        self.temperature = 298.15
        self.pressure = 101325
        self.diffusion_coefficient = 2e-5
        self.accommodation_coefficient = 1.0

    def _build_binned_data(
        self,
        distribution_type: str,
        n_bins: int,
        densities: np.ndarray | None = None,
    ) -> tuple[ParticleData, GasData, dict]:
        """Build a binned particle/gas fixture for discrete or PDF bins."""
        if densities is None:
            densities = np.array([1.0e3, 1.77e3, 1.2e3])

        molar_mass = np.array([18.015e-3, 132.14e-3, 150.0e-3])
        vp_water = par.gas.VaporPressureFactory().get_strategy("water_buck")
        vp_constant = par.gas.VaporPressureFactory().get_strategy(
            "constant", {"vapor_pressure": 1e-24, "vapor_pressure_units": "Pa"}
        )
        water_sat = vp_water.saturation_concentration(
            molar_mass[0], self.temperature
        )
        water_conc = water_sat * 1.02
        gas_species = (
            par.gas.GasDataBuilder()
            .set_names(["H2O", "NH4HSO4", "ORG"])
            .set_molar_mass(molar_mass, "kg/mol")
            .set_concentration(np.array([water_conc, 1e-30, 1e-30]), "kg/m^3")
            .set_partitioning([True, True, True])
            .build()
        )

        activity = (
            par.particles.ActivityKappaParameterBuilder()
            .set_density(densities, "kg/m^3")
            .set_kappa(np.array([0.0, 0.61, 0.2]))
            .set_molar_mass(molar_mass, "kg/mol")
            .set_water_index(0)
            .build()
        )
        surface = (
            par.particles.SurfaceStrategyVolumeBuilder()
            .set_density(densities, "kg/m^3")
            .set_surface_tension(np.array([0.072, 0.092, 0.08]), "N/m")
            .build()
        )

        radius_bins = np.logspace(-9, -6, n_bins)
        mode = np.array([100e-9])
        gsd = np.array([1.4])
        number_concentration = np.array([1e4]) * 1e6
        distribution = (
            par.particles.get_lognormal_pmf_distribution
            if distribution_type == "pmf"
            else par.particles.get_lognormal_pdf_distribution
        )
        concentration = distribution(
            x_values=radius_bins,
            mode=mode,
            geometric_standard_deviation=gsd,
            number_of_particles=number_concentration,
        )
        volume_bins = 4 / 3 * np.pi * radius_bins**3
        fractions = np.array([0.4, 0.3, 0.3])
        masses = volume_bins[:, np.newaxis] * densities * fractions
        particle = ParticleData(
            masses=masses[None, ...],
            density=densities,
            concentration=concentration[None, ...],
            charge=np.zeros((1, n_bins)),
            volume=np.array([1.0]),
            distribution_type=(
                "discrete" if distribution_type == "pmf" else "continuous_pdf"
            ),
            radius_grid=radius_bins if distribution_type == "pdf" else None,
        )
        physics = dict(
            activity_strategy=activity,
            surface_strategy=surface,
            vapor_pressure_strategy=[vp_water, vp_constant, vp_constant],
            species_map=CondensationSpeciesMap(
                tuple(gas_species.name), ((0, 0), (1, 1), (2, 2))
            ),
        )
        return particle, gas_species, physics

    def _species_inventory(
        self, particle_data: ParticleData, gas_data: GasData
    ) -> np.ndarray:
        """Independently integrate each species using the declared measure."""
        masses = particle_data.masses[0]
        concentration = particle_data.concentration[0]
        weighted = masses * concentration[:, np.newaxis]
        if particle_data.distribution_type == "continuous_pdf":
            particle_mass = np.trapezoid(
                weighted, x=particle_data.radius_grid, axis=0
            )
        else:
            particle_mass = np.sum(weighted, axis=0)
        return (
            particle_mass + gas_data.concentration[0]
        ) * particle_data.volume[0]

    def test_step_discrete_10_bins_mass_conservation(self):
        """Discrete binned step conserves mass for 10 bins."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 10)
        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        initial_species = self._species_inventory(particle_data, gas_data)
        initial_mass = initial_species.sum()
        particle_new, gas_new = cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        final_species = self._species_inventory(particle_new, gas_new)
        final_mass = final_species.sum()
        np.testing.assert_allclose(
            final_species, initial_species, rtol=1e-12, atol=1e-30
        )
        np.testing.assert_allclose(final_mass, initial_mass, rtol=1e-12)

    def test_step_discrete_50_bins_mass_conservation(self):
        """Discrete binned step conserves mass for 50 bins."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 50)
        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        initial_species = self._species_inventory(particle_data, gas_data)
        initial_mass = initial_species.sum()
        particle_new, gas_new = cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        final_species = self._species_inventory(particle_new, gas_new)
        final_mass = final_species.sum()
        np.testing.assert_allclose(
            final_species, initial_species, rtol=1e-12, atol=1e-30
        )
        np.testing.assert_allclose(final_mass, initial_mass, rtol=1e-12)

    def test_step_discrete_100_bins_mass_conservation(self):
        """Discrete binned step conserves mass for 100 bins."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 100)
        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        initial_species = self._species_inventory(particle_data, gas_data)
        initial_mass = initial_species.sum()
        particle_new, gas_new = cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        final_species = self._species_inventory(particle_new, gas_new)
        final_mass = final_species.sum()
        np.testing.assert_allclose(
            final_species, initial_species, rtol=1e-12, atol=1e-30
        )
        np.testing.assert_allclose(final_mass, initial_mass, rtol=1e-12)

    def test_step_continuous_distribution_mass_conservation(self):
        """Continuous binned step conserves total mass."""
        particle_data, gas_data, physics = self._build_binned_data("pdf", 50)
        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        initial_species = self._species_inventory(particle_data, gas_data)
        initial_mass = initial_species.sum()
        particle_new, gas_new = cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        final_species = self._species_inventory(particle_new, gas_new)
        final_mass = final_species.sum()
        np.testing.assert_allclose(
            final_species, initial_species, rtol=1e-12, atol=1e-30
        )
        np.testing.assert_allclose(final_mass, initial_mass, rtol=1e-12)

    def test_step_discrete_isothermal_parity(self):
        """Discrete binned step matches isothermal when latent heat is zero."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 50)
        latent_cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
            latent_heat=0.0,
        )
        iso_cond = CondensationIsothermal(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
        )

        latent_particle, latent_gas = latent_cond.step(
            particle=copy.deepcopy(particle_data),
            gas_species=copy.deepcopy(gas_data),
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        iso_particle, iso_gas = iso_cond.step(
            particle=copy.deepcopy(particle_data),
            gas_species=copy.deepcopy(gas_data),
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        np.testing.assert_allclose(
            latent_particle.masses[0], iso_particle.masses[0], rtol=1e-15
        )
        np.testing.assert_allclose(
            latent_gas.concentration[0], iso_gas.concentration[0], rtol=1e-15
        )

    def test_step_continuous_isothermal_parity(self):
        """Continuous bins match isothermal when latent heat is zero."""
        particle_data, gas_data, physics = self._build_binned_data("pdf", 50)
        latent_cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
            latent_heat=0.0,
        )
        iso_cond = CondensationIsothermal(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
        )

        latent_particle, latent_gas = latent_cond.step(
            particle=copy.deepcopy(particle_data),
            gas_species=copy.deepcopy(gas_data),
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        iso_particle, iso_gas = iso_cond.step(
            particle=copy.deepcopy(particle_data),
            gas_species=copy.deepcopy(gas_data),
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        np.testing.assert_allclose(
            latent_particle.masses[0],
            iso_particle.masses[0],
            rtol=1e-15,
            atol=1e-17,
        )
        np.testing.assert_allclose(
            latent_gas.concentration[0],
            iso_gas.concentration[0],
            rtol=1e-15,
            atol=1e-17,
        )

    def test_step_discrete_energy_tracking(self):
        """Latent heat energy tracking matches mass transfer for binned data."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 50)
        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        mass_rate = cond.mass_transfer_rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        volume = particle_data.volume[0]
        # Discrete concentration is already number density.
        norm_conc = particle_data.concentration[0]
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate,
            time_step=1.0,
            gas_mass=gas_data.concentration[0],
            particle_mass=particle_data.masses[0],
            particle_concentration=norm_conc,
        )
        latent_heat = cond._latent_heat_strategy.latent_heat(self.temperature)
        expected_energy = np.sum(
            get_latent_heat_energy_released(mass_transfer * volume, latent_heat)
        )

        cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        np.testing.assert_allclose(
            cond.last_latent_heat_energy, expected_energy, rtol=1e-14
        )

    def test_step_discrete_2d_mass_shapes(self):
        """Binned discrete step preserves 2D mass shapes."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 10)
        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
        )

        particle_new, _ = cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        self.assertEqual(
            particle_new.masses[0].shape,
            (
                particle_data.masses[0].shape[0],
                particle_data.masses[0].shape[1],
            ),
        )

    def test_rate_discrete_shape(self):
        """rate() returns binned shape for discrete and continuous inputs."""
        for distribution_type in ("pmf", "pdf"):
            with self.subTest(distribution_type=distribution_type):
                particle_data, gas_data, physics = self._build_binned_data(
                    distribution_type, 10
                )
                cond = CondensationLatentHeat(
                    molar_mass=gas_data.molar_mass,
                    diffusion_coefficient=self.diffusion_coefficient,
                    accommodation_coefficient=self.accommodation_coefficient,
                    **physics,
                )

                rate = cond.rate(
                    particle=particle_data,
                    gas_species=gas_data,
                    temperature=self.temperature,
                    pressure=self.pressure,
                )
                self.assertEqual(rate.shape, particle_data.masses[0].shape)

    def test_mass_transfer_rate_discrete_shape(self):
        """mass_transfer_rate returns binned shape for discrete/continuous."""
        for distribution_type in ("pmf", "pdf"):
            with self.subTest(distribution_type=distribution_type):
                particle_data, gas_data, physics = self._build_binned_data(
                    distribution_type, 10
                )
                cond = CondensationLatentHeat(
                    molar_mass=gas_data.molar_mass,
                    diffusion_coefficient=self.diffusion_coefficient,
                    accommodation_coefficient=self.accommodation_coefficient,
                    **physics,
                )

                mass_rate = cond.mass_transfer_rate(
                    particle=particle_data,
                    gas_species=gas_data,
                    temperature=self.temperature,
                    pressure=self.pressure,
                )
                self.assertEqual(mass_rate.shape, particle_data.masses[0].shape)

    def test_mass_transfer_rate_rejects_nonfinite_latent_heat(self):
        """mass_transfer_rate rejects non-finite latent heat values."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 10)

        class NonFiniteLatentHeat:
            """Latent heat strategy returning non-finite values."""

            def latent_heat(self, temperature: float) -> float:
                return float("nan")

        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            latent_heat_strategy=NonFiniteLatentHeat(),
            **physics,
        )

        masses_before = particle_data.masses.copy()
        particle_concentration_before = particle_data.concentration.copy()
        gas_before = gas_data.concentration.copy()
        with self.assertRaisesRegex(ValueError, "latent heat must be finite"):
            cond.mass_transfer_rate(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
            )

        np.testing.assert_array_equal(particle_data.masses, masses_before)
        np.testing.assert_array_equal(
            particle_data.concentration, particle_concentration_before
        )
        np.testing.assert_array_equal(gas_data.concentration, gas_before)

    def test_step_rejects_negative_concentration(self):
        """Binned step rejects negative concentrations."""
        particle_data, gas_data, physics = self._build_binned_data("pmf", 10)
        particle_data.concentration[0][0] = -1.0
        cond = CondensationLatentHeat(
            molar_mass=gas_data.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            **physics,
        )

        masses_before = particle_data.masses.copy()
        particle_concentration_before = particle_data.concentration.copy()
        gas_before = gas_data.concentration.copy()
        with self.assertRaisesRegex(
            ValueError, "particle data must have finite physical values"
        ):
            cond.step(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=1.0,
            )

        np.testing.assert_array_equal(particle_data.masses, masses_before)
        np.testing.assert_array_equal(
            particle_data.concentration, particle_concentration_before
        )
        np.testing.assert_array_equal(gas_data.concentration, gas_before)
