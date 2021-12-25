"""class for creating a air parcel to put particles and gases in.
"""

import numpy as np
from particula import u
from particula.aerosol_dynamics import particle


class Parcel:
    """
    Creates an air parcel to track and interact with a list of particle objects.
    This starts the base for particle and gas dynamic simulations.
    For example, eveolution of a size distribution with time.

    Attributes:
        name            (str)               The name of the parcel.
        particle_data   (particle list)     Particle objects.
        enviroment      (object)            Parcel's enviroment.
    """

    def __init__(self, name: str, parcel_environment):
        """
        Constructs the parcel object.

        Parameters:
            name           (str),
            particle_data  (object numpy array),
            enviroment     (object).
        """
        self._name = name
        self._particle_data = np.array([])
        self._enviroment = parcel_environment

    def name(self) -> str:
        """Returns the name of the parcel."""
        return self._name

    def add_particle(self, particle_object):
        """Adds a particle to the parcel.
        Parameters:
            particle_object (object)
        """
        self._particle_data = np.append(self._particle_data, particle_object)

    def create_and_add_particle(
        self, name: str,
        radius,
        density=1e3 * u.kg / u.m**3,
        charge=0 * u.dimensionless
    ):
        """Create and then add a particle to the parcel.

        Parameters:
            name    (str)   [no units],
            radius  (float) [m]
        Optional:
            density (float) [kg/m**3]       default = 1e3 [kg/m**3],
            charge  (int)   [dimensionless] default = 0 [dimensionless]
        """

        try:  # maybe move to particle class
            radius.check('m')
        except AttributeError:
            print('Please add units to radius. E.g., radius = 150 * u.nm]')

        self.add_particle(
            particle.Particle(name, radius, density, charge)
        )

    def create_and_add_list_of_particle(
        self,
        name_of_particles,
        radi_of_particles,
        densities_of_particles=None,
        charges_of_particles=None
    ):
        """Adds a list of particles to the parcel based on size only (simple).
            or delcaring denisties and charges (detailed).
        Parameters:
            name_of_particles   (str)   [no units],
            radi_of_particles (list)  [m]
        Optional:
            densities_of_particles (list) [kg/m**3],
            charges_of_particles  (list) [dimensionless]
        Examples:
            simple: parcel.create_and_add_list_of_particle(
                'org2',
                [1e-9, 2e-9, 3e-9] * u.m
                )
            detailed: create_and_add_list_of_particle(
                'org3',
                [1e-9, 2e-9, 3e-9] * u.m,
                [1.8, 1, 1] * u.kg / u.m ** 3,
                [1, 0, 2] * u.dimensionless)
        """
        if densities_of_particles is None:
            for size_add in radi_of_particles:
                self.create_and_add_particle(name_of_particles, size_add)
        else:
            for size_add, density_add, charge_add in zip(
                radi_of_particles,
                densities_of_particles,
                charges_of_particles
            ):
                self.create_and_add_particle(
                    name_of_particles, size_add, density_add, charge_add
                )

    def remove_particle(self, particle_index):
        """Removes a particle from the parcel.
        Parameters:
            particle_index (list): int or array of ints
        """
        self._particle_data = np.delete(self._particle_data, particle_index)

    def remove_all_particles(self):
        """Removes all particles from the parcel."""
        self._particle_data = np.array([])

    def particle_classes_list(self) -> list:
        """Returns the particle data of the parcel."""
        return self._particle_data

    def particle_masses_list(self) -> float:
        """Returns the mass of the particle. Checks units. [kg]"""
        return [i.mass() for i in self.particle_classes_list()]

    def particle_radi_list(self) -> float:
        """Returns list of radi of particles"""
        return [i.radius() for i in self.particle_classes_list()]

    def particle_densities_list(self) -> float:
        """Returns list of densities of particles."""
        return [i.density() for i in self.particle_classes_list()]

    def particle_charges_list(self) -> float:
        """Returns list of charges of particles."""
        return [i.charge() for i in self.particle_classes_list()]

    def particle_knudsen_numbers_list(self) -> float:
        """Returns list of knudsen numbers of particles."""
        return [
            i.knudsen_number(self._enviroment) for i in self.particle_classes_list()
        ]
