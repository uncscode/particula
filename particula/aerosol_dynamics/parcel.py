"""
Class for creating a air parcel to put particles and gases in.
"""
import numpy as np
from particula.aerosol_dynamics import particle, u


class Parcel:
    """
    Sets the class for creating a parcel.
    This starts the base for particle and gas dynamic simulations.
    Like the eveolution of a size distribution with time.
    Or change in temperature causing particle evaporation or gas condensation.

    Attributes:
        name            (str)               The name of the parcel.
        particle_data   (object list)       Particle objects.
        enviroment      (object)            The enviroment of the parcel.
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
        self, name: str, radius, density=1e3 * u.kg / u.m**3,
        charge=0 * u.dimensionless
    ):
        """creats and then Adds a particle to the parcel.
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
        self, name_of_particles, radius_of_particles,
        density_of_particles=None, charge_of_particles=None
    ):
        """Adds a list of particles to the parcel based on size only
            or delcaring denisty and charge.
        Parameters:
            name_of_particles   (str)   [no units],
            radius_of_particles (list)  [m]
        Optional:
            density_of_particles (list) [kg/m**3],
            charge_of_particles  (list) [dimensionless]
        """
        if density_of_particles is None:
            for size_add in radius_of_particles:
                self.create_and_add_particle(name_of_particles, size_add)
        else:
            for size_add, density_add, charge_add in zip(
                    radius_of_particles, density_of_particles,
                    charge_of_particles
                    ):
                self.create_and_add_particle(
                    name_of_particles, size_add, density_add, charge_add
                )

    def remove_particle(self, particle_index):
        """Removes a particle from the parcel.
        Parameters:
            particle_index (obj): int or array of ints
            Indicate indices of sub-arrays to remove along the specified axis.
        """
        self._particle_data = np.delete(self._particle_data, particle_index)

    def remove_all_particles(self):
        """Removes all particles from the parcel."""
        self._particle_data = np.array([])

    def particle_classes(self) -> list:
        """Returns the particle data of the parcel."""
        return self._particle_data

    def particle_mass(self) -> float:
        """Returns the mass of the particle. Checks units. [kg]"""
        return [i.mass() for i in self.particle_classes()]

    def particle_radius(self) -> float:
        """Returns list of radi of particles"""
        return [i.radius() for i in self.particle_classes()]

    def particle_density(self) -> float:
        """Returns list of densities of particles."""
        return [i.density() for i in self.particle_classes()]

    def particle_charge(self) -> float:
        """Returns list of charges of particles."""
        return [i.charge() for i in self.particle_classes()]

    def particle_knudsen_number(self) -> float:
        """Returns list of knudsen numbers of particles."""
        return [
            i.knudsen_number(self._enviroment) for i in self.particle_classes()
        ]
