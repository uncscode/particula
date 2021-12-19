"""
Class for creating a air parcel to put particles and gases in.
"""
from aerosol_dynamics import particle
from aerosol_dynamics import environment

class Parcel:
    """Sets the class for creating a parcel.
    This starts the base for particle and gas dynamic simulations.
    Like the eveolution of a size distribution with time.
    Or a change in temperature causing particle evaporation or gas condensation.

    Attributes:
        name (str): The name of the parcel.
        TODO: after enviromnetal class is added to main.
        temperature (float) default: The air temperature in Kelvin.
        pressure (float) default: The air pressure in atm?.
        add gases class
    """

    def __init__(self, name: str, temperature = 300, pressure = 101325):
        """
        Constructs the parcel object.

        Parameters:
            name (str): The name of the particle.
            particle_data (list): The radius of the particle.
            temperature (float), default = 300 K: The air temperature in Kelvin.
            pressure (float), default = 101325 Pa: The air pressure in Pascals.
        """
        self._name = name
        self._particle_data = []

        self._enviroment = environment.Environment(
                                temperature,
                                pressure,
                            )

    def name(self) -> str:
        """Returns the name of the parcel."""
        return self._name

    def add_particle(self, particle_object):
        """Adds a particle to the parcel.
        Parameters:
            particle_data (list): The radius of the particle.
        """
        self._particle_data.append(particle_object)

    def create_and_add_particle(self, name: str, radius, density = 1e3, charge = 0):
        """creats and then Adds a particle to the parcel.
        Parameters:
            particle_data (list): The radius of the particle.
        """
        self._particle_data.append(particle.Particle(name, radius, density, charge))

    def create_and_add_list_of_particle(self, name_of_particles, radius_of_particles):
        """Adds a list of particles to the parcel based on size only.
        Parameters:
            radius_of_particles (list): The radius of the particle.
            TODO: add charge and mass as passible parameters.
        """
        for size_add in radius_of_particles:
            self.create_and_add_particle(name_of_particles, size_add)

    def remove_particle(self, particle_data):
        """Removes a particle from the parcel.
        Parameters:
            particle_data (list): The radius of the particle.
        """
        self._particle_data.remove(particle_data)

    def remove_all_particles(self):
        """Removes all particles from the parcel."""
        self._particle_data = []

    def particle_classes(self) -> list:
        """Returns the particle data of the parcel."""
        return self._particle_data

    def particle_mass(self) -> float:
        """Returns the mass of the particle. Checks units. [kg]"""
        return [i.mass() for i in self.particle_classes()]

    def particle_radius(self) -> float:
        """Returns the radius of the particle. Checks units. [m]"""
        return [i.radius() for i in self.particle_classes()]

    def particle_charge(self) -> float:
        """Returns the charge of the particle. Checks units. [unitless]"""
        return [i.charge() for i in self.particle_classes()]

    def knudsen_number_particle(self) -> float:
        """Returns the knudsen number of the particle. Checks units. [unitless]"""
        return [i.knudsen_number(self._enviroment) for i in self.particle_classes()]
