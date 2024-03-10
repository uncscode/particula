"""Aerosol class definition."""

from typing import Union
from particula.base.gas import Gas
from particula.base.particle import Particle


class Aerosol:
    """
    A class that acts as a facade for interacting with a single Gas and a
    single Particle object. It dynamically attaches methods and properties of
    Gas and Particle objects to itself.
    """

    def __init__(self, gas: Gas, particle: Particle):
        """
        Initializes an Aerosol instance with a single Gas and Particle.
        Dynamically attaches methods and properties of these objects to the
        Aerosol instance.

        Parameters:
        - gas (Gas): A single Gas instance.
        - particle (Particle): A single Particle instance.
        """
        self.replace_gas(gas)
        self.replace_particle(particle)

    def attach_methods_and_properties(
            self,
            obj_to_attach: Union[Gas, Particle],
            prefix: str):
        """
        Dynamically attaches methods and properties from the given object to
        the Aerosol instance, prefixing them to distinguish between Gas and
        Particle attributes.

        Parameters:
        - obj: The object whose methods and properties are to be attached.
        - prefix (str): The prefix to apply to the attached attributes.
        """
        for attr_name in dir(obj_to_attach):
            if not attr_name.startswith("__"):
                attribute = getattr(obj_to_attach, attr_name)
                # Prefix the attribute name to distinguish between Gas and
                # Particle attributes
                prefixed_attr_name = f"{prefix}_{attr_name}"
                setattr(self, prefixed_attr_name, attribute)

    def replace_gas(self, gas: Gas):
        """
        Replaces the current Gas instance with a new one and dynamically
        attaches its methods and properties.

        Parameters:
        - gas (Gas): A new Gas instance to replace the current one.
        """
        self.gas = gas
        self.attach_methods_and_properties(gas, 'gas')

    def replace_particle(self, particle: Particle):
        """
        Replaces the current Particle instance with a new one and dynamically
        attaches its methods and properties.

        Parameters:
        - particle (Particle): A new Particle instance to replace the current
        one.
        """
        self.particle = particle
        self.attach_methods_and_properties(particle, 'particle')
