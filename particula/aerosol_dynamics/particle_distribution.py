""" Tracking an array of particles.

    This module contains the Particle class, which is used to
    instantiate an array (distributions) of particles and calculate their 
    base properties. Particles distributions are introduced and defined by 
    calling Particle class, for example:

    >>> from particula import Particle
    >>> p1 = Particle(name='my_particle', radii=1e-9, density=1e3, charge=1)

    Then, it is possible to return the properties of the particle p1:

    >>> p1.mass()

    The environment is defined by the following parameters:
    >>> from particula.aerosol_dynamics import environment
    >>> env = environment.Environment(temperature=300, pressure=1e5)

    If another particle is introduced, it is possible to calculate
    the binary coagulation coefficient:

    >>> p2 = Particle(name='my_particle2', radii=1e-9, density=1e3, charge=1)
    >>> p1.dimensioned_coagulation_kernel(p2, env)

    For more details, see below. More information to follow.
"""

import numpy as np



class Particle_Distribution:
    """Class to instantiate particle distributions.

    This class represents the underlying framework for both
    particle--particle and gas--particle interactions. See detailed
    methods and functions below.

    Attributes:

        radii       (np array)     [m]
        density     (np array)     [kg/m**3]
        charge      (np array)     [dimensionless]
        mass        (np array)     [kg]
        name        (str)          [no units]

    """

    def __init__(self, radii, density, charge, number, name: str='Distribution'):
        """Constructs particle objects.

        Parameters:

            radii       (np array)     [m]
            density     (np array)     [kg/m**3]
            charge      (np array)     [dimensionless]
            number      (np array)     [#/m**3]
            name        (str)          [no units]       default = Distribution
        """

        self._name = name
        self._radii = radii
        self._density = density
        self._charge = charge
        self._mass = density * (4*np.pi/3) * (radii**3)
        self._number = number

    def name(self) -> str:
        """Returns the name of particle.
        """

        return self._name

    def masses(self) -> float:
        """Returns mass of particles of that size.

        units: [kg]
        """

        return self._mass * self._number

    def radii(self) -> float:
        """Returns radii of particle.

        units: [m]
        """

        return self._radii

    def densities(self) -> int:
        """Returns density of particle.

        units: [kg/m**3]
        """

        return self._density

    def charges(self) -> int:
        """Returns number of charges on particle.

        units: [dimensionless]
        """

        return self._charge
    
    def number(self) -> int:
        """Returns number of charges on particle.

        units: [dimensionless]
        """

        return self._number

    def number_concentration(self) -> int:
        """" Returns the number of distribution of particles.
        units: [#/m**3]
        """

        return np.sum(self._number)

    def mass_concentration(self) -> int:
        """" Returns the number of distribution of particles.
        units: [kg/m**3]
        """

        return np.sum(self.masses())