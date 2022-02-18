"""Tracking an array of particles.

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


class Particle_distribution:
    """Class to instantiate particle distributions.

    This class represents the underlying framework for
    storing particle distribution data

    Attributes:

        radii       (np array)     [m]
        density     (np array)     [kg/m**3]
        charge      (np array)     [dimensionless]
        mass        (np array)     [kg]
        name        (str)          [no units]

    """

    def __init__(
        self,
        radii,
        density,
        charge,
        number,
        name: str = 'Distribution',
    ):
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
        self._number = number
        self._mass = density * (4*np.pi/3) * (radii**3)


    def name(self) -> str:
        """Returns the name of the distribution.
        """
        return self._name

    def masses(self) -> float:
        """Returns total mass of particles of that radii.

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
        """Returns number of charges on each particle.

        units: [dimensionless]
        """
        return self._charge
    
    def number(self) -> int:
        """Returns number of particles.

        units: [dimensionless]
        """
        return self._number

    def number_concentration(self) -> int:
        """" Returns the number of distribution of particles.
        
        units: [#/m**3]
        """
        return np.sum(self._number)

    def mass_concentration(self) -> int:
        """" Returns the mass of the distribution.
        
        units: [kg/m**3]
        """
        return np.sum(self.masses())

    def rasterization(self, bins):
        """ Returns binned radii and number using numpy histogram methods

        Parameters:

        bins : int or sequence of scalars or str, optional
            If `bins` is an int, it defines the number of equal-width
            bins in the given range (10, by default). If `bins` is a
            sequence, it defines the bin edges, including the rightmost
            edge, allowing for non-uniform bin widths.

            If `bins` is a string from the list below, `histogram_bin_edges`
            will use the method chosen to calculate the optimal bin width and
            consequently the number of bins (see `Notes` for more detail on
            the estimators) from the data that falls within the requested
            range. While the bin width will be optimal for the actual data
            in the range, the number of bins will be computed to fill the
            entire range, including the empty portions. For visualisation,
            using the 'auto' option is suggested.

            'auto'
                Maximum of the 'sturges' and 'fd' estimators. Provides good
                all around performance.

            'fd' (Freedman Diaconis Estimator)
                Robust (resilient to outliers) estimator that takes into
                account data variability and data size.

            'doane'
                An improved version of Sturges' estimator that works better
                with non-normal datasets.

            'scott'
                Less robust estimator that that takes into account data
                variability and data size.

            'stone'
                Estimator based on leave-one-out cross-validation estimate of
                the integrated squared error. Can be regarded as a generalization
                of Scott's rule.

            'rice'
                Estimator does not take variability into account, only data
                size. Commonly overestimates number of bins required.

            'sturges'
                R's default method, only accounts for data size. Only
                optimal for gaussian data and underestimates number of bins
                for large non-gaussian datasets.

            'sqrt'
                Square root (of data size) estimator, used by Excel and
                other programs for its speed and simplicity.

        returns: 
            particle_number     array of dtype float
            radii               array of dtype float
        """

        # histogram method to rasterize the radii and 
        # tracking the number concentration
        bin_edges = np.histogram_bin_edges(self.radii(), bins=bins)
        particle_number, bin_edges = np.histogram(
            self.radii(),
            bins=bin_edges,
            weights = self.number()
        )

        # calculates bin centers and applies it as the radii
        radii = np.diff(bin_edges)+bin_edges[0:-1] 

        # drop the zero concentration bins, in radii and number
        non_zeros = particle_number>0
        particle_number = particle_number[non_zeros]
        radii = radii[non_zeros]

        return particle_number, radii

    def update_distribution(self, rasterized_radii, rasterized_number):
        """" updates the particle distribution properties, based on a 
        new rasterization. 
        
        Conserves total particle mass.
        Does not conserve number.
        Assumes same density and charge.

        This could be built out to have different options.

        Parameters:

            rasterized_radii:       array of dtype float
            rasterized_number:      array of dtype float
        """
        old_mass = self.mass_concentration()

        self._radii = rasterized_radii
        self._density = np.ones(len(rasterized_radii))*self._density[0]
        self._charge = np.ones(len(rasterized_radii))*self._density[0]
        self._number = rasterized_number
        self._mass = self._density * (4*np.pi/3) * (self._radii**3)

        # adjustment to number so mass is conserved
        new_mass = self.mass_concentration()
        self._number = rasterized_number * old_mass/new_mass

        return
    
    def rasterization_and_update(self, bins):
        """ Bins the radii and number using numpy histogram methods, and updates
        the class data

        Parameters:

        bins : int or sequence of scalars or str, optional
            If `bins` is an int, it defines the number of equal-width
            bins in the given range (10, by default). If `bins` is a
            sequence, it defines the bin edges, including the rightmost
            edge, allowing for non-uniform bin widths.

            If `bins` is a string from the list below, `histogram_bin_edges`
            will use the method chosen to calculate the optimal bin width and
            consequently the number of bins (see `Notes` for more detail on
            the estimators) from the data that falls within the requested
            range. While the bin width will be optimal for the actual data
            in the range, the number of bins will be computed to fill the
            entire range, including the empty portions. For visualisation,
            using the 'auto' option is suggested.

            'auto'
                Maximum of the 'sturges' and 'fd' estimators. Provides good
                all around performance.

            'fd' (Freedman Diaconis Estimator)
                Robust (resilient to outliers) estimator that takes into
                account data variability and data size.

            'doane'
                An improved version of Sturges' estimator that works better
                with non-normal datasets.

            'scott'
                Less robust estimator that that takes into account data
                variability and data size.

            'stone'
                Estimator based on leave-one-out cross-validation estimate of
                the integrated squared error. Can be regarded as a generalization
                of Scott's rule.

            'rice'
                Estimator does not take variability into account, only data
                size. Commonly overestimates number of bins required.

            'sturges'
                R's default method, only accounts for data size. Only
                optimal for gaussian data and underestimates number of bins
                for large non-gaussian datasets.

            'sqrt'
                Square root (of data size) estimator, used by Excel and
                other programs for its speed and simplicity.
        """

        particle_number, radii = self.rasterization(bins)
        self.update_distribution(radii, particle_number)

        return
