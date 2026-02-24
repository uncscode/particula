"""Coagulation strategy module.

Defines an abstract base class and supporting methods for particle
coagulation processes in aerosol simulations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from particula import gas, particles
from particula.dynamics.coagulation import (
    charged_dimensional_kernel,
    charged_kernel_strategy,
    coagulation_rate,
)
from particula.dynamics.coagulation.particle_resolved_step import (
    particle_resolved_method,
)
from particula.particles import change_particle_representation
from particula.particles.representation import ParticleRepresentation

logger = logging.getLogger("particula")


def _resolve_radius_indices(
    particle_radius: NDArray[np.float64],
    lookup_values: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Resolve radius values to unique particle indices.

    Args:
        particle_radius: Available particle radii.
        lookup_values: Radii to resolve against particle_radius.

    Returns:
        Index array aligned with lookup_values.

    Raises:
        ValueError: If a lookup value has zero or multiple matches.
    """
    indices = np.empty(lookup_values.size, dtype=np.int64)
    for idx, value in enumerate(lookup_values):
        matches = np.flatnonzero(np.isclose(particle_radius, value))
        if matches.size == 0:
            raise ValueError(f"Direct kernel lookup failed for radius {value}.")
        if matches.size > 1:
            raise ValueError(
                "Direct kernel lookup ambiguous for radius "
                f"{value}; {matches.size} matches found."
            )
        indices[idx] = matches[0]
    return indices


def _build_direct_kernel_index_func(
    particle_radius: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_charge: NDArray[np.float64],
    kernel_builder: Callable[..., NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Callable[[NDArray[np.int64], NDArray[np.int64]], NDArray[np.float64]]:
    """Build an index-based kernel function for particle-resolved sampling."""

    def direct_kernel_index_func(
        small_index: NDArray[np.int64],
        large_index: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        small_arr, large_arr = np.broadcast_arrays(
            np.atleast_1d(small_index), np.atleast_1d(large_index)
        )
        small_flat = small_arr.ravel().astype(np.int64)
        large_flat = large_arr.ravel().astype(np.int64)
        kernel_values = np.empty(small_flat.shape, dtype=np.float64)
        for i, (small_idx, large_idx) in enumerate(
            zip(small_flat, large_flat, strict=True)
        ):
            kernel_matrix = kernel_builder(
                particle_radius=np.array(
                    [
                        particle_radius[small_idx],
                        particle_radius[large_idx],
                    ],
                    dtype=np.float64,
                ),
                particle_mass=np.array(
                    [
                        particle_mass[small_idx],
                        particle_mass[large_idx],
                    ],
                    dtype=np.float64,
                ),
                particle_charge=np.array(
                    [
                        particle_charge[small_idx],
                        particle_charge[large_idx],
                    ],
                    dtype=np.float64,
                ),
                temperature=temperature,
                pressure=pressure,
            )
            kernel_values[i] = kernel_matrix[0, 1]
        return kernel_values.reshape(small_arr.shape)

    return direct_kernel_index_func


class CoagulationStrategyABC(ABC):
    """Abstract base class for defining a coagulation strategy.

    This class defines the methods that must be implemented by any
    coagulation strategy (e.g., for discrete, continuous, or
    particle-resolved distributions).

    Attributes:
        - distribution_type : The type of distribution to be used, one of
          ("discrete", "continuous_pdf", or "particle_resolved").
        - use_direct_kernel : Whether to compute pairwise kernels directly
          for particle-resolved coagulation.

    Methods:
    - dimensionless_kernel : Calculate the dimensionless coagulation kernel.
    - kernel : Obtain the dimensioned coagulation kernel [m^3/s].
    - loss_rate : Calculate the coagulation loss rate.
    - gain_rate : Calculate the coagulation gain rate.
    - net_rate : Get the net coagulation rate (gain - loss).
    - step : Perform a single step of coagulation.
    - diffusive_knudsen : Calculate the diffusive Knudsen number.
    - coulomb_potential_ratio : Compute Coulomb potential ratio.
    - friction_factor : Compute the effective friction factor.

    Examples:
        ```py
        class ExampleCoagulation(CoagulationStrategyABC):
            def dimensionless_kernel(self, diff_kn, coulomb_phi):
                return diff_kn + coulomb_phi
            def kernel(self, particle, temperature, pressure):
                return 1.0
        strategy = ExampleCoagulation("discrete")
        ```

    References:
        Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and
        Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.
    """

    def __init__(
        self,
        distribution_type: str,
        particle_resolved_kernel_radius: Optional[NDArray[np.float64]] = None,
        particle_resolved_kernel_bins_number: Optional[int] = None,
        particle_resolved_kernel_bins_per_decade: int = 10,
        use_direct_kernel: bool = False,
    ):
        """Initialize the coagulation strategy.

        Arguments:
            distribution_type: Type of distribution ("discrete",
                "continuous_pdf", or "particle_resolved").
            particle_resolved_kernel_radius: Kernel radius for
                particle-resolved simulations.
            particle_resolved_kernel_bins_number: Number of bins for
                particle-resolved kernel.
            particle_resolved_kernel_bins_per_decade: Bins per decade for
                particle-resolved kernel (default 10).
            use_direct_kernel: Whether to compute kernel values directly for
                particle-resolved coagulation instead of using interpolation.

        Raises:
            ValueError: If distribution_type is not valid.
        """
        if distribution_type not in [
            "discrete",
            "continuous_pdf",
            "particle_resolved",
        ]:
            raise ValueError(
                "Invalid distribution type. Must be one of 'discrete', "
                + "'continuous_pdf', or 'particle_resolved'."
            )
        self.distribution_type = distribution_type
        self.random_generator = np.random.default_rng()
        # for particle resolved coagulation strategy
        self.particle_resolved_radius = particle_resolved_kernel_radius
        self.particle_resolved_bins_number = (
            particle_resolved_kernel_bins_number
        )
        self.particle_resolved_bins_per_decade = (
            particle_resolved_kernel_bins_per_decade
        )
        self.use_direct_kernel = use_direct_kernel

    @abstractmethod
    def dimensionless_kernel(
        self,
        diffusive_knudsen: NDArray[np.float64],
        coulomb_potential_ratio: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the dimensionless coagulation kernel.

        Arguments:
            - diffusive_knudsen : The diffusive Knudsen number
              [dimensionless].
            - coulomb_potential_ratio : The Coulomb potential ratio
              [dimensionless].

        Returns:
            - NDArray[np.float64] : Dimensionless kernel for particle
              coagulation.

        Examples:
            ```py
            H = strategy.dimensionless_kernel(kn_array, phi_array)
            # H might be array([...]) representing the dimensionless kernel
            ```
        """

    @abstractmethod
    def kernel(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation kernel [m^3/s].

        Uses particle attributes (e.g., radius, mass) along with temperature
        and pressure to return a dimensional kernel for coagulation.

        Arguments:
            particle : The ParticleRepresentation object, providing radius
            and concentration.
            - temperature : The temperature in Kelvin [K].
            - pressure : The pressure in Pascals [Pa].

        Returns:
            - float or NDArray[np.float64] : The coagulation kernel [m^3/s].

        Examples:
            ```py
            k_val = strategy.kernel(particle, 298.15, 101325)
            # k_val can be a scalar or array
            ```
        """

    def loss_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation loss rate [kg/s].

        Arguments:
            particle : The particle representation for which the loss rate
            is calculated.
            - kernel : The coagulation kernel [m^3/s].

        Returns:
            - float or NDArray[np.float64] : The loss rate [kg/s].

        Raises:
            - ValueError : If the distribution type is invalid.

        Examples:
            ```py
            loss = strategy.loss_rate(particle, k_val)
            ```
        """
        if self.distribution_type == "discrete":
            return coagulation_rate.get_coagulation_loss_rate_discrete(
                concentration=particle.concentration,
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_loss_rate_continuous(
                radius=particle.get_radius(),
                concentration=particle.concentration,
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def gain_rate(
        self,
        particle: ParticleRepresentation,
        kernel: NDArray[np.float64],
    ) -> Union[float, NDArray[np.float64]]:
        """Calculate the coagulation gain rate [kg/s].

        Arguments:
            particle : The particle representation used in the
            calculation.
            - kernel : The coagulation kernel [m^3/s].

        Returns:
            - float or NDArray[np.float64] : The gain rate [kg/s].

        Raises:
            - ValueError : If the distribution type is invalid.

        Examples:
            ```py
            gain = strategy.gain_rate(particle, k_val)
            ```
        """
        if self.distribution_type == "discrete":
            return coagulation_rate.get_coagulation_gain_rate_discrete(
                radius=particle.get_radius(),
                concentration=particle.concentration,
                kernel=kernel,
            )
        if self.distribution_type == "continuous_pdf":
            return coagulation_rate.get_coagulation_gain_rate_continuous(
                radius=particle.get_radius(),
                concentration=particle.concentration,
                kernel=kernel,
            )
        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete' or 'continuous_pdf'."
        )

    def net_rate(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> Union[float, NDArray[np.float64]]:
        """Compute the net coagulation rate = gain - loss [kg/s].

        Arguments:
            - particle : The particle representation.
            - temperature : The gas-phase temperature [K].
            - pressure : The gas-phase pressure [Pa].

        Returns:
            float or NDArray[np.float64] : The net coagulation rate
            [kg/s]. (positive => net gain, negative => net loss).

        Examples:
            ```py
            net = strategy.net_rate(particle, 298.15, 101325)
            ```
        """
        kernel = self.kernel(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # Type narrowing: cast to NDArray for method compatibility
        kernel_arr = cast(NDArray[np.float64], np.atleast_1d(kernel))
        loss_rate = self.loss_rate(particle=particle, kernel=kernel_arr)
        gain_rate = self.gain_rate(particle=particle, kernel=kernel_arr)
        return gain_rate - loss_rate

    def step(  # noqa: C901
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
        time_step: float,
    ) -> ParticleRepresentation:
        """Perform a single coagulation step over a specified time interval.

        Updates the particle distribution or representation based on the
        net_rate calculated for the given time_step.

        Arguments:
            - particle : The particle representation to update.
            - temperature : The gas-phase temperature [K].
            - pressure : The gas-phase pressure [Pa].
            - time_step : The timestep over which to integrate [s].

        Returns:
            ParticleRepresentation : Updated particle representation after
            this step.

        Raises:
            - ValueError : If the distribution type is invalid or unsupported.

        Examples:
            ```py
            updated_particle = strategy.step(particle, 298.15, 101325, 1.0)
            ```
        """
        if self.distribution_type in ["discrete", "continuous_pdf"]:
            net_rate_value = self.net_rate(
                particle=particle,
                temperature=temperature,
                pressure=pressure,
            )
            # Type narrowing: cast to NDArray for add_concentration
            concentration_change = cast(
                NDArray[np.float64], np.atleast_1d(net_rate_value * time_step)
            )
            particle.add_concentration(concentration_change)
            return particle

        if self.distribution_type == "particle_resolved":
            # get the kernel radius
            func = (  # noqa: E501
                change_particle_representation.get_particle_resolved_binned_radius
            )
            kernel_radius = func(
                particle=particle,
                bin_radius=self.particle_resolved_radius,
                total_bins=self.particle_resolved_bins_number,
                bins_per_radius_decade=self.particle_resolved_bins_per_decade,  # noqa: E501
            )
            # convert particle representation to calculate kernel
            func2 = change_particle_representation.get_speciated_mass_representation_from_particle_resolved  # noqa: E501
            kernel_particle = func2(
                particle=particle,
                bin_radius=kernel_radius,
            )
            kernel_radius = kernel_particle.get_radius()
            if np.any(kernel_radius) == 0:
                raise ValueError(
                    "The kernel radius cannot be zero. "
                    "Check the particle representation."
                )
            kernel = self.kernel(
                particle=kernel_particle,
                temperature=temperature,
                pressure=pressure,
            )
            # Type narrowing: cast to NDArray for step function
            kernel_arr = cast(NDArray[np.float64], np.atleast_1d(kernel))
            # Ensure kernel_radius is monotonically increasing for binning
            # and reorder kernel matrix accordingly
            sort_indices = np.argsort(kernel_radius)
            kernel_radius = kernel_radius[sort_indices]
            if kernel_arr.ndim == 2:
                kernel_arr = kernel_arr[sort_indices][:, sort_indices]
            # calculate step
            step_func = (
                particle_resolved_method.get_particle_resolved_coagulation_step
            )
            kernel_func: Optional[
                Callable[
                    [NDArray[np.float64], NDArray[np.float64]],
                    NDArray[np.float64],
                ]
            ] = None
            kernel_index_func: Optional[
                Callable[
                    [NDArray[np.int64], NDArray[np.int64]],
                    NDArray[np.float64],
                ]
            ] = None
            if self.use_direct_kernel:
                kernel_strategy = getattr(self, "kernel_strategy", None)
                kernel_mapping = {
                    charged_kernel_strategy.HardSphereKernelStrategy: (
                        charged_dimensional_kernel.get_hard_sphere_kernel_via_system_state
                    ),
                    charged_kernel_strategy.CoulombDyachkov2007KernelStrategy: (
                        charged_dimensional_kernel.get_coulomb_kernel_dyachkov2007_via_system_state
                    ),
                    charged_kernel_strategy.CoulombGatti2008KernelStrategy: (
                        charged_dimensional_kernel.get_coulomb_kernel_gatti2008_via_system_state
                    ),
                    (
                        charged_kernel_strategy.CoulombGopalakrishnan2012KernelStrategy
                    ): (
                        charged_dimensional_kernel.get_coulomb_kernel_gopalakrishnan2012_via_system_state
                    ),
                    charged_kernel_strategy.CoulumbChahl2019KernelStrategy: (
                        charged_dimensional_kernel.get_coulomb_kernel_chahl2019_via_system_state
                    ),
                }
                kernel_builder = None
                if kernel_strategy is not None:
                    kernel_builder = kernel_mapping.get(type(kernel_strategy))
                if kernel_builder is None:
                    logger.warning(
                        "Direct kernel disabled for unsupported kernel "
                        "strategy."
                    )
                else:
                    particle_radius = np.asarray(
                        particle.get_radius(), dtype=np.float64
                    )
                    particle_mass = np.asarray(
                        particle.get_mass(), dtype=np.float64
                    )
                    particle_charge = particle.get_charge()
                    if particle_charge is None:
                        particle_charge = np.zeros_like(
                            particle_radius, dtype=np.float64
                        )
                    charge_array = np.asarray(particle_charge, dtype=np.float64)
                    kernel_index_func = _build_direct_kernel_index_func(
                        particle_radius=particle_radius,
                        particle_mass=particle_mass,
                        particle_charge=charge_array,
                        kernel_builder=kernel_builder,
                        temperature=temperature,
                        pressure=pressure,
                    )

            loss_gain_indices = step_func(
                particle_radius=particle.get_radius(),
                kernel=kernel_arr,
                kernel_radius=kernel_radius,
                volume=particle.get_volume(),
                time_step=time_step,
                random_generator=self.random_generator,
                kernel_func=kernel_func,
                kernel_index_func=kernel_index_func,
            )
            particle.collide_pairs(loss_gain_indices)
            return particle

        raise ValueError(
            "Invalid distribution type. "
            "Must be either 'discrete', 'continuous_pdf', or"
            " 'particle_resolved'."
        )

    def diffusive_knudsen(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Calculate the diffusive Knudsen number for each particle.

        The Knudsen number is used to characterize the relative importance of
        diffusion-controlled processes.

        Arguments:
            - particle : The ParticleRepresentation.
            - temperature : The gas-phase temperature [K].
            - pressure : The gas-phase pressure [Pa].

        Returns:
            - NDArray[np.float64] : Diffusive Knudsen number(s) [dimensionless].

        Examples:
            ```py
            knudsen_nums = strategy.diffusive_knudsen(particle, 298.15, 101325)
            ```
        """
        # properties calculation for friction factor
        friction_factor = self.friction_factor(
            particle=particle, temperature=temperature, pressure=pressure
        )
        # coulomb potential ratio
        coulomb_potential_ratio = self.coulomb_potential_ratio(
            particle=particle, temperature=temperature
        )
        return particles.get_diffusive_knudsen_number(
            particle_radius=particle.get_radius(),
            particle_mass=particle.get_mass(),
            friction_factor=friction_factor,
            coulomb_potential_ratio=coulomb_potential_ratio,
            temperature=temperature,
        )  # type: ignore

    def coulomb_potential_ratio(
        self, particle: ParticleRepresentation, temperature: float
    ) -> NDArray[np.float64]:
        """Calculate the Coulomb potential ratio for each particle.

        This ratio characterizes the influence of electrostatic forces on
        coagulation processes.

        Arguments:
            - particle : The ParticleRepresentation.
            - temperature : The gas-phase temperature [K].

        Returns:
            - NDArray[np.float64] : Coulomb potential ratio(s) [dimensionless].

        Examples:
            ```py
            phi = strategy.coulomb_potential_ratio(particle, 298.15)
            ```
        """
        return particles.get_coulomb_enhancement_ratio(
            particle_radius=particle.get_radius(),
            charge=particle.get_charge(),
            temperature=temperature,
        )  # type: ignore

    def friction_factor(
        self,
        particle: ParticleRepresentation,
        temperature: float,
        pressure: float,
    ) -> NDArray[np.float64]:
        """Compute the friction factor for each particle in the aerosol.

        Considers dynamic viscosity, mean free path, and slip correction to
        determine the friction factor [dimensionless].

        Arguments:
            particle : The ParticleRepresentation for which to compute
            friction factor.
            - temperature : Gas temperature [K].
            - pressure : Gas pressure [Pa].

        Returns:
            - NDArray[np.float64] : Friction factor(s) [dimensionless].

        Examples:
            ```py
            fr = strategy.friction_factor(particle, 298.15, 101325)
            ```
        """
        # assume standard atmospheric composition
        dynamic_viscosity = gas.get_dynamic_viscosity(temperature=temperature)
        mean_free_path = gas.get_molecule_mean_free_path(
            temperature=temperature,
            pressure=pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        knudsen_number = particles.get_knudsen_number(
            mean_free_path=mean_free_path,
            particle_radius=particle.get_radius(),
        )
        slip_correction = particles.get_cunningham_slip_correction(
            knudsen_number=knudsen_number
        )
        return particles.get_friction_factor(
            particle_radius=particle.get_radius(),
            dynamic_viscosity=dynamic_viscosity,
            slip_correction=slip_correction,
        )  # type: ignore
