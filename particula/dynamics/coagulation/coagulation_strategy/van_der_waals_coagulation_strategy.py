"""CPU strategy for dispersion and electrostatic capture by compact spheres."""

import numpy as np

from particula.dynamics.coagulation.van_der_waals_kernel import (
    TABLE_VERSION,
    get_van_der_waals_kernel_via_system_state,
    validate_material,
)
from particula.particles.particle_data import ParticleData

from ._native_validation import native_kernel
from .coagulation_strategy_abc import (
    CoagulationStrategyABC,
    _get_charge,
    _get_mass,
    _get_radius,
)


class VanDerWaalsCoagulationStrategy(CoagulationStrategyABC):
    """Complete thermal encounter rate with dispersion and polarization.

    One effective Hamaker constant [J] and relative permittivity apply to
    every sphere. Charges use signed elementary-charge units.
    """

    def __init__(
        self,
        distribution_type: str,
        hamaker_constant: float,
        relative_permittivity: float,
    ):
        """Set the population representation and material constants."""
        super().__init__(distribution_type=distribution_type)
        validate_material(hamaker_constant, relative_permittivity)
        self.capture_table_version = TABLE_VERSION
        self.hamaker_constant = float(hamaker_constant)
        self.relative_permittivity = float(relative_permittivity)

    def dimensionless_kernel(self, diffusive_knudsen, coulomb_potential_ratio):
        """Require pair geometry and both particle charges."""
        raise NotImplementedError(
            "use kernel() with both particle radii and charges"
        )

    @native_kernel
    def kernel(
        self,
        particle: ParticleData,
        temperature: float,
        pressure: float,
    ) -> np.ndarray:
        """Return the thermal pair kernel [m³/s] in the native slot layout."""
        return get_van_der_waals_kernel_via_system_state(
            _get_radius(particle),
            _get_mass(particle),
            _get_charge(particle),
            temperature,
            pressure,
            self.hamaker_constant,
            self.relative_permittivity,
        )
