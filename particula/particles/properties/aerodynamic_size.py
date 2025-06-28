"""Module for aerodynamic size and shape factor of a particle in a fluid."""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

AERODYNAMIC_SHAPE_FACTOR_DICT = {
    "sphere": 1.0,
    "cube": 1.08,
    "cylinder_avg_aspect_2": 1.10,
    "cylinder_avg_aspect_5": 1.35,
    "cylinder_avg_aspect_10": 1.68,
    "spheres_cluster_3": 1.15,
    "spheres_cluster_4": 1.17,
    "bituminous_coal": 1.08,
    "quartz": 1.36,
    "sand": 1.57,
    "talc": 1.88,
}


@validate_inputs(
    {
        "physical_length": "nonnegative",
        "physical_slip_correction_factor": "nonnegative",
        "aerodynamic_slip_correction_factor": "nonnegative",
        "density": "positive",
    }
)
def get_aerodynamic_length(
    physical_length: Union[float, NDArray[np.float64]],
    physical_slip_correction_factor: Union[float, NDArray[np.float64]],
    aerodynamic_slip_correction_factor: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    reference_density: float = 1000,
    aerodynamic_shape_factor: float = 1.0,
) -> Union[float, NDArray[np.float64]]:
    # pylint: disable=too-many-positional-arguments, too-many-arguments
    """Calculate the aerodynamic length scale of a particle for a given shape.

    The aerodynamic length (d_a) is determined by:

    - d_a = d_p × √( (C_p / C_a) × (ρ / (ρ₀ × χ)) )
        - d_a is the aerodynamic size (m).
        - d_p is the physical size (m).
        - C_p is the slip correction factor for the physical size.
        - C_a is the slip correction factor for the aerodynamic size.
        - ρ is the particle density (kg/m³).
        - ρ₀ is the reference density (kg/m³).
        - χ is the shape factor (dimensionless).

    Arguments:
        - physical_length : Physical length scale of the particle (m).
        - physical_slip_correction_factor : Slip correction factor for the
            particle's physical size (dimensionless).
        - aerodynamic_slip_correction_factor : Slip correction factor for the
            particle's aerodynamic size (dimensionless).
        - density : Density of the particle in kg/m³.
        - reference_density : Reference density in kg/m³, typically water
            (1000 by default).
        - aerodynamic_shape_factor : Shape factor
            (dimensionless, 1.0 for spheres).

    Returns:
        - Aerodynamic length scale (m).

    Examples:
        ``` py title="Example"
        import particula as par
        par.particles.get_aerodynamic_length(
            physical_length=0.00005,
            physical_slip_correction_factor=1.1,
            aerodynamic_slip_correction_factor=1.0,
            density=1000,
            reference_density=1000,
            aerodynamic_shape_factor=1.0,
        )
        # Output: ...
        ```

    References:
        - "Aerosol: Aerodynamic diameter," Wikipedia,
          https://en.wikipedia.org/wiki/Aerosol#Aerodynamic_diameter
        - Hinds, W.C. (1998). Aerosol Technology: Properties, behavior, and
          measurement of airborne particles (2nd ed.). Wiley-Interscience.
          (pp. 51–53, Section 3.6).
    """
    return physical_length * np.sqrt(
        (physical_slip_correction_factor / aerodynamic_slip_correction_factor)
        * (density / (reference_density * aerodynamic_shape_factor))
    )


def get_aerodynamic_shape_factor(shape_key: str) -> float:
    """Retrieve the aerodynamic shape factor for a given particle shape.

    The shape factor (χ) accounts for non-sphericity in aerodynamic
    calculations. For spheres, χ=1.0. Larger values indicate more deviation
    from spherical shape.

    Arguments:
        - shape_key : String representing the particle's shape
            (e.g. "sphere", "sand").

    Returns:
        - The shape factor (dimensionless).

    Examples:
        ``` py title="Example"
        shape_factor = get_aerodynamic_shape_factor("sand")
        # shape_factor = 1.57
        ```

    Raises:
        - ValueError : If the shape key is not found in the predefined
            dictionary.

    References:
        - Hinds, W.C. (1998). Aerosol Technology: Properties, behavior, and
          measurement of airborne particles (2nd ed.). Wiley-Interscience.
    """
    shape_key = shape_key.strip().lower()  # Clean up the input

    # Retrieve the shape factor from the dictionary, or raise an error
    try:
        return AERODYNAMIC_SHAPE_FACTOR_DICT[shape_key]
    except KeyError as exc:
        raise ValueError(
            f"The shape factor for the shape '{shape_key}' is not available."
        ) from exc
