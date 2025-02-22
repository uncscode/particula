"""
Convert between spherical volume and length properties (radius or diameter).

This module provides functions to:
- Convert a spherical volume to its radius or diameter.
- Convert a radius or diameter to a spherical volume.

Examples:
    ``` py
    import numpy as np
    from particula.util.converting.convert_shapes import volume_to_length, length_to_volume

    # Convert volume to radius
    vol = 1e-21
    rad = volume_to_length(vol, length_type="radius")
    print(rad)

    # Convert diameter to volume
    dia = 1e-6
    vol = length_to_volume(dia, length_type="diameter")
    print(vol)
    ```

References:
    - "Volume of a sphere," Wikipedia, The Free Encyclopedia.

To be removed, likely particula_beta only. -kyle
"""

from typing import Union

from numpy.typing import NDArray
import numpy as np


def get_length_from_volume(
    volume: Union[float, NDArray[np.float64]], length_type: str = "radius"
) -> Union[float, NDArray[np.float64]]:
    """
    Convert a spherical volume (m³) to either radius or diameter (m).

    The relationship for a sphere is:
    - V = (4/3) × π × r³
      Thus:
    - r = ((3 × V) / (4 × π))^(1/3)

    Arguments:
        - volume : The spherical volume(s) in m³ (float or NDArray).
        - length_type : Either "radius" or "diameter". Defaults to "radius".

    Returns:
        - The converted length in meters (float or NDArray).

    Raises:
        - ValueError : If length_type is not "radius" or "diameter".

    Examples:
        ``` py
        import particula as par
        vol = 1e-21
        dia = par.get_length_from_volume(vol, length_type="diameter")
        print(dia)
        # ~1.24e-07
        ```

    References:
        - "Sphere," Wikipedia.
    """

    if length_type not in ["radius", "diameter"]:
        raise ValueError("length_type must be radius or diameter")

    radius = (volume * 3 / (4 * np.pi)) ** (1 / 3)

    return radius if length_type == "radius" else radius * 2


def get_volume_from_length(
    length: Union[float, np.ndarray], length_type: str = "radius"
) -> Union[float, np.ndarray]:
    """
    Convert a radius or diameter (m) to the volume (m³) of a sphere.

    The relationship for a sphere is:
    - V = (4/3) × π × r³
      If length_type is "diameter", then r = length / 2.

    Arguments:
        - length : The radius or diameter in meters (float or NDArray).
        - length_type : Either "radius" or "diameter". Defaults to "radius".

    Returns:
        - The spherical volume in m³ (float or NDArray).

    Raises:
        - ValueError : If length_type is not "radius" or "diameter".

    Examples:
        ``` py
        import particula as par
        rad = 5e-8
        vol = par.get_volume_from_length(rad, length_type="radius")
        print(vol)
        # ~5.236e-22
        ```

    References:
        - "Sphere," Wikipedia.
    """
    if length_type == "diameter":
        length = length / 2
    elif length_type != "radius":
        raise ValueError("length_type must be radius or diameter")
    return (4 / 3) * np.pi * (length**3)
