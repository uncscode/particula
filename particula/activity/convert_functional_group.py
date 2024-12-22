"""
Convert the oxygen to carbon ratio and molar mass ratio
to OH equivalent.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Optional, Union, Tuple
import numpy as np
from numpy.typing import NDArray


def convert_to_oh_equivalent(
    oxygen2carbon: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    functional_group: Optional[Union[list[str], str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Just a pass through now, but will
    add the oh equivalent conversion.

    Args:
        - oxygen2carbon : The oxygen to carbon ratio.
        - molar_mass_ratio : The molar mass ratio of water to organic
          matter.
        - functional_group : Optional functional group(s) of the organic
          compound, if applicable.

    Returns:
        - A tuple containing the converted oxygen to carbon ratio and
          molar mass ratio.
    """

    if functional_group is None:
        return oxygen2carbon, molar_mass_ratio
    if functional_group == "alcohol":
        return oxygen2carbon + 1, molar_mass_ratio + 16  # fix this from SI
    raise ValueError("BAT functional group not recognized")
