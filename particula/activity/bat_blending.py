"""
Blending weights for the BAT model.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.activity import phase_separation


def bat_blending_weights(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Function to estimate the blending weights for the BAT model.

    Args:
        - molar_mass_ratio : The molar mass ratio of water to organic
          matter.
        - oxygen2carbon : The oxygen to carbon ratio.

    Returns:
        - blending_weights : List of blending weights for the BAT model
          in the low, mid, and high oxygen2carbon regions.
    """
    # check types
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)

    oxygen2carbon_ml = phase_separation.organic_water_single_phase(
        molar_mass_ratio=molar_mass_ratio
    )

    blending_weights = np.zeros(3)  # [low, mid, high] oxygen2carbon regions

    # lower to middle oxygen2carbon region
    if oxygen2carbon <= oxygen2carbon_ml * 0.75:
        b_ml = 0.189974476118418
        b_1 = 79.2606902175984
        b_2 = 0.0604293454322489

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml * b_ml
        weight_b = 1 / (
            1 + np.exp(-b_1 * (oxygen2carbon_b - b_2))
        )  # logistic transfer function

        oxygen2carbon_b_norm = oxygen2carbon - (0.75 * oxygen2carbon_ml * b_ml)

        weight_norm = 1 / (1 + np.exp(-b_1 * (oxygen2carbon_b_norm - b_2)))

        blending_weights[1] = weight_b / weight_norm
        blending_weights[0] = 1 - blending_weights[1]

    # middle to high oxygen2carbon region
    elif oxygen2carbon <= oxygen2carbon_ml * 2:
        b_1 = 75.0159268221068
        b_2 = 0.000947111285750515

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml
        blending_weights[2] = 1 / (
            1 + np.exp(-b_1 * (oxygen2carbon_b - b_2))
        )  # logistic transfer function

        blending_weights[1] = 1 - blending_weights[2]

    else:  # high only region
        blending_weights[2] = 1

    return blending_weights
