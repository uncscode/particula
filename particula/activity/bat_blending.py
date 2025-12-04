"""Blending weights for the BAT model.

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
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "molar_mass_ratio": "positive",
        "oxygen2carbon": "nonnegative",
    }
)
def bat_blending_weights(
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Function to estimate the blending weights for the BAT model.

    Args:
        - molar_mass_ratio : The molar mass ratio of water to organic
            matter.
        - oxygen2carbon : The oxygen to carbon ratio.

    Returns:
        - blending_weights : Array of blending weights for the BAT model
            in the low, mid, and high oxygen2carbon regions. The weights
            size is (3,) if oxygen2carbon is a single value, or (n, 3)
            if oxygen2carbon is an array of size n.
    """
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)

    oxygen2carbon_ml = phase_separation.organic_water_single_phase(
        molar_mass_ratio=molar_mass_ratio
    )

    if np.size(oxygen2carbon) == 1:
        return _calculate_blending_weights(
            float(oxygen2carbon.flat[0]), float(oxygen2carbon_ml.flat[0])
        )

    return np.array(
        [
            _calculate_blending_weights(float(oc), float(oxygen2carbon_ml[i]))
            for i, oc in enumerate(oxygen2carbon)
        ]
    )


def _calculate_blending_weights(
    oxygen2carbon: float, oxygen2carbon_ml: float
) -> NDArray[np.float64]:
    """Helper function to calculate blending weights for oxygen2carbon.

    Args:
        - oxygen2carbon : The oxygen to carbon ratio.
        - oxygen2carbon_ml : The single-phase oxygen to carbon ratio.

    Returns:
        - blending_weights : List of blending weights for the BAT model
            in the low, mid, and high oxygen2carbon regions.
    """
    blending_weights = np.zeros(3)  # [low, mid, high] oxygen2carbon regions

    if oxygen2carbon <= oxygen2carbon_ml * 0.75:
        b_ml = 0.189974476118418
        b_1 = 79.2606902175984
        b_2 = 0.0604293454322489

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml * b_ml
        weight_b = 1 / (1 + np.exp(-b_1 * (oxygen2carbon_b - b_2)))

        oxygen2carbon_b_norm = oxygen2carbon - (0.75 * oxygen2carbon_ml * b_ml)
        weight_norm = 1 / (1 + np.exp(-b_1 * (oxygen2carbon_b_norm - b_2)))

        blending_weights[1] = weight_b / weight_norm
        blending_weights[0] = 1 - blending_weights[1]

    elif oxygen2carbon <= oxygen2carbon_ml * 2:
        b_1 = 75.0159268221068
        b_2 = 0.000947111285750515

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml
        blending_weights[2] = 1 / (1 + np.exp(-b_1 * (oxygen2carbon_b - b_2)))

        blending_weights[1] = 1 - blending_weights[2]

    else:
        blending_weights[2] = 1

    return blending_weights
