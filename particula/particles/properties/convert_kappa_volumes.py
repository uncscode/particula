"""Convert between volumes using κ-Köhler relation.

These functions compute water activity and kappa parameter influence on
volume partitioning between solute and water.

References:
    - P. Väisänen et al., "Kappa-Köhler theory for water activity and
      hygroscopic growth of aerosol particles." Journal of Aerosol Science,
      2016.
    - Wikipedia contributors, "Köhler theory," Wikipedia.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def get_solute_volume_from_kappa(
    volume_total: Union[float, np.ndarray],
    kappa: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculate the solute volume from the total solution volume using κ-Köhler
    theory.

    The relation for κ-Köhler can be written as:
    - V_solute = V_total × F
      where F depends on kappa and water activity (aw), ensuring that
      for aw → 0, V_solute → V_total.

    Arguments:
        - volume_total : Volume of the total solution (float or NDArray).
        - kappa : Kappa parameter (float or NDArray).
        - water_activity : Water activity (float or NDArray, 0 < aw ≤ 1).

    Returns:
        - Solute volume (float or NDArray).

    Examples:
        ``` py  title="Example Usage"
        import particula as par
        v_sol = par.get_solute_from_kappa_volume(1e-18, 0.8, 0.9)
        print(v_sol)
        # ~some fraction of the total volume
        ```

    References:
        - Petters, M. D. & Kreidenweis, S. M. (2007). "A single parameter
          representation of hygroscopic growth and cloud condensation nucleus
          activity." Atmos. Chem. Phys.
    """
    kappa = max(kappa, 1e-16)  # Avoid division by zero
    if water_activity <= 1e-16:  # early return for low water activity
        return volume_total

    vol_factor = (water_activity - 1) / (
        water_activity * (1 - kappa - 1 / water_activity)
    )
    return volume_total * np.array(vol_factor)


def get_water_volume_from_kappa(
    volume_solute: Union[float, NDArray[np.float64]],
    kappa: Union[float, NDArray[np.float64]],
    water_activity: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the water volume from the solute volume, κ parameter, and water
    activity.

    This uses κ-Köhler-type relations where:
    - V_water = V_solute × ( κ / (1/aw - 1) ), ensuring that for aw → 0,
      V_water → 0.

    Arguments:
        - volume_solute : Volume of solute (float or NDArray).
        - kappa : Kappa parameter (float or NDArray).
        - water_activity : Water activity (float or NDArray, 0 < aw ≤ 1).

    Returns:
        - Water volume (float or NDArray).

    Examples:
        ``` py title="Example Usage"
        import particula as par
        v_water = par.get_water_volume_from_kappa(1e-19, 0.5, 0.95)
        print(v_water)
        # ~some fraction of the solute volume
        ```

    References:
        - Petters, M. D. & Kreidenweis, S. M. (2007). "A single parameter
          representation of hygroscopic growth and cloud condensation nucleus
          activity." Atmos. Chem. Phys.
    """
    # Avoid division by zero
    water_activity = min(water_activity, 1 - 1e-16)

    if water_activity <= 1e-16:  # early return for low water activity
        return volume_solute * 0

    return volume_solute * kappa / (1 / water_activity - 1)


def get_kappa_from_volumes(
    volume_solute: Union[float, np.ndarray],
    volume_water: Union[float, np.ndarray],
    water_activity: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Compute the κ parameter from known volumes of solute and water, given
    water activity.

    Rearranging κ-Köhler-based relationships, we have:
    - κ = ( (1/aw) - 1 ) × (V_water / V_solute).

    Arguments:
        - volume_solute : Solute volume (float or NDArray).
        - volume_water : Water volume (float or NDArray).
        - water_activity : Water activity (float or NDArray, 0 < aw ≤ 1).

    Returns:
        - The kappa parameter (float or NDArray).

    Examples:
        ``` py
        import particula as par
        kappa_val = par.get_kappa_from_volumes(1e-19, 4e-19, 0.95)
        print(kappa_val)
        # ~indicative value for the solute's hygroscopicity
        ```

    References:
        - Petters, M. D. & Kreidenweis, S. M. (2007). "A single parameter
          representation of hygroscopic growth and cloud condensation nucleus
          activity." Atmos. Chem. Phys.
    """
    # Avoid division by zero
    water_activity = np.where(
        water_activity > 1 - 1e-16, 1 - 1e-16, water_activity
    )

    return (1 / water_activity - 1) * volume_water / volume_solute


def get_water_volume_in_mixture(
    volume_solute_dry: Union[float, np.ndarray],
    volume_fraction_water: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculate water volume in mixture from specified water fraction.

    The relationship is:

    - V_water = (φ_water × V_solute_dry) / (1 - φ_water)
        - φ_water is the water volume fraction.

    Arguments:
        - volume_solute_dry : Volume of the solute (float), excluding water.
        - volume_fraction_water : Fraction of water volume in the total mixture
          (float, 0 ≤ φ_water < 1).

    Returns:
        - The water volume (float), in the same units as volume_solute_dry.

    Examples:
        ``` py title="Example Usage"
        import particula as par
        v_water = par.get_water_volume_in_mixture(100.0, 0.8)
        print(v_water)
        # 400.0
        ```

    References:
        - "Volume Fractions in Mixture Calculations," Standard Chemistry Texts.
    """
    return (
        volume_fraction_water * volume_solute_dry / (1 - volume_fraction_water)
    )
