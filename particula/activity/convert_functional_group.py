"""OH equivalent for the oxygen to carbon ratio and molar mass ratio.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

FloatArray = Union[float, NDArray[np.float64]]

_FUNCTIONAL_GROUP_ADJUSTMENTS = {
    "alcohol": (1.0, 16.0),
    "ether": (1.0, 16.0),
    "carboxylic_acid": (2.0, 45.0),
}


SUPPORTED_FUNCTIONAL_GROUPS = (
    None,
    *tuple(_FUNCTIONAL_GROUP_ADJUSTMENTS.keys()),
)


def _normalize_functional_group(
    functional_group: Optional[Union[str, Sequence[str]]],
) -> Optional[str]:
    if functional_group is None:
        return None
    if isinstance(functional_group, str):
        return functional_group
    if len(functional_group) != 1:
        supported_values = ", ".join(
            "None" if value is None else f'"{value}"'
            for value in SUPPORTED_FUNCTIONAL_GROUPS
        )
        raise ValueError(
            "BAT functional group lists must contain exactly one supported "
            f"value. Supported values: {supported_values}."
        )
    return functional_group[0]


def convert_to_oh_equivalent(
    oxygen2carbon: FloatArray,
    molar_mass_ratio: FloatArray,
    functional_group: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Convert oxygen and molar mass ratios for BAT functional groups.

    Args:
        oxygen2carbon: The oxygen to carbon ratio.
        molar_mass_ratio: The molar mass ratio of water to organic matter.
        functional_group: Optional functional group requiring adjustments.
            Supported values are ``None``, ``"alcohol"``, ``"ether"``, and
            ``"carboxylic_acid"``.

    Returns:
        Tuple containing the converted oxygen2carbon and molar_mass_ratio.
        Scalar inputs return scalars, array inputs return ``np.ndarray``.

    Raises:
        ValueError: If ``functional_group`` is not one of the supported values.

    Examples:
        >>> from particula.activity.convert_functional_group import (
        ...     convert_to_oh_equivalent,
        ... )
        >>> convert_to_oh_equivalent(
        ...     oxygen2carbon=0.3,
        ...     molar_mass_ratio=0.1,
        ...     functional_group="carboxylic_acid",
        ... )
        (2.3, 45.1)
    """
    normalized_group = _normalize_functional_group(functional_group)
    if normalized_group is None:
        return oxygen2carbon, molar_mass_ratio

    adjustments = _FUNCTIONAL_GROUP_ADJUSTMENTS.get(normalized_group)
    if adjustments is None:
        supported_values = ", ".join(
            "None" if value is None else f'"{value}"'
            for value in SUPPORTED_FUNCTIONAL_GROUPS
        )
        raise ValueError(
            f"BAT functional group must be one of: {supported_values}."
        )

    delta_oxygen, delta_mass = adjustments
    return oxygen2carbon + delta_oxygen, molar_mass_ratio + delta_mass
