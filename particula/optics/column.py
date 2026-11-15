"""Path-integrated extinction and direct-beam attenuation.

Beer-Lambert transmission describes the unscattered direct beam. It does not
include diffuse multiple scattering or infer path geometry from box volume.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.optics.population import _nonnegative


class ColumnOptics(NamedTuple):
    """Layer and integrated optical depth and direct-beam transmission.

    Attributes:
        layer_optical_depth: Extinction times geometric path length [1].
        optical_depth: Sum of layer depths along the selected layer axis [1].
        transmission: Unscattered direct-beam fraction exp(-optical_depth).
    """

    layer_optical_depth: NDArray[np.float64]
    optical_depth: NDArray[np.float64]
    transmission: NDArray[np.float64]


def get_column_optics(
    extinction: ArrayLike,
    path_length: ArrayLike,
    *,
    layer_axis: int = -1,
) -> ColumnOptics:
    """Integrate extinction along explicit path segments.

    Args:
        extinction: Nonnegative extinction coefficient [m⁻¹].
        path_length: Nonnegative path through each layer [m], broadcastable
            with extinction. Supply slant lengths explicitly if applicable.
        layer_axis: Layer axis after broadcasting; scalar input is one layer.

    Returns:
        Dimensionless layer depths, total depth and direct transmission.
        Empty columns have zero depth and unit transmission.

    Raises:
        ValueError: If inputs, resulting depths or layer axis are invalid.
    """
    coefficient, length = np.broadcast_arrays(
        _nonnegative(extinction, "Extinction"),
        _nonnegative(path_length, "Path length"),
    )
    with np.errstate(over="ignore"):
        layer = coefficient * length
    if np.any(~np.isfinite(layer)):
        raise ValueError("Layer optical depth overflow")
    if layer.ndim == 0:
        if layer_axis not in (-1, 0):
            raise ValueError("Invalid scalar layer axis")
        total = layer.copy()
    else:
        if not -layer.ndim <= layer_axis < layer.ndim:
            raise ValueError("Invalid layer axis")
        with np.errstate(over="ignore"):
            total = np.sum(layer, axis=layer_axis)
        if np.any(~np.isfinite(total)):
            raise ValueError("Column optical depth overflow")
    return ColumnOptics(layer, total, np.exp(-total))


def get_mass_specific_opacity(
    coefficient: ArrayLike,
    mass_density: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Divide an optical coefficient by a specified material mass density.

    Args:
        coefficient: Nonnegative extinction or absorption [m⁻¹].
        mass_density: Nonnegative density [kg/m³], broadcastable with
            coefficient. The caller specifies which material inventory it is.

    Returns:
        Mass-specific opacity [m²/kg] and positive-density validity mask.
        A zero/zero vacuum returns zero and False.

    Raises:
        ValueError: If values are invalid or nonzero optics has zero density.
    """
    optical, density = np.broadcast_arrays(
        _nonnegative(coefficient, "Optical coefficient"),
        _nonnegative(mass_density, "Mass density"),
    )
    valid = density > 0
    if np.any((~valid) & (optical > 0)):
        raise ValueError("Nonzero optical coefficient requires mass density")
    result = np.divide(
        optical, density, out=np.zeros_like(optical), where=valid
    )
    return result, valid
