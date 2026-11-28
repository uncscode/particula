"""Path-integrated extinction and direct-beam attenuation.

Beer-Lambert transmission describes the unscattered direct beam. It does not
include diffuse multiple scattering or infer path geometry from box volume.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from particula.optics._spectral import (
    finite_array,
    interfaces,
    require_same_grid,
    wavelengths,
)
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
        ValueError: If inputs/results are invalid or nonzero optics has zero
            density.
    """
    optical, density = np.broadcast_arrays(
        _nonnegative(coefficient, "Optical coefficient"),
        _nonnegative(mass_density, "Mass density"),
    )
    valid = density > 0
    if np.any((~valid) & (optical > 0)):
        raise ValueError("Nonzero optical coefficient requires mass density")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.divide(
            optical, density, out=np.zeros_like(optical), where=valid
        )
    if np.any(~np.isfinite(result)):
        raise ValueError("Mass-specific opacity overflow")
    return result, valid


@dataclass(frozen=True)
class SpectralColumn:
    """Top-first layer optical properties for scalar shortwave transport.

    Optical depth and single scattering albedo have shape (layer, wavelength).
    Normalized Legendre moments have a final order axis, starting at beta_0=1
    where scattering is nonzero. Wavelengths and altitudes are in metres.
    """

    wavelength_m: NDArray
    interface_height_m: NDArray
    optical_depth: NDArray
    single_scattering_albedo: NDArray
    phase_moments: NDArray
    sources: tuple[str, ...] = ("user layer optics and interface heights",)

    def __post_init__(self):
        """Freeze and validate the layer and wavelength axes."""
        wave = wavelengths(self.wavelength_m)
        height = interfaces(self.interface_height_m)
        depth = finite_array(self.optical_depth, "Optical depth")
        albedo = finite_array(self.single_scattering_albedo, "Layer albedo")
        moments = finite_array(self.phase_moments, "Phase moments")
        if (
            depth.shape != (height.size - 1, wave.size)
            or albedo.shape != depth.shape
        ):
            raise ValueError("Layer optical fields require (layer, wavelength)")
        if np.any(depth < 0) or np.any((albedo < 0) | (albedo > 1)):
            raise ValueError("Require nonnegative depth and albedo in [0, 1]")
        if (
            moments.ndim != 3
            or moments.shape[:2] != depth.shape
            or not moments.shape[2]
        ):
            raise ValueError("Phase moments require (layer, wavelength, order)")
        scattering = (albedo > 0) & (depth > 0)
        if np.any(abs(moments) > 1) or np.any(moments[..., 0][scattering] != 1):
            raise ValueError("Require normalized scattering phase moments")
        for name, value in (
            ("wavelength_m", wave),
            ("interface_height_m", height),
            ("optical_depth", depth),
            ("single_scattering_albedo", albedo),
            ("phase_moments", moments),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_population(
        cls,
        wavelength_m,
        interface_height_m,
        population,
        box_indices=None,
        *,
        hg_order=32,
    ):
        """Map bulk (box, wavelength) optics to layers with an HG phase law.

        Each layer needs its homogeneous box index and geometrical thickness.
        The Henyey–Greenstein approximation uses beta_l = g**l. Supply a
        SpectralColumn directly to use measured or calculated phase moments.
        """
        wave = wavelengths(wavelength_m)
        height = interfaces(interface_height_m)
        nlayer = height.size - 1
        if (
            not isinstance(hg_order, (int, np.integer))
            or not 1 <= hg_order <= 255
        ):
            raise ValueError("HG order must be an integer from 1 to 255")
        extinction = np.asarray(population.extinction)
        if extinction.ndim != 2 or extinction.shape[1] != wave.size:
            raise ValueError("Population optics require (box, wavelength)")
        if box_indices is None:
            if extinction.shape[0] != nlayer:
                raise ValueError("Specify a box index for each layer")
            box_indices = np.arange(nlayer)
        indices = np.asarray(box_indices)
        if (
            indices.shape != (nlayer,)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= extinction.shape[0])
        ):
            raise ValueError("Invalid box index for layer")
        depth = get_column_optics(
            extinction[indices], -np.diff(height)[:, None], layer_axis=0
        )
        return cls(
            wave,
            height,
            depth.layer_optical_depth,
            np.asarray(population.single_scattering_albedo)[indices],
            np.asarray(population.asymmetry)[indices, :, None]
            ** np.arange(hg_order + 1),
            (
                f"native population, HG order {hg_order}",
                "user interface heights",
            ),
        )

    def scaled(self, factor):
        """Scale the extinction and scattering depth by the same factor."""
        if np.ndim(factor) != 0 or not np.isfinite(factor) or factor < 0:
            raise ValueError("Column scale must be finite and nonnegative")
        return replace(
            self,
            optical_depth=self.optical_depth * factor,
            sources=(*self.sources, f"optical depth multiplier {factor:g}"),
        )

    def plus(self, other):
        """Combine independent scatterers using scattering-depth weights."""
        require_same_grid(self.wavelength_m, other.wavelength_m, "Wavelengths")
        require_same_grid(
            self.interface_height_m, other.interface_height_m, "Heights"
        )
        depth = self.optical_depth + other.optical_depth
        sca_a = self.optical_depth * self.single_scattering_albedo
        sca_b = other.optical_depth * other.single_scattering_albedo
        scattering = sca_a + sca_b
        count = max(self.phase_moments.shape[-1], other.phase_moments.shape[-1])
        moments = np.zeros((*depth.shape, count))
        for column, weight in ((self, sca_a), (other, sca_b)):
            moments[..., : column.phase_moments.shape[-1]] += (
                weight[..., None] * column.phase_moments
            )
        np.divide(
            moments,
            scattering[..., None],
            out=moments,
            where=scattering[..., None] > 0,
        )
        moments[..., 0] = 1
        albedo = np.divide(
            scattering, depth, out=np.zeros_like(depth), where=depth > 0
        )
        return SpectralColumn(
            self.wavelength_m,
            self.interface_height_m,
            depth,
            albedo,
            moments,
            (*self.sources, *other.sources),
        )
