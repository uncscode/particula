"""Scalar shortwave reflection over horizontally uniform land and ocean.

The RossThick/LiSparseReciprocal kernels follow the MODIS MCD43 algorithm.
The idealized wind-roughened ocean uses the isotropic Cox–Munk mean square
slope (1954, doi:10.1364/JOSA.44.000838) and unpolarized Fresnel reflectance.
"""

from dataclasses import dataclass

import numpy as np

from particula.optics._spectral import spectral_values, wavelengths


@dataclass(frozen=True)
class Surface:
    """Spectral lower boundary constructed from the land or ocean models."""

    wavelength_m: np.ndarray
    model: str
    coefficients: tuple
    source: str
    leaving_radiance: np.ndarray | None = None

    def brdf(self, mu_in, mu_out, relative_azimuth_rad):
        """Return BRDF [sr⁻¹], with wavelength first and geometry axes last.

        The azimuth difference is measured between directions from the
        surface toward the source and observer. Incidence and emergence
        cosines are positive. Specular reflection occurs at azimuth pi.
        """
        incoming, outgoing, phi = np.broadcast_arrays(
            mu_in, mu_out, relative_azimuth_rad
        )
        for cosine in (incoming, outgoing):
            if np.any(~np.isfinite(cosine)) or np.any(
                (cosine <= 0) | (cosine > 1)
            ):
                raise ValueError("Direction cosines must lie in (0, 1]")
        if not np.all(np.isfinite(phi)):
            raise ValueError("Relative azimuth must be finite")
        expand = (slice(None),) + (None,) * phi.ndim
        if self.model == "lambertian":
            return self.coefficients[0][expand] * np.ones_like(phi) / np.pi
        if self.model == "ross_li":
            iso, vol, geo = (array[expand] for array in self.coefficients)
            kv, kg = _land_kernels(incoming, outgoing, phi)
            return (iso + vol * kv + geo * kg) / np.pi
        n, slope_variance = self.coefficients
        # The reflecting facet normal bisects source and observer directions.
        cos_angle = _dot(incoming, outgoing, phi)
        cos_facet = np.sqrt(
            np.maximum((1 + cos_angle) / 2, np.finfo(float).tiny)
        )
        cos_tilt = np.clip(
            (incoming + outgoing) / (2 * cos_facet), np.finfo(float).tiny, 1
        )
        tan_sq = (1 - cos_tilt**2) / cos_tilt**2
        slope_density = np.exp(-tan_sq / slope_variance) / (
            np.pi * slope_variance
        )
        index = n[expand]
        refracted = np.sqrt(index**2 - (1 - cos_facet**2))
        rs = ((cos_facet - refracted) / (cos_facet + refracted)) ** 2
        rp = (
            (index**2 * cos_facet - refracted)
            / (index**2 * cos_facet + refracted)
        ) ** 2
        # Smith masking for the isotropic Gaussian facet distribution.
        shadow = 1 / (
            1
            + _shadow(incoming, slope_variance)
            + _shadow(outgoing, slope_variance)
        )
        return (
            (rs + rp)
            / 2
            * slope_density
            * shadow
            / (4 * incoming * outgoing * cos_tilt**4)
        )


def _shadow(mu, variance):
    from scipy.special import erfc

    a = mu / np.sqrt(np.maximum(1 - mu**2, 1e-30) * variance)
    return (np.exp(-(a**2)) / (np.sqrt(np.pi) * a) - erfc(a)) / 2


def _dot(mu_a, mu_b, phi):
    return np.clip(
        mu_a * mu_b + np.sqrt((1 - mu_a**2) * (1 - mu_b**2)) * np.cos(phi),
        -1,
        1,
    )


def _land_kernels(mu_a, mu_b, phi):
    cos_xi = _dot(mu_a, mu_b, phi)
    xi = np.arccos(cos_xi)
    vol = ((np.pi / 2 - xi) * cos_xi + np.sin(xi)) / (mu_a + mu_b) - np.pi / 4
    tan_a = np.sqrt(1 - mu_a**2) / mu_a
    tan_b = np.sqrt(1 - mu_b**2) / mu_b
    dist_sq = np.maximum(
        tan_a**2 + tan_b**2 - 2 * tan_a * tan_b * np.cos(phi), 0
    )
    sec_sum = 1 / mu_a + 1 / mu_b
    cos_t = np.clip(
        2 * np.sqrt(dist_sq + (tan_a * tan_b * np.sin(phi)) ** 2) / sec_sum,
        0,
        1,
    )
    t = np.arccos(cos_t)
    overlap = (t - np.sin(t) * cos_t) * sec_sum / np.pi
    geo = overlap - sec_sum + (1 + cos_xi) / (2 * mu_a * mu_b)
    return vol, geo


def lambertian_surface(
    wavelength_m, albedo, *, source="prescribed Lambertian surface"
):
    """Prescribe spectral hemispheric albedo between zero and one."""
    wave = wavelengths(wavelength_m)
    rho = spectral_values(albedo, wave, "Surface albedo")
    if np.any((rho < 0) | (rho > 1)):
        raise ValueError("Surface albedo must be in [0, 1]")
    return Surface(wave, "lambertian", (rho,), source)


def _reference_albedo(wave, values):
    # Deliberately idealized reference spectra, not measured site properties.
    knots = np.array([280, 400, 550, 700, 1000, 1600, 2200, 4000]) * 1e-9
    if np.min(wave) < knots[0] - 1e-18 or np.max(wave) > knots[-1] + 1e-18:
        raise ValueError("Ideal surface curves span 280–4000 nm")
    return np.interp(wave, knots, values)


def ocean_surface(
    wavelength_m,
    *,
    wind_speed=None,
    refractive_index=1.34,
    leaving_radiance=None,
):
    """Return ideal dark-water albedo or a wind-roughened Fresnel BRDF.

    Wind is in m/s. The BRDF represents surface glint. An optional
    isotropic water-leaving radiance [W m⁻² sr⁻¹ m⁻¹] adds a source.
    Refractive index can vary with wavelength. The bounded wind domain
    spans 1–14 m/s in the original sea-surface slope measurements.
    """
    wave = wavelengths(wavelength_m)
    leaving = None
    if leaving_radiance is not None:
        leaving = spectral_values(
            leaving_radiance, wave, "Water-leaving radiance"
        )
        if np.any(leaving < 0):
            raise ValueError("Water-leaving radiance must be nonnegative")
    if wind_speed is None:
        albedo = _reference_albedo(
            wave, [0.05, 0.055, 0.045, 0.03, 0.025, 0.02, 0.02, 0.02]
        )
        return Surface(wave, "lambertian", (albedo,), "ideal ocean v1", leaving)
    if not np.isfinite(wind_speed) or not 1 <= wind_speed <= 14:
        raise ValueError("Ocean wind speed must be in [1, 14] m/s")
    index = spectral_values(refractive_index, wave, "Water index")
    if np.any(index <= 1):
        raise ValueError("Water index must exceed one")
    variance = 0.003 + 0.00512 * wind_speed
    return Surface(
        wave,
        "cox_munk",
        (index, variance),
        f"isotropic Cox–Munk, {wind_speed:g} m/s",
        leaving,
    )


def dusty_ground(wavelength_m):
    """Return ideal dry mineral-soil Lambertian albedo for 280–4000 nm."""
    wave = wavelengths(wavelength_m)
    albedo = _reference_albedo(
        wave, [0.08, 0.12, 0.24, 0.34, 0.43, 0.46, 0.37, 0.24]
    )
    return lambertian_surface(wave, albedo, source="ideal dusty ground v1")


def ross_li_surface(
    wavelength_m, isotropic, volumetric, geometric, *, source="user Ross–Li"
):
    """Build a MODIS RossThick/LiSparseReciprocal BRDF from coefficients."""
    wave = wavelengths(wavelength_m)
    coeffs = tuple(
        spectral_values(x, wave, "BRDF coefficients")
        for x in (isotropic, volumetric, geometric)
    )
    return Surface(wave, "ross_li", coeffs, source)
