"""Offline shortwave illumination and an idealized Earth reference atmosphere.

Solar position uses the apparent solar longitude and equation of time
(Meeus, Astronomical Algorithms, ch. 25). Solar zenith is geometrical.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from particula.optics._spectral import (
    finite_array,
    interfaces,
    spectral_values,
    wavelengths,
)
from particula.optics.column import SpectralColumn
from particula.util.constants import BOLTZMANN_CONSTANT

_DATA = Path(__file__).with_name("reference_data")


@dataclass(frozen=True, init=False)
class SolarInput:
    """Upper boundary irradiance densities [W m⁻² m⁻¹].

    The collimated beam is normal to the ray. Diffuse horizontal irradiance
    illuminates the downward hemisphere with uniform radiance. Optional
    quadrature weights in metres integrate representative spectral nodes.
    """

    wavelength_m: np.ndarray
    mu0: float
    direct_normal: np.ndarray
    diffuse_horizontal: np.ndarray
    quadrature_weights_m: np.ndarray | None = None
    source: str = "user upper boundary"
    band_bounds_m: np.ndarray | None = None

    def __init__(
        self,
        wavelength_m: ArrayLike,
        mu0: float,
        direct_normal: ArrayLike,
        diffuse_horizontal: ArrayLike,
        quadrature_weights_m: ArrayLike | None = None,
        source: str = "user upper boundary",
        band_bounds_m: ArrayLike | None = None,
    ):
        """Freeze the boundary and broadcast scalar densities to its grid."""
        wave = wavelengths(wavelength_m)
        if np.ndim(mu0) or not np.isfinite(mu0) or not 0 <= mu0 <= 1:
            raise ValueError("Solar zenith cosine must be in [0, 1]")
        object.__setattr__(self, "wavelength_m", wave)
        object.__setattr__(self, "mu0", mu0)
        object.__setattr__(self, "source", source)
        for name, value in (
            ("direct_normal", direct_normal),
            ("diffuse_horizontal", diffuse_horizontal),
            ("quadrature_weights_m", quadrature_weights_m),
        ):
            if name == "quadrature_weights_m" and value is None:
                continue
            array = spectral_values(value, wave, name)
            if np.any(array < 0) or (
                name == "quadrature_weights_m" and np.any(array <= 0)
            ):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, array)
        if band_bounds_m is not None:
            bounds = finite_array(band_bounds_m, "Spectral band bounds")
            if (
                self.quadrature_weights_m is None
                or bounds.shape != (len(wave), 2)
                or np.any(bounds[:, 0] >= bounds[:, 1])
                or np.any((wave < bounds[:, 0]) | (wave > bounds[:, 1]))
            ):
                raise ValueError(
                    "Require a containing quadrature band for each node"
                )
            object.__setattr__(self, "band_bounds_m", bounds)


@dataclass(frozen=True)
class SolarPosition:
    """Geometric Sun direction [degrees] and Earth–Sun distance [au]."""

    zenith_deg: float
    azimuth_deg: float
    distance_au: float


def solar_position(latitude_deg, longitude_deg, time: datetime):
    """Calculate true solar zenith and azimuth clockwise from north.

    Supply geodetic latitude [-90,90], east-positive longitude [-180,180]
    and a timezone-aware instant. No refraction or terrain horizon is applied.
    """
    if not -90 <= latitude_deg <= 90 or not -180 <= longitude_deg <= 180:
        raise ValueError("Invalid Earth latitude or longitude")
    if time.utcoffset() is None:
        raise ValueError("Require a timezone-aware datetime")
    utc = time.astimezone(timezone.utc)
    jd = utc.timestamp() / 86400 + 2440587.5
    t = (jd - 2451545) / 36525
    mean_lon = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    anomaly = np.deg2rad(357.52911 + t * (35999.05029 - 0.0001537 * t))
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    center = (
        (1.914602 - t * (0.004817 + 0.000014 * t)) * np.sin(anomaly)
        + (0.019993 - 0.000101 * t) * np.sin(2 * anomaly)
        + 0.000289 * np.sin(3 * anomaly)
    )
    omega = np.deg2rad(125.04 - 1934.136 * t)
    lon = np.deg2rad(mean_lon + center - 0.00569 - 0.00478 * np.sin(omega))
    eps = np.deg2rad(
        23
        + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60)
        / 60
        + 0.00256 * np.cos(omega)
    )
    decl = np.arcsin(np.sin(eps) * np.sin(lon))
    y = np.tan(eps / 2) ** 2
    longitude = np.deg2rad(mean_lon)
    eq_minutes = 4 * np.rad2deg(
        y * np.sin(2 * longitude)
        - 2 * e * np.sin(anomaly)
        + 4 * e * y * np.sin(anomaly) * np.cos(2 * longitude)
        - 0.5 * y * y * np.sin(4 * longitude)
        - 1.25 * e * e * np.sin(2 * anomaly)
    )
    minutes = (
        utc.hour * 60 + utc.minute + (utc.second + utc.microsecond / 1e6) / 60
    )
    hour = np.deg2rad(
        ((minutes + eq_minutes + 4 * longitude_deg) % 1440) / 4 - 180
    )
    lat = np.deg2rad(latitude_deg)
    cosine = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(
        hour
    )
    azimuth = np.arctan2(
        np.sin(hour), np.cos(hour) * np.sin(lat) - np.tan(decl) * np.cos(lat)
    )
    radius = (
        1.000001018
        * (1 - e**2)
        / (1 + e * np.cos(anomaly + np.deg2rad(center)))
    )
    return SolarPosition(
        float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))),
        float((np.rad2deg(azimuth) + 180) % 360),
        float(radius),
    )


@lru_cache(maxsize=1)
def _extraterrestrial():
    return np.loadtxt(_DATA / "tsis1.txt", unpack=True)


def solar_irradiance(wavelength_m):
    """TSIS-1 irradiance density [W m⁻² m⁻¹] at 1 au over 280–4000 nm."""
    wave = wavelengths(wavelength_m)
    nm, flux = _extraterrestrial()
    if wave.min() < 280e-9 - 1e-18 or wave.max() > 4000e-9 + 1e-18:
        raise ValueError("TSIS-1 reference covers 280–4000 nm")
    return np.interp(wave * 1e9, nm, flux) * 1e9


@dataclass(frozen=True)
class EarthReference:
    """Ideal clear atmosphere, solar position and upper illumination.

    The wavelength quadrature and bounds are available for interpreting
    band averages. Field sources describe user coordinates, prescribed
    profiles and reference approximations.
    """

    column: SpectralColumn
    sun: SolarInput
    position: SolarPosition
    band_bounds_m: np.ndarray | None
    sources: tuple
    latitude_deg: float
    longitude_deg: float
    time: datetime


@lru_cache(maxsize=2)
def _gas_reference(grid):
    with np.load(_DATA / f"gas_{grid}.npz") as data:
        result = {key: data[key].astype(float) for key in data.files}
        for value in result.values():
            value.setflags(write=False)
        return result


def _solar_band_energy(bounds):
    # Integrate the original 1-nm bin means by bin overlap. Partial end bins
    # contribute only their occupied wavelength interval.
    nm, flux = _extraterrestrial()
    edges = np.r_[nm - 0.5, nm[-1] + 0.5]
    cumulative = np.r_[0.0, np.cumsum(flux)]
    return np.diff(np.interp(bounds * 1e9, edges, cumulative), axis=-1).ravel()


def _profile_grid(data, height):
    z = data["interface_height_m"][::-1]
    if height[-1] < -500 or height[0] > z[-1]:
        raise ValueError(
            "Reference atmosphere spans −500–120000 m above sea level"
        )
    # Within a database layer interpolate absorber column in altitude.
    # Integrating cumulative depth preserves exact native-layer totals.
    cumulative = np.vstack(
        (
            np.zeros(data["wavelength_m"].size),
            np.cumsum(data["absorption_tau"], axis=0),
        )
    )
    x = np.interp(height, z, np.arange(z.size - 1, -1, -1))
    lower = x.astype(int)
    frac = (x - lower)[:, None]
    sampled = cumulative[lower] + frac * (
        cumulative[np.minimum(lower + 1, z.size - 1)] - cumulative[lower]
    )
    pressure = np.exp(np.interp(height, z, np.log(data["pressure_pa"][::-1])))
    below = np.minimum(height, 0)
    # Continue the sea-level log-pressure slope and the lowest layer's
    # absorption per pressure increment down to exposed below-sea-level land.
    p0, p1 = data["pressure_pa"][-1:-3:-1]
    extra = p0 * np.expm1(below * np.log(p1 / p0) / (z[1] - z[0]))
    pressure += extra
    sampled += extra[:, None] / (p0 - p1) * data["absorption_tau"][-1]
    return np.maximum(np.diff(sampled, axis=0), 0), pressure


def _gas_band_equivalent(data, depth, wave):
    # A requested wavelength represents the containing REPTRAN native band.
    # Match vertical direct transmittance over the overlying modeled column;
    # this is an approximation for arbitrary zenith angle or diffuse paths.
    from scipy.special import logsumexp

    bounds, inverse = np.unique(
        data["band_bounds_m"], axis=0, return_inverse=True
    )
    ids = np.searchsorted(bounds[:, 1], wave, side="left")
    if np.any(ids >= len(bounds)) or np.any(wave < bounds[ids, 0] - 1e-17):
        raise ValueError("Requested wavelengths exceed gas table coverage")
    cumulative = np.vstack((np.zeros(depth.shape[1]), np.cumsum(depth, axis=0)))
    effective = np.empty((depth.shape[0], len(wave)))
    for group in np.unique(ids):
        select = inverse == group
        weights = data["weight_m"][select] * solar_irradiance(
            data["wavelength_m"][select]
        )
        tau = -logsumexp(
            -cumulative[:, select], b=weights / weights.sum(), axis=1
        )
        effective[:, ids == group] = np.maximum(np.diff(tau), 0)[:, None]
    return effective, bounds[ids]


def profile_gas_optical_depth(
    interface_height_m,
    pressure_pa,
    temperature_k,
    mole_fraction,
    absorption_cross_section_m2,
):
    """Compute layer absorption depths from a local sounding and cross sections.

    Pressure [Pa], temperature [K], and gas mole fractions [mol/mol] refer
    to top-first interfaces. Fractions have axes (interface, gas). Supply
    layer-effective cross sections [m² molecule⁻¹] with axes
    (layer, wavelength, gas) in the same species order. Trapezoidally
    integrate absorber number density assuming a linear vertical profile
    per layer. Supply cross sections at the appropriate temperature,
    pressure and spectroscopic resolution for the application.
    """
    height = interfaces(interface_height_m)
    pressure = finite_array(pressure_pa, "Interface pressures")
    temp = finite_array(temperature_k, "Interface temperatures")
    frac = finite_array(mole_fraction, "Gas mole fractions")
    cross = finite_array(absorption_cross_section_m2, "Gas cross sections")
    if (
        pressure.shape != height.shape
        or temp.shape != height.shape
        or np.any(pressure < 0)
        or np.any(np.diff(pressure) <= 0)
        or np.any(temp <= 0)
    ):
        raise ValueError(
            "Require increasing top-first pressure and positive temperature"
        )
    if (
        frac.ndim != 2
        or frac.shape[0] != len(height)
        or not frac.shape[1]
        or np.any(frac < 0)
        or np.any(frac.sum(axis=1) > 1 + 1e-12)
    ):
        raise ValueError("Invalid gas mole fractions on profile interfaces")
    if (
        cross.ndim != 3
        or cross.shape[0] != len(height) - 1
        or cross.shape[2] != frac.shape[1]
        or not cross.shape[1]
        or np.any(cross < 0)
    ):
        raise ValueError("Gas cross sections require (layer, wavelength, gas)")
    number = frac * (pressure / (BOLTZMANN_CONSTANT * temp))[:, None]
    amount = (number[:-1] + number[1:]) * (-0.5 * np.diff(height)[:, None])
    return np.einsum("lwg,lg->lw", cross, amount)


def molecular_column(
    wavelength_m,
    interface_height_m,
    pressure_pa,
    gas_optical_depth,
    *,
    source="user profile",
):
    """Combine pressure-derived Rayleigh scattering and gas absorption.

    Supply interface pressures in Pa (increasing downward) and gas
    absorption optical depth per layer and wavelength. Use this with local
    sounding/reanalysis profiles or a band-resolved external gas model.
    """
    wave = wavelengths(wavelength_m)
    height = interfaces(interface_height_m)
    pressure = finite_array(pressure_pa, "Interface pressures")
    gas_depth = finite_array(gas_optical_depth, "Gas optical depth")
    if (
        pressure.shape != height.shape
        or np.any(pressure < 0)
        or np.any(np.diff(pressure) <= 0)
    ):
        raise ValueError("Interface pressure must increase toward the ground")
    if gas_depth.shape != (len(height) - 1, len(wave)) or np.any(gas_depth < 0):
        raise ValueError(
            "Gas depth must be nonnegative, with layer and wavelength axes"
        )
    # Hansen & Travis (1974), vertical sea-level molecular scattering depth.
    microns = wave * 1e6
    rayleigh = (
        0.008569
        * microns**-4
        * (1 + 0.0113 * microns**-2 + 0.00013 * microns**-4)
        * (np.diff(pressure) / 101325)[:, None]
    )
    total = rayleigh + gas_depth
    moments = np.zeros((*total.shape, 3))
    moments[..., 0] = 1.0
    moments[..., 2] = 0.1
    return SpectralColumn(
        wave,
        height,
        total,
        rayleigh / total,
        moments,
        (source, "Hansen–Travis Rayleigh", "user interface heights"),
    )


def earth_reference(
    latitude_deg,
    longitude_deg,
    time,
    interface_height_m,
    wavelength_m=None,
    *,
    grid="reduced",
    pressure_pa=None,
    gas_optical_depth=None,
    profile_name="user profile",
):
    """Build an offline AFGL US standard column illuminated by TSIS-1.

    Elevation is the lowest interface. The upper interface receives
    extraterrestrial sunlight, so include the overlying atmosphere when
    interpreting a surface or TOA flux. Native reduced/full REPTRAN grids
    use positive band quadrature. User wavelength grids use containing-band
    effective gas depth, sampled Rayleigh scattering and TSIS illumination.
    Override interface pressure (Pa) or layer gas depths independently.
    Their provenance is recorded under profile_name. The lowest reference
    layer extends hydrostatically to land surfaces down to −500 m.
    """
    if grid not in ("full", "reduced"):
        raise ValueError("Choose a full or reduced gas reference grid")
    position = solar_position(latitude_deg, longitude_deg, time)
    height = interfaces(interface_height_m)
    data = _gas_reference(grid)
    custom = wavelength_m is not None and not np.array_equal(
        wavelength_m, data["wavelength_m"]
    )
    if custom:
        data = _gas_reference("full")
    depth, pressure = _profile_grid(data, height)
    if custom:
        wave = wavelengths(wavelength_m)
        gas_depth, bounds = _gas_band_equivalent(data, depth, wave)
        weights = None
        density = solar_irradiance(wave)
        gas_source = "REPTRAN native band, vertical effective optical depth"
    else:
        wave = data["wavelength_m"]
        weights = data["weight_m"]
        bounds = data["band_bounds_m"]
        bands, idx = np.unique(bounds, axis=0, return_inverse=True)
        energy = _solar_band_energy(bands)
        sampled = solar_irradiance(wave)
        normalization = energy / np.bincount(idx, weights * sampled)
        density = sampled * normalization[idx]
        gas_depth = depth
        gas_source = f"AFGL US standard REPTRAN {grid} quadrature"
    profile_source = "AFGL US standard pressure"
    if pressure_pa is not None:
        pressure = pressure_pa
        profile_source = f"{profile_name} pressure"
    if gas_optical_depth is not None:
        gas_depth = gas_optical_depth
        gas_source = f"{profile_name} gas absorption"
    column = molecular_column(
        wave,
        height,
        pressure,
        gas_depth,
        source=f"{profile_source}; {gas_source}",
    )
    cosine = max(0.0, float(np.cos(np.deg2rad(position.zenith_deg))))
    direct = (
        density / position.distance_au**2
        if cosine > 0
        else np.zeros_like(density)
    )
    source = (
        f"Earth reference v1, TSIS-1 at {position.distance_au:.8f} au, "
        f"{gas_source}; {latitude_deg:g} N, {longitude_deg:g} E, "
        f"{time.astimezone(timezone.utc).isoformat()}"
    )
    sun = SolarInput(
        wave,
        cosine,
        direct,
        np.zeros_like(density),
        weights,
        source,
        None if custom else bounds,
    )
    return EarthReference(
        column,
        sun,
        position,
        bounds,
        (
            "user location and UTC time",
            "user interface heights",
            gas_source,
            profile_source,
            "TSIS-1 HSRS",
            "Hansen–Travis Rayleigh",
        ),
        latitude_deg,
        longitude_deg,
        time,
    )
