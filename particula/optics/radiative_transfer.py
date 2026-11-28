"""Plane-parallel shortwave transport by scalar discrete ordinates.

A matrix exponential solves a thin homogeneous slab. Matrix adding/doubling
avoids propagating growing upward solutions through optically thick layers.
See Hansen & Travis (1974), doi:10.1007/BF00168069, and Stamnes et al.
(1988), doi:10.1364/AO.27.002502. Azimuthal Fourier modes are independent
for the randomly oriented particles and isotropic surface textures used here.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, lpmv

from particula.optics._spectral import require_same_grid
from particula.optics.column import SpectralColumn
from particula.optics.solar import SolarInput
from particula.optics.surface import Surface


@dataclass(frozen=True)
class ShortwaveFlux:
    """Spectral shortwave fluxes at every interface and optional top radiance.

    Attributes:
        wavelength_m: Increasing vacuum wavelengths [m].
        interface_height_m: Top-first interface heights [m].
        direct_down: Direct downward horizontal irradiance density
            [W m⁻² m⁻¹], shape (interface, wavelength).
        diffuse_down: Diffuse downward horizontal irradiance density on
            the same grid and in the same units as direct_down.
        up: Hemispheric upward irradiance density [W m⁻² m⁻¹] with axes
            (interface, wavelength).
        radiance: Optional spectral radiance [W m⁻² sr⁻¹ m⁻¹] leaving the
            top interface toward the requested observer, one per wavelength.
        quadrature_weights_m: Optional integration widths [m] per node.
        sources: Provenance of solar input, surface and layer optics.
        view: Optional (view zenith, solar-relative azimuth) in degrees.
        band_bounds_m: Optional quadrature band edges [m], shape
            (wavelength, 2).
    """

    wavelength_m: np.ndarray
    interface_height_m: np.ndarray
    direct_down: np.ndarray
    diffuse_down: np.ndarray
    up: np.ndarray
    radiance: np.ndarray | None
    quadrature_weights_m: np.ndarray | None
    sources: tuple
    view: tuple | None
    band_bounds_m: np.ndarray | None = None

    @property
    def net_down(self):
        """Net downward irradiance density [W m⁻² m⁻¹] at each interface.

        Returns:
            (ndarray): Direct plus diffuse downward flux minus upward flux,
                shape (interface, wavelength). Positive values carry energy
                downward.
        """
        return self.direct_down + self.diffuse_down - self.up

    @property
    def absorption(self):
        """Absorbed shortwave energy per unit area, time and wavelength.

        Returns:
            (ndarray): Net downward flux at each layer top minus that at its
                bottom [W m⁻² m⁻¹], shape (layer, wavelength). Surface
                absorption is outside the atmospheric layers.
        """
        return -np.diff(self.net_down, axis=0)

    def integrate(self, spectrum, *, band_m=None):
        """Integrate a spectral density over the grid or a selected interval.

        Args:
            spectrum (ArrayLike): Spectral densities with wavelength on the
                final axis. An irradiance input in W m⁻² m⁻¹ integrates to W
                m⁻².
            band_m (tuple[float, float] | None): Optional (lower, upper)
                wavelength limits [m]. Sampled spectra use trapezoids with
                linearly interpolated endpoints. For representative nodes the
                limits must match native band edges, and the integration uses
                the supplied quadrature widths.

        Returns:
            (numpy.floating | ndarray): Integral over wavelength. All leading
                input axes are preserved.

        Raises:
            ValueError: The spectral axis or interval is invalid, a sampled
                grid lacks spectral width, or quadrature band edges are absent
                or incompatible with the requested interval.
        """
        spectrum = np.asarray(spectrum)
        if not spectrum.ndim or spectrum.shape[-1] != self.wavelength_m.size:
            raise ValueError("Spectrum must have a final wavelength axis")
        if band_m is not None:
            low, high = band_m
            if not 0 < low < high:
                raise ValueError("Require increasing positive band bounds")
            if self.quadrature_weights_m is not None:
                bounds = self.band_bounds_m
                if bounds is None:
                    raise ValueError(
                        "Quadrature integration requires known band edges"
                    )
                # Match edges to rounding precision in metres.
                if not (
                    np.any(np.isclose(bounds[:, 0], low, atol=1e-18, rtol=0))
                    and np.any(
                        np.isclose(bounds[:, 1], high, atol=1e-18, rtol=0)
                    )
                ):
                    raise ValueError("Select whole quadrature bands")
                keep = (bounds[:, 0] >= low - 1e-18) & (
                    bounds[:, 1] <= high + 1e-18
                )
                return np.sum(
                    spectrum[..., keep] * self.quadrature_weights_m[keep],
                    axis=-1,
                )
            return _integrate_interval(self.wavelength_m, spectrum, low, high)
        if self.quadrature_weights_m is not None:
            return np.sum(spectrum * self.quadrature_weights_m, axis=-1)
        if self.wavelength_m.size < 2:
            raise ValueError("A single wavelength requires a spectral width")
        return np.trapezoid(spectrum, self.wavelength_m, axis=-1)

    def difference(self, reference):
        """Calculate the signed effect of this case relative to a reference.

        Args:
            reference (ShortwaveFlux): ShortwaveFlux with matching vertical and
                spectral coordinates, integration weights, band edges and view.

        Returns:
            (ShortwaveFlux): Componentwise differences (this case minus
                reference), retaining spectral densities and both source
                records.

        Raises:
            ValueError: Coordinates, integration metadata or views differ.
        """
        for name in (
            "wavelength_m",
            "interface_height_m",
            "quadrature_weights_m",
            "band_bounds_m",
        ):
            require_same_grid(
                getattr(self, name), getattr(reference, name), name
            )
        if self.view != reference.view:
            raise ValueError("Viewing geometries must match")
        return ShortwaveFlux(
            self.wavelength_m,
            self.interface_height_m,
            self.direct_down - reference.direct_down,
            self.diffuse_down - reference.diffuse_down,
            self.up - reference.up,
            None
            if self.radiance is None
            else self.radiance - reference.radiance,
            self.quadrature_weights_m,
            (self.sources, reference.sources),
            self.view,
            self.band_bounds_m,
        )


def _integrate_interval(wave, spectrum, low, high):
    """Integrate a sampled spectrum with linearly interpolated edges."""
    if low < wave[0] or high > wave[-1]:
        raise ValueError("Requested band exceeds the sampled spectrum")
    inside = (wave > low) & (wave < high)
    edges = []
    for x in (low, high):
        j = max(1, np.searchsorted(wave, x))
        f = (x - wave[j - 1]) / (wave[j] - wave[j - 1])
        edges.append(
            (1 - f) * spectrum[..., j - 1 : j] + f * spectrum[..., j : j + 1]
        )
    return np.trapezoid(
        np.concatenate((edges[0], spectrum[..., inside], edges[1]), axis=-1),
        np.r_[low, wave[inside], high],
        axis=-1,
    )


def _phase_basis(mu, order, count):
    degree = np.arange(order, count)[:, None]
    norm = np.exp(
        0.5 * (gammaln(degree - order + 1) - gammaln(degree + order + 1))
    )
    return np.sqrt(2 * degree + 1) * norm * lpmv(order, degree, mu)


def _thin_exponential(generator, depth):
    # Bound the infinity norm on each wavelength batch before Taylor
    # expansion. Once ||A|| <= 1/2 the 16th-order remainder is < 4e-20.
    matrix = generator * depth[:, None, None]
    bound = np.linalg.norm(matrix, ord=np.inf, axis=(-2, -1)).max()
    doubles = max(0, int(np.ceil(np.log2(max(1.0, 2 * bound)))))
    scaled = np.ldexp(matrix, -doubles)
    ident = np.broadcast_to(np.eye(matrix.shape[-1]), matrix.shape)
    result = ident.copy()
    term = ident
    for k in range(1, 17):
        term = (term @ scaled) / k
        result += term
    return result, doubles


def _slab(depth, albedo, moments, basis, mu, weight, mu0):
    """Return reflection, transmission, beam sources and beam transmission."""
    n = mu.size
    signed = np.r_[mu, -mu]
    # The last ordinate in the phase basis is the incident beam. Its source
    # is per unit direct-normal irradiance, hence the 1/(4 pi) factor.
    p = (basis.T[None, :, :] * moments[:, None, :]) @ basis
    matrix = np.zeros((depth.size, 2 * n + 1, 2 * n + 1))
    matrix[:, :-1, :-1] = (
        0.5 * albedo[:, None, None] * p[:, :-1, :-1] * np.tile(weight, 2)
        - np.eye(2 * n)
    ) / signed[:, None]
    matrix[:, :-1, -1] = albedo[:, None] * p[:, :-1, -1] / (4 * np.pi * signed)
    matrix[:, -1, -1] = -1 / mu0
    exp, doubles = _thin_exponential(matrix, depth)
    # Eliminate top upwelling intensity in favor of bottom upwelling.
    # An isotropic medium gives identical diffuse operators from either side.
    outgoing = np.linalg.solve(
        exp[:, n : 2 * n, n : 2 * n],
        -np.concatenate(
            (exp[:, n : 2 * n, :n], exp[:, n : 2 * n, -1:]), axis=-1
        ),
    )
    r, su = outgoing[:, :, :n], outgoing[:, :, -1:]
    t = exp[:, :n, :n] + exp[:, :n, n : 2 * n] @ r
    sd = exp[:, :n, -1:] + exp[:, :n, n : 2 * n] @ su
    beam = np.exp(-np.ldexp(depth, -doubles) / mu0)[:, None, None]
    eye = np.eye(n)
    for _ in range(doubles):
        mid = np.linalg.solve(
            eye - r @ r, np.concatenate((t, sd + r @ (su * beam)), axis=-1)
        )
        tmid, dmid = mid[:, :, :n], mid[:, :, -1:]
        su, sd = su + t @ (r @ dmid + su * beam), t @ dmid + sd * beam
        r, t = r + t @ (r @ tmid), t @ tmid
        beam = beam * beam
    return r, t, su, sd, beam


def _surface_modes(surface, mu, weight, mu0, modes, wave_slice):
    """Compute azimuth moments of the diffuse and direct surface response."""
    n = len(mu)
    count = wave_slice.stop - wave_slice.start
    if surface.model == "lambertian":
        rho = surface.coefficients[0][wave_slice, None, None]
        r = np.broadcast_to(2 * rho * (mu * weight), (count, n, n))
        s = np.broadcast_to(rho * mu0 / np.pi, (count, n, 1))
        return [(r, s)] + [(np.zeros_like(r), np.zeros_like(s))] * modes
    # Midpoints form a cosine quadrature over [0, pi] for even BRDFs.
    phi = (
        (np.arange(max(192, 4 * (modes + 1))) + 0.5)
        * np.pi
        / max(192, 4 * (modes + 1))
    )
    incident = np.r_[mu, mu0]
    # Evaluate only the current spectral batch to bound working memory.
    coeffs = tuple(
        x[wave_slice] if isinstance(x, np.ndarray) else x
        for x in surface.coefficients
    )
    part = Surface(
        surface.wavelength_m[wave_slice], surface.model, coeffs, surface.source
    )
    brdf = part.brdf(incident[None, :, None], mu[:, None, None], np.pi - phi)
    out = []
    for order in range(modes + 1):
        moment = brdf @ (np.cos(order * phi) / len(phi))
        out.append(
            (
                moment[:, :, :-1] * (2 * np.pi * mu * weight),
                moment[:, :, -1:] * mu0,
            )
        )
    return out


def _mode_interfaces(slabs, surface_boundary, diffuse, direct, leaving, n):
    # For every interface store the incident direct-normal beam and the
    # diffuse relation U = R D + S, including any independent ocean source.
    beams = [direct[:, None, None]]
    for _, _, _, _, transmission in slabs:
        beams.append(beams[-1] * transmission)
    r_ground, s_ground = surface_boundary
    s_ground = s_ground * beams[-1] + leaving
    relations = []
    for slab, incident in zip(
        reversed(slabs), reversed(beams[:-1]), strict=True
    ):
        r, t, su, sd, _ = slab
        mid = np.linalg.solve(
            np.eye(n) - r @ r_ground,
            np.concatenate((t, sd * incident + r @ s_ground), axis=-1),
        )
        trans, source = mid[:, :, :n], mid[:, :, -1:]
        relations.append((trans, source, r_ground, s_ground))
        s_ground = su * incident + t @ (r_ground @ source + s_ground)
        r_ground = r + t @ (r_ground @ trans)
    d = np.broadcast_to(diffuse[:, None, None] / np.pi, (len(diffuse), n, 1))
    down, up = [d[..., 0]], [(r_ground @ d + s_ground)[..., 0]]
    for trans, source, r_ground, s_ground in reversed(relations):
        d = trans @ d + source
        down.append(d[..., 0])
        up.append((r_ground @ d + s_ground)[..., 0])
    return np.stack(down), np.stack(up)


def _transport_optics(column, index, streams, delta_m):
    depth = column.optical_depth[:, index]
    albedo = column.single_scattering_albedo[:, index]
    moments = column.phase_moments[:, index]
    limit = min(2 * streams, moments.shape[-1])
    if delta_m and moments.shape[-1] > limit:
        f = np.clip(moments[..., limit], 0, 1)
        den = 1 - albedo * f
        depth = depth * den
        albedo = np.divide(
            albedo * (1 - f), den, out=np.zeros_like(albedo), where=den > 0
        )
        moments = np.divide(
            moments[..., :limit] - f[..., None],
            (1 - f)[..., None],
            out=np.zeros_like(moments[..., :limit]),
            where=f[..., None] < 1,
        )
    return depth, albedo, moments[..., :limit]


def solve_shortwave(
    column: SpectralColumn,
    sun: SolarInput,
    surface: Surface,
    *,
    streams=8,
    view_zenith_deg=None,
    relative_azimuth_deg=0.0,
    fourier_order=None,
    delta_m=True,
):
    """Solve scalar plane-parallel shortwave transfer through a layered column.

    The discrete-ordinate solver includes multiple scattering and surface
    reflection. Returned direct irradiance uses the unscaled extinction depth.
    With delta-M scaling, unresolved forward-scattered light contributes to
    diffuse downward irradiance.

    Args:
        column: Top-first layer optical properties on a common wavelength grid.
        sun: Spectral direct-normal and diffuse-horizontal irradiance densities
            [W m⁻² m⁻¹] at the upper interface, with the solar zenith cosine.
        surface: Reflecting lower boundary on the column wavelength grid.
        streams (int): Number of Gauss–Legendre ordinates per hemisphere, 2–32.
        view_zenith_deg (float | None): Optional angle [degrees] from the
            upward normal to the observer in [0, 90). A zero-weight ordinate
            resolves this view.
        relative_azimuth_deg (float): Azimuth [degrees] between surface-to-Sun
            and surface-to-observer directions. At 180 the observer lies across
            the surface normal from the Sun, in the specular plane.
        fourier_order (int | None): Highest azimuthal harmonic for a
            directional solve, 0–64. The default is 2 * streams - 1. Flux alone
            uses order zero.
        delta_m (bool): Apply delta-M scaling to represent a forward lobe if a
            phase moment beyond the retained orders is available.

    Returns:
        (ShortwaveFlux): Spectral irradiance densities [W m⁻² m⁻¹] at each
            interface and optionally the spectral radiance leaving the column
            top [W m⁻² sr⁻¹ m⁻¹] in the requested viewing direction.

    Raises:
        ValueError: Spectral grids differ or the stream count, viewing
            geometry or Fourier order is invalid.
    """
    require_same_grid(
        column.wavelength_m, sun.wavelength_m, "Solar wavelengths"
    )
    require_same_grid(
        column.wavelength_m, surface.wavelength_m, "Surface wavelengths"
    )
    if not isinstance(streams, (int, np.integer)) or not 2 <= streams <= 32:
        raise ValueError("Require 2–32 streams per hemisphere")
    if not np.isfinite(relative_azimuth_deg):
        raise ValueError("Relative azimuth must be finite")
    if view_zenith_deg is not None and not 0 <= view_zenith_deg < 90:
        raise ValueError("View zenith must be in [0, 90) degrees")
    if fourier_order is not None and (
        not isinstance(fourier_order, (int, np.integer))
        or not 0 <= fourier_order <= 64
    ):
        raise ValueError("Fourier order must be an integer from 0 to 64")
    mu, weight = leggauss(streams)
    mu, weight = (mu + 1) / 2, weight / 2
    if view_zenith_deg is not None:
        mu, weight = (
            np.r_[mu, np.cos(np.deg2rad(view_zenith_deg))],
            np.r_[weight, 0.0],
        )
    modes = (
        0
        if view_zenith_deg is None
        else (2 * streams - 1 if fourier_order is None else fourier_order)
    )
    mu0 = sun.mu0 if sun.mu0 > 0 else 1.0
    beam = (
        sun.direct_normal if sun.mu0 > 0 else np.zeros_like(sun.direct_normal)
    )
    nlayer, nwave = column.optical_depth.shape
    cumulative = np.vstack(
        (np.zeros(nwave), column.optical_depth.cumsum(axis=0))
    )
    direct = beam * mu0 * np.exp(-cumulative / mu0)
    diffuse = np.zeros_like(direct)
    up = np.zeros_like(direct)
    radiance = None if view_zenith_deg is None else np.zeros(nwave)
    max_moments = min(2 * streams, column.phase_moments.shape[-1])
    bases = [
        _phase_basis(np.r_[mu, -mu, mu0], m, max_moments)
        for m in range(modes + 1)
    ]
    # Band chunks also bound allocations for angular BRDF quadrature.
    for start in range(0, nwave, 16):
        index = slice(start, min(start + 16, nwave))
        props = _transport_optics(column, index, streams, delta_m)
        depth, albedo, moments = props
        surface_modes = _surface_modes(surface, mu, weight, mu0, modes, index)
        for m, boundary in enumerate(surface_modes):
            slabs = [
                _slab(
                    depth[k],
                    albedo[k],
                    moments[k, :, m:],
                    bases[m],
                    mu,
                    weight,
                    mu0,
                )
                for k in range(nlayer)
            ]
            d, u = _mode_interfaces(
                slabs,
                boundary,
                sun.diffuse_horizontal[index] * (m == 0),
                beam[index],
                (
                    surface.leaving_radiance[index, None, None]
                    if m == 0 and surface.leaving_radiance is not None
                    else 0.0
                ),
                len(mu),
            )
            if m == 0:
                scaled_beam = (
                    beam[index]
                    * mu0
                    * np.exp(
                        -np.vstack(
                            (np.zeros(depth.shape[1]), depth.cumsum(axis=0))
                        )
                        / mu0
                    )
                )
                diffuse[:, index] = d @ (2 * np.pi * mu * weight) + (
                    scaled_beam - direct[:, index]
                )
                up[:, index] = u @ (2 * np.pi * mu * weight)
            if radiance is not None:
                factor = (
                    1
                    if m == 0
                    else 2
                    * np.cos(m * (np.pi - np.deg2rad(relative_azimuth_deg)))
                )
                radiance[index] += u[0, :, -1] * factor
    return ShortwaveFlux(
        column.wavelength_m,
        column.interface_height_m,
        direct,
        diffuse,
        up,
        radiance,
        sun.quadrature_weights_m,
        (sun.source, surface.source, *column.sources),
        None if radiance is None else (view_zenith_deg, relative_azimuth_deg),
        sun.band_bounds_m,
    )


def sensor_band_radiance(result, response):
    """Calculate the energy-response-weighted mean top-of-column radiance.

    Args:
        result (ShortwaveFlux): ShortwaveFlux with a directional radiance
            spectrum.
        response (ArrayLike): Nonnegative energy response, one value per solved
            wavelength. Its normalization cancels in the mean. For a
            photon-counting detector supply the calibrated energy response or
            include the instrument's required photon weighting.

    Returns:
        (numpy.floating): Band-mean radiance density [W m⁻² sr⁻¹ m⁻¹], the
            integral of radiance times response divided by the integrated
            response.

    Raises:
        ValueError: Radiance is absent, the response has invalid shape,
            sign or values, or the integrated response lacks positive width.
    """
    if result.radiance is None:
        raise ValueError("A sensor band requires directional radiance")
    response = np.asarray(response, dtype=float)
    if (
        response.shape != result.wavelength_m.shape
        or not np.all(np.isfinite(response))
        or np.any(response < 0)
    ):
        raise ValueError("Require a nonnegative response on the solver grid")
    width = result.integrate(response)
    if width <= 0:
        raise ValueError("Sensor response must have positive integral")
    return result.integrate(response * result.radiance) / width
