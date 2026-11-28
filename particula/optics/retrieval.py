"""Optical closure with prescribed size distributions and measurement errors."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from particula.optics.humidity import hydrate_index, subsaturated_growth
from particula.optics.mie import get_mie_efficiencies
from particula.optics.state import mix_refractive_index
from particula.particles.properties.number_spectrum import (
    get_bin_number_density,
)

CHANNELS = ("extinction", "scattering", "absorption")


def _real(value):
    if np.iscomplexobj(value):
        raise ValueError("Expected real optical coordinates")
    result = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError("Optical coordinates must be finite")
    return result


def _freeze(value):
    frozen = np.array(value, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


def _physical_index(value):
    result = np.asarray(value, dtype=np.complex128)
    if (
        np.any(~np.isfinite(result))
        or np.any(result.real <= 0)
        or np.any(result.imag < 0)
    ):
        raise ValueError("Indices must have n>0 and k>=0")
    return result


@dataclass(frozen=True)
class OpticalGrid:
    """One particle spectrum per optical wavelength and measurement window.

    Radius and number_density have shape (group, bin). A one-dimensional
    spectrum broadcasts over groups. Radii and vacuum wavelengths are in
    metres and number densities are integrated counts per bin in m⁻³.
    Groups can repeat a wavelength at different humidity or time conditions.
    """

    radius: np.ndarray
    number_density: np.ndarray
    wavelength: np.ndarray

    def __post_init__(self):
        """Validate and detach fixed measurement coordinates."""
        wavelength = np.atleast_1d(_real(self.wavelength))
        if (
            wavelength.ndim != 1
            or not wavelength.size
            or np.any(wavelength <= 0)
        ):
            raise ValueError("Require positive vacuum wavelengths")
        radius, density = np.broadcast_arrays(
            _real(self.radius), _real(self.number_density)
        )
        slots = radius.shape[-1] if radius.ndim else 0
        if not slots or np.any(radius <= 0) or np.any(density < 0):
            raise ValueError("Require positive radii and nonnegative counts")
        shape = (wavelength.size, slots)
        for name, value in (("radius", radius), ("number_density", density)):
            object.__setattr__(
                self, name, _freeze(np.broadcast_to(value, shape))
            )
        object.__setattr__(self, "wavelength", _freeze(wavelength))

    @classmethod
    def from_diameter_bins(cls, edges, values, wavelength, *, measure):
        """Build a log-midpoint grid from mobility-diameter edges in metres.

        Use `bin_density` for number per m³ already integrated across the
        bin or `dN_dlog10D` for number per m³ per decade. The spectrum
        must already use the same volume basis as the optical coefficients.
        """
        edges, values = _real(edges), _real(values)
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(edges <= 0)
            or np.any(np.diff(edges) <= 0)
            or values.ndim == 0
            or values.shape[-1] != edges.size - 1
        ):
            raise ValueError("Diameter edges and size-bin values do not match")
        log_edges = np.log10(edges)
        if measure == "dN_dlog10D":
            values = get_bin_number_density(values, edges)
        elif measure != "bin_density":
            raise ValueError("Unknown size spectrum measure")
        centres = 10 ** ((log_edges[:-1] + log_edges[1:]) / 2)
        return cls(centres / 2, values, wavelength)


def _bulk(radius, index, density, wave):
    q = get_mie_efficiencies(radius, wave[:, None], index)
    area_density = np.pi * radius**2 * density
    return np.stack(
        [np.sum(getattr(q, key) * area_density, axis=-1) for key in CHANNELS],
        axis=-1,
    )


class _OpticalModel:
    def _setup(self, wavelength, shape, backend):
        self._device = None
        if backend != "numpy":
            from particula.optics.gpu.retrieval import DeviceRetrieval

            self._device = DeviceRetrieval(
                self, shape, wavelength, device=backend
            )

    def predictions(self, parameters):
        """Return bulk extinction, scattering and absorption [m⁻¹]."""
        return self._eval(parameters, None)[0]

    def _eval(self, parameters, seeds):
        values = self._parameters(parameters)
        if self._device is not None:
            return self._device.evaluate(values, seeds)
        if seeds is not None:
            raise ValueError("Optical model autodiff requires a Warp device")
        return _bulk(*self._population(values), self.wavelength), None


def _humidity(value, groups):
    coordinate = _freeze(np.broadcast_to(_real(value), (groups,)))
    if np.any((coordinate < 0) | (coordinate >= 1)):
        raise ValueError("Relative humidity must be in [0,1)")
    return coordinate


def _measured_water(growth_factor, shape, kappa):
    factor = np.broadcast_to(_real(growth_factor), shape)
    if kappa is not None or np.any(factor < 1):
        raise ValueError("Supply growth factors >=1 in place of kappa")
    with np.errstate(over="ignore"):
        ratio = factor**3 - 1
    if np.any(~np.isfinite(ratio)):
        raise ValueError("Growth factor is outside the usable range")
    return _freeze(ratio)


class EffectiveIndexModel(_OpticalModel):
    """Fit a common complex index, with optional water uptake.

    With wet sizes use the default configuration for an index at the optical
    RH. With dry sizes supply relative_humidity and either a fixed kappa or
    leave kappa open for fitting. Alternatively supply a measured diameter
    growth factor to determine water volume directly. Supply sizing RH if
    the measured distribution retains water at a known equilibrium RH.
    This model shares one dry index across all wavelengths and RH groups.
    """

    def __init__(
        self,
        grid,
        *,
        relative_humidity=None,
        sizing_relative_humidity=None,
        kappa=None,
        growth_factor=None,
        kelvin_length=1.05e-9,
        water_index=1.333,
        device="numpy",
    ):
        """Prepare the measured grid and a CPU or explicit Warp backend."""
        self.grid = grid
        self.wavelength = grid.wavelength
        groups, _ = grid.radius.shape
        self.water_index = _freeze(
            np.broadcast_to(_physical_index(water_index), (groups,))
        )
        self.kelvin = _freeze(np.broadcast_to(_real(kelvin_length), (groups,)))
        self._rh = None
        self._ratio = None
        self._kappa = kappa
        if np.any(self.kelvin < 0):
            raise ValueError("Kelvin coefficient must be nonnegative")
        if relative_humidity is not None:
            self._rh = _humidity(relative_humidity, groups)
        if growth_factor is not None:
            self._ratio = _measured_water(
                growth_factor, grid.radius.shape, kappa
            )
        elif kappa is not None:
            if self._rh is None or _real(kappa).ndim or kappa < 0:
                raise ValueError(
                    "A fixed nonnegative kappa requires optical RH"
                )
        self._sizing_rh = np.zeros(groups)
        if sizing_relative_humidity is not None:
            if self._rh is None or self._ratio is not None:
                raise ValueError("Sizing RH requires the kappa growth model")
            self._sizing_rh = _freeze(
                np.broadcast_to(
                    _real(sizing_relative_humidity),
                    (groups,),
                )
            )
            if np.any((self._sizing_rh < 0) | (self._sizing_rh >= 1)):
                raise ValueError("Sizing RH must be in [0,1)")
        self.parameters = (
            ("n", "k", "kappa")
            if (self._rh is not None and self._ratio is None and kappa is None)
            else ("n", "k")
        )
        self._setup(self.wavelength, grid.radius.shape, device)

    def _parameters(self, values):
        p = _real(values)
        if p.shape != (len(self.parameters),) or p[0] <= 0 or np.any(p[1:] < 0):
            raise ValueError("Require n>0 and nonnegative k and optional kappa")
        return p

    def _population(self, values):
        radius = self.grid.radius
        dry_index = values[0] + 1j * values[1]
        index = dry_index
        if self._ratio is not None:
            ratio = self._ratio
            radius = radius * np.cbrt(1 + ratio)
        elif self._rh is not None:
            kappa = values[2] if self._kappa is None else self._kappa
            measured_rh = self._sizing_rh[:, None]
            kelvin = self.kelvin[:, None]
            with np.errstate(over="ignore"):
                measured_water = (
                    kappa
                    * measured_rh
                    / (np.exp(kelvin / radius) - measured_rh)
                )
            dry_radius = radius / np.cbrt(1 + measured_water)
            radius, ratio = subsaturated_growth(
                dry_radius, self._rh[:, None], kappa, kelvin
            )
        else:
            return radius, index, self.grid.number_density
        index = hydrate_index(dry_index, self.water_index[:, None], ratio)
        return radius, index, self.grid.number_density


class MixingStateModel(_OpticalModel):
    """Prescribed composition with internal/external number fractions by size.

    Each dry bin has a monodisperse internal population and size-matched
    pure-species external populations. The endpoints have the same dry
    number density and per-bin species inventories. Volume fractions may
    be uniform across bins or specified per group and bin.
    """

    def __init__(
        self,
        grid,
        *,
        volume_fractions,
        species_index,
        species_kappa,
        relative_humidity,
        water_index=1.333,
        kelvin_length=1.05e-9,
        dry_boundary=None,
        device="numpy",
    ):
        """Prepare optics inputs for both endpoint populations."""
        groups, slots = grid.radius.shape
        kappa = _real(species_kappa)
        if kappa.ndim != 1 or kappa.size < 2 or np.any(kappa < 0):
            raise ValueError(
                "At least two species with nonnegative kappa are required"
            )
        species = kappa.size
        indices = np.broadcast_to(
            _physical_index(species_index), (groups, species)
        )
        fraction = np.broadcast_to(
            _real(volume_fractions), (groups, slots, species)
        )
        internal_index = mix_refractive_index(indices[:, None, :], fraction)
        internal_kappa = np.sum(fraction * kappa, axis=-1)
        endpoint_index = np.concatenate(
            (
                internal_index[..., None],
                np.broadcast_to(indices[:, None, :], fraction.shape),
            ),
            axis=-1,
        )
        endpoint_kappa = np.concatenate(
            (internal_kappa[..., None], np.broadcast_to(kappa, fraction.shape)),
            axis=-1,
        )
        rh = np.broadcast_to(_real(relative_humidity), (groups,))
        kelvin = np.broadcast_to(_real(kelvin_length), (groups,))
        water = np.broadcast_to(_physical_index(water_index), (groups,))
        wet, ratio = subsaturated_growth(
            grid.radius[..., None],
            rh[:, None, None],
            endpoint_kappa,
            kelvin[:, None, None],
        )
        wet_index = hydrate_index(endpoint_index, water[:, None, None], ratio)
        self.grid = grid
        self.wavelength = grid.wavelength
        self._subpopulations = species + 1
        shape = (groups, slots * self._subpopulations)
        self._radius = _freeze(wet.reshape(shape))
        self._index = _freeze(wet_index.reshape(shape))
        self._endpoint_density = _freeze(
            (
                grid.number_density[..., None]
                * np.concatenate(
                    (np.ones_like(fraction[..., :1]), fraction), axis=-1
                )
            ).reshape(shape)
        )
        self._band = _freeze(
            np.zeros(grid.radius.shape, dtype=np.int32)
            if dry_boundary is None
            else (2 * grid.radius >= dry_boundary).astype(np.int32)
        )
        self.parameters = ("internal_fraction",)
        if dry_boundary is not None:
            if (
                not np.isfinite(dry_boundary)
                or dry_boundary <= 0
                or (len(np.unique(self._band[grid.number_density > 0])) < 2)
            ):
                raise ValueError(
                    "The dry diameter boundary must split occupied bins"
                )
            self.parameters = (
                "internal_fraction_small",
                "internal_fraction_large",
            )
        self._setup(self.wavelength, shape, device)

    def _parameters(self, values):
        p = _real(values)
        if p.shape != (len(self.parameters),) or np.any((p < 0) | (p > 1)):
            raise ValueError("Internal fractions must lie between zero and one")
        return p

    def _population(self, values):
        f = values[self._band]
        multiplier = np.broadcast_to(
            (1 - f)[..., None],
            (*f.shape, self._subpopulations),
        ).copy()
        multiplier[..., 0] = f
        density = self._endpoint_density * multiplier.reshape(
            self._radius.shape
        )
        return self._radius, self._index, density


def dry_volume_fractions(mass, density):
    """Convert measured dry species mass and density to volume fractions."""
    masses, densities = _real(mass), _real(density)
    if np.any(masses < 0) or np.any(densities <= 0):
        raise ValueError("Require nonnegative masses and positive densities")
    volume = masses / densities
    total = volume.sum(axis=-1, keepdims=True)
    if np.any(total <= 0) or np.any(~np.isfinite(total)):
        raise ValueError(
            "Each dry composition must have finite positive volume"
        )
    return volume / total


class OpticalObservations:
    """Calibrated measurements projected from (group, E/S/A) coefficients.

    Response has shape (observation, group, 3). Use it to select a channel
    or average predicted spectra across measured time or spectral windows.
    Use only measured channels with an independent uncertainty model.
    Observations and standard deviations have units m⁻¹. Covariance entries
    have units m⁻². Include environmental and sizing uncertainty through
    sensitivity cases or a supplied covariance.
    """

    def __init__(
        self, response, values, *, standard_deviation=None, covariance=None
    ):
        """Validate the response and factor its measurement covariance."""
        response, values = _real(response), _real(values)
        if (
            response.ndim != 3
            or response.shape[2] != len(CHANNELS)
            or not response.shape[0]
            or not response.shape[1]
            or values.shape != response.shape[:1]
            or np.any(~np.any(response, axis=(1, 2)))
        ):
            raise ValueError(
                "Require one nonzero response per optical observation"
            )
        if (standard_deviation is None) == (covariance is None):
            raise ValueError("Supply standard deviations or covariance")
        if covariance is None:
            sigma = np.broadcast_to(_real(standard_deviation), values.shape)
            if np.any(sigma <= 0):
                raise ValueError("Standard deviations must be positive")
            factor = np.diag(sigma)
        else:
            cov = _real(covariance)
            if cov.shape != (values.size, values.size) or not np.allclose(
                cov,
                cov.T,
                rtol=1e-12,
                atol=0,
            ):
                raise ValueError("Covariance must be symmetric on observations")
            try:
                factor = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as err:
                raise ValueError(
                    "Covariance must be positive definite"
                ) from err
        self.response = _freeze(response)
        self.values = _freeze(values)
        self._factor = _freeze(factor)

    @classmethod
    def channels(cls, groups, selectors, values, **uncertainty):
        """Construct observations from (group index, channel name) selectors."""
        selectors = tuple(selectors)
        response = np.zeros((len(selectors), groups, len(CHANNELS)))
        for row, (group, channel) in enumerate(selectors):
            if (
                not isinstance(group, (int, np.integer))
                or not 0 <= group < groups
            ):
                raise ValueError(
                    "Observation group is outside the optical grid"
                )
            response[row, group, CHANNELS.index(channel)] = 1
        return cls(response, values, **uncertainty)

    def project(self, coefficients):
        """Apply the calibrated linear response in measurement coordinates."""
        coefficients = np.asarray(coefficients)
        if coefficients.shape != self.response.shape[1:]:
            raise ValueError("Optical coefficients and response grid differ")
        return np.einsum("ogc,gc->o", self.response, coefficients)

    def _whiten(self, residual):
        return solve_triangular(self._factor, residual, lower=True)


@dataclass(frozen=True)
class OpticalFit:
    """Best bounded fit, with sensitivity diagnostics in measurement space."""

    names: tuple[str, ...]
    parameters: np.ndarray
    predictions: np.ndarray
    residual: np.ndarray
    whitened_residual: np.ndarray
    sensitivity: np.ndarray
    singular_values: np.ndarray
    active_bounds: np.ndarray
    chi_square: float
    success: bool
    forward_evaluations: int
    backward_evaluations: int


class _Residual:
    def __init__(self, model, observations, method):
        if model.wavelength.size != observations.response.shape[1]:
            raise ValueError(
                "Observation and model have different group counts"
            )
        if method == "autodiff" and model._device is None:
            raise ValueError("Use a Warp device for optical autodiff")
        self.model = model
        self.observations = observations
        self.autodiff = method == "autodiff"
        self.point = None
        self.values = None
        self.jacobian = None
        self.forwards = 0
        self.backwards = 0

    def calculate(self, point, *, jacobian=False):
        if np.array_equal(point, self.point) and (
            not jacobian or self.jacobian is not None
        ):
            return
        seeds = self.observations.response if jacobian else None
        bulk, derivative = self.model._eval(point, seeds)
        self.forwards += 1
        if jacobian:
            self.backwards += len(seeds)
        self.point = point.copy()
        self.values = self.observations.project(bulk)
        self.jacobian = (
            self.observations._whiten(derivative) if jacobian else None
        )

    def residual(self, point):
        self.calculate(point)
        return self.observations._whiten(self.values - self.observations.values)

    def derivative(self, point):
        self.calculate(point, jacobian=True)
        return self.jacobian


def _fit_inputs(model, bounds, starts, method):
    if method not in ("autodiff", "finite_difference"):
        raise ValueError("Choose autodiff or finite_difference")
    n = len(model.parameters)
    bounds = _real(bounds)
    starts = np.atleast_2d(_real(starts))
    if (
        bounds.shape != (2, n)
        or not np.all(bounds[1] > bounds[0])
        or starts.ndim != 2
        or starts.shape[1] != n
        or not starts.shape[0]
        or np.any(starts < bounds[0])
        or np.any(starts > bounds[1])
    ):
        raise ValueError("Provide ordered bounds and feasible starts")
    for row in bounds:
        model._parameters(row)
    return bounds, starts


def fit_optical(
    model,
    observations,
    *,
    bounds,
    starts,
    method="autodiff",
    max_nfev=None,
):
    """Fit optically constrained parameters from one or more bounded starts.

    Sensitivity is the whitened Jacobian per parameter in its native units.
    Singular values include structural zeros when there are fewer independent
    observations than unknowns. Inspect profiles and held-out predictions
    before treating a least-squares solution as a unique retrieval.
    """
    bounds, starts = _fit_inputs(model, bounds, starts, method)
    evaluate = _Residual(model, observations, method)
    jac = evaluate.derivative if evaluate.autodiff else "3-point"
    results = [
        least_squares(
            evaluate.residual,
            start,
            bounds=tuple(bounds),
            jac=jac,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=max_nfev,
        )
        for start in starts
    ]
    successful = [result for result in results if result.success]
    best = min(successful or results, key=lambda result: result.cost)
    evaluate.calculate(best.x)
    residual = evaluate.values - observations.values
    singular = np.linalg.svd(best.jac, compute_uv=False)
    singular = np.pad(singular, (0, len(model.parameters) - singular.size))
    return OpticalFit(
        model.parameters,
        best.x,
        evaluate.values.copy(),
        residual,
        observations._whiten(residual),
        best.jac,
        singular,
        best.active_mask,
        2 * best.cost,
        best.success,
        evaluate.forwards,
        evaluate.backwards,
    )


def profile_optical(
    model,
    observations,
    fit,
    parameter,
    coordinates,
    *,
    bounds,
    method="autodiff",
    delta_chi_square=3.841459,
):
    """Profile a fitted parameter by optimizing all others at fixed values.

    Returns the grid, profiled chi square, and the intervals resolved by the
    grid and threshold. A component touching either search bound is limited
    by that bound. The default delta is the asymptotic 95% cutoff for one
    parameter under a regular Gaussian measurement model.
    """
    bounds, _ = _fit_inputs(model, bounds, fit.parameters, method)
    position = model.parameters.index(parameter)
    grid = _real(coordinates)
    if (
        grid.ndim != 1
        or grid.size < 2
        or np.any(np.diff(grid) <= 0)
        or grid[0] < bounds[0, position]
        or grid[-1] > bounds[1, position]
        or not np.isfinite(delta_chi_square)
        or delta_chi_square <= 0
    ):
        raise ValueError("Profile requires increasing feasible coordinates")
    free = np.arange(len(model.parameters)) != position
    evaluate = _Residual(model, observations, method)
    point = fit.parameters.copy()
    cost = np.empty(grid.size)
    for i, fixed in enumerate(grid):
        point[position] = fixed

        def residual(values):
            point[free] = values
            return evaluate.residual(point)

        def jacobian(values):
            point[free] = values
            return evaluate.derivative(point)[:, free]

        if np.any(free):
            result = least_squares(
                residual,
                fit.parameters[free],
                bounds=tuple(bounds[:, free]),
                jac=jacobian if evaluate.autodiff else "3-point",
            )
            if not result.success:
                raise RuntimeError(
                    "Optical profile nuisance optimization failed"
                )
            cost[i] = 2 * result.cost
        else:
            vector = residual(np.empty(0))
            cost[i] = vector @ vector

    cutoff = min(fit.chi_square, float(np.min(cost))) + delta_chi_square
    accepted = cost <= cutoff
    starts = np.flatnonzero(accepted & ~np.r_[False, accepted[:-1]])
    ends = np.flatnonzero(accepted & ~np.r_[accepted[1:], False])
    intervals = []
    for low, high in zip(starts, ends, strict=True):
        left = (
            grid[low]
            if low == 0
            else np.interp(
                cutoff,
                cost[low - 1 : low + 1][::-1],
                grid[low - 1 : low + 1][::-1],
            )
        )
        right = (
            grid[high]
            if high == grid.size - 1
            else np.interp(
                cutoff,
                cost[high : high + 2],
                grid[high : high + 2],
            )
        )
        intervals.append((float(left), float(right)))
    return grid.copy(), cost, tuple(intervals)
