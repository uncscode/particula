# Warp DSL types compiled by Warp.
# mypy: disable-error-code="index,operator,valid-type"
"""Warp parameter transforms for optical retrieval."""

import numpy as np
import warp as wp

from particula.optics.gpu._complex import divide, multiply
from particula.optics.gpu.inverse import BulkOpticsAdjoint
from particula.optics.retrieval import EffectiveIndexModel
from particula.optics.state import lorentz_lorenz

_ZERO = wp.constant(wp.float64(0))
_ONE = wp.constant(wp.float64(1))
_TWO = wp.constant(wp.float64(2))
_THIRD = wp.constant(wp.float64(1 / 3))
_NEG_THIRD = wp.constant(wp.float64(-1 / 3))


@wp.func
def _water_ratio(kappa: wp.float64, rh: wp.float64, bend: wp.float64):
    rhs = kappa * rh
    upper = rhs / (_ONE - rh)
    lower = wp.float64(0.0)
    for _ in range(58):
        w = (lower + upper) / _TWO
        curvature = wp.exp(bend * wp.pow(_ONE + w, _NEG_THIRD))
        if w < rhs / (curvature - rh):
            lower = w
        else:
            upper = w
    return (lower + upper) / _TWO


@wp.func_grad(_water_ratio)
def _water_ratio_grad(
    kappa: wp.float64, rh: wp.float64, bend: wp.float64, adj: wp.float64
):
    # Implicit derivative of w(exp(bend/(1+w)^(1/3)) - RH) - kappa*RH.
    water = _water_ratio(kappa, rh, bend)
    temperature_term = wp.pow(_ONE + water, _NEG_THIRD)
    exponent = bend * temperature_term
    curvature = wp.exp(exponent)
    dw = (
        curvature
        - rh
        - (water * curvature * exponent * _THIRD / (_ONE + water))
    )
    wp.adjoint[kappa] += adj * rh / dw
    wp.adjoint[rh] += adj * (kappa + water) / dw
    wp.adjoint[bend] -= adj * water * curvature * temperature_term / dw


@wp.func
def _refractive_index(polar: wp.vec2d):
    one = wp.vec2d(_ONE, _ZERO)
    dielectric = divide(one + _TWO * polar, one - polar)
    magnitude = wp.length(dielectric)  # type: ignore[call-overload]
    if dielectric[0] >= _ZERO:
        real_index = wp.sqrt((magnitude + dielectric[0]) / _TWO)
        return wp.vec2d(real_index, dielectric[1] / (_TWO * real_index))
    imag_index = wp.sqrt((magnitude - dielectric[0]) / _TWO)
    return wp.vec2d(dielectric[1] / (_TWO * imag_index), imag_index)


@wp.kernel
def _effective_population(
    p: wp.array(dtype=wp.float64),
    dry: wp.array2d(dtype=wp.float64),
    relative_humidity: wp.array(dtype=wp.float64),
    sizing_rh: wp.array(dtype=wp.float64),
    kelvin: wp.array(dtype=wp.float64),
    water_polar: wp.array(dtype=wp.vec2d),
    water_ratio: wp.array2d(dtype=wp.float64),
    fixed_kappa: wp.float64,
    fitted_kappa: int,
    humidify: int,
    measured_growth: int,
    radius: wp.array2d(dtype=wp.float64),
    index: wp.array2d(dtype=wp.vec2d),
):
    group, slot = wp.tid()  # type: ignore[misc]
    dry_radius = dry[group, slot]
    dry_index = wp.vec2d(p[0], p[1])
    if humidify == 0:
        radius[group, slot] = dry_radius
        index[group, slot] = dry_index
        return
    kappa = fixed_kappa
    if fitted_kappa > 0:
        kappa = p[2]
    if measured_growth > 0:
        water = water_ratio[group, slot]
    else:
        s = sizing_rh[group]
        initial_water = kappa * s / (wp.exp(kelvin[group] / dry_radius) - s)
        dry_radius = dry_radius / wp.pow(_ONE + initial_water, _THIRD)
        water = _water_ratio(
            kappa, relative_humidity[group], kelvin[group] / dry_radius
        )
    dielectric = multiply(dry_index, dry_index)
    dry_polar = divide(
        dielectric - wp.vec2d(_ONE, _ZERO),
        dielectric + wp.vec2d(_TWO, _ZERO),
    )
    polar = (dry_polar + water * water_polar[group]) / (_ONE + water)
    radius[group, slot] = dry_radius * wp.pow(_ONE + water, _THIRD)
    index[group, slot] = _refractive_index(polar)


@wp.kernel
def _partition_weights(
    p: wp.array(dtype=wp.float64),
    band: wp.array2d(dtype=wp.int32),
    endpoint: wp.array2d(dtype=wp.float64),
    subpopulations: int,
    weight: wp.array2d(dtype=wp.float64),
):
    group, population = wp.tid()  # type: ignore[misc]
    fraction = p[band[group, population // subpopulations]]
    if population % subpopulations == 0:
        weight[group, population] = endpoint[group, population] * fraction
    else:
        weight[group, population] = endpoint[group, population] * (
            _ONE - fraction
        )


class DeviceRetrieval:
    """Bind a small-parameter optical retrieval to an explicit Warp device."""

    def __init__(self, model, shape, wavelength, *, device):
        """Upload coordinates fixed across all evaluations."""
        self._bulk = BulkOpticsAdjoint(shape, wavelength, device=device)
        self._shape = shape
        self._device = self._bulk.device
        self._effective = isinstance(model, EffectiveIndexModel)
        if self._effective:
            zeros = np.zeros(shape[0])
            rh = zeros if model._rh is None else model._rh
            ratio = np.empty((0, 0)) if model._ratio is None else model._ratio
            self._inputs = (
                self._array(model.grid.radius),
                self._array(rh),
                self._array(model._sizing_rh),
                self._array(model.kelvin),
                self._complex_array(lorentz_lorenz(model.water_index)),
                self._array(ratio),
                wp.float64(model._kappa or 0),
                int(len(model.parameters) == 3),
                int(model._rh is not None or model._ratio is not None),
                int(model._ratio is not None),
            )
            self._density = self._array(model.grid.number_density)
        else:
            self._inputs = (
                self._array(model._band, dtype=wp.int32),
                self._array(model._endpoint_density),
                model._subpopulations,
            )
            self._radii = self._array(model._radius)
            self._index = self._complex_array(model._index)

    def _array(self, value, *, dtype=wp.float64, requires_grad=False):
        return wp.array(
            np.ascontiguousarray(value),
            dtype=dtype,
            device=self._device,
            requires_grad=requires_grad,
        )

    def _complex_array(self, value):
        return self._array(
            np.stack((value.real, value.imag), axis=-1),
            dtype=wp.vec2d,
        )

    def _population(self, p, grad):
        options = dict(device=self._device, requires_grad=grad)
        if self._effective:
            radius = wp.empty(self._shape, dtype=wp.float64, **options)
            index = wp.empty(self._shape, dtype=wp.vec2d, **options)
            wp.launch(
                _effective_population,
                self._shape,
                inputs=[p, *self._inputs],
                outputs=[radius, index],
                device=self._device,
            )
            return radius, index, self._density

        weight = wp.empty(self._shape, dtype=wp.float64, **options)
        wp.launch(
            _partition_weights,
            self._shape,
            inputs=[p, *self._inputs],
            outputs=[weight],
            device=self._device,
        )
        return self._radii, self._index, weight

    def evaluate(self, parameters, seeds):
        """Return predictions and optional projected Jacobian rows."""
        grad = seeds is not None
        p = self._array(parameters, requires_grad=grad)
        if not grad:
            return self._bulk.evaluate(
                *self._population(p, False)
            ).numpy(), None

        with wp.Tape() as tape:
            bulk = self._bulk.evaluate(*self._population(p, True), tape=tape)
        jacobian = np.empty((len(seeds), len(parameters)))
        for i, seed in enumerate(seeds):
            tape.backward(grads={bulk: self._array(seed.reshape(bulk.shape))})
            jacobian[i] = p.grad.numpy()
            tape.zero()
        if not np.all(np.isfinite(jacobian)):
            raise ValueError(
                "Optical retrieval produced a nonfinite derivative"
            )
        return bulk.numpy(), jacobian
