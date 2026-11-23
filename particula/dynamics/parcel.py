"""Closed, warm-cloud parcel with one soluble, size-resolved aerosol mode.

Mixing ratios use kg per kg dry air. Hydrostatic balance includes suspended
liquid loading. The dilute first law uses constant dry-air heat capacity and
latent heat, and neglects condensate volume, dry aerosol loading, sensible
heat of water, fallout and entrainment. Diffusion and thermal resistances act
in series (continuum growth with unit ventilation). See Rothenberg & Wang
(2016), doi:10.1175/JAS-D-15-0223.1, and Petters & Kreidenweis (2007),
doi:10.5194/acp-7-1961-2007. The concrete runner operates on parcel counts
and physical volume. Its binned number densities are counts divided by volume.
"""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from particula.aerosol import Aerosol
from particula.dynamics.multivapor_parcel import (
    MultiVaporParcelInitialState,
    MultiVaporParcelTrajectory,
    cpu_multivapor_reference,
)
from particula.gas.environment_data import EnvironmentData
from particula.gas.gas_data import GasData
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.particles.particle_data import ParticleData
from particula.util.constants import (
    GAS_CONSTANT,
    MOLECULAR_WEIGHT_AIR,
    STANDARD_GRAVITY,
)

RD = GAS_CONSTANT / MOLECULAR_WEIGHT_AIR  # J/(kg dry air K)
RV = GAS_CONSTANT / 0.01801528  # J/(kg water vapor K)
CP = 1005.0  # J/(kg dry air K)
LATENT = 2.5e6  # J/kg
RHO_W = 1000.0  # kg/m³
SIGMA = 0.072  # N/m
WATER_FACTOR = 4 * np.pi / 3 * RHO_W
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParcelInitialState:
    """Initial physical state before equilibration at subsaturated humidity.

    Attributes:
        dry_radius: Positive dry radius of each class (m).
        number: Nonnegative represented particle count per class in the parcel.
        kappa: Positive dimensionless hygroscopicity shared by the classes.
        pressure: Initial pressure (Pa).
        temperature: Initial temperature (K).
        vapor: Initial vapor mixing ratio (kg/kg dry air).
        dry_air_mass: Fixed dry air inventory (kg).
        height: Initial altitude (m).
    """

    dry_radius: FloatArray
    number: FloatArray
    kappa: float
    pressure: float
    temperature: float
    vapor: float
    dry_air_mass: float = 1.0
    height: float = 0.0


@dataclass(frozen=True)
class _ParcelAdaptiveControls:
    """Shared local error tolerance and adaptive trial limits."""

    rtol: float = 1e-5
    min_step: float = 1e-9
    max_trials: int = 100_000

    def __post_init__(self):
        """Reject invalid controller settings before device allocation."""
        if not np.isfinite(self.rtol) or not 0 < self.rtol < 1:
            raise ValueError("parcel rtol must be between zero and one")
        if not np.isfinite(self.min_step) or self.min_step <= 0:
            raise ValueError("parcel min_step must be positive and finite")
        if (
            isinstance(self.max_trials, (bool, np.bool_))
            or not isinstance(self.max_trials, (int, np.integer))
            or self.max_trials < 1
        ):
            raise ValueError("parcel max_trials must be a positive integer")


@dataclass(frozen=True)
class ParcelRK23(_ParcelAdaptiveControls):
    """Adaptive Warp controls for the embedded Bogacki–Shampine 3(2) pair.

    rtol controls pressure, each class's liquid volume, vapor and temperature.
    The relative scales include floors of 1e-8 in normalized pressure and
    liquid/dry class volume, and 1e-12 kg/kg dry air for vapor.
    min_step is the smallest permitted retry in seconds. max_trials bounds
    all evaluated trials, including unused speculative Warp trials. A linearized
    droplet relaxation limit also protects sensitivities near equilibrium.
    """


@dataclass(frozen=True)
class ParcelROS3(_ParcelAdaptiveControls):
    """Adaptive third-order Rosenbrock solver with a second-order estimator.

    Uses the coupled parcel Jacobian and explicit time derivative at each
    trial. rtol, min_step (s) and max_trials use the ParcelRK23 scales.
    """


@dataclass(frozen=True)
class ParcelStepStatistics:
    """Controller counts and smallest/largest accepted timestep (s).

    evaluated_trials includes speculative candidates discarded after the
    controller changes step size. rejected_steps counts controller rejections.
    """

    accepted_steps: int
    rejected_steps: int
    minimum_step: float
    maximum_step: float
    evaluated_trials: int


@dataclass(frozen=True)
class ParcelTrajectory:
    """Float64 samples, including the initial state, on the requested cadence.

    time (s), height (m), pressure (Pa), temperature (K), volume (m³),
    vapor and liquid (kg/kg dry air), and supersaturation (fraction above
    saturation) each have one entry per time. wet_radius (m) has axes time
    and class. updraft_gradient is the terminal liquid mixing ratio derivative
    per m/s at fixed elapsed time, or None when unrequested. statistics
    contains work counts and step extrema for adaptive Warp execution.
    """

    time: FloatArray
    height: FloatArray
    pressure: FloatArray
    temperature: FloatArray
    volume: FloatArray
    vapor: FloatArray
    liquid: FloatArray
    supersaturation: FloatArray
    wet_radius: FloatArray
    updraft_gradient: float | None
    initial: ParcelInitialState
    statistics: ParcelStepStatistics | None = None

    def aerosol_at(self, sample: int, *, dry_density: float) -> Aerosol:
        """Return an independent native one-box snapshot at a time index.

        The discrete particle lanes are dry material and water, respectively.
        Supply dry material density (kg/m³) to convert the fixed dry volumes
        to mass. Gas holds water vapor in kg/m³. The native discrete volume
        is 1 m³. Multiply its density inventories by the physical parcel
        volume at this sample to recover extensive quantities.
        """
        if not np.isfinite(dry_density) or dry_density <= 0:
            raise ValueError("dry_density must be finite and positive")
        dry = self.initial.dry_radius
        wet = self.wet_radius[sample]
        volume = self.volume[sample]
        dry_mass = (4 * np.pi / 3 * dry_density) * dry**3
        water_mass = WATER_FACTOR * (wet - dry) * (wet**2 + wet * dry + dry**2)
        particles = ParticleData(
            masses=np.column_stack((dry_mass, water_mass))[None, :, :],
            concentration=self.initial.number[None, :] / volume,
            charge=np.zeros((1, len(dry))),
            density=np.array([dry_density, RHO_W]),
            volume=np.ones(1),
            distribution_type="discrete",
        )
        gas = GasData(
            name=["water"],
            molar_mass=np.array([GAS_CONSTANT / RV]),
            concentration=np.array(
                [[self.vapor[sample] * self.initial.dry_air_mass / volume]]
            ),
            partitioning=np.array([True]),
        )
        environment = EnvironmentData(
            [self.temperature[sample]],
            [self.pressure[sample]],
            [[self.supersaturation[sample] + 1]],
        )
        return Aerosol(particles=particles, gas=gas, environment=environment)


def _saturation_pressure(temperature):
    """Liquid-water saturation pressure (Pa), constant-L Clausius-Clapeyron."""
    return 611.2 * np.exp(LATENT / RV * (1 / 273.15 - 1 / temperature))


def _equilibrium(radius, dry_radius, temperature, kappa):
    water_volume = (radius - dry_radius) * (
        radius**2 + radius * dry_radius + dry_radius**2
    )
    activity = water_volume / (water_volume + kappa * dry_radius**3)
    return activity * np.exp(2 * SIGMA / (RV * RHO_W * temperature * radius))


def _initial_radii(initial):
    dry = np.asarray(initial.dry_radius, dtype=np.float64)
    number = np.asarray(initial.number, dtype=np.float64)
    if dry.ndim != 1 or dry.shape != number.shape or not dry.size:
        raise ValueError(
            "dry_radius and number need matching nonempty 1-D classes"
        )
    if not np.isfinite(dry).all() or np.any(dry <= 0):
        raise ValueError("dry radii must be finite and positive")
    if not np.isfinite(number).all() or np.any(number < 0):
        raise ValueError("represented counts must be finite and nonnegative")
    for value in (
        initial.pressure,
        initial.temperature,
        initial.dry_air_mass,
        initial.kappa,
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "require positive finite pressure, T, dry mass and kappa"
            )
    if (
        not np.isfinite(initial.height)
        or not np.isfinite(initial.vapor)
        or initial.vapor < 0
    ):
        raise ValueError("invalid initial height or vapor inventory")
    q = initial.vapor
    rh = (
        initial.pressure
        * q
        / (RD / RV + q)
        / _saturation_pressure(initial.temperature)
    )
    if not 0 <= rh < 1:
        raise ValueError("initial vapor must be subsaturated")
    if rh == 0:
        return dry.copy()
    # Omitting the Kelvin factor gives an upper bound on the equilibrium root.
    upper = np.cbrt(1 + initial.kappa * rh / (1 - rh))
    return np.array(
        [
            rd
            * brentq(
                lambda ratio, rd=rd: _equilibrium(
                    ratio * rd, rd, initial.temperature, initial.kappa
                )
                - rh,
                1.0,
                max(upper, 1 + np.finfo(float).eps),
                xtol=1e-14,
            )
            for rd in dry
        ]
    )


def _cpu_reference(initial, wet, time, updraft, phase_exchange):
    """Integrate pressure, temperature, vapor and squared radii."""
    dry = np.asarray(initial.dry_radius)
    weights = WATER_FACTOR * np.asarray(initial.number) / initial.dry_air_mass
    ql_initial = weights @ (wet**3 - dry**3)
    loading = 1 + initial.vapor + ql_initial

    def tendency(_, state):
        pressure, temperature, qv = state[:3]
        # Normalize squared radii to their initial values for the stiff solver.
        radius = wet * np.sqrt(state[3:])
        saturation = _saturation_pressure(temperature)
        rh = pressure * qv / (RD / RV + qv) / saturation
        diffusion = (
            2.11e-5 * (temperature / 273.15) ** 1.94 * (101325 / pressure)
        )
        resistance = RHO_W * (
            RV * temperature / (diffusion * saturation)
            + LATENT
            / (get_thermal_conductivity(temperature) * temperature)
            * (LATENT / (RV * temperature) - 1)
        )
        dr2 = (
            2
            * (rh - _equilibrium(radius, dry, temperature, initial.kappa))
            / resistance
        )
        if not phase_exchange:
            dr2 = np.zeros_like(radius)
        dliquid = weights @ (1.5 * radius * dr2)
        specific_volume = (RD + qv * RV) * temperature / pressure
        dp = -loading * STANDARD_GRAVITY * updraft / specific_volume
        dtemp = (specific_volume * dp + LATENT * dliquid) / CP
        return np.r_[dp, dtemp, -dliquid, dr2 / wet**2]

    y0 = np.r_[
        initial.pressure, initial.temperature, initial.vapor, np.ones_like(wet)
    ]
    sol = solve_ivp(
        tendency,
        (0.0, time[-1]),
        y0,
        t_eval=time,
        method="Radau",
        rtol=2e-10,
        atol=np.r_[1e-7, 1e-10, 1e-15, np.full(wet.size, 1e-13)],
    )
    if not sol.success:
        raise ValueError(f"parcel integration failed: {sol.message}")
    p, temp, qv = sol.y[:3]
    radius = wet * np.sqrt(sol.y[3:].T)
    liquid = (radius**3 - dry**3) @ weights
    volume = initial.dry_air_mass * (RD + qv * RV) * temp / p
    supersaturation = p * qv / (RD / RV + qv) / _saturation_pressure(temp) - 1
    return ParcelTrajectory(
        time,
        initial.height + updraft * time,
        p,
        temp,
        volume,
        qv,
        liquid,
        supersaturation,
        radius,
        None,
        initial,
    )


def _parcel_output_times(steps, timestep, updraft):
    """Validate common parcel forcing and construct the output time grid."""
    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps < 1
    ):
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(timestep) or timestep <= 0 or not np.isfinite(updraft):
        raise ValueError(
            "require a positive finite timestep and finite updraft"
        )
    return np.arange(steps + 1, dtype=np.float64) * timestep


def _run_multivapor(
    initial, time, updraft, device, adaptive, phase_exchange, gradient
):
    """Dispatch the explicit wet-state parcel after validating its controls."""
    if gradient:
        raise ValueError("multi-vapor parcel sensitivities are unsupported")
    if device == "cpu":
        if adaptive is not None:
            raise ValueError("multi-vapor CPU uses the SciPy Radau reference")
        return cpu_multivapor_reference(initial, time, updraft, phase_exchange)
    if device == "cuda" or device.startswith("cuda:"):
        if not isinstance(adaptive, (ParcelRK23, ParcelROS3)):
            raise ValueError(
                "multi-vapor CUDA requires ParcelRK23 or ParcelROS3"
            )
        from particula.execution.multivapor_parcel import (
            run_multivapor_cuda,
        )

        return run_multivapor_cuda(
            initial, time, updraft, phase_exchange, device, adaptive
        )
    raise ValueError("multi-vapor parcel requires CPU or an NVIDIA CUDA device")


def run_parcel(
    initial: ParcelInitialState | MultiVaporParcelInitialState,
    *,
    updraft: float,
    timestep: float,
    steps: int,
    device: str = "cpu",
    gradient: bool = False,
    phase_exchange: bool = True,
    adaptive: ParcelRK23 | ParcelROS3 | None = None,
) -> ParcelTrajectory | MultiVaporParcelTrajectory:
    """Advance a single parcel under constant signed vertical velocity.

    Args:
        initial: A subsaturated κ parcel or explicit multi-vapor wet classes.
            Multi-vapor input specifies each initial material and gas inventory.
        updraft: Vertical velocity in m/s, positive upward.
        timestep: Output spacing (s). Also the fixed Warp step by default.
        steps: Positive number of intervals.
        device: 'cpu' selects the independent SciPy Radau reference.
            'warp:cpu' or a CUDA device selects Warp execution.
        gradient: Use Warp reverse mode for terminal liquid versus updraft.
        phase_exchange: Enable signed condensation and latent feedback.
            A passive run retains the initial equilibrium water on each class.
        adaptive: Optional Warp RK23 or ROS3 controls. Internal steps end at
            the requested output times. The gradient follows the accepted
            discrete schedule.

    Returns:
        State histories and the requested Warp derivative.

    Raises:
        ValueError: Invalid state, backend, integration interval or trajectory.
    """
    time = _parcel_output_times(steps, timestep, updraft)
    if isinstance(initial, MultiVaporParcelInitialState):
        return _run_multivapor(
            initial, time, updraft, device, adaptive, phase_exchange, gradient
        )
    initial = replace(
        initial,
        dry_radius=np.array(initial.dry_radius, dtype=np.float64, copy=True),
        number=np.array(initial.number, dtype=np.float64, copy=True),
    )
    initial.dry_radius.setflags(write=False)
    initial.number.setflags(write=False)
    wet = _initial_radii(initial)
    if adaptive is not None and not isinstance(
        adaptive, (ParcelRK23, ParcelROS3)
    ):
        raise TypeError("adaptive must be ParcelRK23, ParcelROS3 or None")
    if device == "cpu":
        if adaptive is not None:
            raise ValueError(
                f"{type(adaptive).__name__} requires a Warp device"
            )
        if gradient:
            raise ValueError("the gradient requires a Warp device")
        result = _cpu_reference(initial, wet, time, updraft, phase_exchange)
    elif device == "warp:cpu" or device == "cuda" or device.startswith("cuda:"):
        warp_device = "cpu" if device == "warp:cpu" else device
        if adaptive is None:
            from particula.gpu.kernels.parcel import run_warp_parcel

            result = run_warp_parcel(
                initial,
                wet,
                time,
                updraft,
                warp_device,
                gradient,
                phase_exchange,
            )
        else:
            from particula.gpu.kernels.parcel_rk23 import run_adaptive_parcel

            result = run_adaptive_parcel(
                initial,
                wet,
                time,
                updraft,
                warp_device,
                gradient,
                phase_exchange,
                adaptive,
            )
    else:
        raise ValueError(f"unsupported parcel device {device!r}")
    if (
        not all(
            np.isfinite(a).all()
            for a in (
                result.height,
                result.pressure,
                result.temperature,
                result.volume,
                result.vapor,
                result.liquid,
                result.supersaturation,
                result.wet_radius,
            )
        )
        or np.any(result.pressure <= 0)
        or np.any(result.temperature <= 0)
        or np.any(result.volume <= 0)
        or np.any(result.vapor < 0)
        or np.any(result.wet_radius < initial.dry_radius)
        or (gradient and not np.isfinite(result.updraft_gradient))
    ):
        raise ValueError(
            "invalid parcel trajectory; check regime and refine timestep"
        )
    return result
