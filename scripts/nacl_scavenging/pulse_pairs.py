"""Study-specific competing pair clocks for nine equal-weight CUDA slots.

The public capture sampler rejects out-of-domain populations. Here each such
pair uses the same charged hard-sphere completion as the steady calculation.
"""

from functools import lru_cache

import numpy as np
import warp as wp
from capture_study import EPSILON, HAMAKER, material
from instrument import LOG_D, MFP, MU, D
from other_kernels import AIR, DISSIPATION
from particula import dynamics as dyn
from particula.gpu.dynamics.coagulation_funcs import (
    brownian_kernel_pair_wp,
    charged_kernel_wp,
    g_collection_term_wp,
    particle_mean_free_path_wp,
    sedimentation_sp2016_pair_rate_wp,
    turbulent_shear_st1956_pair_rate_wp,
)
from particula.gpu.dynamics.van_der_waals import CaptureData, capture_pair_wp
from particula.gpu.properties.particle_properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    mean_thermal_speed_wp,
    settling_velocity_stokes_from_transport_wp,
)
from particula.util import constants as const
from study import DENSITY, PRESSURE, TEMP

ELEMENTARY_CHARGE_VALUE = wp.constant(wp.float64(const.ELEMENTARY_CHARGE_VALUE))
ELECTRIC_PERMITTIVITY = wp.constant(wp.float64(const.ELECTRIC_PERMITTIVITY))
GAS_CONSTANT = wp.constant(wp.float64(const.GAS_CONSTANT))
MOLECULAR_WEIGHT_AIR = wp.constant(wp.float64(const.MOLECULAR_WEIGHT_AIR))
REF_VISCOSITY_AIR_STP = wp.constant(wp.float64(const.REF_VISCOSITY_AIR_STP))
REF_TEMPERATURE_STP = wp.constant(wp.float64(const.REF_TEMPERATURE_STP))
SUTHERLAND_CONSTANT = wp.constant(wp.float64(const.SUTHERLAND_CONSTANT))
RADIUS_CUBED_PER_KG = wp.constant(wp.float64(3 / (4 * np.pi * DENSITY)))
UINT32_SCALE = wp.constant(wp.float64(2**-32))
T = wp.constant(wp.float64(TEMP))
P = wp.constant(wp.float64(PRESSURE))
KB = wp.constant(wp.float64(const.BOLTZMANN_CONSTANT))
VISC = wp.constant(wp.float64(MU))
PATH = wp.constant(wp.float64(MFP))
RHO = wp.constant(wp.float64(DENSITY))
NU = wp.constant(wp.float64(MU / AIR))
LOG0 = wp.constant(wp.float64(LOG_D[0]))
AO_SCALE = wp.constant(wp.float64(1 / (LOG_D[4] - LOG_D[0])))

# (thermal law, Hamaker J, permittivity, settling, shear epsilon, Ayala)
MODELS = {
    "bare": (0, 0, 1, False, 0, False),
    "dispersion_only": (-1, HAMAKER, 1, False, 0, False),
    "polarization_only": (-1, 0, EPSILON, False, 0, False),
    "full": (-1, HAMAKER, EPSILON, False, 0, False),
    "full_settling": (-1, HAMAKER, EPSILON, True, 0, False),
    "full_settling_shear_10": (-1, HAMAKER, EPSILON, True, 10, False),
    "full_ao2008_0.01": (-1, HAMAKER, EPSILON, False, 0, True),
    **{
        f"full_shear_{e:g}": (-1, HAMAKER, EPSILON, False, e, False)
        for e in DISSIPATION
    },
    "fuchs_neutral": (-2, 0, 1, False, 0, False),
    "dyachkov": (1, 0, 1, False, 0, False),
    "gatti": (2, 0, 1, False, 0, False),
    "gopalakrishnan": (3, 0, 1, False, 0, False),
    "chahl": (4, 0, 1, False, 0, False),
    **{
        f"{name}_neutral": (law, 0, 1, False, 0, False)
        for law, name in enumerate(
            ("bare", "dyachkov", "gatti", "gopalakrishnan", "chahl")
        )
    },
}


@lru_cache(maxsize=2)
def ao_table(enabled):
    """Tabulate the exploratory AO law in SI at log-diameter nodes."""
    values = np.zeros((1, 1))
    if enabled:
        r = D[::4] * 0.5e-9
        values = dyn.get_turbulent_dns_kernel_ao2008_via_system_state(
            r,
            np.full_like(r, DENSITY),
            AIR,
            TEMP,
            re_lambda=100.0,
            relative_velocity=0.0,
            turbulent_dissipation=0.01,
        )
        if not np.all(np.isfinite(values) & (values >= 0)):
            raise ValueError("Invalid Ayala interpolation table")
    return wp.array(values, dtype=wp.float64, device="cuda:0")


@wp.func
def pair_rates(
    mi: wp.float64,
    mj: wp.float64,
    zi: wp.float64,
    zj: wp.float64,
    law: int,
    data: CaptureData,
    settling: int,
    shear: wp.float64,
    ao: wp.array2d(dtype=wp.float64),
):
    """Return thermal, mechanical and fallback pair rates in m³/s."""
    ri = wp.cbrt(mi * RADIUS_CUBED_PER_KG)
    rj = wp.cbrt(mj * RADIUS_CUBED_PER_KG)
    zero = wp.float64(0.0)
    thermal = wp.float64(-1.0)
    tail = zero
    mechanical = zero
    if law < 0:
        di = (
            KB
            * T
            * aerodynamic_mobility_wp(
                ri, cunningham_slip_correction_wp(PATH / ri), VISC
            )
        )
        dj = (
            KB
            * T
            * aerodynamic_mobility_wp(
                rj, cunningham_slip_correction_wp(PATH / rj), VISC
            )
        )
        if law == -1:
            thermal = capture_pair_wp(data, ri, rj, mi, mj, zi, zj, di + dj, T)
            if thermal < zero:
                tail = wp.float64(1.0)
        else:
            vi = mean_thermal_speed_wp(mi, T, KB)
            vj = mean_thermal_speed_wp(mj, T, KB)
            gi = g_collection_term_wp(particle_mean_free_path_wp(di, vi), ri)
            gj = g_collection_term_wp(particle_mean_free_path_wp(dj, vj), rj)
            thermal = brownian_kernel_pair_wp(
                ri, rj, di, dj, gi, gj, vi, vj, wp.float64(1.0)
            )
    if thermal < zero:
        thermal = charged_kernel_wp(
            wp.max(law, 0),
            ri,
            rj,
            mi,
            mj,
            zi,
            zj,
            T,
            P,
            KB,
            ELEMENTARY_CHARGE_VALUE,
            ELECTRIC_PERMITTIVITY,
            GAS_CONSTANT,
            MOLECULAR_WEIGHT_AIR,
            REF_VISCOSITY_AIR_STP,
            REF_TEMPERATURE_STP,
            SUTHERLAND_CONSTANT,
        )
    if settling:
        vi = settling_velocity_stokes_from_transport_wp(ri, RHO, VISC, PATH)
        vj = settling_velocity_stokes_from_transport_wp(rj, RHO, VISC, PATH)
        mechanical += sedimentation_sp2016_pair_rate_wp(ri, rj, vi, vj)
    if shear > zero:
        mechanical += turbulent_shear_st1956_pair_rate_wp(ri, rj, shear, NU)
    if ao.shape[0] > 1:
        x = (wp.log(ri * wp.float64(2e9)) - LOG0) * AO_SCALE
        y = (wp.log(rj * wp.float64(2e9)) - LOG0) * AO_SCALE
        a, b = int(wp.floor(x)), int(wp.floor(y))
        if a < 0 or b < 0 or a + 1 >= ao.shape[0] or b + 1 >= ao.shape[1]:
            mechanical = wp.float64(-1.0)
        else:
            u, v = x - wp.float64(a), y - wp.float64(b)
            mechanical = wp.lerp(
                wp.lerp(ao[a, b], ao[a + 1, b], u),
                wp.lerp(ao[a, b + 1], ao[a + 1, b + 1], u),
                v,
            )
    return wp.vec3d(thermal, mechanical, tail * thermal)


@wp.kernel
def refresh_pairs(
    mass: wp.array3d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    pairs: wp.array(dtype=wp.vec2i),
    left: wp.array(dtype=wp.float64),
    old: wp.array2d(dtype=wp.vec4d),
    rates: wp.array2d(dtype=wp.vec3d),
    law: int,
    neutral: int,
    data: CaptureData,
    settling: int,
    shear: wp.float64,
    ao: wp.array2d(dtype=wp.float64),
    fault: wp.array(dtype=int),
):
    """Recompute rates after a charge, mass or occupancy change."""
    b, p = wp.tid()
    if left[b] <= wp.float64(0.0):
        return
    i, j = pairs[p][0], pairs[p][1]
    mi = mass[b, i, 0] + mass[b, i, 1]
    mj = mass[b, j, 0] + mass[b, j, 1]
    zi, zj = wp.float64(charge[b, i]), wp.float64(charge[b, j])
    if neutral:
        zi, zj = wp.float64(0.0), wp.float64(0.0)
    key = wp.vec4d(mi, mj, zi, zj)
    if key == old[b, p]:
        return
    old[b, p] = key
    result = wp.vec3d()
    if mi > wp.float64(0.0) and mj > wp.float64(0.0):
        result = pair_rates(mi, mj, zi, zj, law, data, settling, shear, ao)
        if (
            not wp.isfinite(result[0] + result[1])
            or result[0] < 0
            or result[1] < 0
        ):
            wp.atomic_add(fault, 0, 1)
    rates[b, p] = result


@wp.kernel
def draw_collision(  # noqa: C901 -- one draw per independent box
    mass: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    pairs: wp.array(dtype=wp.vec2i),
    rates: wp.array2d(dtype=wp.vec3d),
    left: wp.array(dtype=wp.float64),
    inverse_volume: wp.float64,
    seed: int,
    events: wp.array(dtype=int),
    again: wp.array(dtype=int),
    ledger: wp.array2d(dtype=wp.float64),
):
    """Sample the next merger and integrate tracer capture hazards."""
    b = wp.tid()
    if left[b] <= wp.float64(0.0):
        return
    total = wp.float64(0.0)
    tracer = wp.vec3d()
    for p in range(pairs.shape[0]):
        i, j = pairs[p][0], pairs[p][1]
        item = rates[b, p]
        total += item[0] + item[1]
        if (mass[b, i, 1] > 0) != (mass[b, j, 1] > 0):
            tracer += item
    if total <= wp.float64(0.0):
        left[b] = wp.float64(0.0)
        return
    rng = wp.rand_init(seed, b)
    wait = -wp.log(
        wp.max(
            ((wp.float64(wp.randu(rng)) + wp.float64(0.5)) * UINT32_SCALE),
            wp.float64(1e-30),
        )
    ) / (total * inverse_volume)
    interval = wp.min(left[b], wait) * inverse_volume
    for k in range(3):
        ledger[b, k] += interval * tracer[k]
    if wait >= left[b]:
        left[b] = wp.float64(0.0)
        return
    left[b] -= wait
    mark = (
        (wp.float64(wp.randu(rng)) + wp.float64(0.5)) * UINT32_SCALE
    ) * total
    for p in range(pairs.shape[0]):
        item = rates[b, p]
        if mark < item[0] + item[1]:
            i, j = pairs[p][0], pairs[p][1]
            if mark >= item[0] and (mass[b, i, 1] > 0) != (mass[b, j, 1] > 0):
                ledger[b, 3] += wp.float64(1.0)
            for k in range(2):
                mass[b, i, k] += mass[b, j, k]
                mass[b, j, k] = wp.float64(0.0)
            charge[b, i] += charge[b, j]
            charge[b, j] = wp.float64(0.0)
            counts[b, j] = wp.float64(0.0)
            events[b] += 1
            wp.atomic_add(again, 0, 1)
            return
        mark -= item[0] + item[1]


class PulsePairs:
    """Memoize unchanged pair states across stochastic events and time steps."""

    def __init__(self, model, boxes, slots, device, inverse_volume):
        """Allocate the pair cache and event accounting on the GPU."""
        law, a, eps, settling, shear, ao = MODELS[model]
        self.options = [
            law,
            int(model.endswith("_neutral")),
            material(a, eps),
            int(settling),
            shear,
            ao_table(ao),
        ]
        self.device = device
        self.inverse_volume = inverse_volume
        pairs = np.transpose(np.triu_indices(slots, k=1))
        self.pairs = wp.array(pairs, dtype=wp.vec2i, device=device)
        shape = (boxes, len(pairs))
        self.old = wp.zeros(shape, dtype=wp.vec4d, device=device)
        self.rates = wp.zeros(shape, dtype=wp.vec3d, device=device)
        self.left = wp.empty(boxes, dtype=wp.float64, device=device)
        self.events = wp.zeros(boxes, dtype=int, device=device)
        self.again = wp.zeros(1, dtype=int, device=device)
        self.ledger = wp.zeros((boxes, 4), dtype=wp.float64, device=device)

    def step(self, aerosol, dt, seed, fault):
        """Advance until the remaining interval expires in every box."""
        self.left.fill_(dt)
        # Every accepted collision removes a slot. One final draw exhausts dt.
        for draw in range(aerosol.concentration.shape[1]):
            self.again.zero_()
            wp.launch(
                refresh_pairs,
                self.rates.shape,
                inputs=[
                    aerosol.masses,
                    aerosol.charge,
                    self.pairs,
                    self.left,
                    self.old,
                    self.rates,
                    *self.options,
                    fault,
                ],
                device=self.device,
            )
            wp.launch(
                draw_collision,
                self.left.shape,
                inputs=[
                    aerosol.masses,
                    aerosol.concentration,
                    aerosol.charge,
                    self.pairs,
                    self.rates,
                    self.left,
                    self.inverse_volume,
                    seed + draw * 10007,
                    self.events,
                    self.again,
                    self.ledger,
                ],
                device=self.device,
            )
            if self.again.numpy()[0] == 0:
                return
        raise RuntimeError("Exceeded the number of possible pulse collisions")
