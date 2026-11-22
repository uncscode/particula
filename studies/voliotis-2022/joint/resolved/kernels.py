"""Finite-volume, individual-particle chamber processes on CUDA."""

import warp as wp
from engine import (
    KB,
    aerodynamic_mobility_wp,
    brownian_kernel_pair_wp,
    cunningham_slip_correction_wp,
    g_collection_term_wp,
    particle_mean_free_path_wp,
    rectangle_wall_loss_coefficient_from_transport_wp,
)

from gpu import F64, PI, V3, State, advance_channel

# Each occupied slot is one physical particle in a representative air volume.
# State entries carry its contribution to ug/m3. Entry 30 in *particle* state
# holds sulfate; entry 30 in *gas* state is a volatile isoprene daughter.
SEED = wp.constant(30)
T = wp.constant(F64(298.15))


@wp.func
def solid(q: int):
    """Identify a condensed organic reservoir in the chamber mechanism."""
    return (q >= 6 and q <= 9) or (q >= 11 and q <= 25)


@wp.func
def organic(x: State):
    """Sum nonoverlapping organic mass in a particle."""
    mass = F64(0)
    for q in range(6, 26):
        if q != 10:
            mass += x[q]
    return mass


@wp.kernel
def geometry(
    particles: wp.array2d(dtype=State),
    air: wp.array(dtype=F64),
    kw: wp.array(dtype=F64),
    wall_enabled: int,
    viscosity: F64,
    mfp: F64,
    dt: F64,
    step: int,
    seed: int,
    lanes: int,
    wall: wp.array2d(dtype=F64),
    props: wp.array3d(dtype=F64),
    totals: wp.array2d(dtype=F64),
):
    """Remove deposited particles and evaluate surviving pair properties."""
    r, i = wp.tid()
    x = particles[r, i]
    props[r, i, 0] = F64(0)
    if x[SEED] + organic(x) > F64(0):
        oa = organic(x)
        vol = x[SEED] / F64(1770) + oa / F64(1400)
        radius = wp.pow(
            F64(3e-9) * air[r] * vol / (F64(4) * PI), F64(1.0 / 3.0)
        )
        mass = (oa + x[SEED]) * air[r] * F64(1e-9)
        rate = rectangle_wall_loss_coefficient_from_transport_wp(
            kw[r],
            radius,
            (oa + x[SEED]) / vol,
            T,
            F64(3),
            F64(2),
            F64(3),
            viscosity,
            mfp,
            KB,
        )
        rng = wp.rand_init(seed + step, (r % lanes) * particles.shape[1] + i)
        if wall_enabled == 1 and F64(wp.randf(rng)) < F64(1) - wp.exp(
            -rate * dt
        ):
            wp.atomic_add(wall, r, 0, oa)
            wp.atomic_add(wall, r, 1, x[SEED])
            wp.atomic_add(wall, r, 2, F64(1))
            particles[r, i] = State()
        else:
            area = F64(4) * PI * radius * radius / air[r]
            diff = (
                KB
                * T
                * aerodynamic_mobility_wp(
                    radius,
                    cunningham_slip_correction_wp(mfp / radius),
                    viscosity,
                )
            )
            speed = wp.sqrt(F64(8) * KB * T / (PI * mass))
            props[r, i, 0] = radius
            props[r, i, 1] = diff
            props[r, i, 2] = speed
            props[r, i, 3] = g_collection_term_wp(
                particle_mean_free_path_wp(diff, speed), radius
            )
            props[r, i, 4] = area
            props[r, i, 5] = vol
            wp.atomic_add(totals, r, 0, area)
            wp.atomic_add(totals, r, 1, vol)


@wp.kernel
def shared_gas(
    gas: wp.array(dtype=State),
    released: wp.array2d(dtype=F64),
    supply: wp.array3d(dtype=F64),
    extra: wp.array3d(dtype=F64),
    params: wp.array2d(dtype=F64),
    channels: wp.array2d(dtype=int),
    kinetics: wp.array2d(dtype=F64),
    totals: wp.array2d(dtype=F64),
    incoming: wp.array(dtype=State),
    step: int,
    sub: int,
    dt: F64,
):
    """React shared vapours and allocate material ready for condensation."""
    r = wp.tid()
    g = gas[r]
    dh = dt / F64(3600)
    vk = wp.exp(-dh)
    for j in range(31):
        g[j] += released[r, j]
        released[r, j] = F64(0)
    for q in range(3):
        z = dh / params[r, q]
        matured = F64(1) - wp.exp(-z)
        source = supply[r, step // sub, q] / F64(sub)
        ext = extra[r, step // sub, q] / F64(sub)
        move = g[q] * matured + source * (F64(1) - matured / z)
        g[q] += source - move
        g[q + 6] += move
        kept = g[q + 3] * vk + ext * wp.sqrt(vk)
        g[10] += g[q + 3] + ext - kept
        g[q + 3] = kept
    for q in range(26, 31):
        g[10] += g[q] * (F64(1) - vk)
        g[q] *= vk
    for h in range(channels.shape[0]):
        a = channels[h, 0]
        if not solid(a) and kinetics[r, h] > F64(0):
            k = kinetics[r, h]
            if channels[h, 3] == 1:
                k *= totals[r, 0] / F64(0.001)
            g, moved = advance_channel(
                g,
                a,
                channels[h, 1],
                channels[h, 2],
                k,
                dh,
                V3(F64(184), F64(140), F64(118)),
            )
    added = State()
    for q in range(6, 26):
        if solid(q):
            added[q] = g[q]
            g[q] = F64(0)
    incoming[r] = added
    gas[r] = g


@wp.kernel
def particle_chemistry(
    particles: wp.array2d(dtype=State),
    incoming: wp.array(dtype=State),
    released: wp.array2d(dtype=F64),
    totals: wp.array2d(dtype=F64),
    props: wp.array3d(dtype=F64),
    channels: wp.array2d(dtype=int),
    kinetics: wp.array2d(dtype=F64),
    dt: F64,
    allocation_power: wp.array(dtype=F64),
):
    """Advance local chemical inventories and release gas daughters."""
    r, i = wp.tid()
    if props[r, i, 0] > F64(0):
        x = particles[r, i]
        volume_fraction = props[r, i, 5] / totals[r, 1]
        share = props[r, i, 4] / totals[r, 0]
        share = (F64(1) - allocation_power[r]) * share + allocation_power[
            r
        ] * volume_fraction
        fresh = incoming[r]
        for q in range(6, 26):
            if solid(q):
                x[q] += fresh[q] * share
        dt_h = dt / F64(3600)
        for h in range(channels.shape[0]):
            a = channels[h, 0]
            if solid(a) and kinetics[r, h] > F64(0):
                b = channels[h, 1]
                dest = channels[h, 2]
                k = kinetics[r, h]
                if b >= 0:
                    # Bulk-equivalent concentration at this particle's
                    # composition.
                    k /= volume_fraction
                # Released gas is buffered for the next shared-gas step.
                # Pool 30 doubles as the sulfate index in the particle array.
                to = dest
                if not solid(dest):
                    to = 10
                x, moved = advance_channel(
                    x, a, b, to, k, dt_h, V3(F64(184), F64(140), F64(118))
                )
                split = channels[h, 4]
                if split >= 0:
                    x[to] -= F64(0.5) * moved
                    x[split] += F64(0.5) * moved
                    moved *= F64(0.5)
                if not solid(dest):
                    x[to] -= moved
                    wp.atomic_add(released, r, dest, moved)
        particles[r, i] = x


@wp.kernel
def collisions(
    particles: wp.array2d(dtype=State),
    air: wp.array(dtype=F64),
    props: wp.array3d(dtype=F64),
    order: wp.array2d(dtype=int),
    step: int,
    seed: int,
    lanes: int,
    dt: F64,
    viscosity: F64,
    mfp: F64,
    wall: wp.array2d(dtype=F64),
):
    """Sample disjoint Brownian pairs and merge their composition vectors."""
    r, p = wp.tid()
    i = order[step, 2 * p]
    j = order[step, 2 * p + 1]
    if props[r, i, 0] > F64(0) and props[r, j, 0] > F64(0):
        k = brownian_kernel_pair_wp(
            props[r, i, 0],
            props[r, j, 0],
            props[r, i, 1],
            props[r, j, 1],
            props[r, i, 3],
            props[r, j, 3],
            props[r, i, 2],
            props[r, j, 2],
            F64(1),
        )
        probability = k * F64(particles.shape[1] - 1) * dt / air[r]
        wp.atomic_max(wall, r, 4, probability)
        rng = wp.rand_init(seed + step, (r % lanes) * particles.shape[1] + p)
        if F64(wp.randf(rng)) < probability:
            merged = particles[r, i] + particles[r, j]
            particles[r, i] = merged
            particles[r, j] = State()
            props[r, j, 0] = F64(0)
            oa = organic(merged)
            vol = merged[SEED] / F64(1770) + oa / F64(1400)
            radius = wp.pow(
                F64(3e-9) * air[r] * vol / (F64(4) * PI), F64(1.0 / 3.0)
            )
            mass = (oa + merged[SEED]) * air[r] * F64(1e-9)
            diff = (
                KB
                * T
                * aerodynamic_mobility_wp(
                    radius,
                    cunningham_slip_correction_wp(mfp / radius),
                    viscosity,
                )
            )
            speed = wp.sqrt(F64(8) * KB * T / (PI * mass))
            props[r, i, 0] = radius
            props[r, i, 1] = diff
            props[r, i, 2] = speed
            props[r, i, 3] = g_collection_term_wp(
                particle_mean_free_path_wp(diff, speed), radius
            )
            wp.atomic_add(wall, r, 3, F64(1))


@wp.kernel
def nucleation(
    particles: wp.array2d(dtype=State),
    gas: wp.array(dtype=State),
    air: wp.array(dtype=F64),
    strength: wp.array(dtype=F64),
    dt: F64,
    step: int,
    seed: int,
    lanes: int,
    events: wp.array2d(dtype=F64),
):
    """Activate 3 nm organic embryos from a finite common gas reservoir."""
    r = wp.tid()
    g = gas[r]
    vap = g[0] + g[1] + g[2]
    if vap > F64(0) and strength[r] > F64(0):
        # Bounded effective formation rate, cm^-3 s^-1; reference 1 ug/m3.
        rate = strength[r] * vap / (F64(1) + vap)
        rng = wp.rand_init(seed + step, r % lanes)
        demand = int(wp.poisson(rng, float(rate * air[r] * F64(1e6) * dt)))
        cost = PI / F64(6) * F64(3e-9) ** F64(3) * F64(1400e9) / air[r]
        allowed = wp.min(demand, int(wp.min(F64(demand), vap / cost)))
        if allowed > 0:
            start = wp.randi(rng, 0, particles.shape[1])
            made = int(0)
            for offset in range(particles.shape[1]):
                i = (start + offset) % particles.shape[1]
                old = particles[r, i]
                if old[SEED] + organic(old) == F64(0):
                    birth = State()
                    for q in range(3):
                        birth[q + 6] = cost * g[q] / vap
                    particles[r, i] = birth
                    made += 1
                    if made == allowed:
                        break
            for q in range(3):
                g[q] -= F64(made) * cost * g[q] / vap
            gas[r] = g
            events[r, 5] += F64(made)
            events[r, 6] += F64(allowed - made)


@wp.kernel
def record(
    particles: wp.array2d(dtype=State),
    air: wp.array(dtype=F64),
    window: wp.array2d(dtype=F64),
    grid: wp.array(dtype=F64),
    obs: wp.array3d(dtype=F64),
    dist: wp.array3d(dtype=F64),
    sample: int,
):
    """Sample dry particle sizes, selected compositions and organic mass."""
    r, i = wp.tid()
    x = particles[r, i]
    oa = organic(x)
    vol = oa / F64(1400) + x[SEED] / F64(1770)
    if vol > F64(0):
        diam = F64(1e9) * wp.pow(F64(6e-9) * vol * air[r] / PI, F64(1.0 / 3.0))
        wp.atomic_add(obs, r, sample, 0, oa)
        wp.atomic_add(obs, r, sample, 1, x[SEED])
        wp.atomic_add(obs, r, sample, 2, F64(1))
        if diam >= window[r, 0] and diam <= window[r, 1]:
            wp.atomic_add(obs, r, sample, 3, F64(1))
        phi = oa / F64(1400) / vol
        for j in range(3):
            if wp.abs(wp.log(diam / grid[j])) <= F64(0.1):
                wp.atomic_add(obs, r, sample, 4 + j, F64(1))
                wp.atomic_add(obs, r, sample, 7 + j, phi)
        ix = int(wp.floor(wp.log(diam / F64(2)) / wp.log(F64(1.12))))
        if ix >= 0 and ix < 60:
            wp.atomic_add(dist, r, sample, ix, F64(1))
            wp.atomic_add(dist, r, sample, ix + 60, phi)
        for q in range(6, 26):
            if solid(q):
                wp.atomic_add(obs, r, sample, 11 + q, x[q])
        if diam >= F64(40) and diam <= F64(600):
            wp.atomic_add(
                obs, r, sample, 41, PI * diam * diam / air[r] * F64(1e-6)
            )
        wp.atomic_add(
            obs,
            r,
            sample,
            10,
            F64(3.141592653589793) * diam * diam / air[r] * F64(1e-6),
        )
