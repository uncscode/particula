"""NaCl source-off pulses using the particle-resolved CUDA sampler."""

import argparse
import json
import time

import numpy as np
import warp as wp
from instrument import (
    HERE,
    LOG_D,
    PRESSURE,
    TARGETS,
    TEMP,
    D,
    attachment_coefficient,
    fit_walls,
    ion_density,
    probabilities,
    response,
    sample,
    save,
    selected,
    vdw_table,
    wall_table,
)
from particula.gpu import to_warp_particle_data
from particula.gpu.kernels.coagulation import (
    CoagulationMechanismConfig,
    coagulation_step_gpu,
)
from particula.particles import ParticleData

DEVICE = "cuda:0"
DENSITY = 2160.0
N100 = 10000.0
COLLECTORS = 8
ZS = 12


@wp.kernel
def chamber_step(
    m: wp.array3d(dtype=wp.float64),
    c: wp.array2d(dtype=wp.float64),
    z: wp.array2d(dtype=wp.float64),
    neg: wp.array2d(dtype=wp.float64),
    pos: wp.array2d(dtype=wp.float64),
    wall: wp.array(dtype=wp.float64),
    ions: wp.array2d(dtype=wp.float64),
    log_min: float,
    inv_spacing: float,
    dt: float,
    seed: int,
    ledger: wp.array2d(dtype=wp.float64),
    fault: wp.array(dtype=int),
):
    """Sample ion attachment, wall deposition and dilution in each slot."""
    b, s = wp.tid()
    if c[b, s] == wp.float64(0.0):
        return
    rng = wp.rand_init(seed, b * m.shape[1] + s)
    diameter = wp.pow(
        (m[b, s, 0] + m[b, s, 1]) * wp.float64(6.0e27 / (np.pi * 2160.0)),
        wp.float64(1.0 / 3.0),
    )
    g = int(
        wp.round(
            (wp.log(diameter) - wp.float64(log_min)) * wp.float64(inv_spacing)
        )
    )
    if g < 0 or g >= wall.shape[0]:
        wp.atomic_add(fault, 0, 1)
        return
    charge = int(z[b, s])
    elapsed = wp.float64(0.0)
    while True:
        i = charge + 12
        if i < 0 or i >= 25:
            wp.atomic_add(fault, 0, 1)
            return
        an = neg[g, i] * ions[b, 0]
        ap = pos[g, i] * ions[b, 1]
        rate = an + ap + wall[g] + wp.float64(0.6 / 3600.0)
        elapsed -= (
            wp.log(wp.max(wp.float64(1.0e-30), wp.float64(wp.randf(rng))))
            / rate
        )
        if elapsed > wp.float64(dt):
            break
        mark = wp.float64(wp.randf(rng)) * rate
        if mark < an:
            charge -= 1
            wp.atomic_add(ledger, b, 0, wp.float64(-1.0))
        elif mark < an + ap:
            charge += 1
            wp.atomic_add(ledger, b, 0, wp.float64(1.0))
        else:
            mode = int(m[b, s, 1] > wp.float64(0.0))
            cause = int(mark >= an + ap + wall[g])
            wp.atomic_add(ledger, b, 1 + 2 * mode + cause, wp.float64(1.0))
            wp.atomic_add(ledger, b, 5, wp.float64(charge))
            for j in range(2):
                wp.atomic_add(ledger, b, 6 + j, m[b, s, j])
                m[b, s, j] = wp.float64(0.0)
            c[b, s] = wp.float64(0.0)
            z[b, s] = wp.float64(0.0)
            return
    z[b, s] = wp.float64(charge)


@wp.kernel
def risk_time(
    m: wp.array3d(dtype=wp.float64),
    c: wp.array2d(dtype=wp.float64),
    dt: float,
    area: wp.array(dtype=wp.float64),
):
    """Accumulate live tracer exposure to collectors in each box."""
    b = wp.tid()
    tracer = int(0)
    collectors = int(0)
    for s in range(c.shape[1]):
        if c[b, s] > wp.float64(0.0):
            if m[b, s, 1] > wp.float64(0.0):
                collectors += 1
            else:
                tracer += 1
    area[b] += wp.float64(tracer * collectors) * wp.float64(dt * 10000.0 / 8.0)


@wp.kernel
def dispersion(  # noqa: C901 -- one Warp kernel per box
    m: wp.array3d(dtype=wp.float64),
    c: wp.array2d(dtype=wp.float64),
    z: wp.array2d(dtype=wp.float64),
    table: wp.array2d(dtype=wp.float64),
    ln_min: float,
    scale: float,
    dt: float,
    seed: int,
    extra: wp.array(dtype=int),
    fault: wp.array(dtype=int),
):
    """Sample additional collisions from the dispersion rate increment."""
    # Competing pair clocks, with a weighted reservoir draw for the partner.
    b = wp.tid()
    rng = wp.rand_init(seed, b)
    elapsed = wp.float64(0.0)
    while elapsed < wp.float64(dt):
        total = wp.float64(0.0)
        keep = int(-1)
        drop = int(-1)
        for i in range(c.shape[1]):
            if c[b, i] == wp.float64(0.0):
                continue
            di = wp.pow(
                (m[b, i, 0] + m[b, i, 1])
                * wp.float64(6.0e27 / (np.pi * 2160.0)),
                wp.float64(1.0 / 3.0),
            )
            xi = (wp.log(di) - wp.float64(ln_min)) * wp.float64(scale)
            a = int(wp.floor(xi))
            for j in range(i + 1, c.shape[1]):
                if c[b, j] == wp.float64(0.0):
                    continue
                dj = wp.pow(
                    (m[b, j, 0] + m[b, j, 1])
                    * wp.float64(6.0e27 / (np.pi * 2160.0)),
                    wp.float64(1.0 / 3.0),
                )
                xj = (wp.log(dj) - wp.float64(ln_min)) * wp.float64(scale)
                h = int(wp.floor(xj))
                if (
                    a < 0
                    or h < 0
                    or a + 1 >= table.shape[0]
                    or h + 1 >= table.shape[1]
                ):
                    wp.atomic_add(fault, 0, 1)
                    return
                u = xi - wp.float64(a)
                v = xj - wp.float64(h)
                lower = wp.lerp(table[a, h], table[a + 1, h], u)
                upper = wp.lerp(table[a, h + 1], table[a + 1, h + 1], u)
                rate = wp.exp(wp.lerp(lower, upper, v)) * wp.float64(
                    10000.0 / 8.0
                )
                total += rate
                if wp.float64(wp.randf(rng)) * total < rate:
                    keep = i
                    drop = j
        if total == wp.float64(0.0):
            break
        elapsed -= (
            wp.log(wp.max(wp.float64(wp.randf(rng)), wp.float64(1.0e-30)))
            / total
        )
        if elapsed > wp.float64(dt):
            break
        for j in range(2):
            m[b, keep, j] += m[b, drop, j]
            m[b, drop, j] = wp.float64(0.0)
        z[b, keep] += z[b, drop]
        z[b, drop] = wp.float64(0.0)
        c[b, drop] = wp.float64(0.0)
        extra[b] += 1


def simulate(args):  # noqa: C901 -- ordered pulse and inventory checks
    """Evolve paired blank and collector-loaded boxes for one seed."""
    if not wp.is_cuda_available():
        raise RuntimeError("A real CUDA device is required")
    tic = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    p = probabilities(D, args.bound)
    fits = fit_walls()
    hats = selected(
        100, p, large=True, mode_ratio=args.parent, ideal=args.ideal
    )
    reps = args.reps
    groups = [(d, n) for d in TARGETS for n in [0, N100]]
    batches = len(groups) * reps
    mass = np.zeros((batches, 1 + COLLECTORS, 2))
    counts = np.zeros(mass.shape[:2])
    charge = counts.copy()
    ions = np.zeros((batches, 2))
    for k, (target, n) in enumerate(groups):
        sl = slice(k * reps, (k + 1) * reps)
        small = selected(target, p, ideal=args.ideal)
        ds, qs, _ = sample(small, p, reps, rng, args.tube)
        mass[sl, 0, 0] = (np.pi / 6 * DENSITY * 1e-27) * ds**3
        counts[sl, 0] = 1
        charge[sl, 0] = qs
        if n:
            dl, ql, _ = sample(hats, p, reps * COLLECTORS, rng, 1.0)
            mass[sl, 1:, 1] = (np.pi / 6 * DENSITY * 1e-27) * dl.reshape(
                reps, -1
            ) ** 3
            counts[sl, 1:] = 1
            charge[sl, 1:] = ql.reshape(reps, -1)
        ions[sl] = ion_density(p, hats.sum(axis=1), n, args.ion)
    initial = ParticleData(
        mass,
        counts,
        charge,
        np.array([DENSITY, DENSITY]),
        np.full(batches, COLLECTORS / (N100 * 1e6)),
        distribution_type="particle_resolved",
    )
    aerosol = to_warp_particle_data(initial, device=DEVICE)
    initial_mass = mass.sum(axis=1).copy()
    initial_number = counts.sum(axis=1).copy()
    initial_charge = charge.sum(axis=1).copy()
    state = wp.zeros(batches, dtype=wp.uint32, device=DEVICE)
    ledger = wp.zeros((batches, 8), dtype=wp.float64, device=DEVICE)
    fault = wp.zeros(1, dtype=int, device=DEVICE)

    def arr(a):
        return wp.array(a, dtype=wp.float64, device=DEVICE)

    rates = [
        arr(
            1e6
            * attachment_coefficient(
                D[:, None] * 5e-10,
                np.arange(-ZS, ZS + 1)[None, :],
                pol,
                omitted=args.bound,
            )
        )
        for pol in (-1, 1)
    ]
    environment = [
        *rates,
        arr(wall_table(fits)),
        arr(ions),
        float(LOG_D[0]),
        float(1 / (LOG_D[1] - LOG_D[0])),
    ]
    area = wp.zeros(batches, dtype=wp.float64, device=DEVICE)
    extra = wp.zeros(batches, dtype=int, device=DEVICE)
    dispersion_table = arr(vdw_table()) if args.vdw else None
    diam0 = np.cbrt(initial_mass[:, 0] / (np.pi / 6 * DENSITY * 1e-27))
    collision_total = np.zeros(batches)
    params = dict(
        temperature=TEMP,
        pressure=PRESSURE,
        time_step=args.dt,
        max_collisions=(COLLECTORS + 1) // 2,
        rng_seed=args.seed,
        rng_states=state,
        mechanism_config=CoagulationMechanismConfig(("charged_hard_sphere",)),
    )
    smps = {d: response(d, 0.10, 0.03, p, 1) for d in (*TARGETS, 100.0)}
    rows = []
    for step in range(round(args.duration / args.dt) + 1):
        if step:
            wp.launch(
                risk_time,
                dim=batches,
                inputs=[
                    aerosol.masses,
                    aerosol.concentration,
                    args.dt / 2,
                    area,
                ],
                device=DEVICE,
            )
            wp.launch(
                chamber_step,
                dim=counts.shape,
                inputs=[
                    aerosol.masses,
                    aerosol.concentration,
                    aerosol.charge,
                    *environment,
                    args.dt,
                    args.seed + 7919 * step,
                    ledger,
                    fault,
                ],
                device=DEVICE,
            )
            _, _, events = coagulation_step_gpu(
                aerosol, initialize_rng=step == 1, **params
            )
            collision_total += events.numpy()
            if args.vdw:
                wp.launch(
                    dispersion,
                    dim=batches,
                    inputs=[
                        aerosol.masses,
                        aerosol.concentration,
                        aerosol.charge,
                        dispersion_table,
                        float(LOG_D[0]),
                        float(1 / (LOG_D[8] - LOG_D[0])),
                        args.dt,
                        args.seed + 1000003 * step,
                        extra,
                        fault,
                    ],
                    device=DEVICE,
                )
            wp.launch(
                risk_time,
                dim=batches,
                inputs=[
                    aerosol.masses,
                    aerosol.concentration,
                    args.dt / 2,
                    area,
                ],
                device=DEVICE,
            )
        if step * args.dt not in (0, 60, 150, args.duration):
            continue
        m = aerosol.masses.numpy()
        c = aerosol.concentration.numpy().astype(bool)
        z = aerosol.charge.numpy()
        balance = ledger.numpy()
        risk = area.numpy()
        np.testing.assert_array_equal(fault.numpy(), 0)
        np.testing.assert_allclose(
            m.sum(axis=1) + balance[:, 6:], initial_mass, rtol=1e-12, atol=1e-38
        )
        np.testing.assert_array_equal(
            z.sum(axis=1) + balance[:, 5] - balance[:, 0], initial_charge
        )
        np.testing.assert_array_equal(
            c.sum(axis=1)
            + balance[:, 1:5].sum(axis=1)
            + collision_total
            + extra.numpy(),
            initial_number,
        )
        ds = np.cbrt(m.sum(axis=2) / (np.pi / 6 * DENSITY * 1e-27))
        for k, (target, n) in enumerate(groups):
            sl = slice(k * reps, (k + 1) * reps)
            large = m[sl, :, 1] > 0
            for mode, active, d in [
                ("small", c[sl] & ~large, target),
                ("large", large, 100.0),
            ]:
                diam = ds[sl][active]
                ch = z[sl][active]
                spectra = smps[d]
                signal = [
                    np.interp(diam, D, spectra[:, j]).sum() / reps
                    for j in range(3)
                ]
                alive = len(ch) / reps
                weight = np.interp(diam0[sl], D, smps[target].sum(axis=1))
                captured = (
                    1
                    - (c[sl] & ~large).sum(axis=1)
                    - balance[sl, 1:3].sum(axis=1)
                )
                np.testing.assert_equal(
                    np.all((captured == 0) | (captured == 1)), True
                )
                rows.append(
                    dict(
                        weighted_capture_sum=weight @ captured,
                        weighted_exposure_sum=weight @ risk[sl],
                        capture_sum=captured.sum(),
                        exposure_sum=risk[sl].sum(),
                        target_nm=target,
                        initial_collectors_cm3=n,
                        cohort=mode,
                        time_s=step * args.dt,
                        alive_per_box=alive,
                        mean_charge_e=ch.mean() if len(ch) else 0.0,
                        neg_fraction=np.count_nonzero(ch < 0) / max(1, len(ch)),
                        pos_fraction=np.count_nonzero(ch > 0) / max(1, len(ch)),
                        neutral_fraction=np.count_nonzero(ch == 0)
                        / max(1, len(ch)),
                        multiple_fraction=np.count_nonzero(abs(ch) >= 2)
                        / max(1, len(ch)),
                        sample_particles=len(ch),
                        mean_d_nm=diam.mean() if len(ch) else 0.0,
                        detector_z1=signal[0],
                        detector_z2=signal[1],
                        detector_z3=signal[2],
                        detected_per_box=sum(signal),
                        wall_per_box=balance[
                            sl, 1 if mode == "small" else 3
                        ].mean(),
                        diluted_per_box=balance[
                            sl, 2 if mode == "small" else 4
                        ].mean(),
                        ion_neg_cm3=ions[sl, 0].mean(),
                        ion_pos_cm3=ions[sl, 1].mean(),
                        coagulated_tracers_per_box=1
                        - (c[sl] & ~large).sum() / reps
                        - balance[sl, 1:3].sum() / reps,
                    )
                )
    save(args.label + ".csv", rows)
    save("wall-fits.csv", fits)
    meta = dict(
        **vars(args),
        device=wp.get_device(DEVICE).name,
        seconds=time.perf_counter() - tic,
        boxes=batches,
        slots=mass.size // 2,
        accepted_collisions=int(collision_total.sum()),
        dispersion_events=int(extra.numpy().sum()),
        mass_charge_number_checks=True,
    )
    (HERE / (args.label + ".json")).write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(meta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--ion", type=float, default=2.0)
    parser.add_argument("--tube", type=float, default=0.1)
    parser.add_argument("--bound", choices=["lower", "upper"], default="upper")
    parser.add_argument("--parent", type=float, default=1.0)
    parser.add_argument("--ideal", action="store_true")
    parser.add_argument("--vdw", action="store_true")
    parser.add_argument("--label", default="pulse")
    simulate(parser.parse_args())
