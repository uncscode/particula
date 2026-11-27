"""Resolve isothermal exchange at the sizes used for Brownian coagulation."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from water_constants import KELVIN, MEAN_FREE_PATH, SATURATION


def rhs(_, ratio, dry, kappa, weight, total):
    """Evolve water/dry mass ratio with gas eliminated by conservation."""
    water = np.maximum(ratio, 0)
    gas = total - weight @ ratio
    radius = np.cbrt(3 * dry * (1 / 1400 + water / 1000) / (4 * np.pi))
    kn = MEAN_FREE_PATH / radius
    alpha = 0.2
    transition = (
        0.75
        * alpha
        * (1 + kn)
        / (kn**2 + kn + 0.283 * alpha * kn + 0.75 * alpha)
    )
    transport = 4 * np.pi * radius * 2.4e-5 * transition
    activity = water / (water + 1000 / 1400 * kappa)
    return (
        transport
        / dry
        * (gas - SATURATION * activity * np.exp(KELVIN / radius))
    )


def fastest_equilibrium_rate(ratio, dry, kappa, weight):
    """Largest decay rate of the isothermal finite-box equilibrium Jacobian."""
    capacity = 1000 / 1400 * kappa
    radius = np.cbrt(3 * dry * (1 / 1400 + ratio / 1000) / (4 * np.pi))
    kn = MEAN_FREE_PATH / radius
    transport = (
        4
        * np.pi
        * radius
        * 2.4e-5
        * (0.15 * (1 + kn) / (kn**2 + kn + 0.0566 * kn + 0.15))
    )
    activity = ratio / (ratio + capacity)
    dr_du = dry / (4 * np.pi * 1000 * radius**2)
    d_surface = (
        SATURATION
        * np.exp(KELVIN / radius)
        * (
            capacity / (ratio + capacity) ** 2
            - activity * KELVIN / radius**2 * dr_du
        )
    )
    # Similar to a symmetric diagonal-plus-rank-one matrix.
    diag = transport / dry * d_surface
    coupling = np.sqrt(transport / dry * weight)
    return np.linalg.eigvalsh(np.diag(diag) + np.outer(coupling, coupling))[-1]


def stiff_reference(data, rtol):
    """Integrate each box with the gas reset between its two legs."""
    times = data["time"][:9]
    if not np.allclose(times, data["time"][9:] - times[-1], rtol=0, atol=2e-15):
        raise ValueError("Expected two equal-duration forcing legs")
    dense_times = np.unique(np.r_[times, np.geomspace(1e-8, times[-1], 121)])
    cadence = np.searchsorted(dense_times, times)
    all_water, all_gas, traces, rates = [], [], [], []
    for mass, counts, initial, reset in zip(
        data["mass"], data["counts"], data["gas"], data["reset"], strict=True
    ):
        active = counts > 0
        m = mass[active]
        dry = m[:, 1:].sum(1)
        kappa = (0.03 * m[:, 1] + 0.75 * m[:, 2]) / dry
        weight = counts[active] * dry / float(data["volume"])
        ratio = m[:, 0] / dry
        water, gas, trace, rate = [], [], [], []
        for supplied in (initial, reset):
            total = supplied + weight @ ratio
            sol = solve_ivp(
                rhs,
                (0, float(times[-1])),
                ratio,
                args=(dry, kappa, weight, total),
                method="BDF",
                rtol=rtol,
                atol=rtol * 1e-3,
                t_eval=dense_times,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            material = weight @ sol.y
            if np.min(sol.y) < -1e-12 or (total - material).min() < -1e-14:
                raise FloatingPointError("Invalid stiff water solution")
            water.extend(material[cadence])
            gas.extend(total - material[cadence])
            trace.append(material)
            rate.append(
                fastest_equilibrium_rate(sol.y[:, -1], dry, kappa, weight)
            )
            ratio = sol.y[:, -1]
        all_water.append(water)
        all_gas.append(gas)
        traces.append(trace)
        rates.append(rate)
    return dict(
        water=np.array(all_water).T,
        gas=np.array(all_gas).T,
        dense_time=dense_times,
        dense_water=np.array(traces),
        equilibrium_rate_s_inv=np.asarray(rates),
    )


def native_reference(data, steps):
    """Run condensation alone on the identical native collision-study inputs."""
    import numpy.testing as nt
    from collisions import configurations
    from pulse import make_box

    ids = np.unique(data["ids"]).tolist()
    state, process = make_box(
        configurations(ids, "coupled"), data["counts"].shape[1] // 2
    )
    nt.assert_array_equal(state.particles.masses, data["mass"])
    nt.assert_array_equal(state.particles.concentration, data["counts"])
    nt.assert_array_equal(state.gas.concentration[:, 0], data["gas"])
    count = data["counts"]

    def snapshot(owner):
        snap = owner.snapshot()
        nt.assert_array_equal(snap.particles.concentration, count)
        nt.assert_array_equal(
            snap.particles.masses[..., 1:], data["mass"][..., 1:]
        )
        water = (snap.particles.masses[..., 0] * count).sum(1) / float(
            data["volume"]
        )
        return water, snap.gas.concentration[:, 0].copy()

    water, gas, adjacent = [], [], []
    with process.prepare(
        state, device="cuda:0", time_step=0.125 / steps
    ) as owner:
        owner.capture()
        for leg in range(2):
            if leg:
                owner.update_gas(data["reset"][:, None].copy())
            for i in range(9):
                if i:
                    owner.run(steps - 1)
                    earlier, _ = snapshot(owner)
                    owner.run(1)
                w, g = snapshot(owner)
                water.append(w)
                gas.append(g)
                adjacent.append(w - earlier if i else np.zeros_like(w))
    water, gas = np.array(water), np.array(gas)
    for total in np.split(water + gas, 2, axis=0):
        nt.assert_allclose(
            total,
            total[0][None, :] + np.zeros_like(total),
            atol=2e-15,
            rtol=1e-10,
        )
    return water, gas, np.array(adjacent)


def response_times(folder, out):
    """Estimate settling time and fastest local mode at each leg endpoint."""
    with np.load(folder / "bdf-tight.npz") as data:
        traces = data["dense_water"]
        time = data["dense_time"]
        rates = data["equilibrium_rate_s_inv"]
        rows = []
        for leg, label in enumerate(("wet", "dry")):
            signal = traces[:, leg]
            difference = np.abs(signal - signal[:, -1:])
            outside = difference > 0.01 * np.abs(signal[:, -1:] - signal[:, :1])
            idx = np.where(outside, np.arange(len(time)), -1).max(1) + 1
            for i, (fid, settled, rate) in enumerate(
                zip(data["ids"], time[idx], rates[:, leg], strict=True)
            ):
                rows.append(
                    dict(
                        family=fid,
                        density_multiplier=1 + i // 2 % 2,
                        mix=i % 2,
                        leg=label,
                        settled_within_one_percent_ms=1000 * settled,
                        fastest_equilibrium_mode_us=1e6 / rate,
                        forward_euler_linear_stability_limit_us=2e6 / rate,
                    )
                )
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def diagnose(folder, output):
    """Compare native timestep convergence and sign to the stiff reference."""
    with np.load(folder / "bdf-tight.npz") as d:
        ref = d["water"]
        dry = d["dry"]
        source = d["input_sha256"].item()
    rows = []
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as d:
            if d["input_sha256"].item() != source:
                raise ValueError("Input mismatch")
            for leg, frame in (("wet", 8), ("dry", 17)):
                truth = ref[frame] / dry
                pred = d["water"][frame] / dry
                shift = pred[1::2] - pred[::2]
                signal = truth[1::2] - truth[::2]
                rows.append(
                    dict(
                        method=path.stem,
                        leg=leg,
                        boxes=len(dry),
                        mean_abs_member_error_water_dry=np.abs(
                            pred - truth
                        ).mean(),
                        mean_abs_pair_error_water_dry=np.abs(
                            shift - signal
                        ).mean(),
                        mean_abs_reference_pair_water_dry=np.abs(signal).mean(),
                        maximum_abs_member_error_mg_m3=1e6
                        * np.abs(d["water"][frame] - ref[frame]).max(),
                        last_microstep_mean_abs_water_dry=np.abs(
                            d["last_step"][frame] / dry
                        ).mean()
                        if "last_step" in d
                        else None,
                        input_sha256=source,
                    )
                )
    with output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(source, folder, steps):
    """Persist a matched reference and the actual gas and particle series."""
    from ensemble_manifest import checksum

    folder.mkdir(exist_ok=True, parents=True)
    with np.load(source) as archive:
        data = {k: archive[k] for k in archive}
    dry = (data["counts"] * data["mass"][..., 1:].sum(2)).sum(1) / float(
        data["volume"]
    )
    shared = dict(
        dry=dry,
        ids=data["ids"],
        time=data["time"],
        input_sha256=checksum(source),
    )
    if steps:
        for n in steps:
            water, gas, aux = native_reference(data, n)
            np.savez_compressed(
                folder / f"native-s{n}.npz",
                water=water,
                gas=gas,
                last_step=aux,
                **shared,
            )
            print(n, flush=True)
    else:
        for label, rtol in (("nominal", 1e-7), ("tight", 1e-10)):
            result = stiff_reference(data, rtol)
            np.savez_compressed(folder / f"bdf-{label}.npz", **result, **shared)
            print(label, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("--steps", type=int, nargs="+")
    p.add_argument("--report", type=Path)
    p.add_argument("--scales", type=Path)
    args = p.parse_args()
    if args.scales:
        response_times(args.folder, args.scales)
    elif args.report:
        diagnose(args.folder, args.report)
    else:
        generate(args.source, args.folder, args.steps)
