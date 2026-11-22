"""Conditional particle-resolved acid uptake, proton transfer and first passage on CUDA.

Each CUDA thread traces one growing particle in chemostatted CLOUD gas. A first-order
external loss hazard thins its statistical weight exactly between chemical events.
"""

import argparse
import json
import numpy as np
import warp as wp
import particula as par
from attraction import acid_pair
from particula.dynamics import get_brownian_kernel_via_system_state
from particula.gpu.dynamics.condensation_funcs import (
    first_order_mass_transport_k_wp,
)
from particula.gpu.properties.particle_properties import (
    vapor_transition_correction_wp,
)
from reference import ACID, AIR, NH3, OUT, T, dump_csv

wp.config.kernel_cache_dir = str(OUT / "warp-cache")
F = wp.float64
AMU = wp.constant(wp.float64(1.66053906660e-27))
RHO = wp.constant(wp.float64(1500.0))


@wp.func
def transfer(d_nm: F, molar: F, diff: F, temperature: F, sticking: F):
    radius = d_nm * F(0.5e-9)
    speed = wp.sqrt(F(8.0 * 8.314462618 / np.pi) * temperature / molar)
    path = F(3.0) * diff / speed
    transition = vapor_transition_correction_wp(path / radius, sticking)
    return first_order_mass_transport_k_wp(radius, transition, diff)


@wp.kernel
def follow(
    params: wp.array2d(dtype=F),
    bg: wp.array2d(dtype=F),
    states: wp.array3d(dtype=F),
    count: int,
    seed: int,
):
    case, track = wp.tid()
    rng = wp.rand_init(seed, track)  # paired experiments share streams
    temp = params[case, 0]
    acid = params[case, 1]
    base = params[case, 2]
    wall = params[case, 3]
    dil = params[case, 4]
    background = params[case, 5]
    hydrate = params[case, 6]
    capacity = params[case, 7]
    stick = params[case, 8]
    mode = int(params[case, 9])
    chunk = params[case, 10]
    offset = params[case, 11]
    d0 = F(1.7) - offset
    unit_a = F(98.079) + hydrate * F(18.01528)
    seed_mass = F(np.pi / 6.0) * RHO * wp.pow(d0 * F(1.0e-9), F(3.0)) / AMU
    seed_a = seed_mass / (unit_a + capacity * F(17.03052))
    na = seed_a
    nb = seed_a * capacity
    d = d0
    clock = F(0.0)
    weight = F(1.0)
    cross = bool(False)
    wall_count = F(0.0)
    dil_count = F(0.0)
    coag_count = F(0.0)
    parked_a = F(0.0)
    parked_b = F(0.0)
    supply_a = F(0.0)
    supply_b = F(0.0)
    occupancy = F(0.0)
    early = F(0.0)
    early_time = F(0.0)
    for event in range(60000):
        if d + offset >= F(10.0):
            break
        ka = transfer(
            d,
            F(0.098079),
            F(8e-6) * wp.pow(temp / F(293.15), F(1.75)),
            temp,
            F(1.0),
        )
        kb = transfer(
            d,
            F(0.01703052),
            F(2.1e-5) * wp.pow(temp / F(293.15), F(1.75)),
            temp,
            F(1.0),
        )
        if mode == 4 or mode == 5:
            frac = wp.clamp((d - F(1.0)) * F(10.0), F(0.0), F(109.999))
            col = int(frac)
            ka = bg[case, 111 + col] + (frac - F(col)) * (
                bg[case, 112 + col] - bg[case, 111 + col]
            )
        ra = ka * acid * F(1e6) * stick
        vacancies = wp.max(F(0.0), capacity * na - nb)
        rb = kb * base * F(1e6) * wp.min(F(1.0), vacancies / chunk)
        if mode == 1 or mode == 2 or mode == 3:
            # Prescribed diameter-growth controls. Stoichiometric increments
            # follow the source composition so chemistry cannot double-count growth.
            gr = F(2.0)
            if mode == 1:
                gr = (
                    (F(2.68) * wp.pow(d + offset, F(-1.27)) + F(0.81))
                    * acid
                    / F(1e7)
                )
            if mode == 3:
                gr = params[case, 12]
            volume_slope = F(1.5707963267948966) * RHO * d * d * F(1e-27) / AMU
            ra = (
                gr
                / F(3600.0)
                * volume_slope
                / (unit_a + capacity * F(17.03052))
            )
            rb = F(0.0)
        rate = (ra + rb) / chunk
        wait = -wp.log(wp.max(F(1e-12), F(wp.randf(rng)))) / rate
        da = F(0.0)
        db = F(0.0)
        if F(wp.randf(rng)) * (ra + rb) < ra or rb == F(0.0):
            da = chunk
            if mode == 1 or mode == 2 or mode == 3:
                db = capacity * chunk
        else:
            db = wp.min(chunk, vacancies)
        old_d = d
        new_mass = (na + da) * unit_a + (nb + db) * F(17.03052)
        d = wp.pow(new_mass * AMU * F(6.0 / np.pi) / RHO, F(1.0 / 3.0)) * F(1e9)
        if mode == 2:
            wait = (d - old_d) * F(1800.0)  # exact constant 2 nm/h
        wloss = wall / old_d
        # Tabulated Particula Brownian kernel for capture by a 50 nm resident.
        pos = wp.clamp((old_d - F(1.0)) * F(10.0), F(0.0), F(109.999))
        j = int(pos)
        kk = bg[case, j] + (pos - F(j)) * (bg[case, j + 1] - bg[case, j])
        if bg.shape[1] >= 333:
            wloss = wall * (
                bg[case, 222 + j]
                + (pos - F(j)) * (bg[case, 223 + j] - bg[case, 222 + j])
            )
        closs = kk * background * F(1e6)
        hazard = wloss + dil + closs
        remain = wp.exp(-hazard * wait)
        lost = weight * (F(1.0) - remain)
        if hazard > F(0.0):
            wall_count += lost * wloss / hazard
            dil_count += lost * dil / hazard
            coag_count += lost * closs / hazard
            occupancy += lost / hazard
        else:
            occupancy += weight * wait
        parked_a += lost * na
        parked_b += lost * nb
        weight *= remain
        supply_a += weight * da
        supply_b += weight * db
        na += da
        nb += db
        clock += wait
        if not cross and d + offset >= F(3.2):
            cross = True
            early = weight
            early_time = clock
    states[case, track, 0] = early
    states[case, track, 1] = weight
    states[case, track, 2] = early_time
    states[case, track, 3] = clock
    states[case, track, 4] = wall_count
    states[case, track, 5] = dil_count
    states[case, track, 6] = coag_count
    states[case, track, 7] = seed_a + supply_a - parked_a - weight * na
    states[case, track, 8] = (
        seed_a * capacity + supply_b - parked_b - weight * nb
    )
    states[case, track, 9] = supply_a
    states[case, track, 10] = supply_b
    states[case, track, 11] = occupancy
    states[case, track, 12] = d + offset
    states[case, track, 13] = nb / na


COLUMNS = [
    "survival32",
    "survival10",
    "time32_s",
    "time10_s",
    "wall_loss",
    "dil_loss",
    "coag_loss",
    "acid_balance_molecules",
    "base_balance_molecules",
    "acid_uptake_molecules",
    "base_uptake_molecules",
    "person_seconds",
    "endpoint_nm",
    "ammonia_per_acid",
]
# Fixed background concentrations are deliberate chamber sensitivities; the archived
# table gives no run-specific distribution. Self-coagulation is examined separately.
SCENARIOS = {
    "acid_base": {},
    "acid_only": {"capacity": 0},
    "bisulfate": {"capacity": 1},
    "dry": {"hydrate": 0},
    "two_water": {"hydrate": 2},
    "slow_uptake": {"stick": 0.5},
    "weak_wall": {"wall": 0.0005},
    "strong_wall": {"wall": 0.002},
    "no_dilution": {"dil": 0},
    "no_wall": {"wall": 0},
    "bg_1000": {"background": 1000},
    "bg_10000": {"background": 10000},
    "unit_offset_zero": {"offset": 0},
    "warm_law_transfer": {"mode": 1},
    "no_losses": {"wall": 0, "dil": 0},
    "constant_oracle": {"mode": 2, "chunk": 0.125},
    "quarter_event": {"chunk": 0.25},
    "fixed_1nmh": {"mode": 3, "gr": 1},
    "fixed_5nmh": {"mode": 3, "gr": 5},
}
SUPPLEMENT = {
    "acid_pair_hard": {"mode": 4},
    "acid_pair_vdw": {"mode": 5},
    "acid_pair_vdw_bisulfate": {"mode": 5, "capacity": 1, "hydrate": 2},
    "acid_pair_vdw_weakwall": {"mode": 5, "wall": 0.0005},
    "acid_pair_vdw_strongwall": {"mode": 5, "wall": 0.002},
}
BASE = dict(
    wall=0.001,
    dil=9.6e-5,
    background=0,
    hydrate=1.0,
    capacity=2.0,
    stick=1.0,
    mode=0,
    chunk=1.0,
    offset=0.3,
    gr=0.0,
)
LINES = [332, 334, 336, 338, 344, 345, 352, 366, 367, 369, 6, 7]


def run(n, rep, scenarios, lines, folder):

    folder.mkdir(parents=True, exist_ok=True)
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA required for this study replay")
    dev = wp.get_device("cuda:0")
    labels = []
    params = []
    kernels = []
    for line in lines:
        i = line - 2
        diameters = np.r_[np.arange(111) * 0.1 + 1.0, 50.0]
        radius = diameters * 0.5e-9
        mass = 4 * np.pi / 3 * radius**3 * 1500
        brown = get_brownian_kernel_via_system_state(
            radius, mass, T[i], 1.005 * 101325
        )[:-1, -1]
        for scenario in scenarios:
            v = {**BASE, **{**SCENARIOS, **SUPPLEMENT}[scenario]}
            labels.append((line, scenario))
            params.append(
                [T[i], ACID[i] * 1e6, NH3[i] * AIR[i] * 1e-12, *v.values()]
            )
            acid_table = np.zeros(111)
            if v["mode"] in [4, 5]:
                grid, table = acid_pair(
                    T[i],
                    1.005 * 101325,
                    6.4e-20 if v["mode"] == 5 else 0.0,
                    v["hydrate"],
                )
                acid_table = np.interp(diameters[:-1], grid, table)
            kernels.append(np.r_[brown, acid_table])
    p = wp.array(np.array(params), dtype=F, device=dev)
    k = wp.array(np.array(kernels), dtype=F, device=dev)
    summary = []
    for seed in range(rep):
        # Batching constrains host and GPU memory for large campaigns.
        for start in range(0, len(labels), 12):
            end = min(start + 12, len(labels))
            size = end - start
            a = wp.zeros((size, n, len(COLUMNS)), dtype=F, device=dev)
            wp.launch(
                follow,
                dim=(size, n),
                inputs=[p[start:end], k[start:end], a, n, 13514 + seed],
                device=dev,
            )
            values = a.numpy()
            if not np.all(np.isfinite(values)) or np.min(values[:, :, 12]) < 10:
                raise RuntimeError("Invalid or truncated paths")
            if (
                np.max(
                    np.abs(values[:, :, 1] + values[:, :, 4:7].sum(axis=2) - 1)
                )
                > 1e-9
            ):
                raise RuntimeError("Number closure failed")
            if np.max(np.abs(values[:, :, 7:9])) > 1e-7:
                raise RuntimeError("Species closure failed")
            for j, (line, scenario) in enumerate(labels[start:end]):
                item = dict(
                    line=line, scenario=scenario, seed=13514 + seed, tracks=n
                )
                item.update(zip(COLUMNS, values[j].mean(axis=0)))
                item["se32"] = values[j, :, 0].std(ddof=1) / np.sqrt(n)
                item["se10"] = values[j, :, 1].std(ddof=1) / np.sqrt(n)
                summary.append(item)
        dump_csv(folder / "particles.csv", summary)
        print("finished seed", seed, len(summary), flush=True)
    (folder / "device.json").write_text(
        json.dumps(
            dict(
                device=str(dev.name),
                warp=wp.__version__,
                particula=par.__version__,
                source=par.__file__,
                tracks=n,
                replicates=rep,
                scenarios={
                    s: {**BASE, **{**SCENARIOS, **SUPPLEMENT}[s]}
                    for s in scenarios
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--tracks", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--attraction", action="store_true")
    args = parser.parse_args()
    run(
        args.tracks,
        args.reps,
        ["acid_base", "constant_oracle", "no_losses"]
        if args.pilot
        else list(SUPPLEMENT if args.attraction else SCENARIOS),
        [336, 366] if args.pilot else LINES,
        OUT
        / (
            "cuda_pilot"
            if args.pilot
            else ("cuda_attraction" if args.attraction else "cuda")
        ),
    )
