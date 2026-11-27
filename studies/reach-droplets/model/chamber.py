"""Frozen-parameter mesh and observation-operator study for Sagan's two expansions."""

import json, time, os
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from mesh import Grid, air, hologram, gas_path, H, R
from workspace import WORK, measurement, background

HERE = WORK / "optics"
from kernels import Chamber, torch
from geometry import inputs

MESHES = {
    "18": (6, 3, 1),
    "108": (6, 3, 6),
    "576": (8, 6, 12),
    "1944": (12, 9, 18),
    "4608": (16, 12, 24),
}
HEIGHT0 = 0.254
OPTICS = {
    "nominal": dict(z=HEIGHT0),
    "midheight": dict(z=H / 2),
    "lower": dict(z=HEIGHT0 - 0.055),
    "upper": dict(z=HEIGHT0 + 0.055),
    "shift-toward": dict(z=HEIGHT0, axial=0.04),
    "shift-away": dict(z=HEIGHT0, axial=-0.04),
    "shared-near": dict(z=HEIGHT0, axial=-0.00331 * 65 / 2, skew=0.5),
    "shared-far": dict(z=HEIGHT0, axial=0.00331 * 65 / 2, skew=-0.5),
    "cross-axis": dict(z=HEIGHT0, lateral=0.02),
    "constant-cell": dict(z=HEIGHT0, linear=False),
}


def temperature(grid, run, times):
    f = (
        measurement("forcing/figureS2.csv")
        .query("run==@run")
        .pivot(index="time_s", columns="sensor_from_center", values="temperature_K")
    )
    site = np.array([np.interp(times, f.index, f[i]) for i in range(5)]).T
    radial = np.stack([np.interp(grid.rc, np.arange(5) * 0.06, row) for row in site])
    a = np.tile(radial, (1, grid.nz * grid.na))
    return a - (a @ grid.v)[:, None]


def transfer(grid, run, flow, dt, device, scales=None):
    # An implicit update conserves inventory and preserves positivity even across small central sectors.
    G = air(grid, run, flow, scales)
    lu = splu(sparse.eye(grid.n, format="csc") - dt * G)
    mat = lu.solve(np.eye(grid.n))
    assert (
        mat.min() > -1e-12
        and np.max(abs(mat.sum(0) - 1)) < 1e-10
        and np.max(abs(mat @ grid.v - grid.v)) < 1e-12
    )
    return torch.as_tensor(mat, device=device)


def run_case(
    run,
    mesh,
    flow,
    kernel,
    dt,
    width,
    device,
    plume=False,
    ir="separate",
    scales=None,
    pair_schedule=None,
    transport_pulse=None,
    experiment=None,
):
    if kernel != "standard":
        raise ValueError("Supply a pair_schedule for an enhanced kernel")
    begin = time.monotonic()
    grid = Grid(*MESHES[mesh])
    m = Chamber(run, width, dt, device)
    put = m.tensor
    d = m.d
    times, p, base, chi, _, speed, oldloss, init, bands, targets = inputs(run, "local", d, dt)
    x = background(run)
    frac = put(grid.v[:, None])
    opt = {k: hologram(grid, d, **par)[0] for k, par in OPTICS.items()}
    weights = put(np.stack([*opt.values(), np.broadcast_to(grid.v[:, None], (grid.n, len(d)))]))
    chord = put(gas_path(grid, 0.230 if ir == "separate" else HEIGHT0, legacy=ir == "diameter"))
    generators = {
        f: transfer(grid, run, f, dt, device, scales)
        for f in (["jets", "reverse"] if flow == "switch" else [flow])
    }
    if transport_pulse is not None:
        onset, pulse_end, pulse_scales = transport_pulse
        pulse_transfer = transfer(grid, run, flow, dt, device, pulse_scales)
    temp = base[:, None] + x[0] + x[1] * temperature(grid, run, times)
    if experiment is not None:
        experiment.setup(grid, m, times)
        temp = temp + experiment.cooling(times)
    c = temp - 273.15
    es = 611.21 * np.exp((18.678 - c / 234.5) * c / (257.14 + c))
    diffusion = 2.11e-5 * (temp / 273.15) ** 1.94 * 101325 / p
    conduct = 0.024 * (temp / 273.15) ** 0.9
    res = 1000 * (
        461.5 * temp / (diffusion * es) + (2.5e6 / (461.5 * temp) - 1) * 2.5e6 / (conduct * temp)
    )
    growth = put(8e12 * (chi[:, None] * p / es - 1) / res)
    response = put(8e12 * p / (es * res))
    factor = put(p * 0.01801528 / (8.314462618 * temp))
    # Recover separate physical side and end coefficients from the earlier 6-ring x 3-height closure.
    side = oldloss[11] * (11 / 36)
    end = oldloss[0] / 3
    loss = np.zeros((*grid.shape, len(d)))
    loss[:, :, -1] += side / (1 - ((grid.nr - 1) / grid.nr) ** 2)
    loss[0] += grid.nz * end
    loss[-1] += grid.nz * end
    survive = put(np.exp(-dt * loss.reshape(grid.n, -1)))
    fall = speed * dt / grid.dz
    dest = np.arange(grid.nz)[:, None] + np.floor(fall)[None, :].astype(int)
    zi = put(np.minimum(dest, grid.nz - 1))[:, None, None, :].expand(*grid.shape, len(d))
    zj = put(np.minimum(dest + 1, grid.nz - 1))[:, None, None, :].expand_as(zi)
    wa = put((1 - fall % 1)[None, :] * (dest < grid.nz))[:, None, None, :]
    wb = put((fall % 1)[None, :] * (dest + 1 < grid.nz))[:, None, None, :]
    profile = np.exp(x[4] * (grid.rc / R - 1 / 12) / (5 / 6))
    profile = np.broadcast_to(profile, grid.shape).copy()
    # Optional unmeasured upper/lower number contrast (all sizes); antisymmetric and zero at probe heights.
    if plume:
        z = grid.zc / H - 0.5
        contrast = np.sign(z) * np.clip((abs(z) - 0.14) / 0.18, 0, 1)
        profile *= 1 + 0.8 * contrast[:, None, None]
    pr = profile.ravel()[:, None]
    pop = put(grid.v[:, None] * pr * init / (opt["nominal"] * pr).sum(0))
    gas = None
    rows = []
    maps = []
    spans = []
    ledger = []
    mass_ledger = []
    dtotal = 0
    nt = len(d)
    band = put(bands.T)
    with torch.no_grad():
        for i, t in enumerate(times):
            if experiment is not None:
                pop = experiment.relocate(t, pop)
            trans = (
                generators["jets" if t < 4.75 else "reverse"]
                if flow == "switch"
                else generators[flow]
            )
            if transport_pulse is not None and onset <= t < pulse_end:
                trans = pulse_transfer
            if gas is None:
                gas = factor[i] * chi[i]
            path = (chord @ gas) / (chord @ factor[i])
            gas = gas + factor[i] * (chi[i] - path)
            new2 = m.diam**2 + dt * (growth[i] + response[i] * (gas / factor[i] - chi[i]))[:, None]
            water = (
                1000
                * np.pi
                / 6
                * 1e-12
                * ((new2.clamp(min=0) ** 1.5 - m.vol) * pop).sum(1)
                / frac[:, 0]
            )
            wetgas = gas - water
            pos = new2.clamp(min=1e-10).sqrt() / width - 0.5
            j = pos.floor().long()
            alpha = pos - j

            def deposit(idx, weight):
                return torch.zeros_like(pop).scatter_add(
                    1, idx.clamp(0, nt - 1), pop * weight * ((idx >= 0) & (idx < nt) & (new2 > 0))
                )

            wet = deposit(j, 1 - alpha) + deposit(j + 1, alpha)
            if pair_schedule is not None:
                other, mid = pair_schedule(t)
                k = other * (m.base - m.kernel["midpairs"]) + mid * m.kernel["midpairs"]
            else:
                k = m.base
            conc = wet / frac
            chunks = []
            ev = []
            for cell, part in enumerate(conc.split(160)):
                local_kernel = (
                    k if experiment is None else experiment.kernel(t, cell * 160, len(part), k)
                )
                merged, events = m.collision(part, local_kernel, dt)
                chunks.append(merged)
                ev.append(events)
            merged = torch.cat(chunks)
            events = torch.cat(ev)
            loss_before = merged * frac * survive
            local = loss_before.reshape(*grid.shape, nt)
            settled = (local.gather(0, zi) * wa + local.gather(0, zj) * wb).reshape(grid.n, nt)
            moved = trans @ torch.cat([settled, (wetgas * frac[:, 0])[:, None]], 1)
            after = moved[:, :nt]
            gas = moved[:, nt] / frac[:, 0]
            totals = ((pop + after) / 2) / frac
            if experiment is not None:
                experiment.observe(totals)
            rows.append((weights * totals).sum(1).cpu().numpy())
            if t >= 5:
                maps.append((totals @ band).cpu().numpy())
            ledger.append(
                torch.stack(
                    [
                        (events * frac[:, 0]).sum(),
                        abs((after - settled).sum(0)).max(),
                        abs((gas - wetgas) @ frac[:, 0]),
                        merged.min(),
                        gas.min(),
                        abs(((merged - conc) @ m.vol).mul(frac[:, 0]).sum()),
                    ]
                )
                .cpu()
                .numpy()
            )
            pop = after
    ledger = np.stack(ledger)
    assert (
        ledger[:, 1:3].max() < 1e-10
        and ledger[:, 3].min() > -1e-12
        and ledger[:, 4].min() > 0
        and ledger[:, 5].max() < 1e-8
    )
    name = (
        f"{run}-{mesh}-{flow}-{kernel}"
        + ("-plume" if plume else "")
        + ("-" + ir if ir != "separate" else "")
        + f"-{dt}-{width}"
    )
    out = Path(os.getenv("OPTICAL_OUTPUT", HERE / "runs"))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / (name + ".npz"),
        t=times,
        d=d,
        obs=targets,
        bands=bands,
        spectra=np.stack(rows).astype("float32"),
        operators=np.array([*opt, "global"]),
        late_map=np.mean(maps, 0).astype("float32"),
        ledger=ledger,
    )
    meta = dict(
        run=run,
        mesh=mesh,
        n=grid.n,
        flow=flow,
        kernel=kernel,
        dt=dt,
        width=width,
        plume=plume,
        ir=ir,
        seconds=time.monotonic() - begin,
        device=torch.cuda.get_device_name() if device.startswith("cuda") else device,
    )
    (out / (name + ".json")).write_text(json.dumps(meta, indent=2))
    print(name, round(meta["seconds"], 2), flush=True)
