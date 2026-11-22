"""Common schedule, budgets and device dispatch for parcel experiments."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import warp as wp

from particula.dynamics.parcel import RV, WATER_FACTOR, _saturation_pressure
from particula.dynamics.parcel_replay import ReplayCheckpoint, ReplayResult
from particula.gpu.kernels import parcel_replay as kernels


@dataclass
class ClusterResult:
    """Common observations and cumulative donor-to-receiver transfers."""

    time: np.ndarray
    parcels: dict
    transfers: dict
    inventories_transferred: dict
    totals: dict
    schedule: np.ndarray
    flow_gradient: float | None


def _schedule(traces, start, end, max_step, contacts=()):
    records = np.unique(
        np.concatenate(
            [t.time[(t.time >= start) & (t.time <= end)] for t in traces]
            + [np.asarray(contacts, dtype=float)]
        )
    )
    dt = np.diff(records)
    ratios = dt / max_step
    counts = np.maximum(
        1,
        np.ceil(
            ratios
            - 32
            * np.finfo(float).eps
            * np.maximum(
                1, np.maximum(abs(records[:-1]), abs(records[1:])) / max_step
            )
        ),
    ).astype(int)
    steps = np.concatenate(
        [
            [start],
            *[
                np.linspace(a, b, n + 1)[1:]
                for a, b, n in zip(
                    records[:-1], records[1:], counts, strict=True
                )
            ],
        ]
    )
    return steps, np.r_[0, np.cumsum(counts)]


def _forcing(trace, steps, amplitude):
    def interp(values):
        return np.interp(steps, trace.time, values)

    volume = interp(trace.volume)
    q = interp(trace.vapor) * trace.initial.dry_air_mass
    ramp = (steps - trace.time[0]) / (trace.time[-1] - trace.time[0])
    vapor = q * (1 + amplitude * ramp)
    # Midpoint values for growth. The trace linearly prescribes state fields.
    mid = (steps[:-1] + steps[1:]) * 0.5

    def midpoint(a):
        return np.interp(mid, trace.time, a)

    idx = np.searchsorted(trace.time, mid, side="right") - 1
    rate = (
        np.array(trace.organic.rates)[idx]
        if trace.organic.rates
        else np.zeros(len(idx))
    )
    total = np.zeros(len(idx))
    if trace.organic.totals:
        total = np.where(
            steps[1:] == trace.time[idx + 1],
            np.asarray(trace.organic.totals)[idx],
            0.0,
        )
    forcing = np.column_stack(
        (
            midpoint(trace.temperature),
            midpoint(trace.pressure),
            midpoint(trace.vapor)
            * trace.initial.dry_air_mass
            / midpoint(trace.volume),
            (mid - trace.time[0]) / (trace.time[-1] - trace.time[0]),
            rate,
            total,
        )
    )
    return volume, vapor, forcing


def _initial(traces, bins):
    inventory = np.zeros((len(traces), kernels.FIRST_BIN + 3 * bins))
    dry = np.ones((len(traces), bins))
    for c, t in enumerate(traces):
        d = t.initial.dry_radius
        n = t.initial.number
        wet = t.initial_wet
        dry[c, : len(d)] = d
        row = inventory[c]
        row[0] = t.organic.initial_gas
        row[4 : 4 + 3 * len(n) : 3] = n
        row[5 : 5 + 3 * len(n) : 3] = (
            WATER_FACTOR * n * (wet - d) * (wet * wet + wet * d + d * d)
        )
    return inventory, dry


def _restore(trace, species, checkpoint, max_step, amplitude):
    if checkpoint is None:
        return 0
    i = checkpoint.index
    if (
        checkpoint.trace_id != trace.identity()
        or checkpoint.parcel_id != trace.parcel_id
        or checkpoint.species != species
        or type(i) is not int
        or not 0 <= i < len(trace.time)
        or checkpoint.time != trace.time[i]
        or checkpoint.max_step != max_step
        or checkpoint.humidity_amplitude != amplitude
    ):
        raise ValueError(
            "checkpoint does not match trace, owners or integration schedule"
        )
    return i


def _exchange_schedule(ids, connections, steps, volumes):
    pairs = [(e.donor, e.receiver) for e in connections]
    edges = [
        [ids.index(e.donor), ids.index(e.receiver), e.gas, e.particles, e.water]
        for e in connections
    ]
    flows = [e.flow for e in connections]
    dt = np.diff(steps)
    enabled = np.ones((len(dt), len(edges)))
    for e, connection in enumerate(connections):
        if connection.active_intervals:
            enabled[:, e] = np.logical_or.reduce(
                [
                    (steps[:-1] >= a) & (steps[:-1] < b)
                    for a, b in connection.active_intervals
                ]
            )
    for i in range(len(ids)):
        for flag in (2, 3, 4):
            outgoing = sum(
                q * enabled[:, j]
                for j, (edge, q) in enumerate(zip(edges, flows, strict=True))
                if edge[0] == i and edge[flag]
            )
            if np.any(dt * outgoing > volumes[:-1, i]):
                raise ValueError(
                    "aggregate outgoing flow exceeds donor inventory"
                )
    return pairs, edges, flows, enabled


def _observation(ids, pairs, humidity_target, flow_target):
    if flow_target is not None:
        if (
            len(flow_target) not in (3, 4)
            or tuple(flow_target[:2]) not in pairs
            or flow_target[2] not in ids
            or (
                len(flow_target) == 4
                and flow_target[3] not in ("gas", "particle")
            )
        ):
            raise ValueError("unknown gradient connection or observation")
        lane = 0 if len(flow_target) == 4 and flow_target[3] == "gas" else 2
        return ids.index(flow_target[2]), lane
    if humidity_target is not None:
        return ids.index(humidity_target), 1
    return None


def _face_ledgers(fluxes, obs, pairs, edges, dry, traces):
    transfers = {}
    detailed = {}
    if fluxes is not None:
        accumulated = np.concatenate(
            [
                np.zeros((1, len(edges), fluxes.shape[2])),
                np.cumsum(fluxes, axis=0),
            ]
        )[obs]
        for e, pair in enumerate(pairs):
            lanes = accumulated[:, e]
            transfers[pair] = lanes[:, 0]
            detailed[pair] = dict(
                organic_gas=lanes[:, 0],
                water_vapor=lanes[:, 3],
                number=lanes[:, 4::3],
                bin_water=lanes[:, 5::3],
                bin_organic=lanes[:, 6::3],
                dry_volume=lanes[:, 4::3]
                * dry[edges[e][0]] ** 3
                * (4 * np.pi / 3),
            )
            density = traces[edges[e][0]].dry_density
            if density is not None:
                detailed[pair]["dry_mass"] = (
                    detailed[pair]["dry_volume"] * density
                )
    return transfers, detailed


def _parcel_result(
    t,
    s,
    times,
    volume,
    vapor,
    species,
    end,
    max_step,
    amplitude,
    humidity_gradient,
):
    b = len(t.initial.number)
    n = s[:, 4 : 4 + 3 * b : 3]
    water = s[:, 5 : 5 + 3 * b : 3]
    organic = s[:, 6 : 6 + 3 * b : 3]
    liquid = water.sum(axis=1)
    radius = np.cbrt(
        t.initial.dry_radius**3
        + np.divide(
            water, WATER_FACTOR * n, out=np.zeros_like(water), where=n > 0
        )
    )
    temperature = np.interp(times, t.time, t.temperature)
    sat = (
        vapor / volume * RV * temperature / _saturation_pressure(temperature)
        - 1
    )
    restored = vapor - t.vapor[0] * t.initial.dry_air_mass + s[:, 2] - s[:, 3]
    saved = None
    if species is not None:
        saved = ReplayCheckpoint(
            t.identity(),
            t.parcel_id,
            species,
            end,
            float(times[-1]),
            max_step,
            amplitude,
            tuple(s[-1].tolist()),
        )
    gas = {"water": vapor, t.organic.name: s[:, 0]}
    names = tuple(gas) if species is None else species
    dry_mass = (
        None
        if t.dry_density is None
        else (4 * np.pi / 3) * t.dry_density * t.initial.dry_radius**3 * n
    )
    return ReplayResult(
        times,
        volume,
        radius,
        liquid,
        sat,
        s[:, 2],
        restored,
        s[:, 0],
        organic.sum(axis=1),
        s[:, 1],
        n,
        water,
        organic,
        names,
        np.column_stack([gas[name] for name in names]),
        dry_mass,
        humidity_gradient,
        saved,
    )


def _experiment_times(
    traces, connections, max_step, amplitude, species, checkpoint, stop_index
):
    if (
        not np.isfinite(max_step)
        or max_step <= 0
        or not np.isfinite(amplitude)
        or amplitude < -1
    ):
        raise ValueError(
            "require positive max_step and physical humidity amplitude"
        )
    first = traces[0]
    if any(
        t.time[0] != first.time[0] or t.time[-1] != first.time[-1]
        for t in traces
    ):
        raise ValueError("parcel histories must cover the common duration")
    index = _restore(first, species, checkpoint, max_step, amplitude)
    end = len(first.time) - 1 if stop_index is None else stop_index
    if type(end) is not int or not index <= end < len(first.time):
        raise ValueError("restart and stop must be ordered forcing records")
    start_time, end_time = first.time[[index, end]]
    contacts = [
        point
        for e in connections
        for span in e.active_intervals
        for point in span
        if start_time < point < end_time
    ]
    steps, obs = _schedule(traces, start_time, end_time, max_step, contacts)
    return steps, obs, end


def _restore_inventory(checkpoint, initial):
    if checkpoint is not None:
        v = np.asarray(checkpoint.inventory, dtype=float)
        if (
            v.shape != initial[0].shape
            or not np.isfinite(v).all()
            or v[0] < 0
            or np.any(v[4:] < 0)
        ):
            raise ValueError("invalid checkpoint inventory")
        initial[0] = v


def integrate(
    trace_map,
    *,
    connections=(),
    max_step,
    device,
    amplitude=0.0,
    flow_target=None,
    humidity_target=None,
    checkpoint=None,
    stop_index=None,
    species=None,
):
    """Integrate local growth and exchange on a recorded fixed mesh."""
    ids = list(trace_map)
    traces = list(trace_map.values())
    steps, obs, end = _experiment_times(
        traces,
        connections,
        max_step,
        amplitude,
        species,
        checkpoint,
        stop_index,
    )
    target_device = wp.get_device(
        "cpu" if device in ("cpu", "warp:cpu") else device
    )
    nc = len(ids)
    bins = max(len(t.initial.number) for t in traces)
    initial, dry = _initial(traces, bins)
    _restore_inventory(checkpoint, initial)
    data = [_forcing(t, steps, amplitude) for t in traces]
    volumes, vapors, forcing = (
        np.stack(v, axis=1) for v in zip(*data, strict=True)
    )
    pairs, edges, flows, enabled = _exchange_schedule(
        ids, connections, steps, volumes
    )
    dt = np.diff(steps)
    observation = _observation(ids, pairs, humidity_target, flow_target)
    grad = observation is not None
    opts = dict(dtype=wp.float64, device=target_device)

    def arr(v):
        return wp.array(np.ascontiguousarray(v), **opts)

    state = wp.empty((len(steps), *initial.shape), requires_grad=grad, **opts)
    state[0].assign(initial)
    dry_dev = arr(dry)
    kap = arr([t.initial.kappa for t in traces])
    uptake = arr([t.organic.uptake_rate for t in traces])
    amplitude_dev = wp.array(np.full(nc, amplitude), requires_grad=grad, **opts)
    counts = wp.empty((len(steps), nc), requires_grad=grad, **opts)
    if len(dt):
        force_dev = arr(forcing)
    if edges:
        edge_dev = wp.array(edges, dtype=wp.int32, device=target_device)
        rates = wp.array(flows, requires_grad=grad, **opts)
        contacts_dev = arr(enabled)
        volume_dev = arr(volumes)
        vapor_dev = arr(vapors)
        flux = wp.empty(
            (len(dt), len(edges), initial.shape[1]), requires_grad=grad, **opts
        )
        mixed = wp.empty((len(dt), *initial.shape), requires_grad=grad, **opts)
    tape = wp.Tape() if grad else None
    with tape if tape else nullcontext():
        for i, h in enumerate(dt):
            prev = state[i]
            if edges:
                wp.launch(
                    kernels.exchange,
                    (len(edges), initial.shape[1]),
                    [
                        prev,
                        volume_dev[i],
                        edge_dev,
                        rates,
                        contacts_dev[i],
                        vapor_dev[i],
                        h,
                    ],
                    [flux[i]],
                    device=target_device,
                )
                wp.launch(
                    kernels.gather,
                    initial.shape,
                    [prev, edge_dev, flux[i]],
                    [mixed[i]],
                    device=target_device,
                )
                prev = mixed[i]
            wp.launch(
                kernels.population_count,
                nc,
                [prev],
                [counts[i]],
                device=target_device,
            )
            wp.launch(
                kernels.local,
                nc,
                [
                    prev,
                    force_dev[i],
                    dry_dev,
                    kap,
                    uptake,
                    counts[i],
                    h,
                    amplitude_dev,
                ],
                [state[i + 1]],
                device=target_device,
            )
        if grad:
            cell, lane = observation
            loss = wp.empty(1, requires_grad=True, **opts)
            wp.launch(
                kernels.observe,
                1,
                [state[len(steps) - 1], cell, lane],
                [loss],
                device=target_device,
            )
    humidity_gradient = flow_gradient = None
    if tape is not None:
        tape.backward(loss)
        if humidity_target:
            humidity_gradient = float(amplitude_dev.grad.numpy().sum())
        else:
            flow_gradient = float(
                rates.grad.numpy()[pairs.index(tuple(flow_target[:2]))]
            )
    history = state.numpy()
    if (
        not np.isfinite(history).all()
        or np.any(history[:, :, 0] < 0)
        or np.any(history[:, :, 4:] < 0)
    ):
        raise ValueError("nonphysical inventory; check forcing and timestep")
    parcels = {
        t.parcel_id: _parcel_result(
            t,
            history[obs, c],
            steps[obs],
            volumes[obs, c],
            vapors[obs, c],
            species,
            end,
            max_step,
            amplitude,
            humidity_gradient if t.parcel_id == humidity_target else None,
        )
        for c, t in enumerate(traces)
    }
    transfers, detailed = _face_ledgers(
        flux.numpy() if edges else None, obs, pairs, edges, dry, traces
    )
    totals = {
        field: sum(getattr(p, field) for p in parcels.values())
        for field in (
            "liquid",
            "organic_gas",
            "organic_particle",
            "water_exchange",
            "water_restored",
        )
    }
    totals["number"] = sum(p.number.sum(axis=1) for p in parcels.values())
    return ClusterResult(
        steps[obs], parcels, transfers, detailed, totals, steps, flow_gradient
    )
