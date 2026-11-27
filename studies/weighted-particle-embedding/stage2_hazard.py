"""Integrate changing wet-particle hazards and locate sampled mergers."""

import torch


def partial_time(start, end, width, area):
    """Invert the integral of a nonnegative linearly varying total rate."""
    disc = (start.square() + 2 * (end - start) * (area / width)).clamp_min(0)
    return (
        (2 * area / (start + torch.sqrt(disc)).clamp_min(1e-300))
        .clamp_min(0)
        .minimum(width)
    )


def event_times(start, middle, end, dt, target):
    """Locate the first crossing on a positive piecewise-linear hazard."""
    half = dt * 0.5
    first = (start + middle) * half * 0.5
    second = (middle + end) * half * 0.5
    in_first = target <= first
    tau = torch.where(
        in_first,
        partial_time(start, middle, half, target.minimum(first)),
        half
        + partial_time(
            middle, end, half, (target - first).clamp_min(0).minimum(second)
        ),
    )
    return (
        (first + second >= target) & (first + second > 0),
        tau,
        first + second,
    )


def advance(sim, state, dt, idx, vapor_steps, order, exchange):
    """Advance vapor to each merger and restart the event hazard."""
    if not exchange:
        sim.coag(state, dt, idx)
        return
    if sim.kernel == "off":
        sim.vapor(state, dt, 2 * vapor_steps, order)
        return
    if state.pair_memory.shape[-1]:
        raise ValueError("Hazard quadrature requires a memoryless pair rate")
    unfinished = torch.arange(len(idx), device=idx.device)
    remaining_time = torch.full_like(state.volume, dt)
    while len(unfinished):
        original = state.subset(unfinished)
        duration = remaining_time[unfinished]
        start_rate = sim.pair_rates(original)[0].sum(-1) / original.volume
        trial = original.clone()
        sim.vapor(trial, duration * 0.5, vapor_steps, order)
        mid_rate = sim.pair_rates(trial)[0].sum(-1) / original.volume
        sim.vapor(trial, duration * 0.5, vapor_steps, order)
        end_rate = sim.pair_rates(trial)[0].sum(-1) / original.volume
        hit, tau, integral = event_times(
            start_rate, mid_rate, end_rate, duration, original.remaining
        )
        trial.remaining = (original.remaining - integral).clamp_min(0)
        state.put(unfinished, trial)
        local = torch.where(hit)[0]
        if not len(local):
            break
        crossing = original.subset(local)
        sim.vapor(crossing, tau[local], 2 * vapor_steps, order)
        rates = sim.pair_rates(crossing)[0]
        cdf = (rates / crossing.volume[:, None]).cumsum(-1)
        if not torch.all(cdf[:, -1] > 0):
            raise ArithmeticError(
                "A localized collision has zero physical rate"
            )
        rows = torch.arange(len(local), device=idx.device)
        active = unfinished[local]
        sim.merge_selected(crossing, rows, idx[active], cdf)
        state.put(active, crossing)
        remaining_time[active] = (duration[local] - tau[local]).clamp_min(0)
        unfinished = active[remaining_time[active] > 0]
