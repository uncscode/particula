"""Advance water on reconstructed nodes in one shared finite vapor reservoir."""

import torch
from flux_encoder import KELVIN, SATURATION, conductance


def balance(water, gas, number, increment):
    """Limit evaporation per node and allocate the available vapor."""
    gain = increment.clamp_min(0)
    loss = torch.minimum(water, (-increment).clamp_min(0))
    available = gas + (number * loss).sum(1)
    demand = (number * gain).sum(1)
    fraction = (available / demand.clamp_min(1e-30)).clamp(0, 1)
    return water - loss + gain * fraction[:, None], (
        available - demand * fraction
    ).clamp_min(0)


def implicit_increment(water, gas, number, volume_dry, capacity, h):
    """Solve particle responses coupled through the shared gas mass."""
    r = (
        ((3 / (4 * torch.pi)) * (volume_dry + water / 1000))
        .clamp_min(1e-36)
        .pow(1 / 3)
    )
    wet = water + capacity
    activity = water / wet.clamp_min(1e-36)
    kelvin = SATURATION * torch.exp(KELVIN / r.clamp_min(5e-10))
    surface = kelvin * activity
    # Sensitivity of surface concentration to per-particle water mass.
    slope = kelvin * (
        capacity / wet.square().clamp_min(1e-72)
        - activity * KELVIN / (4 * torch.pi * 1000 * r.pow(4))
    )
    hk = h[:, None] * conductance(r) * (number > 0)
    response = hk / (1 + hk * slope.clamp_min(0))
    flux = response * (gas[:, None] - surface)
    total = (number * flux).sum(1) / (1 + (number * response).sum(1))
    return (
        flux - response * total[:, None],
        hk * slope.clamp_min(0),
        hk,
        surface,
    )


def implicit_step(water, gas, number, volume_dry, capacity, h):
    """Advance the linearized exchange and enforce nonnegative inventories."""
    delta, _, _, _ = implicit_increment(
        water, gas, number, volume_dry, capacity, h
    )
    return balance(water, gas, number, delta)


def advance(
    water, gas, number, volume_dry, capacity, dt, substeps, extrapolate
):
    """Optionally extrapolate two implicit half steps for transient accuracy."""
    h = dt / substeps
    for _ in range(substeps):
        whole, g = implicit_step(water, gas, number, volume_dry, capacity, h)
        if extrapolate:
            half, gh = implicit_step(
                water, gas, number, volume_dry, capacity, h / 2
            )
            fine, _ = implicit_step(
                half, gh, number, volume_dry, capacity, h / 2
            )
            water, gas = balance(water, gas, number, 2 * fine - whole - water)
        else:
            water, gas = whole, g
    return water, gas


def rollout(
    mass, number, gas, force, substeps, extrapolate=False, return_nodes=False
):
    """Forecast water from per-node kilograms and number per cubic metre."""
    if substeps < 1:
        raise ValueError("At least one substep is required")
    water = mass[..., 0]
    volume_dry = mass[..., 1:].sum(2) / 1400.0
    capacity = (1000.0 / 1400.0) * (0.03 * mass[..., 1] + 0.75 * mass[..., 2])
    out = [water if return_nodes else (water * number).sum(1)]
    gas_history = [gas]
    for f in force.unbind(1):
        water, gas = advance(
            water,
            gas,
            number,
            volume_dry,
            capacity,
            f[:, 0],
            substeps,
            extrapolate,
        )
        gas = torch.where(f[:, 3] > 0, f[:, 4], gas)
        out.append(water if return_nodes else (water * number).sum(1))
        gas_history.append(gas)
    return torch.stack(out, 1), torch.stack(gas_history, 1)
