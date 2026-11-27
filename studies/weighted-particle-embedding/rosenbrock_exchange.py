"""Structured Rosenbrock stages for exchange with one finite gas reservoir."""

import math

import torch
from flux_encoder import KELVIN, MEAN_FREE_PATH, SATURATION, conductance
from reduced_exchange import balance

# Same autonomous ROS-2 and KPP ROS-3 tableaus as particula.execution.ode.
GAMMA3 = 0.43586652150845899941601945119356


def transport(water, gas, number, volume, capacity, jacobian):
    """Evaluate per-particle rates and the diagonal part of their Jacobian."""
    radius = (
        ((volume + water / 1000) * (3 / (4 * torch.pi)))
        .clamp_min(1e-36)
        .pow(1 / 3)
    )
    wet = (water + capacity).clamp_min(1e-36)
    activity = water / wet
    kelvin = SATURATION * torch.exp(KELVIN / radius.clamp_min(5e-10))
    drive = gas[:, None] - kelvin * activity
    conduct = conductance(radius) * (number > 0)
    rate = conduct * drive
    if not jacobian:
        return rate
    # d log(K) / d log(r) for the native transition-regime conductance.
    kn = MEAN_FREE_PATH / radius
    b = 1 + 0.283 * 0.2
    power = (
        1
        - kn / (1 + kn)
        + kn * (2 * kn + b) / (kn.square() + b * kn + 0.75 * 0.2)
    )
    radius_prime = 1 / (4 * torch.pi * 1000 * radius.square())
    kelvin_slope = torch.where(
        radius > 5e-10, KELVIN * radius_prime / radius.square(), 0.0
    )
    surface_prime = kelvin * (capacity / wet.square() - activity * kelvin_slope)
    diag = conduct * (drive * power * radius_prime / radius - surface_prime)
    # Eliminated gas has derivative -number for each particle water mass.
    return rate, diag, conduct


def factors(diagonal, conduct, number, gamma_h):
    """Factor I - gamma*h*J as a diagonal matrix plus an outer product."""
    scaled = gamma_h * diagonal
    # For positive kinetic feedback a very long step may cross a pole.
    # Use a dissipative approximate Jacobian for those boxes; report its use.
    fallback = (scaled >= 0.95).any(1, keepdim=True)
    diagonal_part = 1 - torch.where(fallback, scaled.clamp_max(0), scaled)
    response = gamma_h * conduct / diagonal_part
    return (
        diagonal_part,
        response,
        (1 + (number * response).sum(1, keepdim=True)),
        fallback,
    )


def solve(rhs, number, diagonal, response, denominator):
    """Apply the rank-one inverse without building a dense particle matrix."""
    first = rhs / diagonal
    return (
        first - response * (number * first).sum(1, keepdim=True) / denominator
    )


def step(water, gas, number, volume, capacity, h, order):
    """Advance a closed interval and report the embedded error and limiting."""
    gamma = GAMMA3 if order == 3 else 1 + 1 / math.sqrt(2)
    h = h[:, None]
    rate, diagonal, conduct = transport(
        water, gas, number, volume, capacity, True
    )
    d, response, denom, fallback = factors(diagonal, conduct, number, gamma * h)
    k1 = solve(gamma * h * rate, number, d, response, denom)
    stage = k1 if order == 3 else k1 / gamma
    stage_water, stage_gas = balance(water, gas, number, stage)
    next_rate = transport(
        stage_water, stage_gas, number, volume, capacity, False
    )
    if order == 3:
        k2 = solve(
            gamma * (h * next_rate - 1.0156171083877702092 * k1),
            number,
            d,
            response,
            denom,
        )
        k3 = solve(
            gamma
            * (
                h * next_rate
                + 4.0759956452537699825 * k1
                + 9.2076794298330791242 * k2
            ),
            number,
            d,
            response,
            denom,
        )
        delta = k1 + 6.1697947043828245593 * k2 - 0.4277225654321857333 * k3
        error = (
            0.5 * k1 - 2.9079558716805469822 * k2 + 0.2235406989781156963 * k3
        )
    elif order == 2:
        k2 = solve(gamma * h * next_rate - 2 * k1, number, d, response, denom)
        delta = (3 * k1 + k2) / (2 * gamma)
        error = (k1 + k2) / (2 * gamma)
    else:
        raise ValueError("Expected ROS order 2 or 3")
    final, g = balance(water, gas, number, delta)
    stage_clipped = ((water + stage < 0) & (number > 0)).any(1) | (
        (stage * number).sum(1) > gas
    )
    final_clipped = ((water + delta < 0) & (number > 0)).any(1) | (
        (delta * number).sum(1) > gas
    )
    return (
        final,
        g,
        error,
        torch.stack((fallback[:, 0], stage_clipped, final_clipped), 1),
    )


def rollout(mass, number, gas, force, substeps, order):
    """Return free node trajectories, gas and per-box numerical diagnostics."""
    if substeps < 1:
        raise ValueError("Expected a positive substep count")
    water = mass[..., 0]
    volume = mass[..., 1:].sum(-1) / 1400
    capacity = (1000 / 1400) * (0.03 * mass[..., 1] + 0.75 * mass[..., 2])
    series, gases = [water], [gas]
    counts = torch.zeros(
        (len(water), 3), dtype=torch.int64, device=water.device
    )
    errors = torch.zeros_like(gas)
    for f in force.unbind(1):
        for _ in range(substeps):
            water, gas, error, limited = step(
                water, gas, number, volume, capacity, f[:, 0] / substeps, order
            )
            counts += limited
            errors += (error.abs() * number).sum(1)
        gas = torch.where(f[:, 3] > 0, f[:, 4], gas)
        series.append(water)
        gases.append(gas)
    return torch.stack(series, 1), torch.stack(gases, 1), counts, errors
