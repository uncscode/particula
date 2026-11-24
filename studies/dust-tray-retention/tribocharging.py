"""Charge inherited from dry grain contacts: collision and field-drift tests."""

import numpy as np
from scipy.constants import elementary_charge as E, epsilon_0, Boltzmann, g
from scipy.linalg import expm
from transport import (
    AIR,
    DENSITY,
    HEIGHT,
    NU,
    TEMP,
    WINDS,
    apply,
    collide,
    initial,
    inputs,
    kernel,
    median,
    normalize,
    properties,
    save,
    transport,
    velocities,
)


def source(wind, n, sigma, pool="inferred"):
    d, _, obs, u, _, monitor = inputs(wind)
    raw, m = initial(wind, n, False)
    # Sum the conservative spin-up inventory to recover the inferred source by size.
    source_number = raw.sum(axis=0)
    area = np.pi * (d * 1e-6) ** 2
    if pool == "bed10":
        # Hypothetical soil-bed contacts span the sieved source sizes. Assign a
        # lognormal area distribution centered at 10 µm per log diameter.
        probability = np.exp(
            -0.5 * (np.log(d / 10.0) / np.log(1.8)) ** 2
        ) * np.gradient(np.log(d))
        probability[d > 63.0] = 0
        source_number = probability / area
    p = normalize(source_number * area)
    # Area-weighted pair contacts. Each pair transfers equal and opposite charge;
    # a grain gains positive charge when its partner is smaller.
    signed_exposure = np.cumsum(p) - p - (1 - np.cumsum(p))
    z = sigma * 1e-6 * area / E * signed_exposure
    assert abs(source_number @ z) <= 1e-12 * max(1.0, source_number @ abs(z))
    x = raw.copy()
    x[:n] *= monitor / (x[:n].mean(axis=0) @ m * 1e6)
    return d, m, obs, u, z, x


def coulomb(phi):
    # phi > 0 means attraction. Evaluate the continuum multiplier stably.
    out = np.ones_like(phi)
    large = phi > 50
    out[large] = phi[large]
    out[phi < -700] = 0.0
    mid = (abs(phi) > 1e-12) & (phi >= -700) & ~large
    out[mid] = phi[mid] / (-np.expm1(-phi[mid]))
    return out


def charge_rate(d, m, n, u):
    i, j = np.triu_indices(len(m))
    ix = np.arange(len(i))
    radius = d * 0.5e-6
    inverse_contact_energy = -(E**2) / (
        4 * np.pi * epsilon_0 * Boltzmann * TEMP * (radius[i] + radius[j])
    )
    brown = kernel(d, n, u, 0, "neutral", "brownian")[:, i, j]
    other = (
        kernel(d, n, u, 0, "neutral", "shear")[:, i, j]
        + kernel(d, n, u, 0, "neutral", "differential")[:, i, j]
    )
    births = np.zeros((len(i), len(d)))
    product = m[i] + m[j]
    hi = np.searchsorted(m, product)
    within = hi < len(d)
    upper = hi[within]
    w = (product[within] - m[upper - 1]) / (m[upper] - m[upper - 1])
    births[ix[within], upper] = w
    births[ix[within], upper - 1] = 1 - w
    births[~within, -1] = 1.0
    net = births.copy()
    net[~within, -1] = product[~within] / m[-1]
    net[ix, i] -= 1.0
    net[ix, j] -= 1.0
    first = births.copy()
    first[ix, i] -= 1.0
    second = births.copy()
    second[ix, j] -= 1.0
    pair_weight = np.where(i == j, 0.5, 1.0)

    def rate(y):
        number = y[0]
        tags = np.divide(
            y[1:], number, out=np.zeros_like(y[1:]), where=number > 0
        )
        z = tags[0] - tags[1]
        enhancement = coulomb(inverse_contact_energy * z[:, i] * z[:, j])
        events = (
            number[:, i]
            * number[:, j]
            * pair_weight
            * (brown * enhancement + other)
        )
        # Conserve positive and negative source tags individually, so net charge
        # of a merger is the sum of its parents' charges.
        dtag = (events[None, :, :] * tags[:, :, i]) @ first + (
            events[None, :, :] * tags[:, :, j]
        ) @ second
        return np.concatenate(((events @ net)[None, :, :], dtag))

    return rate


def interacting(wind, sigma, n, dt, decay_min, pool="inferred"):
    d, m, obs, u, z, x = source(wind, n, sigma, pool)
    # A decay time, if specified, runs from source release. Count leaked charge
    # separately after conditioning the airborne mass at 30 minutes.
    z = z * np.exp(-30 / decay_min)
    y = np.stack((x, x * np.maximum(z, 0), x * np.maximum(-z, 0)))
    tags0 = y[1:].sum(axis=(1, 2))
    leaked = np.zeros(2)
    mass0 = x.sum(axis=0) @ m
    number0 = x.sum()
    steps = int(np.ceil(9000 / dt))
    h = 9000 / steps
    move = transport(d, n, u, 0, 1, h / 2)
    rate = charge_rate(d, m, n, u)
    for _ in range(steps):
        for _half in range(2):
            y = np.stack([apply(move, field) for field in y])
            removed = y[1:] * (-np.expm1(-h / (120 * decay_min)))
            y[1:] -= removed
            leaked += removed.sum(axis=(1, 2))
            if _half == 0:
                y[:, :n] = collide(y[:, :n], rate, h)
    charge_error = max(abs(y[1:].sum(axis=(1, 2)) + leaked - tags0)) / max(
        1.0, sum(tags0)
    )
    mass_error = abs(y[0].sum(axis=0) @ m / mass0 - 1)
    assert charge_error < 5e-9 and mass_error < 5e-9 and np.min(y) >= 0
    return summary(
        "collisions",
        wind,
        d,
        m,
        obs,
        y[0, n],
        sigma,
        0,
        decay_min,
        n,
        dt,
        charge_error,
        mass_error,
        100 * (1 - y[0].sum() / number0),
        pool,
    )


def summary(
    mechanism,
    wind,
    d,
    m,
    obs,
    floor,
    sigma,
    field,
    decay,
    n,
    dt,
    qerr,
    merr,
    mergers,
    pool,
):
    retained = normalize((floor * m)[:-7])
    return dict(
        mechanism=mechanism,
        contact_pool=pool,
        wind_ms=wind,
        sigma_scale_uC_m2=sigma,
        field_upward_V_m=field,
        charge_lifetime_min=decay,
        layers=n,
        step_s=dt,
        median_um=median(d[:-7], retained),
        measured_um=median(d[:-7], obs[:-7]),
        middle_pct=100 * retained[(d[:-7] > 10) & (d[:-7] <= 20)].sum(),
        merger_number_loss_pct=mergers,
        charge_error_rel=qerr,
        mass_error_rel=merr,
    )


def drift(wind, sigma, field, n=12, pool="inferred"):
    d, m, obs, u, z, x = source(wind, n, sigma, pool)
    # Steady imposed vertical field after 30 min. The sign chooses the collecting
    # boundary; the effective buoyant weight gives the terminal speed magnitude.
    vol = m / DENSITY
    net_density = DENSITY - AIR - z * E * field / (vol * g)
    v = np.sign(net_density) * velocities(d, AIR + abs(net_density))
    dz = HEIGHT / n
    edge = dz * np.arange(1, n)
    mix = 0.4 * u * edge * (1 - edge / HEIGHT) / dz**2
    _, _, _, diff = properties(d, 0.0)
    side = 4 / (np.pi * 2) * np.sqrt(u * u / NU * diff)
    generator = np.zeros((len(d), n + 3, n + 3))
    for j, k in enumerate(mix):
        generator[:, j, j + 1] = k
        generator[:, j + 1, j] = k
    for j in range(n):
        generator[:, j - 1 if j else n, j] += np.maximum(v, 0) / dz
        generator[:, j + 1 if j + 1 < n else n + 2, j] += np.maximum(-v, 0) / dz
        generator[:, n + 1, j] += side
        generator[:, j, j] = -generator[:, :, j].sum(axis=1)
    y = apply(expm(generator * 9000), np.vstack((x, np.zeros(len(d)))))
    difference = y.sum(axis=0) - x.sum(axis=0)
    merr = abs(difference @ m) / (x.sum(axis=0) @ m)
    qerr = abs(difference @ z) / max(1.0, x.sum(axis=0) @ abs(z))
    assert y.min() >= 0 and max(qerr, merr) < 5e-9
    return summary(
        "drift",
        wind,
        d,
        m,
        obs,
        y[n],
        sigma,
        field,
        np.inf,
        n,
        9000,
        qerr,
        merr,
        0.0,
        pool,
    )


def main():
    rows = []
    for wind in WINDS:
        for sigma in (0.0, 1.0, 10.0):
            rows.append(interacting(wind, sigma, 12, 120, np.inf))
        for field in (0.0, -1000.0, 1000.0):
            rows.append(drift(wind, 10.0, field))
    for tau in (10.0, 60.0):
        rows.append(interacting(2, 10.0, 12, 120, tau))
    rows.append(interacting(2, 10.0, 24, 60, np.inf))
    for field in (-10000.0, 10000.0):
        rows.append(drift(2, 10.0, field))
    for sigma in (1.0, 10.0):
        rows.append(interacting(2, sigma, 12, 120, np.inf, "bed10"))
    for field in (-1000.0, 1000.0, -10000.0, 10000.0):
        rows.append(drift(2, 10.0, field, pool="bed10"))
    save("tribo-cases.csv", rows)
    zrows = []
    for wind in WINDS:
        d, m, _, _, z, _ = source(wind, 12, 1.0)
        for diameter, mass, charge in zip(d[:-7], m[:-7], z[:-7]):
            zrows.append(
                dict(
                    wind_ms=wind,
                    diameter_um=diameter,
                    charge_e_at_1uC_m2=charge,
                    field_for_buoyant_weight_V_m=(
                        mass * (1 - AIR / DENSITY) * g / abs(charge * E)
                        if charge
                        else np.inf
                    ),
                )
            )
    save("tribo-source-charge.csv", zrows)


if __name__ == "__main__":
    main()
