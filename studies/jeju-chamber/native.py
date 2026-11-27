"""Tabulate Particula's native gas, settling and thermal encounter calculations."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import particula as par
from particula.dynamics.coagulation.brownian_kernel import (
    get_brownian_kernel_via_system_state as brownian,
)
from particula.dynamics.coagulation.van_der_waals_kernel import (
    BOLTZMANN_CONSTANT,
    get_hard_sphere_kernel,
    pair_enhancements,
)
from particula.dynamics.coagulation.van_der_waals_kernel import (
    get_van_der_waals_kernel_via_system_state as dispersion,
)

DENSITY = np.asarray([1000.0, 2160.0, 917.0])
WALL_H = 3.0
WALL_SIDE = np.sqrt(22.4 / WALL_H)
THAM = np.array([-21.0, -20.0, -19.0])
TEMPS = np.array([249.0, 273.0, 295.0])
PRESSURES = np.array([18000.0, 59000.0, 100000.0])


def particle_properties(radius, temperature, pressure):
    visc = par.gas.get_dynamic_viscosity(temperature)
    path = par.gas.get_molecule_mean_free_path(
        temperature=temperature, pressure=pressure, dynamic_viscosity=visc
    )
    slip = par.particles.get_cunningham_slip_correction(path / radius)
    diffusion = par.particles.get_diffusion_coefficient(
        temperature, par.particles.get_aerodynamic_mobility(radius, slip, visc)
    )
    rho = pressure / (287.05 * temperature)
    drag = np.stack(
        [
            par.particles.get_particle_settling_velocity_with_drag(
                radius, d, rho, visc, slip
            )
            for d in DENSITY
        ]
    )
    stokes = np.stack(
        [
            par.particles.get_particle_settling_velocity_via_system_state(
                radius, d, temperature, pressure
            )
            for d in DENSITY
        ]
    )
    return diffusion, stokes, drag, visc / rho


def tables(exp, n):
    grid = np.geomspace(0.004, 1000, n)
    radius = grid / 2 * 1.0e-6
    nt = exp.nmin
    wall = np.empty((6, nt, 7, n), dtype=np.float32)
    for i, r in enumerate(exp.runs):
        for t in range(nt):
            j = min(t * 60 + 30, r["valid_stop"])
            diff, stokes, drag, _ = particle_properties(
                radius, r["temp"][j, 0], r["pressure"][j]
            )
            wall[i, t] = np.vstack([diff, stokes, drag])
    rad = np.tile(radius, len(DENSITY))
    mass = np.repeat(DENSITY, n) * (4 * np.pi / 3 * rad**3)
    nn = len(rad)
    # Independent temperature and pressure knots bracket the chamber records.
    therm = np.empty((len(TEMPS) * len(PRESSURES), 5, nn, nn), dtype=np.float32)
    rij = rad[:, None] + rad[None, :]
    mu = mass[:, None] * mass[None, :] / (mass[:, None] + mass[None, :])
    idx = np.arange(n)
    native_mask = (
        np.maximum(rad[:, None], rad[None, :])
        / np.minimum(rad[:, None], rad[None, :])
        <= 32.0
    )
    local_mask = idx[:, None] - idx[None, :]
    for ti, T in enumerate(TEMPS):
        enhance = []
        for ham in THAM:
            ratios = [(1.0, 1.0)] * n
            for offset in range(n):
                if radius[offset] / radius[0] <= 32.0:
                    ratios[offset] = pair_enhancements(
                        radius[0], radius[offset], 0.0, 0.0, T, 10.0**ham, 80.0
                    )
            ec, ef = np.array(ratios)[np.abs(local_mask)].transpose(2, 0, 1)
            enhance.append((np.tile(ec, (3, 3)), np.tile(ef, (3, 3))))
        for pi, P in enumerate(PRESSURES):
            tab = therm[ti * len(PRESSURES) + pi]
            tab[0] = brownian(rad, mass, T, P) * 1e6
            viscosity = par.gas.get_dynamic_viscosity(T)
            path = par.gas.get_molecule_mean_free_path(
                temperature=T, pressure=P, dynamic_viscosity=viscosity
            )
            slip = par.particles.get_cunningham_slip_correction(path / rad)
            friction = 6.0 * np.pi * viscosity * rad / slip
            drag = (
                friction[:, None]
                * friction[None, :]
                / (friction[:, None] + friction[None, :])
            )
            kn = np.sqrt(BOLTZMANN_CONSTANT * T * mu) / (drag * rij)
            tab[4] = np.where(
                native_mask,
                1e6 * (drag * rij**3 / mu) * get_hard_sphere_kernel(kn),
                tab[0],
            )
            for hi, (ec, ef) in enumerate(enhance):
                active_k = (
                    1e6
                    * (drag * rij**3 / mu)
                    * (ef**2 / ec)
                    * get_hard_sphere_kernel(kn * ec / ef)
                )
                tab[hi + 1] = np.where(native_mask, active_k, tab[0])
            # Independently compare the reconstructed matrix with the native
            # complete public kernel on grid regions within its pair domain.
            selected = np.array([0, n + 2, 2 * n + 4])
            exact = dispersion(
                rad[selected],
                mass[selected],
                np.zeros(len(selected)),
                T,
                P,
                1e-20,
                80.0,
            )
            np.testing.assert_allclose(
                tab[2][np.ix_(selected, selected)] * 1e-6, exact, rtol=2.0e-6
            )
    assert np.isfinite(therm).all() and np.all(therm > 0)
    assert np.allclose(therm, np.swapaxes(therm, 2, 3), rtol=3e-7, atol=0)
    return wall, therm, grid


def make_tables(exp, n, root):
    cache = root / f"native-wall-with-thermal-control-{n}.npz"
    if not cache.exists():
        w, k, d = tables(exp, n)
        with NamedTemporaryFile(dir=root, delete=False) as output:
            temporary = Path(output.name)
        try:
            with temporary.open("wb") as output:
                np.savez_compressed(output, wall=w, thermal=k, grid=d)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    with np.load(cache) as f:
        return f["wall"], f["thermal"], f["grid"]
