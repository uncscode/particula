#!/usr/bin/env python3
"""Bounded, deterministic CPU Particula starters with explicit assumptions and outputs."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", choices=("condensation", "coagulation"), default="condensation")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--duration", type=positive, default=1.0, help="Total seconds (default 1)")
    parser.add_argument("--steps", type=int, default=100, help="Time steps, 1..10000 (default 100)")
    parser.add_argument("--number", type=positive, default=1e8, help="Initial particle number concentration, m^-3")
    parser.add_argument("--radius", type=positive, default=1e-7, help="Mode radius, m")
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000:
        parser.error("--steps must be between 1 and 10000")
    if args.output.exists():
        parser.error("output already exists; choose a fresh run directory")
    return args


def build(args, np, par):
    gas = (par.gas.GasSpeciesBuilder().set_name("illustrative-organic")
           .set_molar_mass(0.18, "kg/mol")
           .set_vapor_pressure_strategy(par.gas.ConstantVaporPressureStrategy(1e-5))
           .set_partitioning(True).set_concentration(np.array([1e-8]), "kg/m^3").build())
    atmosphere = (par.gas.AtmosphereBuilder().set_temperature(298.15, "K")
                  .set_pressure(101325, "Pa").set_more_partitioning_species(gas).build())
    radii = np.geomspace(args.radius / 2, args.radius * 4, 40)
    weights = np.exp(-0.5 * (np.log(radii / args.radius) / np.log(1.2)) ** 2)
    weights = weights / weights.sum() * args.number
    particles = par.particles.ParticleRepresentation(
        strategy=par.particles.SpeciatedMassMovingBin(),
        activity=par.particles.ActivityIdealMass(),
        surface=par.particles.SurfaceStrategyVolume(surface_tension=0.072, density=1000),
        distribution=(4 / 3 * np.pi * radii ** 3 * 1000)[:, np.newaxis],
        density=np.array([1000.0]), concentration=weights, charge=np.zeros(40),
    )
    aerosol = par.AerosolBuilder().set_atmosphere(atmosphere).set_particles(particles).build()
    if args.process == "condensation":
        process = par.dynamics.MassCondensation(par.dynamics.CondensationIsothermal(molar_mass=0.18, diffusion_coefficient=2e-5, accommodation_coefficient=1.0))
    else:
        process = par.dynamics.Coagulation(par.dynamics.BrownianCoagulationStrategy(distribution_type="discrete"))
    return aerosol, process


def snapshot(aerosol, time, np):
    p = aerosol.particles
    for name, value in (("radius", p.get_radius()), ("particle mass", p.get_mass()), ("concentration", p.get_concentration()), ("gas concentration", aerosol.atmosphere.partitioning_species.get_concentration())):
        values = np.asarray(value)
        if not np.isfinite(values).all() or (values < 0).any():
            raise RuntimeError(f"Nonfinite or negative {name} at time {time}; inspect time step and configuration")
    pm = float(p.get_mass_concentration())
    gm = float(np.sum(aerosol.atmosphere.partitioning_species.get_concentration()))
    return {"time_s": time, "number_m-3": float(p.get_total_concentration()), "particle_mass_kg_m-3": pm, "gas_mass_kg_m-3": gm, "total_mass_kg_m-3": pm + gm}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mass_balance_checks(rows, process):
    """Check the conserved mass budget for the selected physical process."""
    start = rows[0]
    total_key = "total_mass_kg_m-3"
    total_error = max(
        abs(row[total_key] - start[total_key]) for row in rows
    ) / start[total_key]
    checks = {
        "max_relative_total_mass_error": total_error,
        "mass_tolerance": 1e-6,
        "mass_balance_pass": total_error < 1e-6,
    }
    if process == "coagulation":
        particle_key = "particle_mass_kg_m-3"
        particle_error = max(
            abs(row[particle_key] - start[particle_key]) for row in rows
        ) / start[particle_key]
        checks["max_relative_particle_mass_error"] = particle_error
        checks["mass_balance_pass"] = (
            checks["mass_balance_pass"] and particle_error < 1e-6
        )
    return checks


def main():
    args = parse_args()
    import numpy as np
    import particula as par
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aerosol, process = build(args, np, par)
    initial_radius = np.array(aerosol.particles.get_radius(), copy=True)
    initial_number = np.array(aerosol.particles.get_concentration(), copy=True)
    rows = [snapshot(aerosol, 0.0, np)]
    for step in range(args.steps):
        aerosol = process.execute(aerosol, time_step=args.duration / args.steps, sub_steps=1)
        rows.append(snapshot(aerosol, (step + 1) * args.duration / args.steps, np))
    start, end = rows[0], rows[-1]
    checks = {"finite_nonnegative": True, **mass_balance_checks(rows, args.process)}
    if args.process == "condensation":
        checks["number_conserved"] = all(np.isclose(r["number_m-3"], start["number_m-3"], rtol=1e-10, atol=0) for r in rows)
    else:
        checks["number_nonincreasing"] = all(b["number_m-3"] <= a["number_m-3"] * (1 + 1e-12) for a, b in zip(rows, rows[1:]))
    if not checks["mass_balance_pass"] or not all(v for k, v in checks.items() if k in ("number_conserved", "number_nonincreasing")):
        raise RuntimeError("Scientific checks failed: " + json.dumps(checks))
    final_radius = aerosol.particles.get_radius()
    final_number = aerosol.particles.get_concentration()
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "time_series.csv", rows)
    write_csv(args.output / "distribution.csv", [{"initial_radius_m": float(r0), "initial_number_per_bin_m-3": float(n0), "final_radius_m": float(r1), "final_number_per_bin_m-3": float(n1)} for r0,n0,r1,n1 in zip(initial_radius,initial_number,final_radius,final_number)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].semilogx(initial_radius * 1e9, initial_number, label="Initial")
    axes[0].semilogx(final_radius * 1e9, final_number, label="Final")
    axes[0].set(xlabel="Particle radius (nm)", ylabel="Number per bin (m⁻³)", title="Size distribution")
    axes[0].legend()
    times = [r["time_s"] for r in rows]
    if args.process == "condensation":
        for key, label in (("particle_mass_kg_m-3", "Particle"), ("gas_mass_kg_m-3", "Gas"), ("total_mass_kg_m-3", "Total")):
            axes[1].plot(times, [r[key] * 1e9 for r in rows], label=label)
        axes[1].set_ylabel("Mass concentration (µg m⁻³)")
    else:
        axes[1].plot(times, [r["number_m-3"] / start["number_m-3"] for r in rows], label="Number / initial")
        axes[1].plot(times, [r["particle_mass_kg_m-3"] / start["particle_mass_kg_m-3"] for r in rows], label="Particle mass / initial")
        axes[1].set_ylabel("Fraction of initial value")
    axes[1].set(xlabel="Time (s)", title="Bulk evolution")
    axes[1].legend()
    fig.suptitle(f"Particula {args.process} — illustrative CPU study")
    fig.savefig(args.output / "result.png", dpi=160)
    plt.close(fig)
    metadata = {"process": args.process, "backend": "CPU", "particula_source": str(par.__file__), "particula_source_version": getattr(par, "__version__", None), "python": platform.python_version(), "versions": {k:version(k) for k in ("particula", "numpy", "scipy", "matplotlib")}, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "parameters": {"duration_s":args.duration,"steps":args.steps,"mode_radius_m":args.radius,"number_m-3":args.number,"temperature_K":298.15,"pressure_Pa":101325,"density_kg_m-3":1000,"gsd":1.2,"molar_mass_kg_mol":0.18,"vapor_pressure_Pa":1e-5,"initial_gas_kg_m-3":1e-8,"diffusion_m2_s":2e-5,"accommodation":1.0}, "representation":"40 discrete speciated-mass bins; radius grid mode/2 to 4*mode", "assumptions":["Illustrative single organic species; not measured or calibrated", "Closed, fixed-temperature CPU model", "No wall loss, dilution, nucleation or temperature feedback", "Time-step/resolution convergence must be checked for quantitative use"], "initial":start,"final":end,"checks":checks,"status":"completed"}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output":str(args.output.resolve()),"status":"completed","checks":checks},indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
