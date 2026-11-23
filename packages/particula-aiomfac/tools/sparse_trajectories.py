"""Time closed-box and moist parcel trajectories at fixed physical loading."""

import argparse
import runpy
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
from particula_aiomfac.adapter import AIOMFACActivity
from sparse_study import record, timed


def example(name):
    root = (
        Path(__file__).resolve().parents[3]
        / "docs/studies/examples/thermodynamics"
    )
    if not root.is_dir():
        root = files("particula").joinpath(
            "docs/studies/examples/thermodynamics"
        )
    return runpy.run_path(str(root.joinpath(name)))


def box_series(args):
    from particula.aerosol import Aerosol

    lib = example("activity/aiomfac_closed_box.py")
    ref = lib["radau"](lib["make_box"]())
    for n in args.sizes or (1, 16, 256, 4096, 16384):
        repeated = [
            {
                k: np.repeat(v, n, axis=0)
                if k in ("flux", "activity", "radius", "mass")
                else v
                for k, v in row.items()
            }
            for row in ref
        ]
        for solver in args.solvers or (
            "native",
            "fixed",
            "host-rk23",
            "resident-rk23",
        ):

            def run(hybrid):
                box = lib["make_box"](
                    table=(args.table or "mixed-two-vapor") if hybrid else None
                )
                base = box.aerosol
                p = base.particles
                if n != 1:
                    p = replace(
                        p,
                        masses=np.repeat(p.masses, n, axis=1),
                        charge=np.repeat(p.charge, n, axis=1),
                        concentration=np.repeat(p.concentration / n, n, axis=1),
                    )
                    box.aerosol = Aerosol(
                        particles=p, gas=base.gas, environment=base.environment
                    )
                provider = box.process.activity_strategy
                calls = 0

                def measured_neutrals(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    return evaluate(*args, **kwargs)

                if solver == "native" and args.stage == "box-accounting":
                    evaluate = provider.neutral_with_error
                    provider.neutral_with_error = measured_neutrals
                if solver == "native":
                    rows, elapsed = timed(lambda: lib["native"](box, 8))
                    stats = dict(
                        advance_s=elapsed,
                        lookup_counts=box.process.activity_strategy.counts.tolist(),
                    )
                else:
                    (rows, stats), elapsed = timed(
                        lambda: lib["gpu_run"](box, solver)
                    )
                if solver == "native" and args.stage == "box-accounting":
                    stats.update(
                        activity_evaluations=calls,
                        control_readbacks=0,
                        explicit_snapshots=0,
                    )
                initial_flux = stats.pop("initial_flux_error_kg_s", None)
                if initial_flux is not None:
                    stats["max_initial_flux_error_kg_s"] = np.max(
                        abs(np.asarray(initial_flux)), axis=0
                    ).tolist()
                return dict(
                    **stats,
                    wall_s=elapsed,
                    errors=lib["compare"](rows, repeated, box.targets),
                )

            if args.stage == "box-accounting":
                for hybrid in (False, True):
                    stats = run(hybrid)
                    stats.pop("device", None)
                    record(
                        args.output,
                        stage=args.stage,
                        n=n,
                        solver=solver,
                        hybrid=hybrid,
                        device="cpu" if solver == "native" else "cuda:0",
                        **stats,
                    )
            else:
                _paired_runs(args, "box", n, solver, run)


def _paired_runs(args, kind, n, solver, run):
    for i in range(-1, args.pairs):
        for hybrid in (False, True) if i % 2 == 0 else (True, False):
            result = run(hybrid)
            if i >= 0:
                result.pop("device", None)
                record(
                    args.output,
                    stage=kind,
                    n=n,
                    solver=solver,
                    hybrid=hybrid,
                    pair=i,
                    table_build_device=args.table_build_device,
                    table_digest=args.table_digest,
                    device="cpu" if solver in ("native", "radau") else "cuda:0",
                    **result,
                )


def parcel_series(args):
    lib = example("parcel-experiments/aiomfac_multivapor_parcel.py")
    long = args.stage == "activation"
    height = 1500 if long else 150
    base = lib["initial_haze" if long else "initial_parcel"]()
    ref = lib["run_parcel"](
        base,
        timestep=0.5 if long else 1.0,
        steps=height * (2 if long else 1),
        updraft=1.0,
    )
    idx = np.arange(len(ref.time))
    if long:
        # Retain half-meter resolution around the known events, 5 m elsewhere.
        z = ref.height
        idx = np.union1d(
            idx[::10],
            np.flatnonzero(
                np.any(abs(z[:, None] - [400, 463, 707]) < 16, axis=1)
            ),
        )
    for n in args.sizes or ((2,) if long else (2, 8, 32, 128)):
        repeat = n // 2
        if n < 2 or n % 2:
            raise ValueError("split the two original classes equally")
        state = replace(
            base,
            number=np.repeat(base.number / repeat, repeat),
            masses=np.repeat(base.masses, repeat, axis=0),
        )
        reference = lib["multivapor_trajectory"](
            state,
            ref.time[idx],
            ref.pressure[idx],
            ref.temperature[idx],
            ref.vapor[idx],
            np.repeat(ref.masses[idx], repeat, axis=1),
            1.0,
        )
        for solver in args.solvers or ("resident-rk23",):

            def run(hybrid):
                initial = state
                if hybrid:
                    provider = AIOMFACActivity(
                        base.activity_strategy.mixture,
                        dict(enumerate(base.material_names)),
                        evaluation="hybrid",
                        table=args.table or "mixed-two-vapor",
                    )
                    initial = replace(initial, activity_strategy=provider)
                oracle = replace(reference, initial=initial)
                (metrics, trajectory), elapsed = timed(
                    lambda: lib["cuda_comparison"](
                        oracle,
                        lib[
                            "ParcelROS3"
                            if solver == "resident-ros3"
                            else "ParcelRK23"
                        ](rtol=2e-6),
                        "cuda:0",
                        restart=args.restart,
                        return_trajectory=True,
                    )
                )
                if long:
                    # Always diagnose the critical curves with native direct activities.
                    events, _, _ = lib["activation_events"](
                        replace(trajectory, initial=state), np.arange(len(idx))
                    )
                    expected, _, _ = lib["activation_events"](
                        reference, np.arange(len(idx))
                    )
                    activation = np.asarray([e["height_m"] for e in events])
                    cloud = lib["upward_crossing"](
                        trajectory.height, trajectory.supersaturation
                    )
                    direct_cloud = lib["upward_crossing"](
                        reference.height, reference.supersaturation
                    )
                    if (
                        max(
                            abs(cloud - direct_cloud),
                            np.max(
                                abs(
                                    activation
                                    - [e["height_m"] for e in expected]
                                )
                            ),
                        )
                        > 1
                    ):
                        raise AssertionError(
                            "activation shifted more than one meter"
                        )
                    metrics.update(
                        cloud_base_m=cloud, activation_m=activation.tolist()
                    )
                return dict(**metrics, wall_s=elapsed)

            if solver == "direct-ros3":
                metrics = lib["cuda_comparison"](
                    reference,
                    lib["ParcelROS3"](rtol=2e-6),
                    "cuda:0",
                    restart=False,
                )
                metrics.pop("device", None)
                record(
                    args.output,
                    stage=args.stage,
                    n=n,
                    solver=solver,
                    device="cuda:0",
                    hybrid=False,
                    **metrics,
                )
            elif solver in ("resident-rk23", "resident-ros3"):
                _paired_runs(args, args.stage, n, solver, run)
            else:
                raise ValueError(
                    "parcel solvers are resident-rk23, resident-ros3 and direct-ros3"
                )


def water_uptake(args):
    p = Path(__file__).resolve().parents[1] / "examples/closed_water.py"
    lib = runpy.run_path(str(p))
    for sat in (0.98, 1.02):
        # Keep scientific observations outside the recorded timing. Use the
        # direct native path as the independent reference for both devices.
        process, p, g, src, water, _ = lib["setup"](sat)
        start_total = p.species_mass_density[0, water] + g.concentration[0, src]
        for _ in range(20):
            process.step(p, g, 298.0, 101325.0, time_step=1e-4)
        ref_radius = p.radii[0, 0]
        ref_gas = g.concentration[0, src]

        def errors(particles, gas):
            radius = abs(particles.radii[0, 0] / ref_radius - 1.0)
            phase = abs(gas.concentration[0, src] - ref_gas) / start_total
            if radius > 1e-3 or phase > 1e-3:
                raise AssertionError(
                    f"water trajectory: radius={radius}, phase={phase}"
                )
            return dict(
                relative_radius_error=radius, normalized_gas_error=phase
            )

        def run(hybrid):
            process, p, g, src, water, _ = lib["setup"](sat)
            if hybrid:
                provider = process.activity_strategy
                process.activity_strategy = AIOMFACActivity(
                    provider.mixture,
                    dict(enumerate(provider.mixture.components)),
                    evaluation="hybrid",
                    table=args.table or "mixed-water",
                )
            start = (
                p.concentration[0] @ p.masses[0, :, water]
                + g.concentration[0, src]
            )
            peak = 0.0

            def integrate():
                nonlocal peak
                for _ in range(20):
                    process.step(p, g, 298.0, 101325.0, time_step=1e-4)
                    peak = max(
                        peak,
                        abs(
                            p.concentration[0] @ p.masses[0, :, water]
                            + g.concentration[0, src]
                            - start
                        )
                        / start,
                    )

            _, elapsed = timed(integrate)
            if peak > 1e-10:
                raise AssertionError("water budget")
            return dict(
                wall_s=elapsed,
                saturation=sat,
                **errors(p, g),
                wet_radius_m=p.radii[0, 0],
                budget=peak,
                lookup_counts=process.activity_strategy.counts.tolist(),
            )

        if "native" in (args.solvers or ("native", "fixed")):
            _paired_runs(args, "water", 1, "native", run)
        if "fixed" in (args.solvers or ("native", "fixed")):
            import warp as wp
            from particula import Aerosol
            from particula.dynamics import MassCondensation
            from particula.gas.environment_data import EnvironmentData

            def gpu(hybrid):
                process, p, g, src, water, _ = lib["setup"](sat)
                if hybrid:
                    provider = process.activity_strategy
                    process.activity_strategy = AIOMFACActivity(
                        provider.mixture,
                        dict(enumerate(provider.mixture.components)),
                        evaluation="hybrid",
                        table=args.table or "mixed-water",
                    )
                env = EnvironmentData(
                    temperature=np.array([298.0]),
                    pressure=np.array([101325.0]),
                    saturation_ratio=np.zeros((1, 1)),
                )
                aerosol = Aerosol(particles=p, gas=g, environment=env)
                total = (
                    p.species_mass_density[0, water] + g.concentration[0, src]
                )
                with MassCondensation(process).prepare(
                    aerosol, device="cuda:0", time_step=1e-4
                ) as owner:
                    owner.capture()
                    wp.synchronize_device("cuda:0")

                    def integrate():
                        owner.run(20)
                        wp.synchronize_device("cuda:0")

                    _, elapsed = timed(integrate)
                    end = owner.snapshot()
                budget = (
                    abs(
                        end.particles.species_mass_density[0, water]
                        + end.gas.concentration[0, src]
                        - total
                    )
                    / total
                )
                if budget > 1e-10:
                    raise AssertionError("water budget")
                return dict(
                    wall_s=elapsed,
                    saturation=sat,
                    **errors(end.particles, end.gas),
                    wet_radius_m=end.particles.radii[0, 0],
                    budget=budget,
                )

            _paired_runs(args, "water", 1, "fixed", gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "box",
            "box-accounting",
            "parcel",
            "activation",
            "water",
            "restart",
        ],
    )
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--solvers", nargs="+")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--table",
        help="override the shipped recipe with a compatible portable NPZ",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from particula_aiomfac.sparse import ActivityTable

    artifact = ActivityTable.load(
        args.table
        or ("mixed-water" if args.stage == "water" else "mixed-two-vapor")
    )
    args.table_build_device = artifact.metadata["device"]
    args.table_digest = artifact.digest
    if args.stage == "restart":
        lib = example("activity/aiomfac_closed_box.py")
        metrics = lib["qualify_restart"](
            "rk23", table=args.table or "mixed-two-vapor"
        )
        record(args.output, stage="restart", device="cuda:0", **metrics)
    elif args.stage in ("box", "box-accounting"):
        box_series(args)
    elif args.stage == "water":
        water_uptake(args)
    else:
        parcel_series(args)
