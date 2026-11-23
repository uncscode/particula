"""Repeatable native table builds, independent accuracy tests and paired sweeps.

Run one stage or one chemistry at a time. JSONL appends a record after each
configuration. Run devices serially on an otherwise idle host.
"""

import argparse
import json
import platform
import resource
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from particula_aiomfac.sparse import (
    ActivityTable,
    NodeEvaluator,
    build_table,
    load_recipe,
)

RECIPES = (
    "mixed-water",
    "mixed-two-vapor",
    "mixed-variable-acid",
    "neutral",
    "shared-ion",
    "sulfate",
    "carbonate",
)
SIZES = (1, 8, 32, 128, 512, 2048, 4096, 16384, 65536)


def timed(function):
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def paired(direct, hybrid, repeats=5):
    direct()
    hybrid()
    times = [[], []]
    for rep in range(repeats):
        for i in (0, 1) if rep % 2 == 0 else (1, 0):
            _, seconds = timed((direct, hybrid)[i])
            times[i].append(seconds)
    return dict(
        direct_s=times[0],
        hybrid_s=times[1],
        speedup=float(np.median(times[0]) / np.median(times[1])),
    )


def record(path, **row):
    row.update(
        numpy=np.__version__,
        host=platform.node(),
        cpu=platform.machine(),
        peak_process_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    from particula_aiomfac import __version__

    row["backend"] = __version__
    if row.get("device") != "cpu" or row.get("build_device", "cpu") != "cpu":
        import warp as wp

        selection = next(
            v
            for v in (row.get("device", "cpu"), row.get("build_device", "cpu"))
            if v != "cpu"
        )
        device = wp.get_device(selection)
        row.update(
            gpu=device.name,
            warp=wp.__version__,
            peak_process_device_pool_bytes=wp.get_mempool_used_mem_high(device),
        )
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def qualify(args, recipe):
    unit = np.random.default_rng(27419).random((4096, len(recipe.bounds)))
    # Independent near-face and interior transition probes supplement holdouts.
    face = np.repeat(unit[:64], 2 * unit.shape[1], axis=0)
    for j in range(unit.shape[1]):
        face[2 * j :: 2 * unit.shape[1], j] = 1e-12
        face[2 * j + 1 :: 2 * unit.shape[1], j] = 1 - 1e-12
    # Concentrated mixtures have rapidly changing activity along the water
    # coordinate. Sample this end independently of the builder's error probes.
    concentrated = np.random.default_rng(1749).random((256, unit.shape[1]))
    concentrated[:, 0] *= 0.08
    u = np.vstack((unit, face, concentrated))
    mass, tk = recipe.states(u)
    oracles = {device: NodeEvaluator(recipe, device) for device in args.devices}
    reference = {}
    for device, oracle in oracles.items():
        reference[device], status = oracle.evaluate(mass, tk)
        if np.any(status < 0):
            raise RuntimeError(
                f"holdout chemistry failed at row {np.flatnonzero(status < 0)[0]}"
            )
    builds = {}
    for device in args.devices:
        table = build_table(recipe, device=device)
        path = args.cache / f"{recipe.name}-{device.replace(':', '-')}.npz"
        _, save = timed(lambda: table.save(path))
        builds[device] = table
        for run_device in args.devices:
            runtime = ActivityTable.load(path).prepare(device=run_device)
            a, e, reason = runtime.evaluate(mass, tk)
            target = reference[run_device]
            log_error = abs(np.log(a / target))
            abs_error = np.max(abs(a[:, 0] - target[:, 0]))
            err = np.max(log_error)
            if abs_error > 1e-4 or err > 1e-3 or not np.any(reason == 0):
                raise AssertionError((recipe.name, abs_error, err))
            if np.any((log_error > e + 1e-13) & (reason == 0)[:, None]):
                raise AssertionError(
                    f"{recipe.name}: held-out error exceeds reported interval"
                )
            record(
                args.output,
                stage="qualification",
                recipe=recipe.name,
                device=run_device,
                build_device=device,
                nodes=table.metadata["unique_nodes"],
                interpolation_nodes=len(table.arrays["coefficients"]),
                qualified_cells=int(
                    np.isfinite(table.arrays["errors"][..., 0]).sum()
                ),
                cells=int(table.arrays["errors"][..., 0].size),
                holdouts=len(unit),
                supplementary=len(face) + len(concentrated),
                concentrated_states=len(concentrated),
                worst_log_unit_coordinates=u[
                    np.unravel_index(np.argmax(log_error), log_error.shape)[0]
                ].tolist(),
                hits=int(np.sum(reason == 0)),
                max_water_abs=float(abs_error),
                max_log=float(err),
                max_log_by_target=log_error.max(axis=0).tolist(),
                interval_exceedances=int(
                    np.sum(
                        np.any(log_error > e + 1e-13, axis=1) & (reason == 0)
                    )
                ),
                **table.timings,
                serialization_s=save,
                artifact_bytes=path.stat().st_size,
                digest=table.digest,
            )
    if len(builds) == 2:
        # Comparing logs at common hierarchical nodes is separate from comparing
        # two independently refined surrogate values on holdouts.
        from particula_aiomfac.sparse import _nodes

        levels = sorted(
            set.intersection(*(set(t.blocks) for t in builds.values()))
        )
        m, t = recipe.states(np.vstack([_nodes(level) for level in levels]))
        direct = [o.evaluate(m, t)[0] for o in oracles.values()]
        record(
            args.output,
            stage="common_nodes",
            recipe=recipe.name,
            device=args.devices[-1],
            common_nodes=len(t),
            max_log=float(np.max(abs(np.log(direct[0] / direct[1])))),
            different_subspaces=len(
                set.symmetric_difference(
                    *(set(x.blocks) for x in builds.values())
                )
            ),
        )


def construction(args, recipe, device):
    oracle = NodeEvaluator(recipe, device)
    fixed = np.random.default_rng(783).random((4096, len(recipe.bounds)))
    m, t = recipe.states(fixed)
    held, _ = oracle.evaluate(m, t)
    from particula_aiomfac.sparse import _nodes

    for factor in (0.5, 1.0, 2.0):
        table = build_table(
            recipe, device=device, tolerance_scale=factor, profile=True
        )
        build_pairs = [
            timed(
                lambda: build_table(
                    recipe, device=device, tolerance_scale=factor
                )
            )[1]
            for _ in range(5)
        ]
        path = (
            args.cache
            / f"{recipe.name}-{device.replace(':', '-')}-{factor}.npz"
        )
        _, save = timed(lambda: table.save(path))
        fit, _, reason = table.prepare(device=device).evaluate(m, t)
        # Ordered nodes from the frozen common CPU artifact isolate direct cost.
        frozen = ActivityTable.load(recipe.name)
        rows, temp = recipe.states(
            np.vstack([_nodes(level) for level in frozen.blocks])
        )
        oracle.evaluate(rows[:1], temp[:1])
        times = [
            timed(lambda: oracle.evaluate(rows, temp))[1] for _ in range(5)
        ]
        record(
            args.output,
            stage="build",
            recipe=recipe.name,
            device=device,
            factor=factor,
            **table.timings,
            refinement_s=table.timings["total_s"]
            - table.timings["setup_s"]
            - table.timings["node_s"],
            serialization_s=save,
            nodes=table.metadata["unique_nodes"],
            warmed_build_s=build_pairs,
            fixed_node_count=len(rows),
            fixed_node_s=times,
            bytes=path.stat().st_size,
            holdout_max_log=float(np.max(abs(np.log(held / fit)))),
            holdout_hits=int((reason == 0).sum()),
        )


def cold_build(args, recipe, device):
    """Measure initialization, JIT, warm rebuild and serialization in a fresh process."""
    with tempfile.TemporaryDirectory(dir=args.cache) as directory:
        if device != "cpu":
            import warp as wp

            wp.config.kernel_cache_dir = directory
        for cold in (True, False):
            table, seconds = timed(
                lambda: build_table(recipe, device=device, profile=True)
            )
            path = Path(directory) / "table.npz"
            _, serialization = timed(lambda: table.save(path))
            record(
                args.output,
                stage="cold_build",
                cold=cold,
                device=device,
                recipe=recipe.name,
                wall_s=seconds,
                serialization_s=serialization,
                nodes=table.metadata["unique_nodes"],
                **table.timings,
            )


def device_pairs(runtime, mass, tk, active):
    """Time both kernel paths on the same retained stream with resident input."""
    import warp as wp

    model = runtime.oracle.model
    table = runtime.device_table
    with wp.ScopedStream(model.stream):
        chunks = []

        def alloc(shape, dtype):
            return wp.empty(shape, dtype=dtype, device=model.device)

        for start in range(0, len(mass), model.capacity):
            end = min(start + model.capacity, len(mass))
            n = end - start

            def up(a, dtype):
                return wp.array(
                    np.ascontiguousarray(a[start:end]),
                    dtype=dtype,
                    device=model.device,
                )

            chunks.append(
                (
                    up(mass, wp.float64),
                    up(tk, wp.float64),
                    up(active, int),
                    model.output[:n],
                    model.status[:n],
                    model.scratch[:n],
                    alloc((n, len(table.columns)), wp.float64),
                    alloc(n, int),
                    alloc(n, int),
                )
            )
        times = [[], []]
        for rep in range(-1, 5):
            for hybrid in (0, 1) if rep % 2 == 0 else (1, 0):
                begin = wp.Event(model.device, enable_timing=True)
                end = wp.Event(model.device, enable_timing=True)
                wp.record_event(begin)
                for m, t, mask, out, status, work, err, reason, miss in chunks:
                    if hybrid:
                        table.lookup_into(
                            m, t, mask, out, err, reason, miss, model.stream
                        )
                    model.evaluate_into(
                        m,
                        t,
                        out,
                        status,
                        work,
                        stream=model.stream,
                        active=miss if hybrid else mask,
                    )
                wp.record_event(end)
                sec = wp.get_event_elapsed_time(begin, end) * 1e-3
                if rep >= 0:
                    times[hybrid].append(sec)
        return dict(direct_device_s=times[0], hybrid_device_s=times[1])


def throughput(args, recipe, device):
    model = ActivityTable.load(recipe.name).prepare(device=device)
    rng = np.random.default_rng(2146)
    for n in args.sizes or SIZES:
        for fraction in (
            (0.0, 0.01, 0.10, 0.50, 1.0) if args.stage == "fallback" else (0.0,)
        ):
            for occupied in (0.25, 1.0) if args.stage == "fallback" else (1.0,):
                for shuffled in (
                    (False, True) if args.stage == "fallback" else (False,)
                ):
                    m, t = recipe.states(
                        rng.uniform(0.1, 0.9, (n, len(recipe.bounds)))
                    )
                    t[:] = 298.0
                    active = np.arange(n) < max(1, round(n * occupied))
                    nmiss = round(active.sum() * fraction)
                    t[:nmiss] = recipe.bounds[1][1] + 1
                    if shuffled:
                        ix = rng.permutation(n)
                        m, t, active = m[ix], t[ix], active[ix]

                    def direct():
                        if device == "cpu":
                            return model.oracle.model.evaluate(
                                m[active], t[active]
                            ).neutral_activities[:, model.oracle.indices]
                        return model.oracle.evaluate(m[active], t[active])[0]

                    truth = direct()
                    a, _, r = model.evaluate(m, t, active=active)
                    np.testing.assert_allclose(
                        a[active], truth, rtol=1e-3, atol=1e-10
                    )
                    assert np.sum(r > 0) == nmiss
                    metrics = paired(
                        direct, lambda: model.evaluate(m, t, active=active)
                    )
                    if device != "cpu":
                        metrics.update(device_pairs(model, m, t, active))
                    record(
                        args.output,
                        stage=args.stage,
                        recipe=recipe.name,
                        device=device,
                        n=n,
                        occupancy=occupied,
                        fallback=fraction,
                        shuffled=shuffled,
                        direct_rows=nmiss,
                        **metrics,
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "qualification",
            "build",
            "cold-build",
            "throughput",
            "fallback",
        ],
    )
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument(
        "--recipes", nargs="+", choices=RECIPES, default=RECIPES
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path(".artifacts/sparse-tables")
    )
    parser.add_argument("--sizes", type=int, nargs="+")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "fallback" and not args.sizes:
        args.sizes = (128, 4096, 65536)
    if args.stage == "cold-build" and (
        len(args.recipes) != 1 or len(args.devices) != 1
    ):
        parser.error(
            "cold-build requires one recipe and one device per Python process"
        )
    for name in args.recipes:
        recipe = load_recipe(name)
        if args.stage == "qualification":
            qualify(args, recipe)
        else:
            for device in args.devices:
                if args.stage == "build":
                    construction(args, recipe, device)
                elif args.stage == "cold-build":
                    cold_build(args, recipe, device)
                else:
                    throughput(args, recipe, device)


if __name__ == "__main__":
    main()
