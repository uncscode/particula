"""Check full campaign derivatives and compare matched fixed-topology refits.

Run from any directory on the RTX 5060 with the Particula study environment.
The same mass and count observations and native Warp simulator provide the
finite differences, timestep, reaction-order and reevaluated candidate scores.
"""

import argparse
import json
import shutil
import subprocess
from functools import partial
from time import perf_counter

import numpy as np
import warp as wp
from ad_campaign import CampaignDerivatives, fit_rates_and_doses
from baseline_replay import HERE, Model, run, save
from growth import BANK, HI, LO, initial


def errors(model, x, ids, doses, sub=4, reverse=False):
    """Per-experiment normalized RMSE at fixed experiment dose scales."""
    return run(
        model,
        x,
        ids,
        sub=sub,
        doses=np.asarray(doses)[:, None],
        reverse=reverse,
    )[1][0, :, :2]


def qualify(model, x, ids, shortlist, doses, sub=4):
    """Check dose, rate and shared derivatives against native differences."""
    native_error = partial(errors, sub=sub)
    positive = x.copy()
    for h in shortlist:
        positive[16 + h] = 0.15
    active = sorted(set(ids) | set(shortlist))
    # Include representative time, yield, eddy and competition sensitivities.
    shared = [0, 3, 9, 12]
    directions = [*shortlist, *(-2 - np.array(shared))]
    engine = CampaignDerivatives(model, active, directions, sub=sub)
    base = np.full_like(doses, 1.0)
    records, parity = [], []
    elapsed = []
    for label, point in (("zero", x), ("positive", positive)):
        start = perf_counter()
        value, grad_d, grad = engine.evaluate(point[:16], point[16:], base)
        elapsed.append(perf_counter() - start)
        native = native_error(model, point, active, base)
        parity.append(float(np.max(abs(native - value))))
        for h in [-1, *directions]:
            if h == -1:
                gradient = grad_d
                name = "per-run precursor dose scale"
            else:
                gradient = grad[:, :, directions.index(h)]
                name = BANK[h].key if h >= 0 else f"shared_p[{-h - 2}]"
            forward = label == "zero" and h >= 0
            backward = h < -1 and point[-h - 2] >= HI[-h - 2] - 1e-10
            for step in (1e-3, 1e-4, 1e-5):
                shift = np.zeros_like(point)
                ddose = np.zeros_like(base)
                if h == -1:
                    ddose[:] = step
                else:
                    shift[16 + h if h >= 0 else -h - 2] = step
                plus = (
                    native
                    if backward
                    else native_error(
                        model, point + shift, active, base + ddose
                    )
                )
                if forward:
                    estimate = (plus - native) / step
                else:
                    minus = native_error(
                        model, point - shift, active, base - ddose
                    )
                    estimate = (plus - minus) / (step if backward else 2 * step)
                magnitude = float(np.max(abs(gradient)))
                absolute = float(np.max(abs(estimate - gradient)))
                records.append(
                    dict(
                        point=label,
                        coordinate=name,
                        increment=step,
                        side="forward"
                        if forward
                        else ("backward" if backward else "centered"),
                        max_abs_error=absolute,
                        max_abs_gradient=magnitude,
                        normalized_error=absolute / max(magnitude, 1e-12),
                    )
                )
                if step == 1e-5 and absolute > 1e-6 + 2e-4 * magnitude:
                    raise ArithmeticError(
                        f"Unqualified {name}: {absolute}, {magnitude}"
                    )
    if max(parity) > 1e-10:
        raise ArithmeticError(f"Primal disagreement {parity}")
    parent = native_error(model, x, ids, doses)
    nested = native_error(model, x, active, doses)
    np.testing.assert_allclose(parent, nested, atol=1e-14, rtol=0)
    native_times, gradient_times = benchmark(
        model, positive, active, base, engine
    )
    return dict(
        fixed_dose_native_seconds=native_times,
        fixed_dose_jacobian_seconds=gradient_times,
        simultaneous_directions=1 + len(directions),
        joint_direction=full_direction(model, x, active, base, sub),
        step_s=60.0 / sub,
        warm_repeats_bitwise_equal=3,
        device=wp.get_device("cuda:0").name,
        warp=wp.__version__,
        cuda_compiler=subprocess.check_output(  # noqa: S603
            [shutil.which("nvcc"), "--version"], text=True
        ).splitlines()[-2],
        max_abs_primal_differences=parity,
        nested_zero=float(np.max(abs(parent - nested))),
        kernel_seconds=elapsed,
        measurements=records,
    )


def full_direction(model, original, ids, doses, sub=4):
    """Verify a joint change in every fitted coordinate at an interior point."""
    native_error = partial(errors, sub=sub)
    free_p = np.array([*range(10), 12], dtype=int)
    columns = np.r_[free_p, 16 + np.array(ids)]
    # Move away from physical bounds to allow feasible centered differences.
    point = original.copy()
    ranges = HI[columns] - LO[columns]
    point[columns] = np.clip(
        point[columns], LO[columns] + 1e-2 * ranges, HI[columns] - 1e-2 * ranges
    )
    rng = np.random.default_rng(20261122)
    shared_step = rng.normal(size=len(columns))
    shared_step *= ranges / np.linalg.norm(shared_step)
    dose_step = rng.uniform(-0.15, 0.15, len(doses))
    engine = CampaignDerivatives(model, ids, [*(-2 - free_p), *ids], sub=sub)
    y, local, shared = engine.evaluate(point[:16], point[16:], doses)
    predicted = shared @ shared_step + local * dose_step[:, None]
    np.testing.assert_allclose(
        y, native_error(model, point, ids, doses), atol=1e-10, rtol=0
    )
    magnitude = float(np.max(np.abs(predicted)))
    differences = []
    for epsilon in (1e-3, 1e-4, 1e-5):
        plus, minus = point.copy(), point.copy()
        plus[columns] += epsilon * shared_step
        minus[columns] -= epsilon * shared_step
        estimate = (
            native_error(model, plus, ids, doses + epsilon * dose_step)
            - native_error(model, minus, ids, doses - epsilon * dose_step)
        ) / (2 * epsilon)
        difference = float(np.max(abs(estimate - predicted)))
        differences.append(dict(epsilon=epsilon, max_abs_error=difference))
    if differences[-1]["max_abs_error"] > 2e-6 + 1e-4 * magnitude:
        raise ArithmeticError("Joint normalized-direction check failed")
    return dict(
        seed=20261122,
        shared_coordinates=len(columns),
        local_doses=len(doses),
        maximum_abs_derivative=magnitude,
        differences=differences,
    )


def benchmark(model, x, ids, doses, engine):
    """Time both backends at the same fixed doses, after warm-up."""
    native_error = partial(errors, sub=engine.sub)
    native = native_error(model, x, ids, doses)
    gradient = engine.evaluate(x[:16], x[16:], doses)
    native_times, gradient_times = [], []
    for _ in range(3):
        clock = perf_counter()
        replay = native_error(model, x, ids, doses)
        native_times.append(perf_counter() - clock)
        clock = perf_counter()
        repeated = engine.evaluate(x[:16], x[16:], doses)
        gradient_times.append(perf_counter() - clock)
        np.testing.assert_array_equal(replay, native)
        for first, second in zip(gradient, repeated, strict=True):
            np.testing.assert_array_equal(first, second)
    return native_times, gradient_times


def native_scores(model, vector, ids, doses):
    """Fixed fit doses across all three independent native simulator replays."""
    settings = [(4, False), (32, False), (32, True)]
    return [
        float(errors(model, vector, ids, doses, sub, reverse).max())
        for sub, reverse in settings
    ]


def campaign_seed(round_number):
    """Recover the exact frozen vectors at which that growth round began."""
    pinned = HERE / f"growth_seed_{round_number:02d}.json"
    if pinned.exists():
        return json.loads(pinned.read_text())
    return json.loads((HERE / "lineage.json").read_text())[round_number - 2]


def starting_point(model, x, ids, extra, i, previous_doses=None):
    """Project common starts and activate a connected motif on the second."""
    start = x.copy()
    if i:
        start[:3] = np.clip(start[:3] * 1.01, LO[:3], HI[:3])
    d = (
        run(model, start, ids)[2][0]
        if previous_doses is None
        else np.array(previous_doses)
    )
    if i and len(extra) > 1:
        # Seed all edges to give the downstream step leverage.
        for h in extra:
            start[16 + h] = 0.1
    return start, d


def finalists(number, extra):
    """Use the recorded shortlist and any declared connected additions."""
    matches = json.loads((HERE / f"shortlist_{number:02d}.json").read_text())
    for group in extra:
        candidate = list(map(int, group.split(",")))
        if all(row["extra"] != candidate for row in matches):
            matches.append(dict(extra=candidate))
    return matches


def main():
    """Qualify sensitivities and refit each screened finalist and its parent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calls", type=int, default=100, help="Per-start SLSQP cost ceiling"
    )
    parser.add_argument("--round", type=int, default=1, dest="number")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a saved round at the same evaluation budget",
    )
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument(
        "--check-sub",
        type=int,
        default=4,
        help="Number of integration substeps per minute for FD checks",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Append comma-separated channel IDs of an additional motif",
    )
    args = parser.parse_args()
    model = Model(BANK)
    x, ids = initial()
    if args.number > 1:
        earlier = campaign_seed(args.number)
        x, ids = np.array(earlier["x"]), earlier["ids"]
    matches = finalists(args.number, args.extra)
    shortlist = sorted({h for m in matches for h in m["extra"]})
    base_doses = (
        np.array(earlier["dose"])
        if args.number > 1
        else run(model, x, ids)[2][0]
    )
    qualifier = qualify(model, x, ids, shortlist, base_doses, args.check_sub)
    suffix = "" if args.check_sub == 4 else f"_sub{args.check_sub}"
    save(f"ad_checks_{args.number:02d}{suffix}.json", qualifier)
    print("qualified", max(qualifier["max_abs_primal_differences"]), flush=True)
    if args.checks_only:
        return
    saved = HERE / f"ad_fits_{args.number:02d}.json"
    results = []
    if args.resume and saved.is_file():
        prior = json.loads(saved.read_text())
        if prior["calls"] != args.calls:
            raise ValueError("Cannot resume with a different call budget")
        results = prior["results"]
    for extra in [[], *(m["extra"] for m in matches)]:
        if any(entry["extra"] == extra for entry in results):
            continue
        active = sorted(set(ids) | set(extra))
        starts = []
        for i in (0, 1):
            start, d = starting_point(
                model,
                x,
                ids,
                extra,
                i,
                earlier["dose"] if args.number > 1 else None,
            )
            clock = perf_counter()
            fit = fit_rates_and_doses(
                model,
                start[:16],
                start[16:],
                active,
                d,
                active,
                sub=4,
                max_evaluations=args.calls,
                tune_physics=[*range(10), 12],
            )
            elapsed = perf_counter() - clock
            fit.update(start=i, elapsed_s=elapsed)
            fit["scores"] = native_scores(
                model, np.asarray(fit["x"]), active, fit["dose"]
            )
            if abs(fit["score"] - fit["scores"][0]) > 1e-10:
                raise ArithmeticError(
                    "AD and Warp disagree at the fitted parameters"
                )
            # Independently reprofile on the archived 61-point dose grid.
            fit["grid_scores"] = [
                float(
                    run(
                        model,
                        np.asarray(fit["x"]),
                        active,
                        sub=sub,
                        reverse=reverse,
                    )[0][0]
                )
                for sub, reverse in [(4, False), (32, False), (32, True)]
            ]
            starts.append(fit)
            print(
                extra,
                i,
                fit["scores"],
                fit["evaluations"],
                fit["status"],
                round(elapsed, 2),
                flush=True,
            )
        best = min(starts, key=lambda s: s["scores"][0])
        results.append(
            dict(
                extra=extra,
                active=active,
                starts=starts,
                best_start=best["start"],
            )
        )
        save(
            f"ad_fits_{args.number:02d}.json",
            dict(
                calls=args.calls,
                results=results,
                ranking="Best native 15 s fixed-dose score per topology.",
                resolution=["15s forward", "1.875s forward", "1.875s reverse"],
            ),
        )


if __name__ == "__main__":
    main()
