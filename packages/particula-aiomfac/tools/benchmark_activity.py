"""Time direct activities for one chemistry in a fresh process per device and size."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from particula_aiomfac import AIOMFACModel, MixtureSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["mixed", "sulfate", "carbonate"])
    parser.add_argument("size", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--kernel-cache",
        type=Path,
        help="use an empty directory to measure cold compilation",
    )
    args = parser.parse_args()
    if args.kernel_cache:
        import warp as wp

        wp.config.kernel_cache_dir = str(args.kernel_cache.resolve())
    ref = Path(__file__).resolve().parents[1] / "reference/cases"
    case = json.loads((ref / (args.case + ".json")).read_text())
    groups = json.loads((ref / "components.json").read_text())
    mixture = MixtureSpec(
        {
            k: {int(i): n for i, n in groups[k].items()}
            for k in case["components"]
        }
    )
    execution = "cpu" if args.device == "cpu" else "cuda"
    t0 = time.perf_counter()
    model = AIOMFACModel.prepare(
        mixture, execution=execution, device=args.device
    )
    setup = time.perf_counter() - t0
    prepared = model if execution == "cpu" else model.prepared
    fractions = np.array(case["fractions"])
    if case["basis"] == "mole":
        mw = np.zeros(prepared.ncomponents + prepared.virtual_count)
        mw[prepared.neutral_indices] = prepared.neutral_mw
        mw[prepared.salt_indices] = prepared.salt_mw
        fractions *= mw[: prepared.ncomponents]
    masses = np.tile(fractions, (args.size, 1))
    masses[:, case["components"].index("water")] *= np.linspace(
        0.9, 1.1, args.size
    )
    temp = case["temperature_K"]
    t0 = time.perf_counter()
    model.evaluate(masses[:1], temp)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    result = model.evaluate(masses, temp, diagnostics=True)
    record = dict(
        case=args.case,
        size=args.size,
        device=args.device,
        setup_s=setup,
        first_call_s=first,
        host_to_host_s=time.perf_counter() - t0,
        mean_evaluations=float(
            np.mean(result.diagnostics.get("evaluations", 0))
        ),
    )
    if execution == "cuda":
        import warp as wp

        # Measure the pool high-water mark of host-facing chunked execution
        # before allocating independent buffers for kernel-only timing.
        record["peak_device_pool_bytes"] = wp.get_mempool_used_mem_high(
            model.device
        )
        times = []
        with wp.ScopedStream(model.stream):
            for start in range(0, args.size, model.capacity):
                inputs = masses[start : start + model.capacity]
                rows = len(inputs)
                device_mass = wp.array(
                    inputs, device=model.device, dtype=wp.float64
                )
                ts = wp.array(
                    np.full(rows, float(temp)),
                    device=model.device,
                    dtype=wp.float64,
                )
                out = wp.empty(
                    (rows, model.output_width),
                    device=model.device,
                    dtype=wp.float64,
                )
                flags = wp.empty(rows, device=model.device, dtype=wp.int32)
                begin = wp.Event(model.device, enable_timing=True)
                end = wp.Event(model.device, enable_timing=True)
                wp.record_event(begin)
                model.evaluate_into(
                    device_mass,
                    ts,
                    out,
                    flags,
                    model.scratch[:rows],
                    stream=model.stream,
                )
                wp.record_event(end)
                times.append(wp.get_event_elapsed_time(begin, end) * 1e-3)
        record["kernel_s"] = sum(times)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
