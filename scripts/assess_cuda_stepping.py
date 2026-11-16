"""Actual CUDA feasibility of per-box adaptive scalar closed exchange.

Research prototype only: y is gas fraction, 1-y particle fraction. Each box
has its own controller and stops at 1 second. Counts include rejected trials;
there is no CPU fallback, capture, transfer inside the loop, or aerosol adapter.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp


@wp.kernel
def integrate(
    forward: wp.array(dtype=wp.float64),
    method: int,
    result: wp.array(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.int32),
):
    """Evaluate RK3(2)/ROS-2 with per-box acceptance on the device."""
    box = wp.tid()
    a = forward[box]
    b = wp.float64(0.2)
    y = wp.float64(1.0)
    time = wp.float64(0.0)
    h = wp.float64(0.1)
    accepted = int(0)
    rejected = int(0)
    trials = int(0)
    gamma = wp.float64(1.0) + wp.float64(1.0) / wp.sqrt(wp.float64(2.0))
    while time < wp.float64(1.0) and trials < 100000:
        h = wp.min(h, wp.float64(1.0) - time)
        trials += 1
        f = b - (a + b) * y
        candidate = wp.float64(0.0)
        error = wp.float64(0.0)
        valid = bool(True)
        power = wp.float64(-1.0 / 3.0)
        if method == 0:
            stage2 = y + h * f / wp.float64(2.0)
            k2 = b - (a + b) * stage2
            stage3 = y + wp.float64(0.75) * h * k2
            k3 = b - (a + b) * stage3
            candidate = y + h * (
                wp.float64(2.0 / 9.0) * f
                + k2 / wp.float64(3.0)
                + wp.float64(4.0 / 9.0) * k3
            )
            k4 = b - (a + b) * candidate
            error = h * (
                -wp.float64(5.0 / 72.0) * f
                + k2 / wp.float64(12.0)
                + k3 / wp.float64(9.0)
                - k4 / wp.float64(8.0)
            )
            valid = (
                stage2 >= wp.float64(0.0)
                and stage2 <= wp.float64(1.0)
                and stage3 >= wp.float64(0.0)
                and stage3 <= wp.float64(1.0)
            )
        else:
            denominator = wp.float64(1.0) / (h * gamma) + a + b
            k1 = f / denominator
            stage = y + k1 / gamma
            k2 = (
                b - (a + b) * stage - wp.float64(2.0) * k1 / (gamma * h)
            ) / denominator
            candidate = y + (wp.float64(3.0) * k1 + k2) / (
                wp.float64(2.0) * gamma
            )
            error = (k1 + k2) / (wp.float64(2.0) * gamma)
            valid = stage >= wp.float64(0.0) and stage <= wp.float64(1.0)
            power = wp.float64(-0.5)
        valid = (
            valid
            and candidate >= wp.float64(0.0)
            and candidate <= wp.float64(1.0)
        )
        gas_scale = wp.float64(1e-10) + wp.float64(1e-7) * wp.max(
            wp.abs(y), wp.abs(candidate)
        )
        particle_scale = wp.float64(1e-10) + wp.float64(1e-7) * wp.max(
            wp.abs(wp.float64(1.0) - y), wp.abs(wp.float64(1.0) - candidate)
        )
        norm = wp.abs(error) / wp.min(gas_scale, particle_scale)
        if valid and norm <= wp.float64(1.0):
            y = candidate
            time += h
            accepted += 1
            factor = wp.float64(5.0)
            if norm > wp.float64(0.0):
                factor = wp.min(
                    wp.float64(5.0),
                    wp.max(
                        wp.float64(0.2), wp.float64(0.9) * wp.pow(norm, power)
                    ),
                )
            h = wp.min(wp.float64(0.1), h * factor)
        else:
            rejected += 1
            h *= wp.float64(0.5)
    result[box] = y
    counts[box, 0] = accepted
    counts[box, 1] = rejected
    counts[box, 2] = int(time >= wp.float64(1.0))


def assess():
    """Measure warmed kernel time and independent endpoint error."""
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("actual CUDA is required")
    device = wp.get_device("cuda:0")
    rows = []
    for boxes in (1, 4096):
        rates = np.geomspace(1.0, 1000.0, boxes)
        forward = wp.array(rates, dtype=wp.float64, device=device)
        output = wp.empty(boxes, dtype=wp.float64, device=device)
        counts = wp.empty((boxes, 3), dtype=wp.int32, device=device)
        exact = 0.2 / (rates + 0.2) + (1 - 0.2 / (rates + 0.2)) * np.exp(
            -(rates + 0.2)
        )
        for method in (0, 1):
            wp.launch(
                integrate,
                dim=boxes,
                inputs=[forward, method, output, counts],
                device=device,
            )
            wp.synchronize_device(device)
            start = perf_counter()
            wp.launch(
                integrate,
                dim=boxes,
                inputs=[forward, method, output, counts],
                device=device,
            )
            wp.synchronize_device(device)
            elapsed = perf_counter() - start
            values, work = output.numpy(), counts.numpy()
            if not np.all(work[:, 2] == 1):
                raise RuntimeError(
                    "trial limit reached; not all boxes completed"
                )
            rows.append(
                {
                    "method": "rk23" if method == 0 else "ros2",
                    "boxes": boxes,
                    "seconds_excluding_upload_download_compile": elapsed,
                    "maximum_endpoint_error": float(
                        np.max(np.abs(values - exact))
                    ),
                    "accepted_range": [
                        int(work[:, 0].min()),
                        int(work[:, 0].max()),
                    ],
                    "rejected_range": [
                        int(work[:, 1].min()),
                        int(work[:, 1].max()),
                    ],
                    "device_array_bytes": boxes * (8 + 8 + 3 * 4),
                }
            )
    return {
        "device": str(device),
        "name": device.name,
        "warp": wp.__version__,
        "runs": rows,
    }


def main():
    """Save actual CUDA evidence, failing if the device is unavailable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
