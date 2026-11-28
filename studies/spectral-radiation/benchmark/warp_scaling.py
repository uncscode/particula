"""Time warmed shortwave column batches on one explicit CUDA device."""

from datetime import UTC, datetime
from statistics import median
from time import perf_counter

import numpy as np
import warp as wp
from particula.optics import dusty_ground, earth_reference, solve_shortwave
from particula.optics.gpu.shortwave import prepare_shortwave


def main():
    """Report setup and solve times in seconds at three batch sizes."""
    ref = earth_reference(
        35, -106, datetime(2024, 6, 21, 19, tzinfo=UTC), [120000, 2000, 0]
    )
    ground = dusty_ground(ref.column.wavelength_m)
    print("columns,nodes,prepare_s,warm_gpu_median_s,cpu_loop_s", flush=True)
    for count in (1, 16, 64):
        columns = [ref.column] * count
        start = perf_counter()
        work = prepare_shortwave(columns, ref.sun, ground, device="cuda:0")
        wp.synchronize_device("cuda:0")
        setup = perf_counter() - start
        work.enqueue()
        work.check()
        times = []
        for _ in range(3):
            start = perf_counter()
            work.enqueue()
            wp.synchronize_device("cuda:0")
            times.append(perf_counter() - start)
        actual = work.download()
        start = perf_counter()
        expected = [solve_shortwave(c, ref.sun, ground) for c in columns]
        cpu = perf_counter() - start
        for result, oracle in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                result.up, oracle.up, rtol=2e-9, atol=0.002
            )
        print(
            f"{count},{len(ref.column.wavelength_m)},{setup:.5f},"
            f"{median(times):.5f},{cpu:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
