"""Profile four warmed resident graphs on one NVIDIA device."""

import ctypes
import runpy

import warp as wp
from particula.dynamics.parcel import ParcelRK23, ParcelROS3
from particula.execution import _condensation_device_control as control
from particula.execution import _multivapor_parcel_kernels as kernels
from particula.execution.gpu_resident_condensation import (
    GPUResidentCondensation,
)
from particula.execution.multivapor_parcel import ResidentMultiVaporParcel


def profile_graphs():
    """Capture only replay, excluding preparation and explicit observations."""
    boxmod = runpy.run_path(
        "docs/studies/examples/thermodynamics/activity/aiomfac_closed_box.py"
    )
    parcelmod = runpy.run_path(
        "docs/studies/examples/thermodynamics/parcel-experiments/aiomfac_multivapor_parcel.py"
    )
    box = boxmod["make_box"]()
    parcel = parcelmod["initial_parcel"]()
    owners = []
    graphs = []
    for method, cls in [("rk23", ParcelRK23), ("ros3", ParcelROS3)]:
        b = GPUResidentCondensation(
            box.aerosol,
            box.process,
            device="cuda:0",
            method=method,
            first_step=1e-5,
        )
        b.advance(0.00005)
        b._launch(
            control.set_bound,
            1,
            [b._control, b._condition, b._counters, wp.float64(0.0001)],
        )
        p = ResidentMultiVaporParcel(
            parcel,
            device="cuda:0",
            adaptive=cls(rtol=2e-5),
            timestep=1.0,
            updraft=1.0,
        )
        p.advance(0.5)
        p._launch(kernels.start, 1, p.clock, p.counters, 1.0, p.cond)
        owners.extend((b, p))
        graphs.extend((b._graph, p.graph))
    wp.synchronize_device("cuda:0")
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuProfilerStart() != 0:
        raise RuntimeError("CUDA profiler could not start")
    for graph in graphs:
        wp.capture_launch(graph)
    wp.synchronize_device("cuda:0")
    if driver.cuProfilerStop() != 0:
        raise RuntimeError("CUDA profiler could not stop")
    for i in (0, 2):
        if not (
            owners[i]._counters.numpy()[3] == 0
            and owners[i]._control.numpy()[1] == 0.0001
        ):
            raise RuntimeError("closed box stopped before the observation")
        if not (
            owners[i + 1].counters.numpy()[3] == 0
            and owners[i + 1].clock.numpy()[0] == 1.0
        ):
            raise RuntimeError("parcel stopped before the observation")


if __name__ == "__main__":
    profile_graphs()
