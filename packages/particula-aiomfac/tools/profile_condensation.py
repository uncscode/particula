"""Profile four warmed resident graphs on one NVIDIA device."""

import argparse
import ctypes
import json
import runpy
import sqlite3
from dataclasses import replace
from pathlib import Path

import warp as wp
from particula.dynamics.parcel import ParcelRK23, ParcelROS3
from particula.execution import _condensation_device_control as control
from particula.execution import _multivapor_parcel_kernels as kernels
from particula.execution.gpu_resident_condensation import (
    GPUResidentCondensation,
)
from particula.execution.multivapor_parcel import ResidentMultiVaporParcel


def profile_graphs(hybrid=False, method=None):
    """Capture only replay, excluding preparation and explicit observations."""
    boxmod = runpy.run_path(
        "docs/studies/examples/thermodynamics/activity/aiomfac_closed_box.py"
    )
    parcelmod = runpy.run_path(
        "docs/studies/examples/thermodynamics/parcel-experiments/aiomfac_multivapor_parcel.py"
    )
    box = boxmod["make_box"](table="mixed-two-vapor" if hybrid else None)
    parcel = parcelmod["initial_parcel"]()
    schemes = {"rk23": ParcelRK23, "ros3": ParcelROS3}
    methods = (method,) if method else (("rk23",) if hybrid else tuple(schemes))
    if hybrid:
        from particula_aiomfac.adapter import AIOMFACActivity

        parcel = replace(
            parcel,
            activity_strategy=AIOMFACActivity(
                parcel.activity_strategy.mixture,
                dict(enumerate(parcel.material_names)),
                evaluation="hybrid",
                table="mixed-two-vapor",
            ),
        )
    owners = []
    graphs = []
    for method in methods:
        cls = schemes[method]
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
    for i in range(0, len(owners), 2):
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


def summarize_trace(path):
    """Summarize a CUDA-profiler-range Nsight SQLite export after replay."""
    with sqlite3.connect(path) as db:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        runtime = dict(
            db.execute(
                "SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON r.nameId=s.id GROUP BY s.value"
            )
        )
        kernels = dict(
            db.execute(
                "SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.shortName=s.id GROUP BY s.value"
            )
        )
        copies = 0
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            copies = db.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY"
            ).fetchone()[0]
    return dict(
        graph_launches=sum(v for k, v in runtime.items() if "GraphLaunch" in k),
        kernel_executions=sum(kernels.values()),
        lookup_kernels=sum(
            v for k, v in kernels.items() if k.startswith("_lookup_")
        ),
        cuda_copies=copies,
        allocation_calls=sum(
            v for k, v in runtime.items() if "Alloc" in k or "Malloc" in k
        ),
        runtime_calls=runtime,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--method", choices=("rk23", "ros3"))
    parser.add_argument(
        "--summary", type=Path, help="summarize an exported Nsight trace"
    )
    args = parser.parse_args()
    if args.summary:
        print(json.dumps(summarize_trace(args.summary), indent=2))
    else:
        profile_graphs(args.hybrid, args.method)
