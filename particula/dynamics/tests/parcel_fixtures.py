"""Manufactured soluble parcel and execution devices for parcel controls."""

import numpy as np
import pytest
from particula.dynamics.parcel import ParcelInitialState, run_parcel
from particula.dynamics.parcel_trace import ParcelTrace


@pytest.fixture(name="trace")
def parcel_trace_fixture():
    """Record a small, supersaturation-sensitive soluble aerosol."""
    initial = ParcelInitialState(
        np.array([45e-9, 100e-9]),
        np.array([4e7, 2e7]),
        0.5,
        90000.0,
        288.15,
        0.0118,
    )
    return ParcelTrace.record(
        run_parcel(initial, updraft=1.0, timestep=0.25, steps=4)
    )


@pytest.fixture(params=["cpu", "cuda:0"], name="device")
def parcel_device_fixture(request):
    """Select CUDA only on a real device."""
    import warp as wp

    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("native CUDA unavailable")
    return request.param
