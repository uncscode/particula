"""Build the study-specific size-selective jump sampler and native kernel lookup."""

import ctypes, subprocess, tempfile
from pathlib import Path
import numpy as np
from simulate import DYN, RHO, RUN_EPS
from particula.dynamics.coagulation.sedimentation_kernel import (
    get_sedimentation_kernel_sp2016_via_system_state,
)


def sampler(run, temp, p):
    folder = tempfile.TemporaryDirectory(prefix="sagan-collision-")
    so = Path(folder.name) / "pair.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-shared",
            "-std=c++17",
            "-fPIC",
            str(Path(__file__).with_name("collisions.cpp")),
            "-o",
            str(so),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(so))
    arr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    fn = lib.evolve
    fn.argtypes = [
        arr,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        arr,
        arr,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_ulonglong,
    ]
    fn.restype = ctypes.c_int
    d = np.arange(0.5, 351, 1.0)
    r = d * 0.5e-6
    k = 1e6 * (
        DYN.get_brownian_kernel_via_system_state(r, 4 * np.pi * RHO * r**3 / 3, temp, p)
        + get_sedimentation_kernel_sp2016_via_system_state(r, np.full_like(r, RHO), temp, p, False)
        + DYN.get_turbulent_shear_kernel_st1956_via_system_state(
            r, RUN_EPS[run], temp, p * 0.0280134 / (8.314462618 * temp)
        )
    )
    k = np.ascontiguousarray(k)
    bounds = np.maximum.accumulate(np.tril(k).max(1))
    assert np.allclose(k, k.T)
    upper = k[20:49, 20:49].max()

    def advance(mass, vol, dt, other, mid, seed):
        m = mass.copy()
        n = fn(m, len(m), vol * 1e6, dt, other, mid, k, bounds, len(d), upper, seed)
        assert n >= 0, n
        return m[:n]

    base_kernel = k

    def set_multiplier(matrix):
        nonlocal k, bounds, upper
        assert (
            matrix.shape == base_kernel.shape
            and np.isfinite(matrix).all()
            and matrix.min() > 0
            and np.allclose(matrix, matrix.T)
        )
        k = np.ascontiguousarray(base_kernel * matrix)
        bounds = np.maximum.accumulate(np.tril(k).max(1))
        upper = k[20:49, 20:49].max()

    advance.set_multiplier = set_multiplier
    advance.temporary = folder
    return advance
