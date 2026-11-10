#!/usr/bin/env python3
"""Read-only local Python and Particula diagnostics; no installation or GPU initialization."""
import importlib
from importlib.metadata import version, PackageNotFoundError
import json
import platform
import sys


def main():
    report = {"python": platform.python_version(), "executable": sys.executable, "packages": {}, "api": {}}
    for name in ("particula", "numpy", "matplotlib", "scipy", "warp-lang"):
        try:
            report["packages"][name] = version(name)
        except PackageNotFoundError:
            report["packages"][name] = None
    try:
        par = importlib.import_module("particula")
        report["particula_source"] = str(par.__file__)
        report["particula_source_version"] = getattr(par, "__version__", None)
        for name in ("AerosolBuilder", "gas.GasSpeciesBuilder", "gas.AtmosphereBuilder", "particles.ParticleRepresentation", "particles.SpeciatedMassMovingBin", "particles.ActivityIdealMass", "particles.SurfaceStrategyVolume", "dynamics.MassCondensation", "dynamics.CondensationIsothermal", "dynamics.BrownianCoagulationStrategy", "dynamics.Coagulation"):
            obj = par
            for part in name.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            report["api"][name] = obj is not None
    except Exception as exc:
        report["import_error"] = f"{type(exc).__name__}: {exc}"
    report["starter_ready"] = bool(report["api"]) and all(report["api"].values()) and all(report["packages"][p] for p in ("particula", "numpy", "matplotlib"))
    report["gpu_tested"] = False
    print(json.dumps(report, indent=2))
    return 0 if report["starter_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
