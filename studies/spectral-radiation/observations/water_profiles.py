"""Generate clear-sky REPTRAN gas tables for prescribed precipitable water.

The optional independent libRadtran installation is documented in ../benchmark.
Gas tables are temporary during extraction. The footprint-layer subset suffices
to replay all reported fluxes and radiances without an external executable.
"""

import argparse
import concurrent.futures
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WATER_CM = (0.5, 1.5, 3.0, 5.0)


def generator():
    """Load the shared, pinned REPTRAN reference generator."""
    path = HERE.parent / "benchmark/generate_reference.py"
    spec = importlib.util.spec_from_file_location("gas_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_table(prefix, out, water_cm):
    """Recompute gas absorption with a prescribed water number profile."""
    data = prefix / "share/libRadtran/data"
    with tempfile.TemporaryDirectory() as tmp:
        supplied = Path(tmp)
        for source in data.iterdir():
            if source.name != "atmmod":
                (supplied / source.name).symlink_to(source)
        (supplied / "atmmod").mkdir()
        for source in (data / "atmmod").iterdir():
            if source.name != "afglus.dat":
                (supplied / "atmmod" / source.name).symlink_to(source)
        profile = np.loadtxt(data / "atmmod/afglus.dat")
        # Height in km and H2O in molecules/cm3 give molecule column in cm^-2.
        mass_g_cm2 = (
            -np.trapezoid(profile[:, 6], profile[:, 0] * 1e5)
            * 18.01528
            / 6.02214076e23
        )
        profile[:, 6] *= water_cm / mass_g_cm2
        np.savetxt(supplied / "atmmod/afglus.dat", profile, fmt="%.11e")
        full, _, model = generator().full_reference(prefix, supplied)
        np.savez_compressed(
            out / f"water-{water_cm:.1f}.npz",
            **full,
            interface_height_m=model[:, 0] * 1e3,
            pressure_pa=model[:, 1] * 100,
            water_number_cm3=profile[:, 6],
            temperature_k=model[:, 2],
            water_cm=water_cm,
        )
        print(f"Generated sea-level water {water_cm:g} cm", flush=True)


def main():
    """Generate the four water-column tables required for footprint staging."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(len(WATER_CM)) as pool:
        jobs = [
            pool.submit(make_table, prefix, args.output, w) for w in WATER_CM
        ]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
