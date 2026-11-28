"""Run a downloaded historical study with its matching reporting helpers."""

import runpy
import sys
from pathlib import Path


def load_reporting(particula, root, *, spectrum):
    """Load the study diagnostics and style alongside the installed engine."""
    from importlib.util import module_from_spec, spec_from_file_location

    import matplotlib as mpl
    from matplotlib import style

    if spectrum:
        # These archived engines predate the host-side reporting functions.
        # The study declares the current diagnostic as a hashed science input.
        spec = spec_from_file_location(
            "study_number_spectrum",
            root / "particula/particles/properties/number_spectrum.py",
        )
        diagnostic = module_from_spec(spec)
        spec.loader.exec_module(diagnostic)
        particula.particles.get_particle_number_spectrum = (
            diagnostic.get_particle_number_spectrum
        )
        particula.particles.get_number_per_decade = (
            diagnostic.get_number_per_decade
        )
    spec = spec_from_file_location(
        "particula.plotting", root / "particula/plotting/__init__.py"
    )
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    particula.plotting = module
    style.library["particula.default"] = mpl.rc_params_from_file(
        root / "particula/default.mplstyle", use_default_template=False
    )


if __name__ == "__main__":
    import particula

    root = Path(__file__).resolve().parent
    load_reporting(particula, root, spectrum=True)
    runpy.run_path(str(root / "study.py"), run_name="__main__")
