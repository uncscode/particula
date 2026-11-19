"""Compare neutral NaCl rates with Cai et al. (2022), Table S1 and Fig. S1."""

import argparse
import json
from pathlib import Path

import numpy as np
from particula.dynamics import get_van_der_waals_kernel_via_system_state
from particula.util.constants import BOLTZMANN_CONSTANT


def compare():
    """Use the reported Kn and strength in the published interpolations."""
    data = []
    hamaker = 8.9e-20
    temperature = 298.0
    # Table S1 diameter [nm], particle Kn and measured beta [cm3/s].
    for small, kn, measured in [
        (3.6, 1.72, 1.89e-7),
        (3.7, 1.70, 1.74e-7),
        (4.0, 1.63, 1.76e-7),
        (5.0, 1.45, 1.22e-7),
        (10.0, 0.99, 4.25e-8),
    ]:
        radius = np.array([small, 100.0]) * 0.5e-9
        mass = (4 * np.pi / 3) * 2160 * radius**3
        inputs = (radius, mass, [0, 0], temperature, 101325.0)
        rate = get_van_der_waals_kernel_via_system_state(*inputs, hamaker, 1)[
            0, 1
        ]
        hard = get_van_der_waals_kernel_via_system_state(*inputs, 0.0, 1)[0, 1]
        strength = (
            hamaker
            / (BOLTZMANN_CONSTANT * temperature)
            * 4
            * np.prod(radius)
            / sum(radius) ** 2
        )
        log_strength = np.log1p(strength)
        ec = 1 + 0.0757 * log_strength + 0.0015 * log_strength**3
        ef = (
            1
            + np.sqrt(strength / 3) / (1 + 0.0151 * np.sqrt(strength))
            - 0.186 * log_strength
            - 0.0163 * log_strength**3
        )
        # S10–S14. x is beta_continuum / beta_ballistic for hard spheres.
        x = 4 * kn / 3
        alam = ec * (1 + x) / (1 + x * ec / ef)
        t = ef / (2 * x * ec)
        t0 = 1 / (2 * x)
        sceats = ef * (np.hypot(1, t0) + t0) / (np.hypot(1, t) + t)
        data.append(
            dict(
                diameter_nm=small,
                reported_kn=kn,
                measured_m3_s=measured * 1e-6,
                model_m3_s=rate,
                model_enhancement=rate / hard,
                alam_enhancement=alam,
                sceats_enhancement=sceats,
            )
        )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.write_text(json.dumps(compare(), indent=2) + "\n")
