"""Rebuild single-H2SO4 loss channels via detailed balance from published binding energies."""
import argparse
import csv
import hashlib
import io
import math
from pathlib import Path
import re
import subprocess
import zipfile

from scipy import constants as c

HERE = Path(__file__).resolve().parent
T = 298.15
ARCHIVE_SHA = "d939c55871d021a42854a7511188b858eb5f17943330caccc21a005003b5dd81"
SUPPLEMENT_PDF = "Acid-Base-Ions_xyz/SupportingInfo-Acid-Base-Ions.pdf"
MA = 0.098079 / c.Avogadro
MN = 0.01703052 / c.Avogadro
PROTON = c.proton_mass
# Ortega 2014 supplementary Table S2, acid monomer only
ALPHA_VOLUME = 6.2e-30  # m3
DIPOLE = 2.96 * 3.33564e-30  # C m


def read_energies(archive):
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Unexpected Myllys 2019 supplement ZIP digest")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        pdf = z.read(SUPPLEMENT_PDF)
    text = subprocess.run(["pdftotext", "-layout", "-", "-"], input=pdf,
                          capture_output=True, check=True).stdout.decode()
    table = text.split("Table S1: Binding energies", 1)[1].split("Evaporation Rates", 1)[0]
    # Keys are (sulfate equivalents, ammonia equivalents, charge). P and B are ionic cores.
    # Standard state monomer energies are zero; 1N1P = NH4+.
    energies = {(1, 0, 0): 0., (0, 1, 0): 0., (1, 0, -1): 0., (0, 1, 1): 0.}
    for label, _, _, g in re.findall(
        r"^\s*([0-9ABNP]+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
        table, re.M
    ):
        units = {s: int(n) for n, s in re.findall(r"(\d+)([ABNP])", label)}
        anion = units.get("B", 0)
        key = units.get("A", 0) + anion, units.get("N", 0), units.get("P", 0) - anion
        if key in energies:
            raise ValueError(f"Repeated composition {label}")
        energies[key] = float(g)
    assert energies[2, 2, 0] == -28.926 and energies[2, 2, -1] == -30.645
    return energies


def acid_capture(sulfate, ammonia, charge):
    """Acid collision coefficient of daughter molecule/cluster, in m3/s."""
    mass = sulfate * MA + ammonia * MN + charge * PROTON
    reduced_mass = 1 / (1 / mass + 1 / MA)
    if charge:
        beta_langevin = c.e * math.sqrt(math.pi * ALPHA_VOLUME / (c.epsilon_0 * reduced_mass))
        x = DIPOLE / math.sqrt(8 * math.pi * c.epsilon_0 * ALPHA_VOLUME * c.k * T)
        enhancement = (0.4767 * x + 0.6200 if x >= 2
                       else (x + 0.5090)**2 / 10.526 + 0.9754)
        return beta_langevin * enhancement
    volume = sulfate * MA / 1830 + ammonia * MN / 696
    radii = [(3 * v / (4 * math.pi))**(1/3) for v in (volume, MA / 1830)]
    return math.pi * sum(radii)**2 * math.sqrt(8 * c.k * T / (math.pi * reduced_mass))



def hydrate_factor(sulfate, ammonia, charge, water_pa):
    """Effective acid activity correction at fixed capture, for covered parent/daughter only.

    Equilibrate water before each acid step, retain the supplied dry capture coefficient,
    and let the incoming acid concentration mean the sum over its water hydrates.
    The result is an equilibrium sensitivity, not a microscopic wet loss rate.
    """
    # Neutral cumulative hydration standard free energies, kcal/mol. Henschel 2014 Table 1.
    neutral = {
        (1, 0): [-2.60,-4.40,-5.83,-7.05,-6.81],
        (2, 0): [-3.36,-6.38,-8.61,-10.97,-12.75],
        (3, 0): [-5.17,-8.22,-12.75,-16.16,-17.78],
        (4, 0): [-2.67,-7.80,-13.96,-13.49,-15.78],
        (1, 1): [-1.67,-5.93,-7.16,-7.09,-8.23],
        (2, 1): [-2.68,-4.00,-6.67,-7.33,-8.48],
        (3, 1): [-3.26,-6.70,-9.08,-9.48,-10.92],
        (2, 2): [-4.14,-7.56,-7.75,-8.18,-9.58],
        (3, 2): [-3.30,-3.40,-7.16,-7.76,-9.46],
        (3, 3): [-1.47,-3.49,-3.36,-5.84],  # Henschel 2016 Table 1
    }
    # Negative stepwise hydration free energies from Tsona et al. 2015 Table 1.
    negative_steps = {
        (1, 0): [-4.20,-2.51,-2.45,-1.88,-1.84],
        (2, 0): [-1.05,0.05,-1.39,0.02,-1.22],
        (3, 0): [-2.45,-3.28,1.11,1.51,-4.02],
        (4, 0): [-4.66,-3.89,-1.96,4.53,-1.17],
        (1, 1): [-4.76,-2.56,-2.15,-0.30,-0.16],
        (2, 1): [-2.38,0.04,-2.23,-4.45,0.17],
        (2, 2): [2.21,-0.32,-6.66,3.06,-6.11],
        (3, 1): [-3.78,-1.25,-0.36,-1.63,-4.08],
        (3, 2): [1.96,-5.12,-1.39,-0.36,-5.27],
        (4, 1): [-5.91,-0.22,0.86],
        (4, 2): [-0.30,0.36],
    }
    if charge > 0:
        return None
    lookup = neutral if not charge else {
        key: [sum(steps[:i+1]) for i in range(len(steps))]
        for key, steps in negative_steps.items()
    }
    parent, daughter = (sulfate, ammonia), (sulfate - 1, ammonia)
    if parent not in lookup or daughter not in lookup:
        return None
    def partition(cumulative):
        return 1 + sum((water_pa / c.atm)**i * math.exp(-g*4184/(c.R*T))
                       for i, g in enumerate(cumulative, start=1))
    return partition(lookup[daughter]) * partition(neutral[1,0]) / partition(lookup[parent])


def channels(energies):
    for (sulfate, ammonia, charge), g in sorted(energies.items()):
        daughter = (sulfate - 1, ammonia, charge)
        if daughter not in energies:
            continue
        beta = acid_capture(*daughter)
        dg_add = g - energies[daughter]
        # Identical-products case: use 1/2 kinetic event counting, not 2 acid losses.
        symmetry = 0.5 if daughter == (1, 0, 0) else 1.0
        gamma = symmetry * beta * c.atm / (c.k*T) * math.exp(dg_add * 4184 / (c.R*T))
        yield [sulfate, ammonia, charge, T, daughter[0], g, energies[daughter], dg_add,
               beta * 1e6, symmetry, gamma]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("myllys_zip", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "evaporation-reference.csv")
    args = parser.parse_args()
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["sulfate_equivalents", "ammonia_equivalents", "charge_e", "T_K",
                         "daughter_sulfates", "G_parent_kcal_mol", "G_daughter_kcal_mol",
                         "delta_G_add_acid_kcal_mol", "acid_capture_beta_cm3_s",
                         "association_pair_factor", "single_H2SO4_loss_s-1"])
        writer.writerows(channels(read_energies(args.myllys_zip)))


if __name__ == "__main__":
    main()
