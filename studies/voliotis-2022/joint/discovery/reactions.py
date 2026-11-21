"""Candidate transfers between parent-tagged effective organic product pools."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Reaction:
    """Describe a fitted transfer and its literature-search hypothesis."""

    key: str
    equation: str
    hypothesis: str
    a: int
    b: int
    dest: int
    area_scaled: bool = False


# G: delayed gas intermediate, V: oxygenated vapor, P: exchangeable particle.
# A/C/I denote alpha-pinene, o-cresol and isoprene. Stable products and terminal
# volatile pools aggregate for physics, with reaction IDs retained.
CANDIDATES = [
    Reaction(
        "a_evap",
        "P_A → V_A",
        "Evaporation of semivolatile alpha-pinene SOA",
        6,
        -1,
        3,
    ),
    Reaction(
        "c_evap",
        "P_C → V_C",
        "Evaporation of semivolatile cresol SOA",
        7,
        -1,
        4,
    ),
    Reaction(
        "i_evap",
        "P_I → V_I",
        "Evaporation of semivolatile isoprene SOA",
        8,
        -1,
        5,
    ),
    Reaction(
        "a_oxid",
        "V_A → S_A,ox",
        "Further gas oxidation of "
        "pinonaldehyde-like vapor to condensable products",
        3,
        -1,
        9,
    ),
    Reaction(
        "c_oxid",
        "V_C → S_C,ox",
        "Further gas oxidation of "
        "cresol-derived phenolics to condensable material",
        4,
        -1,
        9,
    ),
    Reaction(
        "i_uptake",
        "V_I → S_I,upt",
        "Reactive uptake of an "
        "isoprene-derived oxygenated vapor (epoxydiol-like)",
        5,
        -1,
        9,
        True,
    ),
    Reaction(
        "a_uptake",
        "V_A → S_A,upt",
        "Reactive uptake of a pinonaldehyde-like vapor onto seed",
        3,
        -1,
        9,
        True,
    ),
    Reaction(
        "c_uptake",
        "V_C → S_C,upt",
        "Reactive uptake of aromatic oxidation carbonyls on aqueous aerosol",
        4,
        -1,
        9,
        True,
    ),
    Reaction(
        "aa_gas",
        "2 G_A → S_AA",
        "Gas-phase alpha-pinene peroxy radical accretion products",
        0,
        0,
        9,
    ),
    Reaction(
        "cc_gas",
        "2 G_C → S_CC",
        "Gas-phase aromatic peroxy radical accretion in cresol oxidation",
        1,
        1,
        9,
    ),
    Reaction(
        "ac_gas",
        "G_A + G_C → S_AC",
        "Cross-accretion of oxygenated intermediates "
        "from alpha-pinene and cresol",
        0,
        1,
        9,
    ),
    Reaction(
        "ai_term",
        "G_A + G_I → F_AI",
        "Mixed isoprene and monoterpene "
        "peroxy termination to volatile products",
        0,
        2,
        10,
    ),
    Reaction(
        "ci_gas",
        "G_C + G_I → S_CI",
        "Mixed isoprene and aromatic gas products condense",
        1,
        2,
        9,
    ),
    Reaction(
        "aa_particle",
        "2 P_A → S_AA,p",
        "Particle-phase dimer or oligomer formation in alpha-pinene SOA",
        6,
        6,
        9,
    ),
    Reaction(
        "cc_particle",
        "2 P_C → S_CC,p",
        "Condensed-phase coupling of cresol-derived phenolics",
        7,
        7,
        9,
    ),
    Reaction(
        "ii_particle",
        "2 P_I → S_II,p",
        "Condensed-phase oligomerization of isoprene SOA products",
        8,
        8,
        9,
    ),
    Reaction(
        "a_photo",
        "P_A → F_A,p",
        "Photochemical fragmentation and volatilization of alpha-pinene SOA",
        6,
        -1,
        10,
    ),
    Reaction(
        "c_photo",
        "P_C → F_C,p",
        "Photochemical volatilization of oxidized aromatic aerosol",
        7,
        -1,
        10,
    ),
    Reaction(
        "i_photo",
        "P_I → F_I,p",
        "Photochemical volatilization of isoprene-derived organic aerosol",
        8,
        -1,
        10,
    ),
    Reaction(
        "a_gasfrag",
        "V_A → F_A,g",
        "Gas-phase pinonaldehyde photolysis/fragmentation",
        3,
        -1,
        10,
    ),
    Reaction(
        "c_gasfrag",
        "V_C → F_C,g",
        "Photolysis of ring-opening carbonyls from aromatic oxidation",
        4,
        -1,
        10,
    ),
    Reaction(
        "i_gasfrag",
        "V_I → F_I,g",
        "Photolysis of isoprene carbonyl products such as methacrolein",
        5,
        -1,
        10,
    ),
    Reaction(
        "ii_gas",
        "2 G_I → S_II",
        "Gas-phase accretion of isoprene-derived peroxy radicals",
        2,
        2,
        9,
    ),
    Reaction(
        "a_return",
        "V_A → P_A",
        "Recondensation of semivolatile alpha-pinene products",
        3,
        -1,
        6,
        True,
    ),
    Reaction(
        "c_return",
        "V_C → P_C",
        "Recondensation of semivolatile aromatic oxidation products",
        4,
        -1,
        7,
        True,
    ),
    Reaction(
        "i_return",
        "V_I → P_I",
        "Recondensation of semivolatile isoprene oxidation products",
        5,
        -1,
        8,
        True,
    ),
    Reaction(
        "a_carbonyl",
        "G_A → V_A",
        "Alpha-pinene radical oxidation "
        "yielding pinonaldehyde-like carbonyl vapor",
        0,
        -1,
        3,
    ),
    Reaction(
        "c_ringopen",
        "G_C → V_C",
        "Ring opening of cresol radical products to volatile carbonyls",
        1,
        -1,
        4,
    ),
    Reaction(
        "i_carbonyl",
        "G_I → V_I",
        "Isoprene peroxy chemistry yielding methyl "
        "vinyl ketone or methacrolein-like products",
        2,
        -1,
        5,
    ),
]


def catalog():
    """Save model equations, units and the numerical exploration bounds."""
    return [
        dict(
            asdict(r),
            rate_units="h^-1" if r.b < 0 else "m3 air (umol h)^-1",
            upper=5.0 if r.b < 0 else 25.0,
        )
        for r in CANDIDATES
    ]
