"""Define product-specific pools and the ordered 40 multigeneration trials."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

JOINT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOINT / "discovery"))
from reactions import Reaction  # noqa: E402

OUT = JOINT / "results/chains"
PRIOR = json.loads(
    (JOINT / "results/discovery/best_seen_mechanism.json").read_text()
)
FAMILIES = ("AA", "AC", "AU", "CC", "II")
PARENTS = {
    "AA": ("aa_gas", "aa_particle"),
    "AC": ("ac_gas",),
    "AU": ("a_uptake",),
    "CC": ("cc_gas", "cc_particle"),
    "II": ("ii_gas", "ii_particle"),
}
# D = original labelled dimer or uptake product, W = oxidized particle,
# L = smaller retained residue, U = secondary vapor, F = terminal volatiles.
TRANSFERS = {
    "oxidize": ("D", "W", None),
    "split": ("D", "U", "L"),
    "oxsplit": ("W", "U", "L"),
    "gasage": ("U", "W", None),
    "cleave": ("D", "L", None),
    "resloss": ("L", "F", None),
    "volatile": ("D", "F", None),
    "gasfrag": ("U", "F", None),
    "oxcleave": ("W", "L", None),
    "resoxid": ("L", "W", None),
}
MOTIFS = (
    ("Retained oxidation", ("oxidize",)),
    ("Oxidation then fragmentation", ("oxidize", "oxsplit")),
    ("Direct partial fragmentation", ("split",)),
    ("Oxidation of released vapor", ("split", "gasage")),
    ("Retained cleavage then fragment loss", ("cleave", "resloss")),
    ("Extensive fragmentation", ("volatile",)),
    ("Released vapor fragmentation", ("split", "gasfrag")),
    ("Oxidized residue and further aging", ("oxcleave", "resoxid")),
)


@dataclass(frozen=True)
class Channel(Reaction):
    """A second destination receives half the transferred mass if present."""

    split: int = -1


def make_bank():
    """Map original product labels and specify common-rate chamber trials."""
    pools = {
        f: dict(D=11 + j, W=16 + j, L=21 + j, U=26 + j, F=10)
        for j, f in enumerate(FAMILIES)
    }
    bank = []
    for row in PRIOR["reactions"]:
        fields = {key: row[key] for key in Reaction.__dataclass_fields__}
        for family, parents in PARENTS.items():
            if row["key"] in parents:
                fields["dest"] = pools[family]["D"]
        bank.append(Channel(**fields))
    edges = {}
    trials = []
    for name, steps in MOTIFS:
        for f in FAMILIES:
            ids = []
            for step in steps:
                if (f, step) not in edges:
                    src, dest, split = TRANSFERS[step]
                    eq = (
                        f"{src}_{f} → {dest}_{f}"
                        if dest != "F"
                        else f"{src}_{f} → F"
                    )
                    if split:
                        eq = f"{src}_{f} → ½ {dest}_{f} + ½ {split}_{f}"
                    edges[f, step] = len(bank)
                    bank.append(
                        Channel(
                            f"{f}_{step}",
                            eq,
                            name,
                            pools[f][src],
                            -1,
                            pools[f][dest],
                            False,
                            pools[f][split] if split else -1,
                        )
                    )
                ids.append(edges[f, step])
            trials.append(dict(family=f, motif=name, ids=ids))
    return bank, trials


BANK, TRIALS = make_bank()
CAP = np.array([25.0 if r.b >= 0 else 5.0 for r in BANK])
DOSES = np.linspace(0.85, 1.15, 61)


def manifest():
    """Export the full pool map, bounds, proposals and inherited reference."""
    return dict(
        reactions=[
            dict(asdict(r), upper=cap) for r, cap in zip(BANK, CAP, strict=True)
        ],
        trials=TRIALS,
        families=FAMILIES,
        initial=PRIOR,
    )
