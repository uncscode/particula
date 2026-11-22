"""Load the exported reaction network and reproduce its chamber fits."""

import argparse
import json
from pathlib import Path

import numpy as np
from reactions import Reaction
from smoke import DOSES, OUT

from gpu import Model


def replay(path):
    """Read the reaction specifications and return both fitted scores."""
    spec = json.loads(Path(path).read_text())
    rows = spec["reactions"]
    cards = [
        Reaction(**{k: r[k] for k in Reaction.__dataclass_fields__})
        for r in rows
    ]
    model = Model(
        cards,
        spec["surrogate_molar_masses_g_mol"],
        spec["secondary_vapor_background_loss_h_inv"],
    )
    result = {}
    for obj in ("mass", "joint"):
        rates = np.array([r[f"{obj}_fit_rate"] for r in rows])
        out = model.run(
            spec["baseline_parameters"][obj],
            rates,
            list(range(len(cards))),
            DOSES,
            obj,
            sub=4,
            share=spec["secondary_vapor_share"],
        )
        result[obj] = float(out[0][0])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism", nargs="?", default=OUT / "mechanism.json")
    print(json.dumps(replay(parser.parse_args().mechanism), indent=2))
