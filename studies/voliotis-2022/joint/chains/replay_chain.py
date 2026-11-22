"""Reconstruct the product-resolved CUDA simulation from its saved mechanism."""

import argparse
import json
from pathlib import Path

import numpy as np
from bank import OUT, Channel

from gpu import Model


def replay(path):
    """Reprofile each calibration on the original 15 runs and compare scores."""
    archive = json.loads(Path(path).read_text())
    channels = [
        Channel(**{k: row[k] for k in Channel.__dataclass_fields__})
        for row in archive["reactions"]
    ]
    conf = archive["conditions"]
    model = Model(channels, vapor_loss_h=conf["secondary_vapor_loss_per_hour"])
    scores = {}
    for obj, record in archive["fits"].items():
        fit = record["fit"]
        x = np.array(fit["x"])
        score, _, _, _ = model.run(
            x[:16],
            x[16:],
            record["active"],
            np.linspace(0.85, 1.15, 61),
            obj,
            sub=round(60 / conf["integration_seconds"]),
            split_residue=conf["retained_split_fraction"],
        )
        np.testing.assert_allclose(score[0], fit["score"], atol=1e-10)
        scores[obj] = float(score[0])
        if obj in archive.get("selection", {}):
            selected = archive["selection"][obj]
            for label, reverse in (("forward", False), ("reverse", True)):
                value = model.run(
                    x[:16],
                    x[16:],
                    record["active"],
                    np.linspace(0.85, 1.15, 61),
                    obj,
                    sub=round(60 / selected["seconds"]),
                    reverse=reverse,
                    split_residue=conf["retained_split_fraction"],
                )[0][0]
                np.testing.assert_allclose(value, selected[label], atol=1e-10)
                scores[f"{obj}_{label}"] = float(value)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mechanism", nargs="?", default=OUT / "mechanism.json")
    print(json.dumps(replay(parser.parse_args().mechanism), indent=2))
