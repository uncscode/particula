"""Archive AIOMFAC-web submissions for the published chemistry."""

import csv
import gzip
import json
import re
import time
from math import gcd
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import requests
from scripts.mt_hua_salts.study import ANION_EQ, CATION_EQ, MASS_PER_EQ

ROOT = Path(__file__).resolve().parents[2] / "docs/data/mt-hua-salts"
ENDPOINT = "https://www.aiomfac.lab.mcgill.ca/cgi_bin/model.pl"
RATIOS = np.geomspace(0.35, 30, 33)


def compositions():
    """Preserve the same balanced, proportional neutral-salt dry masses."""
    with (ROOT / "matched.csv").open() as f:
        samples = list(csv.DictReader(f))
    mixes = []
    for i, row in enumerate(samples):
        cat = (
            np.array([float(row[k]) for k in ("nh4", "na", "k", "ca", "mg")])
            / CATION_EQ
        )
        an = np.array([float(row[k]) for k in ("so4", "no3", "cl")]) / ANION_EQ
        if min(cat.sum(), an.sum()) == 0:
            continue
        mass = np.outer(cat / cat.sum(), an / an.sum()) * MASS_PER_EQ
        mixes.append((str(i), mass.ravel() / mass.sum()))
    for i in [0, 5, 10, 12] + [k for k in range(15) if k not in (0, 5, 10, 12)]:
        mixes.append((f"pure_{i}", np.eye(15)[i]))
    return mixes


def fetch():
    """Submit three grids per public form, sequentially; save exact I/O."""
    dest = ROOT / "aiomfac"
    dest.mkdir(exist_ok=True)
    mixes = compositions()
    original = [
        r
        for r in mixes
        if r[0] in {"pure_0", "pure_5", "pure_10", "pure_12"}
        or not r[0].startswith("pure")
    ]
    extra = [
        r
        for r in mixes
        if r[0].startswith("pure")
        and r[0] not in {"pure_0", "pure_5", "pure_10", "pure_12"}
    ]
    batches = [
        group[i : i + 3]
        for group in (original, extra)
        for i in range(0, len(group), 3)
    ]
    for index, group in enumerate(batches):
        path = dest / f"batch-{index:02d}.json.gz"
        if path.exists():
            continue
        ids = np.flatnonzero(sum(m for _, m in group) > 0)
        form = dict(
            water="yes",
            comp_type="mass",
            counter="0",
            num_compounds=str(len(ids) + 1),
            honig="",
        )
        for k, idx in enumerate(ids, 1):
            c, a = divmod(idx, 3)
            pc = [1, 1, 1, 2, 2][c]
            pa = [2, 1, 1][a]
            form.update(
                {
                    f"inorganic_{k}": f"salt{idx}",
                    f"cation_{k}": str([204, 202, 203, 221, 223][c]),
                    f"anion_{k}": str([261, 245, 242][a]),
                    f"cation_qty_{k}": str(pa // gcd(pc, pa)),
                    f"anion_qty_{k}": str(pc // gcd(pc, pa)),
                }
            )
        form["conc"] = "\r\n".join(
            "298.15 " + " ".join(f"{x:.12g}" for x in mass[ids] / (1 + r))
            for _, mass in group
            for r in RATIOS
        )
        res = requests.post(ENDPOINT, data=form, timeout=90)
        res.raise_for_status()

        def artifact(pattern, body=res.text):
            match = re.search(pattern, body)
            if not match:
                raise ValueError("AIOMFAC returned no official output link")
            url = urljoin(ENDPOINT, match[1])
            raw = requests.get(url, timeout=60)
            raw.raise_for_status()
            return raw.text

        inp = artifact(r"href=['\"]([^'\"]*/input_\d+\.txt)")
        out = artifact(r"href=['\"]([^'\"]*/AIOMFAC_output_\d+\.txt)")
        if "Component's name      : 'Water'" not in out:
            raise ValueError(out[:500])
        record = dict(
            mixture_ids=[name for name, _ in group],
            endpoint=ENDPOINT,
            form=form,
            input=inp,
            output=out,
        )
        temp = path.with_suffix(".tmp")
        temp.write_bytes(
            gzip.compress(
                json.dumps(record, ensure_ascii=False).encode(), mtime=0
            )
        )
        temp.replace(path)
        print(path.name, flush=True)
        time.sleep(1.5)


if __name__ == "__main__":
    fetch()
