"""Export or verify the initialized v3.15 tables from the verified Fortran tree."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from oracle_build import CACHE, PACKAGE, PIN, SOURCE, fetch, replace

DEST = PACKAGE / "src/particula_aiomfac/data"
# Each axis uses upstream identifiers minus the corresponding offset.
TABLES = {}


def register(names, offsets, units):
    for name in names.split():
        TABLES[name] = (offsets, units)


register("NKTAB SR_RR SR_QQ", [1], "1")
register("GroupMW", [1], "g/mol")
register("SMWC", [201], "g/mol")
register("SMWA", [241], "g/mol")
register("lambdaIN", [201], "kg/mol")
register("Ioncharge", [201], "elementary charge")
register("ARR", [1, 1], "K")
register("bTABnc cTABnc", [1, 201], "kg/mol")
register("bTABna cTABna", [1, 241], "kg/mol")
register("bTABAC cTABAC", [201, 241], "kg/mol")
register("Cn1TABAC Cn2TABAC", [201, 241], "(kg/mol)^2")
register("omegaTAB omega2TAB", [201, 241], "(kg/mol)^0.5")
register("RccTAB", [201, 201], "kg/mol")
register("qcca1TAB", [201, 201, 241], "(kg/mol)^2")


def export(update=False):
    fetch(offline=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
        src = Path(tmp)
        for name in PIN["sha256"]:
            if name.startswith("FortranCode/"):
                shutil.copyfile(SOURCE / name, src / Path(name).name)
        mr = src / "ModMRpart.f90"
        for name in ("RccTAB", "qcca1TAB", "lambdaIN"):
            anchor = next(
                line
                for line in mr.read_text().splitlines()
                if "private ::" in line and name.lower() in line.lower()
            )
            replace(mr, anchor, anchor.replace("private", "public"))
        lines = [
            "program export_tables",
            "use ModSubgroupProp",
            "use ModSRparam",
            "use ModMRpart",
            "implicit none",
            "call SRdata()",
            "call MRdata()",
        ]
        for name in TABLES:
            lines += [
                f"open(99,file='{name}.bin',access='stream',form='unformatted',convert='little_endian')",
                f"write(99) int(shape({name}),4)",
                f"write(99) {name}",
                "close(99)",
            ]
        (src / "Main_IO_driver.f90").write_text(
            "\n".join([*lines, "end program"]) + "\n"
        )
        fc = os.environ.get("FC", "gfortran")
        subprocess.run(
            [
                "make",
                "-s",
                f"FC={fc}",
                f"LD={fc}",
                "FFLAGS=-c -O2 -ffree-line-length-none -fwrapv",
            ],
            cwd=src,
            check=True,
        )
        subprocess.run(["./AIOMFAC-web.out"], cwd=src, check=True)
        arrays, metadata = {}, {}
        for name, (offsets, units) in TABLES.items():
            dtype = "<i4" if name in {"NKTAB", "Ioncharge"} else "<f8"
            with (src / f"{name}.bin").open("rb") as f:
                shape = np.fromfile(f, dtype="<i4", count=len(offsets))
                arr = (
                    np.fromfile(f, dtype=dtype).reshape(shape, order="F").copy()
                )
            arrays[name] = arr
            metadata[name] = dict(
                shape=list(map(int, shape)),
                dtype=dtype,
                units=units,
                index_offset=offsets,
                sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
            )
        # Explicitly identified analog sources in the audited ModMRpart::MRdata.
        manifest = dict(
            revision=PIN["revision"],
            dataset=1,
            use_latest_param=False,
            source_manifest="reference/upstream.toml",
            tables=metadata,
            estimated_pairs={
                "201/250": "203/250",
                "204/246": "204/245",
                "204/247": "203/247",
                "221/246": "221/245",
                "221/247": "221/244",
                "223/247": "223/244",
            },
        )
        if update:
            DEST.mkdir(exist_ok=True)
            np.savez_compressed(DEST / "parameters.npz", **arrays)
            (DEST / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
        else:
            if json.loads((DEST / "manifest.json").read_text()) != manifest:
                raise ValueError("generated parameter metadata differs")
            with np.load(DEST / "parameters.npz") as existing:
                if set(existing.files) != set(arrays):
                    raise ValueError("parameter inventory differs")
                for name, arr in arrays.items():
                    np.testing.assert_array_equal(
                        existing[name], arr, strict=True
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    export(parser.parse_args().update)
