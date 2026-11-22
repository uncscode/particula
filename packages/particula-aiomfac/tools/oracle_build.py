"""Fetch and instrument only the audited AIOMFAC source revision."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import tomllib

PACKAGE = Path(__file__).resolve().parents[1]
REFERENCE = PACKAGE / "reference"
PIN = tomllib.loads((REFERENCE / "upstream.toml").read_text())
# Override when regenerating from an unpacked standalone source distribution.
CACHE = Path(
    os.environ.get("AIOMFAC_CACHE", Path.cwd() / ".artifacts/aiomfac-reference")
).resolve()
SOURCE = CACHE / "upstream"
BUILD = CACHE / "build"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, cwd=SOURCE):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def fetch(offline=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        if offline:
            raise ValueError("offline cache is missing")
        with tempfile.TemporaryDirectory(dir=CACHE) as temp:
            git("init", "-q", cwd=temp)
            git("fetch", "--depth=1", PIN["repository"], PIN["revision"], cwd=temp)
            git("checkout", "--detach", "FETCH_HEAD", cwd=temp)
            Path(temp).rename(SOURCE)
    if git("rev-parse", "HEAD") != PIN["revision"] or git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("cached source has changes or a different HEAD")
    for name, expected in PIN["sha256"].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")


def replace(path, old, new):
    s = path.read_text()
    if s.count(old) != 1:
        raise ValueError(f"instrumentation anchor changed in {path.name}: {old}")
    path.write_text(s.replace(old, new))


def instrument(src):
    # Keep the web driver's input conversion, implicit-component mapping and flags.
    replace(
        src / "AIOMFAC_inout.f90", "calcviscosity = .true.", "calcviscosity = .false."
    )
    core = src / "ModCalcActCoeff.f90"
    rows = [
        "block",
        "use ModSubgroupProp, only: Ioncharge",
        "write(*,*) 'REF scale ', Tmolal, SolvMixRefnd",
        "do i=1,nneutral",
        "write(*,*) 'REF neutral ', i,x(i),lnactcoeff_n(i),activity(i),gnsrln(i),gnmrln(i),gnlrln(i),wtf(i)",
        "enddo",
    ]
    for c, count, ids, mol in [
        ("c", "Ncation", "Ication", "SMC"),
        ("a", "Nanion", "Ianion", "SMA"),
    ]:
        rows += [
            f"do i=1,{count}",
            f"ii={ids}(i)",
            f"write(*,*) 'REF ion ',ii,Ioncharge(ii),{mol}(i),lnactcoeff_{c}(i),g{c}srln(i),g{c}mrln(i),g{c}lrln(i)",
            "enddo",
        ]
    rows += [
        "do i=1,nelectrol",
        "write(*,*) 'REF salt ',ElectComps(i,1:2),ElectNues(i,1:2),meanmolalactcoeff(i),ionactivityprod(i)",
        "enddo",
        "end block",
    ]
    end = "    end subroutine AIOMFAC_calc"
    replace(core, end, "\n".join(rows) + "\n" + end)
    diss = src / "SubModDissociationEquil.f90"
    replace(
        diss,
        "diffKHSO4 = abs(diffK)",
        """diffKHSO4 = abs(diffK)
        write(*,*) 'REF sulfate ', diffK, mSulfmax, SMA(idHSO4)+SMA(idSO4)""",
    )
    anchor = "        molarinp(map_solver2molarinp(1:n)) = exp(solve_var(1:n))"
    replace(
        diss,
        anchor,
        """        write(*,*) 'REF carbonate ',info, diffK
"""
        + anchor,
    )
    anchor = "        !predict the equil. partial pressure"
    replace(
        diss,
        anchor,
        "        write(*,*) 'REF carbon ', nCarbmax, nCarb+nCO2+nHCO3\n"
        "        if (bisulfsyst) write(*,*) 'REF sulfur ', nSulfmax, nSulf+nHSO4\n" + anchor,
    )
    main = src / "Main_IO_driver.f90"
    anchor = "            call AIOMFAC_inout("
    replace(main, anchor, "            write(*,*) 'REF point ',pointi\n" + anchor)
    anchor = "            if (warningflag > 0 .or. any(errflag_list))"
    replace(
        main,
        anchor,
        """            write(*,*) 'REF status ',warningflag, pack([(nc,nc=1,size(errflag_list))],errflag_list)
            do nc=1,nspecies
                write(*,*) 'REF species ',nc,outputvars(:,nc)
            enddo
"""
        + anchor,
    )


def build():
    fetch(offline=True)
    compiler = shutil.which(os.environ.get("FC", "gfortran"))
    if compiler is None:
        raise ValueError("GNU Fortran is required for reference generation; set FC")
    flags = "-c -O2 -fcheck=bounds -ffree-line-length-none -fwrapv"
    BUILD.mkdir(exist_ok=True)
    (BUILD / "generation.json").unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        src = Path(temp)
        for name in PIN["sha256"]:
            if name.startswith("FortranCode/"):
                shutil.copyfile(SOURCE / name, src / Path(name).name)
        for name in ("standard", "activity"):
            if name == "activity":
                instrument(src)
            with (BUILD / f"{name}.log").open("w") as log:
                subprocess.run(
                    ["make", f"FC={compiler}", f"LD={compiler}", f"FFLAGS={flags}"],
                    cwd=src,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            shutil.copy2(src / "AIOMFAC-web.out", BUILD / name)
    metadata = dict(
        revision=PIN["revision"],
        flags=flags,
        working_precision="selected_real_kind(15)",
        compiler=subprocess.check_output(
            [compiler, "--version"], text=True
        ).splitlines()[0],
        generator_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(REFERENCE / "upstream.toml"),
        executables={name: sha(BUILD / name) for name in ("standard", "activity")},
    )
    (BUILD / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
