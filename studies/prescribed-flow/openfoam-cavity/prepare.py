"""Generate native v2312 dictionaries and run three cavity cases."""

import argparse
from pathlib import Path

CASES = {
    "reference": (16, 0.002),
    "half-step": (16, 0.001),
    "fine-mesh": (32, 0.001),
}
IMAGE = (
    "opencfd/openfoam-default@sha256:"
    "097e45046a74ee1dae3e6135df0c42723ba55092d246e7c9932f3bfa1e89b7b2"
)


def dictionary(path, body, kind="dictionary", name=None):
    """Write a native dictionary or field with its OpenFOAM header."""
    header = (
        f"FoamFile {{ version 2.0; format ascii; class {kind}; "
        f"object {name or path.name}; }}\n"
    )
    path.write_text(header + body + "\n", encoding="ascii")


def prepare(root):
    """Generate all solver inputs in a new case directory."""
    root.mkdir(parents=True, exist_ok=True)
    for case, (size, step) in CASES.items():
        path = root / case
        for folder in ("constant", "system", "0"):
            (path / folder).mkdir(parents=True)
        system = path / "system"
        dictionary(
            system / "blockMeshDict",
            f"""
scale 1;
vertices ((0 0 0) (.1 0 0) (.1 .1 0) (0 .1 0)
          (0 0 .01) (.1 0 .01) (.1 .1 .01) (0 .1 .01));
blocks (hex (0 1 2 3 4 5 6 7) ({size} {size} 1) simpleGrading (1 1 1));
edges ();
boundary (
 lid {{type wall; faces ((3 7 6 2));}}
 walls {{type wall; faces ((0 4 7 3)(2 6 5 1)(1 5 4 0));}}
 frontBack {{type empty; faces ((0 3 2 1)(4 5 6 7));}}
);
mergePatchPairs ();
""",
        )
        dictionary(
            path / "constant/transportProperties",
            """
nu [0 2 -1 0 0 0 0] 0.001;
DT [0 2 -1 0 0 0 0] 0.00001;
""",
        )
        dictionary(
            system / "fvSchemes",
            """
ddtSchemes {default Euler;}
gradSchemes {default Gauss linear;}
divSchemes {default none; div(phi,U) Gauss linear; div(phi,T) Gauss upwind;}
laplacianSchemes {default Gauss linear orthogonal;}
interpolationSchemes {default linear;}
snGradSchemes {default orthogonal;}
fluxRequired {default no; p;}
""",
        )
        dictionary(
            system / "fvSolution",
            """
solvers {
 p {solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0;}
 pFinal {$p; relTol 0;}
 U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0;}
 T {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-13; relTol 0;}
}
PISO {nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0;}
SIMPLE {nNonOrthogonalCorrectors 0;}
""",
        )
        for stage, application, start, end, dt, write in (
            ("spin", "icoFoam", 0, 0.5, 0.001, 500),
            ("tracer", "scalarTransportFoam", 0.5, 0.58, step, 1),
        ):
            dictionary(
                system / stage,
                f"""
application {application};
startFrom startTime; startTime {start}; stopAt endTime; endTime {end};
deltaT {dt}; adjustTimeStep no; writeControl timeStep; writeInterval {write};
writeFormat ascii; writePrecision 15; writeCompression off;
timeFormat general; timePrecision 10; runTimeModifiable false;
""",
                name="controlDict",
            )
        stationary = "lid {type zeroGradient;} walls {type zeroGradient;}"
        for field, kind, dims, initial, walls in (
            ("p", "volScalarField", "0 2 -2 0 0 0 0", "0", stationary),
            ("T", "volScalarField", "1 -3 0 0 0 0 0", "0", stationary),
            (
                "U",
                "volVectorField",
                "0 1 -1 0 0 0 0",
                "(0 0 0)",
                "lid {type fixedValue; value uniform (1 0 0);} "
                "walls {type fixedValue; value uniform (0 0 0);}",
            ),
        ):
            dictionary(
                path / "0" / field,
                f"""
dimensions [{dims}]; internalField uniform {initial};
boundaryField {{{walls} frontBack {{type empty;}}}}
""",
                kind,
            )
        dictionary(
            system / "setFieldsDict",
            """
defaultFieldValues (volScalarFieldValue T 0);
regions (boxToCell {
 box (0.025 0.0625 -1) (0.05 0.0875 1);
 fieldValues (volScalarFieldValue T 1e-6);
});
""",
        )
    (root / "run.sh").write_text(
        """#!/bin/bash
set -eu
for c in reference half-step fine-mesh; do
  cd "/work/$c"
  cp system/spin system/controlDict
  blockMesh > log.mesh
  setFields > log.initial
  icoFoam > log.flow
  test -s 0.5/phi
  cp 0/T 0.5/T
  cp system/tracer system/controlDict
  scalarTransportFoam > log.scalar
  postProcess -func writeCellCentres -time 0.5 > log.centres
  postProcess -func writeCellVolumes -time 0.5 > log.volumes
done
""",
        encoding="ascii",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    prepare(parser.parse_args().directory)
