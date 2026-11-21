"""Regenerate the MAM4xx host reference from the Fast et al. pinned source."""

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

HEADER_SHA256 = (
    "2de7a14a09cc6320e3a42dc9c96fff6f9b85d3d625b263bf303f5939700bc4c4"
)


def generate(header, output):
    """Compile and execute the pinned MAM4xx one-subarea driver."""
    raw = header.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HEADER_SHA256:
        raise ValueError(
            "Use MAM4xx revision 0f51633242168c90018f6e296d02776f86dc9be6"
        )
    cpp = raw.decode()
    start = cpp.index("namespace coagulation {")
    stop = cpp.rindex(
        "KOKKOS_INLINE_FUNCTION",
        start,
        cpp.index("void coagulation_rates_1box("),
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "modal_core.inc").write_text(cpp[start:stop] + "}\n")
        binary = root / "reference"
        source = Path(__file__).with_name("reference_driver.cpp")
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("A C++17 compiler is required")
        subprocess.run(  # noqa: S603 - fixed driver and hash-checked header
            [
                compiler,
                "-O2",
                "-std=c++17",
                "-I",
                directory,
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
        )
        data = subprocess.check_output([str(binary)])  # noqa: S603
    output.write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("header", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.header, args.output)
