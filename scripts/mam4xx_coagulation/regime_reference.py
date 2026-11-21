"""Reproduce the archived two-fine-mode atmospheric reference cases."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.mam4xx_coagulation.build_reference import HEADER_SHA256


def generate(header, output):
    """Compile the pinned unaltered microphysics with the atmospheric driver."""
    raw = header.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HEADER_SHA256:
        raise ValueError("Unexpected MAM4xx header")
    cpp = raw.decode()
    start = cpp.index("namespace coagulation {")
    end = cpp.rindex(
        "KOKKOS_INLINE_FUNCTION",
        start,
        cpp.index("void coagulation_rates_1box("),
    )
    here = Path(__file__).parent
    host = (here / "reference_driver.cpp").read_text().split("int main()")[0]
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        (tmp / "modal_core.inc").write_text(cpp[start:end] + "}\n")
        (tmp / "regime_core.inc").write_text(host)
        exe = tmp / "regime"
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("A C++17 compiler is required")
        subprocess.run(  # noqa: S603 - fixed compiler arguments and verified header
            [
                compiler,
                "-O2",
                "-std=c++17",
                "-I",
                folder,
                str(here / "regime_driver.cpp"),
                "-o",
                str(exe),
            ],
            check=True,
        )
        output.write_bytes(subprocess.check_output([str(exe)]))  # noqa: S603


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
