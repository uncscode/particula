"""Run a nonempty CUDA test selection and retain strict qualification evidence.

Unlike ordinary portable pytest runs, qualification cannot succeed when CUDA
is unavailable or a selected test skips. Run from a host environment exposing
the GPU. This runner does not change device configuration or select Warp CPU.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (
    "particula/execution/tests/native_resident_test.py",
    "particula/execution/tests/native_checkpoint_test.py",
    "particula/execution/tests/native_full_loop_test.py",
    "particula/execution/tests/captured_full_loop_test.py",
    "particula/execution/tests/graph_capture_test.py",
    "particula/execution/tests/prepared_test.py",
    "particula/execution/tests/prepared_restart_capture_test.py",
    "particula/gpu/kernels/tests/coagulation_native_test.py",
    "particula/gpu/kernels/tests/brownian_parallel_test.py",
    "particula/gpu/kernels/tests/condensation_native_test.py",
    "particula/gpu/kernels/tests/condensation_parallel_test.py",
    "scripts/tests/gpu_studies_test.py",
)


def summarize_results(root: ET.Element) -> dict:
    """Require executed passing cases, retaining every skip/failure reason."""
    cases = list(root.iter("testcase"))
    problems = []
    passed = 0
    for case in cases:
        outcomes = [
            child
            for child in case
            if child.tag in ("skipped", "failure", "error")
        ]
        if not outcomes:
            passed += 1
        for child in outcomes:
            problems.append(
                {
                    "case": f"{case.get('classname')}.{case.get('name')}",
                    "outcome": child.tag,
                    "message": child.get("message", child.text or ""),
                }
            )
    return {
        "cases": len(cases),
        "passed": passed,
        "problems": problems,
        "qualified": bool(cases) and not problems,
    }


def qualify(paths: list[str], output: Path) -> int:
    """Execute CUDA cases and save JSON/XML with actual command provenance."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    junit = output.with_suffix(".xml")
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- trusted developer PATH
            cwd=ROOT,
            text=True,
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],  # noqa: S607 -- trusted developer PATH
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        "qualified": False,
    }
    started = time.perf_counter()
    try:
        import warp as wp

        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("CUDA qualification requires an available GPU")
        device = wp.get_device("cuda:0")
        report.update(
            warp=importlib.metadata.version("warp-lang"),
            device=str(device),
            gpu_name=device.name,
            gpu_arch=device.arch,
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            *paths,
            "-m",
            "cuda and not slow and not performance",
            "-q",
            "-Werror",
            "--no-cov",
            "-rs",
            f"--junitxml={junit}",
        ]
        report["command"] = command
        # An interrupted earlier run must not leave usable-looking XML.
        junit.unlink(missing_ok=True)
        result = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603 -- explicit local pytest arguments
        report["returncode"] = result.returncode
        if junit.exists():
            # Parse only the report just written by our own pytest subprocess.
            report.update(summarize_results(ET.parse(junit).getroot()))  # noqa: S314
        report["qualified"] = result.returncode == 0 and report["qualified"]
    except (ImportError, RuntimeError, OSError, ET.ParseError) as error:
        report["error"] = str(error)
        report["qualified"] = False
    finally:
        report["duration_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"CUDA qualification {'PASS' if report['qualified'] else 'FAIL'}")
    print(f"Evidence: {output}")
    return 0 if report["qualified"] else 1


def main() -> int:
    """Parse optional explicit test paths and evidence destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".artifacts/qualification/gpu.json",
    )
    args = parser.parse_args()
    return qualify(args.paths, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
