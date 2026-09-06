"""Closed native-CUDA worker used only by Nsight collection subprocesses."""

from __future__ import annotations

import sys
from typing import Sequence

UNAVAILABLE_PREFIX = "PROFILING_WORKLOAD_UNAVAILABLE: "
EXPECTED_ARGUMENTS = (
    "--workload",
    "small",
    "--mode",
    "captured-replay",
)


def _arguments_are_valid(arguments: Sequence[str]) -> bool:
    """Return whether arguments match the one closed workload invocation."""
    return tuple(arguments) == EXPECTED_ARGUMENTS


def run(arguments: Sequence[str]) -> int:
    """Run the closed worker or return a bounded unavailable status.

    CUDA imports deliberately occur only after strict argument validation.
    """
    if not _arguments_are_valid(arguments):
        print("Invalid profiling worker arguments.", file=sys.stderr)
        return 2
    from particula.execution.tests.resident_benchmark_cuda_support import (
        ResidentBenchmarkUnavailableError,
        cuda_capture_availability,
        qualified_cuda_resident_benchmark,
    )
    from particula.gpu.tests.profiling_support import (
        build_default_profiling_workload_matrix,
    )

    availability = cuda_capture_availability()
    if not availability.available:
        print(f"{UNAVAILABLE_PREFIX}{availability.reason}")
        return 3
    workload = build_default_profiling_workload_matrix()[0]
    try:
        with qualified_cuda_resident_benchmark(
            duration=0.5,
            n_boxes=1,
            n_particles=16,
            n_species=2,
            root_seed=1582,
            case_id=workload.workload_id,
            availability=availability,
        ) as binding:
            binding.validate_identities()
            binding.reset()
            for _ in range(2):
                binding.replay()
                binding.synchronize()
            binding.replay()
            binding.synchronize()
    except ResidentBenchmarkUnavailableError as error:
        print(f"{UNAVAILABLE_PREFIX}{error}")
        return 3
    return 0


def main() -> None:
    """Exit with the closed worker result."""
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
