"""Hardware-free tests for the closed profiling workload child process."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from particula.gpu.tests import profiling_support as support
from particula.gpu.tests import profiling_workload_runner as runner


def test_invalid_arguments_reject_before_cuda_import(capsys) -> None:
    """Test malformed argv returns before importing the CUDA support module."""
    module = "particula.execution.tests.resident_benchmark_cuda_support"
    previous = sys.modules.pop(module, None)
    try:
        assert runner.run(("--workload", "medium")) == 2
        assert module not in sys.modules
        assert "Invalid profiling worker arguments" in capsys.readouterr().err
    finally:
        if previous is not None:
            sys.modules[module] = previous


def test_worker_argument_shape_is_closed() -> None:
    """Test the worker accepts only the fixed Nsight invocation."""
    assert runner._arguments_are_valid(runner.EXPECTED_ARGUMENTS)
    assert not runner._arguments_are_valid((*runner.EXPECTED_ARGUMENTS, "x"))


def test_unavailable_cuda_capture_returns_bounded_status(
    monkeypatch, capsys
) -> None:
    """Test an unavailable capture binding reports no substitute execution."""
    module = ModuleType("resident_benchmark_cuda_support")
    module.ResidentBenchmarkUnavailableError = RuntimeError
    module.cuda_capture_availability = lambda: SimpleNamespace(
        available=False,
        reason="native capture unavailable",
    )
    module.qualified_cuda_resident_benchmark = lambda **_: None
    monkeypatch.setitem(
        sys.modules,
        "particula.execution.tests.resident_benchmark_cuda_support",
        module,
    )

    assert runner.run(runner.EXPECTED_ARGUMENTS) == 3
    assert capsys.readouterr().out == (
        "PROFILING_WORKLOAD_UNAVAILABLE: native capture unavailable\n"
    )


def test_available_worker_executes_closed_reset_warmup_and_sample(
    monkeypatch,
) -> None:
    """Test the available worker uses the frozen small captured-replay setup."""
    calls: list[str] = []
    captured: dict[str, object] = {}

    class Binding:
        """Record the closed binding lifecycle without CUDA execution."""

        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("exit")
            return False

        def validate_identities(self) -> None:
            calls.append("validate")

        def reset(self) -> None:
            calls.append("reset")

        def replay(self) -> None:
            calls.append("replay")

        def synchronize(self) -> None:
            calls.append("synchronize")

    binding = Binding()
    module = ModuleType("resident_benchmark_cuda_support")
    module.ResidentBenchmarkUnavailableError = RuntimeError
    module.cuda_capture_availability = lambda: SimpleNamespace(available=True)

    def qualified_cuda_resident_benchmark(**kwargs):
        captured.update(kwargs)
        return binding

    module.qualified_cuda_resident_benchmark = qualified_cuda_resident_benchmark
    monkeypatch.setitem(
        sys.modules,
        "particula.execution.tests.resident_benchmark_cuda_support",
        module,
    )

    assert runner.run(runner.EXPECTED_ARGUMENTS) == 0
    assert captured == {
        "duration": 0.5,
        "n_boxes": 1,
        "n_particles": 16,
        "n_species": 2,
        "root_seed": 1582,
        "case_id": support.build_default_profiling_workload_matrix()[
            0
        ].workload_id,
        "availability": captured["availability"],
    }
    assert calls == [
        "enter",
        "validate",
        "reset",
        "replay",
        "synchronize",
        "replay",
        "synchronize",
        "replay",
        "synchronize",
        "exit",
    ]
