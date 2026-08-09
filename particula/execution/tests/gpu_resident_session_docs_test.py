"""Regression coverage for the lazy resident-session documentation example."""

from __future__ import annotations

import builtins
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE = _ROOT / "docs" / "Examples" / "gpu_resident_session.py"
_DISABLED = [
    "Canonical path: docs/Examples/gpu_resident_session.py",
    "CPU fixture: not constructed because Warp is unavailable or disabled.",
    "Warp is unavailable or disabled; no resident session was created.",
]


@pytest.fixture
def example_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import a fresh example while proving import-time GPU safety."""
    blocked = {
        "warp",
        "particula.gpu",
        "particula.execution.gpu_session",
        "particula.execution.gpu_resources",
        "particula.execution.checkpoint",
    }
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name in blocked:
            pytest.fail(f"example imported {name} eagerly")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.syspath_prepend(str(_EXAMPLE.parent))
    sys.modules.pop("gpu_resident_session", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("gpu_resident_session")
    yield module
    sys.modules.pop("gpu_resident_session", None)


def test_forced_disable_skips_loader_and_fixture(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced disable is a deterministic no-work path."""
    monkeypatch.setenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", "1")
    monkeypatch.setattr(
        example_module, "_load_enabled_runtime", lambda: pytest.fail("loader")
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    result = example_module.run_example()

    assert result.output == _DISABLED
    assert result.session is result.registry is result.guard is None
    assert (
        result.checkpoint
        is result.restarted
        is result.terminal_checkpoint
        is None
    )


def test_missing_warp_skips_fixture(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a missing Warp import selects the no-work route."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    assert example_module.run_example().output == _DISABLED


@pytest.mark.parametrize(
    "error",
    [
        ImportError("Warp extension failed to load"),
        ModuleNotFoundError(name="warp_transitive_dependency"),
    ],
)
def test_broken_enabled_warp_import_propagates(
    example_module: Any, monkeypatch: pytest.MonkeyPatch, error: ImportError
) -> None:
    """Only a missing top-level Warp module is treated as disabled."""
    monkeypatch.delenv("PARTICULA_EXAMPLE_FORCE_NO_WARP", raising=False)
    monkeypatch.setattr(
        example_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    with pytest.raises(type(error)) as raised:
        example_module.run_example()

    assert raised.value is error


def test_forced_disabled_script_has_exact_stdout() -> None:
    """The standalone forced-disabled command exits successfully."""
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(_EXAMPLE)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )
    assert result.stdout == "\n".join(_DISABLED) + "\n"


def test_loader_orders_concrete_imports_without_gpu_package(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The enabled loader requests only the documented concrete seams."""
    modules = {
        "warp": object(),
        "particula.execution": SimpleNamespace(
            Backend=object(), Device=object()
        ),
        "particula.execution.gpu_session": SimpleNamespace(
            ResidentLifecycle=object(),
            setup_resident_session=object(),
            ResidentStepGuard=object(),
        ),
        "particula.execution.gpu_resources": SimpleNamespace(
            GPUResourceRegistry=object()
        ),
        "particula.execution.checkpoint": SimpleNamespace(
            restart_resident_session=object()
        ),
    }
    calls: list[str] = []

    def load(name: str) -> object:
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(example_module.importlib, "import_module", load)
    example_module._load_enabled_runtime()
    assert calls == list(modules)
    assert "particula.gpu" not in calls


@pytest.mark.parametrize(
    "error", [ImportError("resident"), RuntimeError("bad")]
)
def test_enabled_loader_errors_propagate_without_fixture_or_output(
    example_module: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Selected enabled failures never become disabled-path success output."""
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )
    with pytest.raises(type(error)) as raised:
        example_module.run_example()
    assert raised.value is error


def test_main_propagates_an_enabled_loader_error(
    example_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command entry point does not hide selected enabled-path failures."""
    error = RuntimeError("resident runtime failed")
    monkeypatch.setattr(example_module, "_warp_enabled", lambda: True)
    monkeypatch.setattr(
        example_module,
        "_load_enabled_runtime",
        lambda: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(
        example_module, "_build_cpu_state", lambda: pytest.fail("fixture")
    )

    with pytest.raises(RuntimeError) as raised:
        example_module.main()

    assert raised.value is error


def test_lifecycle_documentation_preserves_published_boundaries() -> None:
    """Resident lifecycle documentation retains its cross-reference links."""
    feature = (
        _ROOT / "docs" / "Features" / "data-containers-and-gpu-foundations.md"
    ).read_text()
    checkpoint = (
        _ROOT / "docs" / "Features" / "gpu_resident_checkpoints.md"
    ).read_text()
    roadmap = (
        _ROOT / "docs" / "Features" / "Roadmap" / "data-oriented-gpu.md"
    ).read_text()
    assert "[GPU resident checkpoints](gpu_resident_checkpoints.md)" in feature
    assert (
        "[GPU-resident deterministic timestep]"
        "(data-containers-and-gpu-foundations.md#gpu-resident-deterministic-timestep)"
        in checkpoint
    )
    assert (
        "[GPU resident checkpoints](../gpu_resident_checkpoints.md)" in roadmap
    )


def test_scheduler_documentation_preserves_published_boundaries() -> None:
    """Resident scheduler documentation retains its cross-reference links."""
    feature = (
        _ROOT / "docs" / "Features" / "data-containers-and-gpu-foundations.md"
    ).read_text()
    roadmap = (
        _ROOT / "docs" / "Features" / "Roadmap" / "data-oriented-gpu.md"
    ).read_text()
    assert "[GPU resident checkpoints](gpu_resident_checkpoints.md)" in feature
    assert (
        "[Data-Oriented Design and GPU Roadmap](Roadmap/data-oriented-gpu.md)"
        in feature
    )
    assert (
        "[GPU resident checkpoints](../gpu_resident_checkpoints.md)" in roadmap
    )


def test_communication_feature_documentation_preserves_direct_and_resident_contracts() -> (
    None
):
    """Feature guidance retains prescribed communication ownership boundaries."""
    feature = (
        _ROOT / "docs" / "Features" / "data-containers-and-gpu-foundations.md"
    ).read_text(encoding="utf-8")
    feature_text = " ".join(feature.split())

    for phrase in (
        "declared `-1` source/sink endpoints",
        "exactly one complete closed-map GAS or PARTICLES family",
        "`(B, S)` `amounts`",
        "amount = concentration * volume",
        "new_concentration = final_amount / new_volume",
        "no fused direct-gas `new_volume` argument",
        "communication first stages with old volumes",
        "invalidates saturation only",
        "required only when its matching open endpoint is enabled",
        "At the standalone direct-kernel boundaries, validated empty or disabled maps and unchanged final volumes are successful write-free no-ops.",
        "Resident composition has its own barrier validation and does not make this general no-op guarantee.",
        "`volume_evolution_step_gpu` is independently callable at its standalone direct-kernel boundary.",
        "Its use in resident composition is the optional scheduled barrier",
        "exact population match",
        "ascending pre-step free slot",
        "Capacity is deterministically gated",
        "Host/schema preflight rejects without primary or buffer mutation.",
        "rollback is not promised after a writer launches.",
        "Restart remains explicit and exact-device with fresh identities.",
        "CFD, adaptive/distributed/multi-GPU transport",
        "E7-F8 RNG policy, graph capture, performance claims, and autodiff.",
    ):
        assert phrase in feature_text
    assert feature_text.count("no fused direct-gas `new_volume` argument") == 1


def test_communication_roadmap_and_architecture_documentation_preserve_boundaries() -> (
    None
):
    """Roadmap and architecture guidance retain concrete communication scope."""
    roadmap = (
        _ROOT / "docs" / "Features" / "Roadmap" / "data-oriented-gpu.md"
    ).read_text(encoding="utf-8")
    guide = (
        _ROOT
        / ".opencode"
        / "guides"
        / "architecture"
        / "architecture_guide.md"
    ).read_text(encoding="utf-8")
    reference = (
        _ROOT / ".opencode" / "guides" / "architecture_reference.md"
    ).read_text(encoding="utf-8")
    roadmap_text = " ".join(roadmap.split())
    guide_text = " ".join(guide.split())
    reference_text = " ".join(reference.split())

    for phrase in (
        "E7-F7/T7 is shipped",
        "concrete-only, caller-owned, explicitly synchronized",
        "enabled open-source or open-sink edges additionally require",
        "Direct-boundary empty/disabled maps and unchanged final volumes are write-free no-ops",
        "resident barriers instead follow their own composition and validation rules.",
        "CFD, pressure/velocity solvers, adaptive meshes",
        "E7-F8 owns scheduled RNG policy while E7-F9 owns complete-loop publication.",
    ):
        if phrase not in roadmap_text:
            pytest.fail(f"roadmap is missing: {phrase}")
    for phrase in (
        "particula.gpu.kernels.communication.gas_communication_step_gpu",
        "amount = concentration * volume",
        "new_concentration = final_amount / new_volume",
        "no hidden transfer, synchronization, or CPU fallback",
        "each enabled open source/sink endpoint additionally requires",
        "Validated empty or disabled maps are write-free no-ops",
        "`volume_evolution_step_gpu` is independently callable",
    ):
        if phrase not in guide_text:
            pytest.fail(f"architecture guide is missing: {phrase}")
    for phrase in (
        "Normal steps validate only that identity and metadata",
        "communication with pre-update volumes before optional prescribed volume evolution",
        "The barriers invalidate saturation ratio only",
        "Standalone direct-kernel empty or disabled maps and unchanged final volumes are write-free no-ops",
        "resident barriers instead follow their own composition and validation rules.",
        "exact device; explicit restart creates fresh identities",
    ):
        if phrase not in reference_text:
            pytest.fail(f"architecture reference is missing: {phrase}")
    assert "no fused direct-gas `new_volume` argument" in guide_text


def _normalized_anchor(heading: str) -> str:
    """Return the MkDocs-style normalized anchor for a Markdown heading."""
    return re.sub(r"[^a-z0-9 -]", "", heading.lower()).replace(" ", "-")


def test_communication_documentation_links_and_anchors_resolve() -> None:
    """Edited communication-document links resolve to existing targets/anchors."""
    sources_and_links = {
        _ROOT
        / "docs"
        / "Features"
        / "data-containers-and-gpu-foundations.md": (
            "gpu_resident_checkpoints.md",
            "Roadmap/data-oriented-gpu.md#e6-roadmap-inventory",
        ),
        _ROOT / "docs" / "Features" / "Roadmap" / "data-oriented-gpu.md": (
            "../data-containers-and-gpu-foundations.md",
        ),
        _ROOT / ".opencode" / "guides" / "architecture_reference.md": (
            "architecture/decisions/ADR-018-resident-communication-integration.md",
        ),
        _ROOT
        / ".opencode"
        / "guides"
        / "architecture"
        / "architecture_outline.md": (
            "decisions/ADR-018-resident-communication-integration.md",
        ),
    }
    for source, links in sources_and_links.items():
        source_text = source.read_text(encoding="utf-8")
        for link in links:
            assert f"]({link})" in source_text, (
                f"{source}: missing source link {link}"
            )
            target_text, _, anchor = link.partition("#")
            target = source.parent / target_text
            assert target.is_file(), f"{source}: missing link target {link}"
            if anchor:
                headings = re.findall(
                    r"^#{1,6}\s+(.+?)\s*$",
                    target.read_text(encoding="utf-8"),
                    flags=re.MULTILINE,
                )
                assert anchor in {
                    _normalized_anchor(item) for item in headings
                }, f"{source}: missing anchor {link}"


def test_e7_f7_plan_state_records_only_completed_publication_evidence() -> None:
    """E7-F7 plan records describe the completed documentation publication."""
    plan = json.loads(
        (_ROOT / ".opencode" / "plans" / "features" / "E7-F7.json").read_text(
            encoding="utf-8"
        )
    )
    phase_details = (
        _ROOT
        / ".opencode"
        / "plans"
        / "sections"
        / "features"
        / "E7-F7"
        / "phase_details.md"
    ).read_text(encoding="utf-8")
    assert plan["status"] == "Shipped"
    assert plan["lifecycle"] == "completed"
    assert plan["completion_date"] == "2026-08-09"
    assert plan["last_updated"] == "2026-08-09"
    assert [phase["status"] for phase in plan["phases"]] == ["Shipped"] * 7
    assert [phase["issue_number"] for phase in plan["phases"]] == list(
        range(1507, 1514)
    )
    assert [phase["completion_date"] for phase in plan["phases"]] == [
        "2026-08-08",
        "2026-08-08",
        "2026-08-08",
        "2026-08-08",
        "2026-08-09",
        "2026-08-09",
        "2026-08-09",
    ]
    assert "Issue: #1513" in phase_details
    assert "TBD" not in phase_details
    for path in (
        "docs/Features/data-containers-and-gpu-foundations.md",
        "docs/Features/Roadmap/data-oriented-gpu.md",
        ".opencode/guides/architecture/architecture_guide.md",
        ".opencode/guides/architecture_reference.md",
        ".opencode/guides/architecture/architecture_outline.md",
        "particula/execution/tests/gpu_resident_session_docs_test.py",
    ):
        assert path in phase_details
    assert (
        "pytest particula/execution/tests/gpu_resident_session_docs_test.py -q -Werror"
        in phase_details
    )
    assert (
        "pytest particula/tests/execution_selection_docs_test.py -q -Werror"
        in phase_details
    )
    assert "mkdocs build --strict" in phase_details


@pytest.mark.warp
def test_real_warp_cpu_lifecycle_example() -> None:
    """The real Warp CPU route preserves the published identity lifecycle."""
    pytest.importorskip("warp")
    spec = importlib.util.spec_from_file_location(
        "resident_example_real", _EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    result = module.run_example()
    restarted_session, restarted_registry, restarted_guard = result.restarted
    assert result.session.lifecycle.value == "finalized"
    assert result.checkpoint.lifecycle.value == "active"
    assert result.terminal_checkpoint is result.session.finalize(
        result.registry, result.guard
    )
    result.guard.assert_step_closed()
    restarted_guard.assert_step_closed()
    assert restarted_session is not result.session
    assert restarted_registry is not result.registry
    assert restarted_guard is not result.guard
    assert restarted_session.particles is not result.session.particles
    assert restarted_session.gas is not result.session.gas
    assert restarted_session.environment is not result.session.environment
    first_view = restarted_registry.acquire_wall_loss()
    second_view = restarted_registry.acquire_wall_loss()
    assert first_view is second_view
    assert first_view.rng_states is second_view.rng_states
    assert all(
        phrase in result.output
        for phrase in (
            "Finalization terminalizes its source; returned checkpoint is ACTIVE "
            "and cached.",
            "Restart is explicit, same-device, and never automatic.",
            "Inspection is lossy; canonical checkpoint bytes are restart authority.",
            "Checkpoint schema version 1 compatibility is exact and fail-closed.",
            "Exclusions: no scheduling, transport, fallback, or physics orchestration.",
        )
    )
