# Phase Details

- [x] **E7-F2-P1:** Map condensation capabilities and configuration semantics with unit tests
  - Issue: #1470 | Size: S | Status: Shipped
  - Delivered: Direct-module-only immutable condensation vocabulary,
    configuration validation, exact four-axis requirements mapping, and a
    36-entry CPU plus 8-entry declarative Warp-profile catalogue.
  - Files: `particula/execution/__init__.py`,
    `particula/tests/execution_test.py`
  - Tests: Mapping/catalogue counts, supported and rejected profiles,
    validation order, immutability/purity, non-composable requirements, and
    fresh-process optional-import isolation.
  - Boundary: No runtime availability or native-device handling, adapter
    selection, exports, or GPU API changes.

- [x] **E7-F2-P2:** Define condensation execution state and sidecar ownership with unit tests
  - Issue: #1471 | Size: S | Status: Shipped
  - Delivered: Migrated `particula.execution` to a package while preserving its
    exact ten-name selection `__all__`; added concrete-only frozen CPU and lazy
    Warp state carriers with identity retention, metadata-only validation, and
    writable-output ownership checks.
  - Files: `particula/execution/adapters/condensation.py`,
    `particula/execution/__init__.py`,
    `particula/execution/tests/condensation_adapter_test.py`
  - Tests: Import/export boundaries; CPU and Warp type/schema/shape/device
    validation; validation order; identity retention; alias/overlap and
    contiguity rejection; no execution, transfer, or synchronization; and
    accepted/rejected construction non-mutation.
  - Boundary: No profile selection, adapter registration or execution, kernel
    physics validation, conversion, allocation, transfer, synchronization, or
    top-level concrete exports.

- [x] **E7-F2-P3:** Implement backend-selected isothermal condensation adapter with unit tests
  - Issue: #1472 | Size: S | Status: Shipped
  - Delivered: Concrete-only selected isothermal CPU and Warp P3 carriers and
    adapters with exact preflight, one unchanged native call, and
    identity-preserving normalized results. Warp kernel resolution is lazy;
    neither path transfers, synchronizes, falls back, or recovers failures.
  - Files: `particula/execution/__init__.py`,
    `particula/execution/adapters/condensation.py`,
    `particula/execution/tests/condensation_adapter_test.py`
  - Tests: CPU/Warp dispatch, exact call arguments/counts, normalized-result
    identity, exception propagation, lazy Warp import, no transfer/sync/fallback,
    and public export boundaries.

- [x] **E7-F2-P4:** Add latent-heat support and explicit unsupported-mode errors with unit tests
  - Issue: #1473 | Size: S | Status: Shipped
  - Delivered: Selected Warp dispatch forwards caller-owned `latent_heat`,
    `energy_transfer`, and deferred `thermal_work` by identity to
    `condensation_step_gpu`; CPU remains isothermal. Capability-profile failure
    occurs before lazy kernel resolution, while direct-kernel thermal validation,
    execution, exceptions, and energy accounting remain authoritative.
  - Files: `particula/execution/adapters/condensation.py`,
    `particula/execution/tests/condensation_adapter_test.py`,
    `.opencode/guides/architecture/architecture_outline.md`
  - Tests: Sidecar and native-result identity; native energy/deferred-work
    validation propagation; omitted/zero heat behavior; energy accounting;
    unsupported-profile preflight; and no transfer, synchronization, fallback,
    or recovery.

- [x] **E7-F2-P5:** Prove CPU and Warp condensation parity, conservation, and transfer boundaries
  - Issue: #1474 | Size: S | Status: Shipped
  - Delivered: Added test-only native legacy-CPU and resident direct-Warp
    integration evidence. Warp uses a local independent NumPy fixed-four-
    substep P2 oracle rather than CPU numerical output; no public API or
    production implementation changed.
  - Files: `particula/execution/tests/condensation_integration_test.py`
  - Tests: CPU uptake, evaporation, zero gas, inactive, skip-partitioning, and
    exact zero-time cases; Warp CPU uptake, evaporation, disabled partitioning,
    zero gas, inactive slots, distinct two-box environment/gas inputs, latent
    heat/energy sidecars, per-case mass/gas tolerances, conservation, exact
    zero-time behavior, and resident-boundary spies. Optional CUDA runs uptake,
    two-box, and latent-heat rows and skips explicitly when unavailable.

- [ ] **E7-F2-P6:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document selected condensation usage, support matrix, ownership, errors, tolerances, limitations, and downstream scheduler handoff.
  - Files: `docs/Features/condensation_strategy_system.md`, data-container/backend-selection guides, focused example or API reference
  - Tests: Documentation regression, import snippets, `mkdocs build --strict`
