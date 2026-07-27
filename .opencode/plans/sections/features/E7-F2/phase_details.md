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

- [ ] **E7-F2-P3:** Implement backend-selected isothermal condensation adapter with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Route explicit CPU and Warp requests to existing implementations while preserving exact arguments, fixed four-substep GPU behavior, and result semantics.
  - Files: `particula/execution/__init__.py`,
    `particula/execution/adapters/condensation.py`,
    `particula/execution/tests/condensation_adapter_test.py`
  - Tests: Dispatch, call arguments/counts, in-place particle/gas mutation, transfer result identity, no fallback or conversion

- [ ] **E7-F2-P4:** Add latent-heat support and explicit unsupported-mode errors with unit tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Map latent-heat and energy sidecars and reject staggered or unsupported BAT configurations before mutation.
  - Files: `particula/execution/adapters/condensation.py`,
    `particula/execution/tests/condensation_adapter_test.py`
  - Tests: Isothermal zero/omitted latent heat, latent energy accounting, output ownership, unsupported modes, failure boundaries

- [ ] **E7-F2-P5:** Prove CPU and Warp condensation parity, conservation, and transfer boundaries
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Record bounded parity and conservation evidence for selected one-box and multi-box fixtures on Warp CPU, with optional CUDA rows.
  - Files: `particula/execution/tests/condensation_adapter_test.py`,
    `particula/execution/tests/condensation_integration_test.py`
  - Tests: Uptake, evaporation, disabled partitioning, latent heat, zero gas, inactive slots, conservation, explicit tolerances, no intermediate transfer

- [ ] **E7-F2-P6:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Document selected condensation usage, support matrix, ownership, errors, tolerances, limitations, and downstream scheduler handoff.
  - Files: `docs/Features/condensation_strategy_system.md`, data-container/backend-selection guides, focused example or API reference
  - Tests: Documentation regression, import snippets, `mkdocs build --strict`
