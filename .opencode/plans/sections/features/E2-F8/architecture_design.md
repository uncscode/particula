# E2-F8 Architecture Design

## High-Level Design

E2-F8 is a boundary-clarification feature layered on top of existing CPU
dynamics strategy code. It does not add a new execution engine; instead it
aligns validation, tests, and docs around the distinction between container
schema support and strategy execution support.

```text
ParticleData / GasData schemas
  support n_boxes >= 1 at the data-container level
        |
        v
CPU condensation strategies
  unwrap data containers -> require n_boxes == 1 -> raise clear errors otherwise
        |
        v
CPU coagulation strategies
  unwrap ParticleData -> currently read/mutate box 0 only through legacy adapters
        |
        v
E2-F8 boundary clarification
  P1 tests document the baseline; later phases may tighten validation/errors and docs
```

## Data / API / Workflow Changes

- **Data Model:** No schema changes to `ParticleData` or `GasData`. Their
  multi-box shapes remain valid containers.
- **API Surface:** No new public API is expected. Existing CPU dynamics methods
  keep their current runtime behavior in P1; later phases may add clearer
  `ValueError` messages for unsupported multi-box inputs.
- **Workflow Hooks:** No ADW workflow changes. Implementation phases should run
  focused pytest suites for condensation/coagulation and standard linters.

## Strategy Boundary Decisions

- Condensation remains single-box only for CPU data-container paths. It should
  reject `ParticleData.n_boxes != 1` and `GasData.n_boxes != 1` with clear
  messages.
- Coagulation currently avoids all-box execution by reading or mutating box 0
  only for `ParticleData` strategy calls. P1 preserves and tests that baseline;
  a later phase may replace it with explicit single-box validation.

## Security & Compliance

No security-sensitive surfaces are introduced. The main robustness concern is
scientific correctness: ambiguous multi-box support can silently produce wrong
simulation results. P1 mitigates this by making the current boundary explicit in
tests and plan sections, with later phases able to add clearer errors or docs.
