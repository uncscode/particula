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
  unwrap ParticleData -> require n_boxes == 1 -> raise clear errors otherwise
        |
        v
E2-F8 boundary clarification
  tests + validation/errors + docs explain unsupported full multi-box execution
```

## Data / API / Workflow Changes

- **Data Model:** No schema changes to `ParticleData` or `GasData`. Their
  multi-box shapes remain valid containers.
- **API Surface:** No new public API is expected. Existing CPU dynamics methods
  may raise clearer `ValueError` messages for unsupported multi-box inputs.
- **Workflow Hooks:** No ADW workflow changes. Implementation phases should run
  focused pytest suites for condensation/coagulation and standard linters.

## Strategy Boundary Decisions

- Condensation remains single-box only for CPU data-container paths. It should
  reject `ParticleData.n_boxes != 1` and `GasData.n_boxes != 1` with clear
  messages.
- Coagulation should avoid silent all-box expectations. The preferred design is
  explicit single-box validation for `ParticleData` strategy calls that would
  otherwise operate only on `particle.*[0]`. Do not keep transitional box-0
  behavior for unsupported multi-box inputs; raise explicit errors instead.

## Security & Compliance

No security-sensitive surfaces are introduced. The main robustness concern is
scientific correctness: ambiguous multi-box support can silently produce wrong
simulation results. Clear errors and docs mitigate this risk.
