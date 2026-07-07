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
  unwrap ParticleData -> require n_boxes == 1 before helper reads or step mutation
        |
        v
E2-F8 boundary clarification
  P1 documented the baseline; P2 tightened runtime validation; P3 can align docs
```

## Data / API / Workflow Changes

- **Data Model:** No schema changes to `ParticleData` or `GasData`. Their
  multi-box shapes remain valid containers.
- **API Surface:** No new public API is expected. Condensation keeps its
  existing single-box-only runtime behavior, while CPU coagulation
  `ParticleData` paths now raise `ValueError` for unsupported multi-box inputs
  instead of falling back to box 0.
- **Workflow Hooks:** No ADW workflow changes. Implementation phases should run
  focused pytest suites for condensation/coagulation and standard linters.

## Strategy Boundary Decisions

- Condensation remains single-box only for CPU data-container paths. It should
  reject `ParticleData.n_boxes != 1` and `GasData.n_boxes != 1` with clear
  messages.
- Coagulation now rejects `ParticleData` strategy calls with `n_boxes != 1`
  before helper-backed reads or `step()` mutation, preserving supported
  single-box behavior while removing silent box-0 fallback.

## Security & Compliance

No security-sensitive surfaces are introduced. The main robustness concern is
scientific correctness: ambiguous multi-box support can silently produce wrong
simulation results. P1 documented the baseline and P2 reduced that risk by
moving CPU coagulation container paths to explicit single-box validation, with
later phases able to align broader docs around the tested contract.
