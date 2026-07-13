# E4-F3: Fixed Four-Substep Integration

## Problem Statement

The production GPU condensation path performs one explicit update and allocates
temporary environment-property buffers on every call. Issue #1272 selected a
fixed four-substep candidate because it improves stiffness behavior while
retaining static control flow, but that candidate remains test-local.

## Value Proposition

Promote exactly four equal condensation substeps into production, refresh state
and E4-F1 thermodynamic physics before every substep, and let callers reuse
fixed-shape fp64 scratch buffers. This provides deterministic, graph-friendly
integration without hidden host transfers or steady-state allocations.

## User Stories

- As a simulation author, I want stable fixed-four integration so stiff cases
  remain finite, deterministic, and nonnegative.
- As a GPU workflow author, I want caller-owned scratch buffers so repeated
  steps preserve allocation and shape stability.
- As a maintainer, I want issue #1272 validation signals promoted with the
  implementation so later E4 physics cannot silently weaken them.
