# E4-F3: Fixed Four-Substep Integration and Reusable Buffers

## Problem Statement

The production GPU condensation path performs one explicit update and previously
provided no reusable operation-sidecar contract. Fixed four-substep integration
remains planned for this feature, but issue #1292 completed the prerequisite
stable-shape scratch-buffer and fail-before-mutation contract.

## Value Proposition

The delivered P1 API lets callers reuse fixed-shape fp64 work-transfer,
total-transfer, dynamic-viscosity, and mean-free-path sidecars for one existing
condensation update. It validates every supplied field atomically before
allocation, normalization, refresh, launch, clear, or mutation. Later phases
will promote exactly four equal substeps and per-substep physics refresh.

## User Stories

- As a simulation author, I want invalid reusable sidecars rejected without
  changing particle, gas, vapor-pressure, or caller-owned scratch state.
- As a GPU workflow author, I want caller-owned scratch buffers so repeated
  one-update calls preserve allocation and shape stability.
- As a maintainer, I want issue #1272 validation signals promoted with the
  implementation so later E4 physics cannot silently weaken them.
