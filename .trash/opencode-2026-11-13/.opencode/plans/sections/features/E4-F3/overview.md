# E4-F3: Fixed Four-Substep Integration and Reusable Buffers

## Problem Statement

The production GPU condensation path now has the reusable operation-sidecar
contract from issue #1292 and fixed four-substep integration from issue #1293.
Each successful call advances four equal unconditional intervals instead of one
full-step explicit update.

## Value Proposition

The delivered path lets callers reuse fixed-shape fp64 work-transfer,
total-transfer, dynamic-viscosity, and mean-free-path sidecars. It validates
every supplied field atomically before allocation, normalization, refresh,
launch, clear, or mutation; then executes four current-state updates. The
returned total contains clamped applied transfer across all four substeps,
while work retains the final raw proposal.

## User Stories

- As a simulation author, I want invalid reusable sidecars rejected without
  changing particle, gas, vapor-pressure, or caller-owned scratch state.
- As a GPU workflow author, I want caller-owned scratch buffers so repeated
  four-substep calls preserve allocation and shape stability.
- As a maintainer, I want issue #1272 validation signals promoted with the
  implementation so later E4 physics cannot silently weaken them.
