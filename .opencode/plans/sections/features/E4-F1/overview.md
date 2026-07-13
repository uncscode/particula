# Overview

## Problem Statement

Before issue #1281, GPU condensation had no explicit, validated thermodynamic
configuration boundary. Callers could provide vapor-pressure state without a
device-local schema tying thermodynamic inputs to the ordered gas species.

## Value Proposition

Issue #1281 shipped E4-F1-P1: a caller-owned, fixed-shape GPU sidecar and
validator required at the `condensation_step_gpu()` boundary. It validates
species ordering, field metadata, device locality, supported mode codes, and
finite non-negative numeric inputs before later-step allocation or launch.
This phase deliberately does not evaluate vapor-pressure formulas, refresh
`gas.vapor_pressure`, or change GPU/CPU data schemas; those remain future work.

## User Stories

- As a simulation developer, I want a required, caller-owned thermodynamic
  sidecar so future thermodynamic behavior has an explicit GPU boundary.
- As a model author, I want explicit model modes and parameter validation so an
  unsupported thermodynamic strategy fails predictably.
- As a GPU user, I want invalid sidecars rejected before condensation allocates
  scratch state, accesses caller mass-transfer storage, or launches a kernel.
