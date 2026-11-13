# Overview

## Problem Statement

`GasData` and `WarpGasData` intentionally diverge, so the feature needed to
publish the tested round-trip contract clearly for users and implementers. CPU
`GasData` owns species names, molar mass, concentration, and boolean
partitioning. GPU `WarpGasData` drops names, stores partitioning as `int32`,
and adds a `(n_boxes, n_species)` vapor-pressure buffer required by GPU
condensation kernels. `E2-F4-P4` in issue `#1200` completed the migration-doc
publication pass without changing runtime behavior.

## Value Proposition

This feature makes gas CPU/GPU schema ownership and round-trip behavior
predictable, tested, and documented. `E2-F4-P1` landed issue `#1197` as a
focused regression-test audit. `E2-F4-P2` landed issue `#1198` by making the
restore contract explicit in `particula/gpu/conversion.py`. `E2-F4-P3` landed
issue `#1199` by clarifying vapor-pressure transfer semantics in
`particula/gpu/conversion.py`, `particula/gpu/warp_types.py`, and
`particula/gpu/tests/conversion_test.py`. `E2-F4-P4` landed issue `#1200` by
updating the migration guide authority table, revising roadmap wording from
unresolved schema drift to an intentional tested contract, and doing a narrow
`GasData` docstring consistency pass only where wording still lagged the shipped
semantics. Users migrating to the data-oriented GPU path can now rely on
test-backed expectations that:
- caller-supplied ordered names are preferred on restore;
- omitted or `None` names restore as placeholder `species_0..n-1` labels;
- wrong-length or empty provided name lists fail with actual/expected count
  messaging; and
- restored `partitioning` accepts only binary `0/1` values before conversion
  back to CPU bool dtype;
- caller-supplied `vapor_pressure` must have shape `(n_boxes, n_species)`;
- omitted `vapor_pressure` becomes a zero-filled GPU buffer with that shape; and
- CPU restore intentionally drops GPU-only `vapor_pressure`, so callers must
  read or save it before `from_warp_gas_data()`.
- migration-facing docs now treat this as the authoritative published contract,
  not an unresolved boundary.

## Parent Epic Context

- Parent epic: `E2` for issue `#1172`.
- Feature: `E2-F4`.
- Dependency: `E2-F1`, which establishes the schema foundation and should be
  treated as the baseline authority for data-container ownership decisions.
- Sibling context: `E2-F2` defines environment container boundaries, and
  `E2-F3` separates gas/environment responsibility. This feature should avoid
  assigning temperature-dependent vapor-pressure ownership to `GasData` unless
  those sibling tracks explicitly require it.

## User Stories

- As a GPU-path user, I want the current name contract made explicit so I know
  that supplied names are restored, omitted names become placeholders, and
  `WarpGasData` itself is not authoritative name storage.
- As a kernel implementer, I want `partitioning` bool-to-int32 conversion rules
  documented and tested so that CPU and GPU schemas remain compatible.
- As a migration reviewer, I want vapor-pressure ownership documented so that
  GPU transient buffers are not confused with CPU `GasData` state.
