# Overview

## Problem Statement

`GasData` and `WarpGasData` intentionally diverge, so the feature must make the
round-trip contract explicit for users and implementers. CPU `GasData` owns
species names, molar mass, concentration, and boolean partitioning. GPU
`WarpGasData` drops names, stores partitioning as `int32`, and adds a
`(n_boxes, n_species)` vapor-pressure buffer required by GPU condensation
kernels. After `E2-F4-P1` and `E2-F4-P2`, the name and partitioning restore
paths are now explicit and test-backed, while vapor-pressure ownership and
broader migration documentation remain for later phases.

## Value Proposition

This feature makes gas CPU/GPU schema ownership and round-trip behavior
predictable, tested, and documented. `E2-F4-P1` landed issue `#1197` as a
focused regression-test audit. `E2-F4-P2` landed issue `#1198` by making the
restore contract explicit in `particula/gpu/conversion.py` and expanding the
focused tests in `particula/gpu/tests/conversion_test.py`. Users migrating to
the data-oriented GPU path can now rely on test-backed expectations that:
- caller-supplied ordered names are preferred on restore;
- omitted or `None` names restore as placeholder `species_0..n-1` labels;
- wrong-length or empty provided name lists fail with actual/expected count
  messaging; and
- restored `partitioning` accepts only binary `0/1` values before conversion
  back to CPU bool dtype.

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
