# Overview

## Problem Statement

`GasData` and `WarpGasData` intentionally diverge, but the current conversion
helpers do not make that divergence explicit enough for users or implementers.
CPU `GasData` owns species names, molar mass, concentration, and boolean
partitioning. GPU `WarpGasData` drops names, stores partitioning as `int32`, and
adds a `(n_boxes, n_species)` vapor-pressure buffer required by GPU
condensation kernels. The current round trip can silently generate placeholder
names, silently default missing vapor pressure to zeros, and drop GPU
vapor-pressure data on return to CPU.

## Value Proposition

This feature makes gas CPU/GPU schema ownership and round-trip behavior
predictable, tested, and documented. In the delivered `E2-F4-P1` slice for
issue `#1197`, the implementation stayed intentionally narrow: it added focused
regression coverage in `particula/gpu/tests/conversion_test.py` to lock the
current contract without changing production semantics. Users migrating to the
data-oriented GPU path can now rely on test-backed expectations for which
metadata survives conversion, which fields must be provided explicitly, and
which losses are intentional transfer-boundary semantics rather than bugs.

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
