# Overview

## Problem Statement

Before P3 shipped, `coagulation_step_gpu` reinitialized Warp RNG state on every
call, including when callers passed a reusable `rng_states` buffer. That prior
behavior replayed correlated random draws unless callers manually incremented
the seed. P3 corrected the runtime behavior: caller-provided `rng_states` now
persist unless `initialize_rng=True` is explicitly requested, while omitted
`rng_states` still use the convenience allocation-and-seed path. This remains
important for low-level timestep APIs and graph-capture workflows where hidden
re-seeding inside a captured step can freeze the RNG sequence.

## Value Proposition

This feature hardens the GPU coagulation RNG API in stages. The shipped work now
covers all four steps: P1 locked the compatibility contract so callers can
distinguish between legacy omitted-`rng_states` behavior, caller-owned reusable
buffers, and an explicit reset path via `initialize_rng=True`; P2 shipped
regressions that make the persisted caller-owned buffer contract explicit for
repeated valid calls and invalid follow-up failures; P3 aligned the runtime
control flow and coagulation benchmark path with that contract; and P4 shipped
the final docstring and roadmap/container-boundary guidance for seed-once
repeated-call usage, caller-owned persistent `rng_states`, and graph-capture
setup expectations. The final phase was documentation-only and did not change
runtime behavior.

## User Stories

- As a simulation author, I want caller-provided `rng_states` to avoid implicit
  reset unless I explicitly request it, so repeated GPU timesteps keep a stable
  low-level ownership contract.
- As a benchmark maintainer, I want reusable `rng_states` buffers to persist and
  avoid implicit reset so repeated benchmark steps can reuse one buffer without
  external seed drift.
- As a GPU workflow developer, I want initialization separated from repeated
  timestep execution so graph-captured loops do not re-seed inside the graph.

## Parent Epic Context

Parent epic: E3, GPU Kernel Correctness and Low-Level API Hardening. This is the
first child feature track (`E3-F1`) and has no upstream feature dependency.
Sibling features (`E3-F2` through `E3-F7`) should be able to build on the same
low-level API hardening pattern: validate before mutation, preserve backwards
compatibility, and document explicit GPU buffer ownership.
