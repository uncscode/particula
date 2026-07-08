# Overview

## Problem Statement

`coagulation_step_gpu` currently reinitializes Warp RNG state on every call,
including when callers pass a reusable `rng_states` buffer. Repeated timesteps
with the same `rng_seed` therefore replay correlated random draws unless callers
manually increment the seed. This is surprising for a low-level GPU timestep API,
prevents natural seed-once usage, and is unsafe for graph-capture workflows where
re-seeding inside a captured step can freeze the RNG sequence.

## Value Proposition

This feature makes GPU coagulation RNG behavior match caller expectations:
callers can seed once, retain `rng_states`, and let repeated
`coagulation_step_gpu` calls advance per-box RNG state without host-managed seed
increments. It improves stochastic correctness, simplifies benchmark and
simulation loops, and prepares the coagulation kernel path for graph-capture
friendly execution.

## User Stories

- As a simulation author, I want to initialize coagulation RNG state once so
  repeated GPU timesteps produce uncorrelated stochastic draws without manual
  seed bookkeeping.
- As a benchmark maintainer, I want reusable `rng_states` buffers to persist and
  advance so timing loops exercise realistic timestep behavior.
- As a GPU workflow developer, I want initialization separated from repeated
  timestep execution so graph-captured loops do not re-seed inside the graph.

## Parent Epic Context

Parent epic: E3, GPU Kernel Correctness and Low-Level API Hardening. This is the
first child feature track (`E3-F1`) and has no upstream feature dependency.
Sibling features (`E3-F2` through `E3-F7`) should be able to build on the same
low-level API hardening pattern: validate before mutation, preserve backwards
compatibility, and document explicit GPU buffer ownership.
