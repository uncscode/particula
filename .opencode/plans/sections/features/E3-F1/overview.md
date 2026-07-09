# Overview

## Problem Statement

`coagulation_step_gpu` currently reinitializes Warp RNG state on every call,
including when callers pass a reusable `rng_states` buffer. Repeated timesteps
with the same `rng_seed` therefore replay correlated random draws unless callers
manually increment the seed. This is surprising for a low-level GPU timestep API,
prevents natural seed-once usage, and is unsafe for graph-capture workflows where
re-seeding inside a captured step can freeze the RNG sequence.

## Value Proposition

This feature hardens the GPU coagulation RNG API in stages. The shipped first
phase locks the compatibility contract so callers can distinguish between
legacy omitted-`rng_states` behavior, caller-owned reusable buffers, and an
explicit reset path via `initialize_rng=True`. That removes ambiguity before
later phases broaden repeated-step semantics, benchmark updates, and GPU docs.

## User Stories

- As a simulation author, I want caller-provided `rng_states` to avoid implicit
  reset unless I explicitly request it, so repeated GPU timesteps keep a stable
  low-level ownership contract.
- As a benchmark maintainer, I want reusable `rng_states` buffers to persist and
  avoid implicit reset so compatibility tests can protect current behavior
  before any broader benchmark guidance changes.
- As a GPU workflow developer, I want initialization separated from repeated
  timestep execution so graph-captured loops do not re-seed inside the graph.

## Parent Epic Context

Parent epic: E3, GPU Kernel Correctness and Low-Level API Hardening. This is the
first child feature track (`E3-F1`) and has no upstream feature dependency.
Sibling features (`E3-F2` through `E3-F7`) should be able to build on the same
low-level API hardening pattern: validate before mutation, preserve backwards
compatibility, and document explicit GPU buffer ownership.
