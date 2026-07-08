# Architecture Design

## High-Level Design

The architecture change is intentionally narrow: keep per-box RNG state in the
existing Warp `uint32` buffer, but stop host orchestration from reinitializing
caller-owned buffers on every timestep.

```text
Caller setup
  ├─ optional: allocate rng_states buffer shaped (n_boxes,)
  ├─ seed/init once according to explicit API contract
  └─ enter repeated timestep loop

coagulation_step_gpu(..., rng_states=rng_states)
  ├─ validate particles/environment/volume/buffers before mutation
  ├─ allocate+initialize RNG only when the API contract says this call owns init
  ├─ launch brownian_coagulation_kernel
  │    ├─ load rng_states[box_idx]
  │    ├─ consume randf/randi draws
  │    └─ store advanced rng_states[box_idx]
  └─ return particle/collision outputs while caller retains rng_states
```

## Data / API / Workflow Changes

- **Data Model:** No persistent schema change. RNG state remains a Warp array of
  shape `(n_boxes,)`, dtype `wp.uint32`, on the same device as particle buffers.
- **API Surface:** `coagulation_step_gpu` keeps existing positional parameters.
  If a new control is needed, add it as keyword-only after existing keyword-only
  parameters. The contract must distinguish these cases:
  - no `rng_states`: allocate and seed internal state for source-compatible
    single-call behavior;
  - provided `rng_states`: validate and preserve the caller-owned state across
    repeated calls unless explicit initialization is requested;
  - explicit seed-once setup: expose an initializer/reset helper or equivalent
    keyword-only initialization path so callers can initialize outside a repeated
    timestep loop without hidden host inspection of buffer contents;
  - caller-supplied `rng_states`: continue to bypass automatic initialization
    unless the caller explicitly invokes the initializer/reset path.
- **Workflow Hooks:** Tests and documentation become the enforcement mechanism.
  Benchmarks that currently increment `rng_seed` every step should either adopt
  seed-once setup or explicitly document why they still vary seeds.

## Validation and Mutation Order

Validation must continue to run before volume normalization, RNG setup, and
kernel launches. Invalid scalar/environment combinations, wrong device buffers,
or wrong shapes must not mutate `rng_states`, particles, collision buffers, or
launch downstream kernels.

## Security & Compliance

No new permissions, external services, serialization formats, or network access
are introduced. The main robustness requirement is deterministic ownership of
GPU buffers: no hidden CPU reads/synchronization and no implicit mutation of
caller-owned RNG state before validation succeeds.
