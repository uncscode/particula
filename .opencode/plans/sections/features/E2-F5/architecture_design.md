# Architecture Design

## Design Principles

- Preserve source compatibility for scalar GPU API callers.
- Treat environment state as a first-class per-box input owned outside
  `GasData`.
- Validate the P1 contract before launching Warp kernels so later phases can add
  normalization without changing the published ambiguity rule.
- Keep conversion explicit; helpers may construct Warp arrays from scalars, but
  CPU-to-GPU environment container transfers should live in conversion utilities.

## Shipped P1 Compatibility Layer

P1 shipped the API shell, not the normalization layer. Both GPU entry points now
accept these contract forms:

1. Legacy scalar `temperature: float` and `pressure: float`.
2. Mixed scalar values plus `environment=...`, which raise an early
   `ValueError`.
3. `temperature=None`, `pressure=None`, and `environment=...`, which raise a
   phase-scoped early `ValueError` in P1.

The reserved `environment` parameter is keyword-only and documented as the
future `WarpEnvironmentData` handoff point with `(n_boxes,)` temperature and
pressure arrays.

## API Options

- Chosen in P1: keep existing scalar parameters and add an optional
  keyword-only `environment=None` that does not break positional callers.
- Deferred: helper-based normalization or dedicated environment wrappers.
- Avoided in P1: changing positional ordering or making environment mandatory in
  the first migration step.

## Kernel Feed Points

- Condensation: replace globally scalar temperature-derived inputs with per-box
  values indexed by `box_idx`. If dynamic viscosity and mean free path remain
  precomputed on CPU, precompute them as `(n_boxes,)`; otherwise compute them in
  device code from per-box temperature/pressure.
- Coagulation: change Brownian kernel inputs from scalar temperature/pressure to
  arrays and use `temperature[box_idx]` and `pressure[box_idx]`.

## Error Handling

- If scalar and environment values are both provided, the implementation should
  raise a clear error instead of applying precedence rules. That keeps the first
  migration path deterministic for callers and avoids silently mixing legacy
  scalar inputs with per-box environment state.
- If `environment` is provided with both scalar inputs omitted, P1 should raise
  a clear phase-scoped error instead of pretending explicit environment
  execution already exists.
- Shape, `n_boxes`, and device validation for real explicit-environment
  execution are deferred to P2+.
