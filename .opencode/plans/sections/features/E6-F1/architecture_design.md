# Architecture Design

## High-Level Design

The shipped P1-P4 reference consists of pure NumPy equation helpers, the
concrete-module-only `dilute_aerosol()` container primitive, and a CPU strategy
plus runnable. `DilutionStrategy` is configured with one precomputed chamber
dilution coefficient and delegates every step to P2. `Dilution` delegates equal
internal substeps to that strategy. Callers can derive the coefficient from
volume and flow with the existing pure helper.

```text
volume [m³] + inlet flow [m³/s]
        -> validate -> alpha = Q / V [s^-1]
                              |
                              v
                 DilutionStrategy.rate/step
                     /                    \
ParticleRepresentation                    GasSpecies
number concentration [1/m³]         mass concentration [kg/m³]
                     \                    /
                      preserve all metadata
                              |
                              v
             Dilution.execute(aerosol, dt, sub_steps)
```

The canonical finite-step update is the exact solution
`c_new = c * exp(-alpha * dt)`. Both particle and gas paths use that same update,
and E6-F2 treats it as the parity oracle. No path may encode dilution by changing
particle mass.

### P1 Implementation Record

Issue #1389 implemented the pure numerical boundary in
`particula/dynamics/dilution.py`. All three helpers validate finite physical
domains, explicitly reject `None`, preflight operands with `np.broadcast_arrays`,
preserve scalar returns for all-scalar calls, and do not mutate input arrays.
`get_dilution_step()` evaluates the exact exponential under a local NumPy
overflow/underflow suppression so finite extreme decay returns zero without a
warning. It remains intentionally unexported from `particula.dynamics`; no
container, strategy, runnable, or GPU workflow was introduced.

### P2 Implementation Record

Issue #1390 added concrete-module-only `dilute_aerosol()` in
`particula/dynamics/dilution.py`. It first validates scalar `coefficient` and
`time_step`, reads particle physical concentration plus
`atmosphere.partitioning_species` and `atmosphere.gas_only_species`, and
calculates each candidate with `get_dilution_step()`. It verifies candidate
shape, finiteness, and nonnegativity, converts the particle candidate through
representation volume, and preflights that stored candidate before any write.
It then writes particle storage followed by the gas groups in declared order.
Snapshots permit best-effort rollback if a later assignment fails. The primitive
returns the same aerosol and remains unexported; strategy/runnable and public
API decisions remain later-phase work.

### P3 Implementation Record

Issue #1391 added `DilutionStrategy` after `dilute_aerosol()` in
`particula/dynamics/dilution.py`. Its coefficient is validated as one finite,
nonnegative scalar; `rate()` returns `get_dilution_rate()` for the physical
particle concentration unchanged; and `step()` is exactly a delegation to P2.
`Dilution(RunnableABC)` was added to
`particula/dynamics/particle_process.py`. Before division, strategy calls, or
mutation, `execute()` rejects invalid `sub_steps`, validates the total duration,
splits it equally, and calls `step()` once per slice without adopting a custom
strategy return value. It therefore returns the original aerosol identity.
Both symbols were unexported pending P4.

### P4 Implementation Record

Issue #1392 centralized concrete-path preflight in
`particula/dynamics/dilution.py`. Before a commit, it validates physical
particle and both gas-group sources, particle volume and backing storage, and
all decay candidates and their public/storage shapes; snapshots are captured
only after that validation. The existing ordered particle/partitioning-gas/
gas-only commit and rollback path remain the mutation boundary.

`DilutionStrategy.step()` receives the same concrete preflight. In
`particula/dynamics/particle_process.py`, `Dilution.execute()` performs it once
after scalar validation and before the first equal substep when its strategy is
a `DilutionStrategy`. Strategy-like custom objects are not inspected and retain
their prior generic delegation contract. `particula/dynamics/__init__.py` now
exports `DilutionStrategy` and `Dilution`, but not `get_dilution_step()` or
`dilute_aerosol()`.

## Data / API / Workflow Changes

- **Data model:** No schema or ownership changes. Only particle and gas
  concentration storage mutates. Particle mass, charge, density, distribution,
  volume and atmospheric temperature/pressure retain identity and value.
- **Shipped API surface (P1/P2):** Preserve
  `get_volume_dilution_coefficient()` and `get_dilution_rate()`; keep
  `get_dilution_step()` and `dilute_aerosol()` concrete-module-only and
  unexported. P2 accepts scalar coefficient and time step only.
- **Shipped container workflow (P2):** Preflight all particle and gas
  candidates and converted particle storage before writes, then commit particle
  storage followed by both gas groups. On an unexpected later write failure,
  restore already-written snapshots. This avoids a half-mutated aerosol state.
- **Shipped P3/P4 API:** `DilutionStrategy` and `Dilution` are public through
  `particula.dynamics`; `get_dilution_step()` and `dilute_aerosol()` remain
  concrete-module-only. The strategy accepts the coefficient in `s^-1`, and
  volume/flow derivation remains in the pure helper.
- **Shipped P3 substeps:** `Dilution.execute()` validates `sub_steps` as a
  positive non-boolean Python/NumPy integer, validates total time before
  delegation, uses `time_step / sub_steps`, and executes exactly that many
  strategy steps.
- **Shipped P4 validation:** Concrete malformed state fails before all writes,
  including zero-duration/zero-coefficient cases. The supported runnable
  preflights before its first substep; custom strategies retain their own
  validation and atomicity.
- **Downstream:** E6-F2 consumes the formula, validation ordering, scalar
  semantics, and deterministic fixtures. E6-F9 consumes the runnable in an
  integrated process example.

## Security & Compliance

There are no permissions, network, secret, or persistence changes. Robustness
requires rejecting malformed, negative, or nonfinite physical inputs before
mutation. Scientific documentation must state units and cite the dilution
equation. Tests must demonstrate that invalid calls cannot leave particle and
gas state partially updated.
