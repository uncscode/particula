# Architecture Design

## High-Level Design

Exhaustion handling is a two-stage transaction. The read-only planner consumes
E6-F5 diagnostics, source demand, configuration, and caller-owned scratch. It
computes a complete feasible policy plan and expected conservation totals.
Only a valid complete plan reaches commit.

### Delivered P1 boundary

Issue #1422 implements only the first, concrete CPU planning boundary in
`particula/particles/exhaustion.py`. It consumes already-discovered fixed-shape
capacity sidecars (`requested_count`, `free_count`, releasable count, and
ascending free-index prefixes), validates all boxes before resolution, and
returns frozen tuple-backed records. Capacity-sufficient boxes expose the exact
activation prefix; exhausted boxes select deferred resampling before deferred
scaling when enabled. Release selection and scale feasibility remain absent:
release tuples are empty and `nan` is the no-scale sentinel. No caller state is
accepted or written, so P1 has neither a commit nor rollback path.

P1 also implements the float64 weighted number, species-mass, and charge
inventory oracle as tuple-backed results. Its documented accounting boundary is
`pre_state + source` without scaling or `scale * pre_state + source` with
scaling; it makes no exact radius-cubed or other moment-preservation claim.

### Delivered P2 CPU resampling boundary

Issue #1423 adds private frozen P2 remap records plus `plan_resampling` and
`apply_resampling` in `particula/particles/exhaustion.py`. Planning validates
the complete fixed `ParticleData` schema and P1 records once, caches active
state, and retains no caller-array or P1-record views. For each eligible box,
it retains the first `K=A-R` active slots, releases the last `R`, stable-sorts
the active sources by radius, mass fractions, charge, and original slot, and
uses a monotonic two-cursor cumulative-weight interval sweep to calculate
float64 equal-weight replacements. The plan independently checks represented
number, species mass, signed charge, weighted radius cubed, mean radius,
surface, and Riemer diversity/mixing bounds.

`apply_resampling` only validates and commits a detached plan; it performs no
policy resolution or remap selection. It validates every box and confirms the
plan still exactly covers current active slots before any assignment, then
bulk-writes retained replacements and literal float64 zeroes to released slots.
P2 is CPU-only and leaves scaling, discovery/activation, package exports, and
Warp behavior deferred.

### Delivered P4 representative-volume scaling boundaries

Issue #1425 adds separate direct scaling commits rather than extending P1
policy resolution. The CPU `apply_representative_volume_scaling` and Warp
`representative_volume_scaling_step_gpu` take caller-owned per-box provisional
source demand, scaling flag, requested/minimum factors, minimum volume, and
resolved-scale output. Every factor and minimum-volume bound is validated for
every row. A row is selected only when its flag is set and its provisional
demand is positive; selected rows must also satisfy the scaled-volume bound.
After successful preflight, selected rows apply exactly `V *= s`,
`concentration *= s`, and `provisional_source_demand *= s`; `resolved_scale`
is set to `s` for selected rows and `1.0` otherwise. Mass, charge, density,
unselected rows, and configuration sidecars are protected.

The CPU implementation uses a vectorized selected-row commit. The Warp
boundary uses read-only active-device validation and a bounded status gate,
writes diagnostics once, and skips its scaling kernel if no row is selected.
Both return caller objects by identity and reject invalid calls without writes.
The helpers are concrete-module-only and neither consumes `ExhaustionPlan` nor
performs policy selection, resampling, activation, resizing, transfer, or
runnable orchestration.

### Delivered P3 direct Warp resampling boundary

Issue #1424 adds `resampling_step_gpu` in
`particula.gpu.kernels.exhaustion`. The direct boundary consumes explicit
already-resolved per-box release counts rather than P1 policy records and uses
concrete-only `ResamplingBuffers` for all particle-scale plan, diagnostic, and
sort storage. Read-only preflight validates particle fields, counts, bounds,
same-device buffer schemas, and buffer nonaliasing without writing caller
storage. For nonzero demand, active-device staged bitonic sorting uses the CPU
source key (radius, mass fractions, charge, original active index), followed by
an interval sweep that writes remap output into caller buffers.

Planning records per-box diagnostic status, performs an aggregate status gate,
and launches one all-box commit only if every box succeeds. The commit writes
replacement rows to retained original slots and clears released slots. Failed
planning may leave attempted plan data but does not mutate particles; rollback
after a launched commit is not promised. The entry point is exported from
`particula.gpu.kernels`, while `ResamplingBuffers` and implementation details
remain concrete-module-only.

```text
particle state + E6-F5 capacity + fixed-shape demand + policy config
                              |
                    read-only validation
                              |
                enough slots? -- yes --> activation plan
                       |
                       no
                       v
             resampling enabled? (default yes)
                       | plan deterministic releases
                       v
              enough capacity after plan? -- yes --> commit plan
                       |
                       no
                       v
       representative-volume scaling enabled? (default no)
                       | plan bounded volume/weight transform
                       v
                all demand representable?
              no -> error, no writes | yes -> commit once
```

Resampling has precedence whenever enabled; scaling cannot replace a feasible
resampling result. Scaling is considered only after the planned resample remains
insufficient. Neither branch may truncate demand. Planning must include all
boxes so one invalid box prevents writes to every caller-owned input/output.

## Conservation and Distribution Contract

For box volume `V`, raw represented count/weight `w_j`, species mass `m_j,s`,
and charge `q_j`, the independent oracle records represented number `sum(w_j)`,
represented species mass `sum(w_j*m_j,s)`, and represented charge
`sum(w_j*q_j)`. Their intensive values divide those totals by `V`. Without
scaling, commit equals pre-state plus represented source. With scale `s`, commit
equals `s*pre_state + represented_source` at recorded float64 tolerances.
Resampling preserves these required moments and uses deterministic ordering/tie
breaks. Distribution shape is assessed using the formulas and configured bounds
frozen in `open_questions.md`, not an unsupported claim of sample identity.

Representative-volume scaling applies `V_new=s*V_old` and `w_new=s*w_old` as
one per-box operation. Source demand transforms as `E_new=s*E_old`, so source
and existing intensive concentrations are preserved. P4 validates caller-supplied
finite `0 < minimum_scale <= requested_scale <= 1` and minimum-volume inputs;
it does not choose a scale or package source records. Density, per-particle
composition, and charge values are not scaling knobs.

## Data / API / Workflow Changes

- **Data Model:** No `ParticleData` or `WarpParticleData` field is added.
  Configuration and fixed-shape plan/diagnostic sidecars remain caller-owned.
- **API Surface:** P4 supplies concrete-only CPU and direct Warp scaling helpers
  in their exhaustion modules; neither is re-exported. They return supplied
  particle/demand/diagnostic objects without host fallback.
- **Defaults:** Resampling `True`; representative-volume scaling `False`.
  Controls are independent. Both `False` is legal only when capacity is already
  sufficient; an exhausted box raises before mutation.
- **Workflow Hooks:** E6-F5 supplies capacity and activation. E6-F7/E6-F8 supply
  provisional gas-admitted demand, consume the policy scale, then finalize
  represented source records. E6-F9 validates the integrated direct sequence.
- **Mutation Boundary:** Commit may mutate selected mass, concentration/weight,
  charge, and (only for scaling) per-box volume. Shapes, dtypes, devices,
  container objects, density, unselected boxes/slots, requests, and sidecar
  identities remain stable.

## Security & Compliance

There are no network or permission changes. Scientific safety requires finite,
nonnegative physical inputs, positive finite volume/scale, checked integer
counts, deterministic tie breaks, explicit tolerances, and failure-before-write
tests. Documentation must not imply dynamic capacity, hidden transfer, exact
stochastic parity, graph capture, or performance evidence.
