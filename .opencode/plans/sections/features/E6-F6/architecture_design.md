# Architecture Design

## High-Level Design

Exhaustion handling is a two-stage transaction. The read-only planner consumes
E6-F5 diagnostics, source demand, configuration, and caller-owned scratch. It
computes a complete feasible policy plan and expected conservation totals.
Only a valid complete plan reaches commit.

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

For box volume `V`, slot weight `w_j`, species mass `m_j,s`, and charge `q_j`,
the independent oracle records represented number `V*sum(w_j)`, represented
species mass `V*sum(w_j*m_j,s)`, and represented charge `V*sum(w_j*q_j)`.
Commit must equal the pre-state plus the explicitly admitted source record at
recorded float64 tolerances; no residual demand is discarded. Resampling
preserves these required moments and uses deterministic ordering/tie breaks.
Distribution shape is assessed using predeclared radius/composition moments and
bounded error rather than an unsupported claim of sample identity.

Representative-volume scaling updates `volume` and all affected active weights
reciprocally as one per-box operation. P1 must freeze the allowed scale range,
rounding, and source-demand transformation before implementation. Density,
per-particle composition, and charge values are not scaling knobs.

## Data / API / Workflow Changes

- **Data Model:** No `ParticleData` or `WarpParticleData` field is added.
  Configuration and fixed-shape plan/diagnostic sidecars remain caller-owned.
- **API Surface:** Add CPU helpers under `particula.particles.exhaustion` and
  direct Warp entry points under `particula.gpu.kernels.exhaustion`. The public
  result reports requested/admitted counts, slots released, policy code, and
  per-box scale factor without host fallback.
- **Defaults:** Resampling `True`; representative-volume scaling `False`.
  Controls are independent. Both `False` is legal only when capacity is already
  sufficient; an exhausted box raises before mutation.
- **Workflow Hooks:** E6-F5 supplies capacity and activation. E6-F7/E6-F8 supply
  finalized source records and consume the policy plan. E6-F9 validates the
  integrated direct sequence.
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
