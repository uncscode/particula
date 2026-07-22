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
one per-box operation. Source demand transforms as `E_new=s*E_old` before final
slot packaging, so source and existing intensive concentrations are preserved.
The planner chooses the largest feasible finite `0<s<=1` that satisfies the
caller-configured minimum scale or volume. Pre-scale demand, represented demand,
and scale remain explicit diagnostics; no demand is truncated within the scaled
domain. Density, per-particle composition, and charge values are not scaling
knobs.

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
