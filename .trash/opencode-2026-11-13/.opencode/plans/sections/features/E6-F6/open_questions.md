# Open Questions

All E6-F6 planning questions were resolved on 2026-07-21 from particle-resolved
raw-count semantics and the developer-approved bounded compatibility policy.

- [x] Which deterministic resampling algorithm and tie break are used?
  - Decision: use stable equal-weight conservative remapping into the required
    retained active count. Stable-sort by radius, lexicographic species mass
    fractions, charge, and original slot index; partition cumulative represented
    weight into equal strata; write retained records in ascending slot order and
    clear released records. This is resampling, not physical coagulation.
- [x] Which moments and error thresholds form the acceptance contract?
  - Decision: represented number, per-species mass, charge, and weighted
    `radius^3` are conserved at `rtol=1e-12`, `atol=1e-30`. Mean radius
    `sum(w*r)/sum(w)` and surface moment `sum(w*r^2)/sum(w)` use relative error.
    Mixing state is `get_mixing_state_index(w[:, None] * mass)` and uses absolute
    change. Their limits are required policy inputs. Canonical tests use 2%, 2%,
    and 0.02 without making them universal defaults; mixed-scale fixtures assess
    small-particle strata independently. When bulk diversity is one (including
    one-species state), mixing state is degenerate: skip that bound and require
    both pre/post states to retain exact species fractions.
- [x] Which representative-volume scale and indivisible-demand rules apply?
  - Decision: stored particle-resolved concentration is represented raw
    count/weight. Scaling uses `V_new=s*V_old`, `w_new=s*w_old`, and
    `E_new=s*E_old`, with finite `0<s<=1`, preserving existing and source
    intensive concentrations. Choose the largest feasible `s`; scaling is
    disabled by default and requires a caller-configured minimum scale or
    volume. Scaled demand is divided equally among `ceil(E_new/W_max)` source
    slots. `gas_admitted_event_count` records pre-scale demand,
    `represented_event_count` records `E_new`, and
    `representation_reduction_event_count` records their difference. No demand
    is truncated within the scaled representative domain.
  - P4 implementation note (#1425): the direct helpers do not select the
    largest feasible scale or package indivisible source demand. They validate
    caller-provided `requested_scale`, `minimum_scale`, and `minimum_volume`,
    then scale only flagged rows with positive provisional demand; P5+ owns
    policy selection and packaging integration.
- [x] Which diagnostics must E6-F7/E6-F8 retain?
  - Decision: fixed caller-owned per-box active/free, requested/released/
    activated counts, policy code, scale factor, and number/charge/`radius^3`/
    per-species-mass residuals. Nucleation-specific event and gas fields remain
    on its own sidecar.
- [x] Which fixed exhaustion work fields does E6-F8 consume?
  - Decision: P3 provides concrete-only `ResamplingBuffers` to
    `resampling_step_gpu`; it contains caller-owned same-device plan outputs,
    diagnostic outputs, planning status, and sort/interval-sweep scratch. It is
    not exported through `particula.gpu.kernels`, not a reusable CPU plan, and
    does not bind future E6-F8 source inputs.
- [x] Does scaling run before resampling when both are enabled?
  - Decision: no. Resampling-first precedence is mandatory; scaling is
    considered only when planned resampling remains insufficient.
- [x] What happens when both policies are disabled?
  - Decision: capacity-sufficient calls proceed; actual exhaustion raises before
    mutation.
- [x] Must all resulting weighted states work with every E5 coagulation mode?
  - Decision: no. Use equal-weight source packaging where possible and test the
    supported integrated sequence. General multiplicity-aware coagulation and
    sedimentation compatibility are explicitly deferred rather than changing
    E5 physics inside E6.
