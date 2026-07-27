# Open Questions

- [x] Should backend selection extend `RunnableABC` or introduce a separate
  typed execution/session protocol?
  - Resolved 2026-07-27: Introduce a separate typed execution/session protocol
    and retain `RunnableABC` as the CPU `Aerosol` adapter contract.
  - Rationale: The existing runnable accepts one `Aerosol`, while the planned
    resident path owns particle, gas, environment, work-buffer, and RNG state.
  - Evidence:
    - `particula/runnable.py:36` - `RunnableABC` is explicitly an `Aerosol`
      process abstraction.
    - `docs/Features/Roadmap/data-oriented-gpu.md:1524` - the GPU loop must keep
      three typed Warp containers resident across enabled dynamics.
  - Resolved by: plan-question-resolver

- [x] What is the minimum versioned checkpoint payload?
  - Resolved 2026-07-27: Store the schema version, process-boundary step/time,
    authoritative particle, gas, and environment state, ordered CPU semantic
    metadata, stable box-to-row identities, persistent per-process RNG state,
    and configuration/capability identity; reconstruct scratch and derived
    thermodynamic buffers.
  - Rationale: Those fields are the minimum authoritative state needed to
    resume the same logical boxes and stochastic streams, while vapor pressure
    and scratch remain derived or replaceable implementation state.
  - Evidence:
    - `particula/gpu/conversion.py:422` - checkpoint restoration covers all
      authoritative particle fields.
    - `particula/gpu/conversion.py:468` - gas restoration requires caller-owned
      ordered names and intentionally drops GPU-only vapor pressure.
    - `particula/gpu/conversion.py:584` - environment restoration preserves
      temperature, pressure, and saturation ratio.
    - `docs/Features/Roadmap/data-oriented-gpu.md:1592` - persistent per-box RNG
      streams are required resident execution state.
  - Resolved by: plan-question-resolver

- [x] Where should explicit CPU fallback occur?
  - Resolved 2026-07-27: Select CPU before GPU upload, or transition only after
    a caller-requested synchronized checkpoint/finalize restores complete CPU
    state; never fall back inside an adapter or scheduler step.
  - Rationale: This keeps every transfer visible and prevents retrying from
    partially mutated resident state.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1494` - missing GPU processes
      require explicit fallback boundaries and no silent movement.
    - `docs/Features/Roadmap/data-oriented-gpu.md:1535` - CPU transfers are
      limited to named observation and final-result boundaries.
  - Resolved by: plan-question-resolver

- [ ] Which transport map representation best preserves fixed-shape and
  deterministic execution?
  - Open: The roadmap permits both fixed-shape maps and sparse edge lists, and
    no workload or memory evidence establishes one representation universally.
  - Recommendation: **A - Use a fixed-capacity canonical edge list**
  - Suggested answer: Choose **A** because it preserves stable allocation while
    avoiding dense storage for prescribed low-degree communication graphs.
  - Options:
    - [ ] A. Fixed-capacity edge list with canonical source/destination order and
      an active-edge count (Recommended)
    - [ ] B. Dense box-to-box transfer matrix with zero entries for absent edges
    - [ ] C. Freeze separate dense and sparse public schemas in the first release
  - Evidence considered:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1573` - both fixed-shape maps
      and sparse edge lists are explicitly allowed.

- [x] How are stream identities preserved when boxes are reordered?
  - Resolved 2026-07-27: Key each process stream by a stable caller-provided
    logical box ID plus a versioned process namespace, checkpoint that mapping,
    and permute row-aligned physical and RNG state together without reseeding.
  - Rationale: Current device rows are positional, so row index alone cannot
    preserve stream identity under reorder, insertion, or disabling.
  - Evidence:
    - `docs/Features/Roadmap/data-oriented-gpu.md:1592` - independent box streams
      must remain reproducible when box count or enablement changes.
    - `particula/gpu/warp_types.py:164` - environment state is row-batched and
      contains no logical box identifier.
  - Resolved by: plan-question-resolver
