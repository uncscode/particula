# Scope

E7-F5 adds a deterministic scheduler in `particula.execution` over the E7-F2,
E7-F3, and E7-F4 contracts. A typed process graph validates supported nodes and
dependencies, then runs one canonical resident timestep with explicit
environment, derived-thermodynamic, gas, and diagnostic boundaries.

## In Scope

- Typed capability nodes and immutable timestep/process declarations.
- Deterministic dependency resolution independent of user registration order.
- Backend-selected condensation and Brownian coagulation from E7-F2/E7-F3.
- Resident direct adapters for shipped dilution, neutral/charged wall loss, and
  fixed-slot nucleation without changing their kernel contracts.
- Prescribed per-box temperature, pressure, and gas updates with strict shape,
  dtype, device, finiteness, positivity/nonnegativity, and alias validation.
- Vapor-pressure and saturation refresh after relevant updates and before every
  consumer; simulation volume remains `ParticleData.volume` state.
- E7-F4 `begin_step()`/`complete_step()` lifecycle integration, stable identity,
  post-launch faulting, and no intermediate transfer or synchronization.
- Optional non-mutating diagnostic hooks and complete-loop Warp CPU tests;
  optional CUDA rows skip cleanly.

## Out of Scope

- Silent CPU fallback, hidden CPU/GPU movement, or runtime retry on another
  backend; E7-F6 owns explicit transition and error policy.
- Multi-box transport, mixing, advection, and expansion (E7-F7).
- Final per-box stream identity and restart policy (E7-F8).
- Epic-wide diagnostics products, complete public example, and closeout matrix
  (E7-F9), beyond the hooks and tests needed by this feature.
- Unsupported physics expansion, GPU staggered condensation, dynamic resizing
  or compaction, multi-GPU/distributed execution, graph capture, performance
  claims, autodiff, or kernel-physics rewrites.
