# Vision and Problem

Particula has shipped fixed-shape CPU and Warp containers, explicit transfer
helpers, and direct GPU kernels for condensation, coagulation, dilution, wall
loss, and nucleation. Those pieces are not yet a complete user-facing
simulation system.

Problems today:

1. **No stable backend choice** -- users cannot select CPU or GPU through one
   documented execution API with explicit capability behavior.
2. **No resident simulation abstraction** -- callers must manually retain and
   coordinate particle, gas, environment, scratch, diagnostic, and RNG state.
3. **No canonical full-loop scheduler** -- ad hoc ordering can use stale
   thermodynamic state or change process semantics.
4. **No complete multi-box contract** -- independent boxes exist, but prescribed
   transport, mixing, and volume evolution are not integrated.
5. **No unified restart story** -- checkpoints and persistent per-box random
   streams lack a system-level ownership and reproducibility contract.

## The Vision

Users select a supported backend, initialize a typed simulation session once,
and execute deterministic multi-timestep simulations while all mutable state
remains on the selected device. Bulk transfers occur only at explicit,
synchronized checkpoints or finalization. CPU behavior remains the reference;
unsupported GPU physics and unavailable devices fail clearly or cross only an
explicitly requested fallback boundary.

## Why Now

Earlier GPU epics supplied the required containers, conversion boundaries, and
bounded direct process kernels. Integrating them is the next ordered roadmap
step and is a prerequisite for later graph-capture/performance work (Epic H)
and autodiff/optimization work (Epic I). Delaying integration encourages
duplicated orchestration, hidden transfers, and incompatible public APIs.
