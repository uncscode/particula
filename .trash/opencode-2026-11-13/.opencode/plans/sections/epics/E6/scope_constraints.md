# Scope and Constraints

## In Scope

- CPU dilution strategy and runnable reference, then direct GPU dilution.
- Neutral spherical/rectangular GPU wall loss and charged GPU wall loss with
  neutral fallback.
- Shared CPU/GPU fixed-slot activation and exact diagnostics.
- Default exhaustion resampling plus optional representative-volume scaling.
- New inventory-limited CPU nucleation process, then direct GPU nucleation.
- CPU reference tests, CPU/Warp parity, stochastic validation, conservation,
  inactive/activation/full-slot cases, documentation, and a direct-step example.

## Out of Scope

- Backend selection, resident-loop scheduling, high-level GPU runnable
  integration, multi-box transport, graph capture, and differentiability.
- Dynamic GPU storage, process-level transfers, hidden fallback, general CFD
  coupling, and unsupported performance claims.

## Constraints

- Python 3.12+, NumPy CPU references, Warp kernels, fixed-shape fp64 container
  conventions, explicit transfer helpers, and caller-owned sidecars.
- CPU references precede corresponding GPU parity work.
- Warp CPU is mandatory GPU evidence; CUDA is optional and non-blocking.
- Public boundaries validate before mutation; invalid calls preserve caller
  state and sidecar identity.
- Resampling defaults on, volume scaling defaults off, both controls remain
  independent, and disabling both fails before mutation.
- Scientific nucleation equations document units, citations, and validity
  domains.
