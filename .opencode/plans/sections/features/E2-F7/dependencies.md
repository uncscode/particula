# E2-F7 Dependencies

## Required Upstream Dependencies

- **E2-F2 / T2:** Environment and schema foundations.
  - Needed for per-box temperature, pressure, and derived environment state.
  - Until available, E2-F7 should use current scalar `temperature` and
    `pressure` compatibility paths and document migration assumptions.
- **E2-F6 / T6:** Dtype and precision envelope.
  - Needed before recommending any fp32 or mixed-precision condensation path.
  - Current plan treats fp64 as reference for all stiffness measurements.

## Sequencing Constraints

- `E2-F7-P1` can start after `E2-F1` using current scalar compatibility paths to
  define stress cases and metrics.
- `E2-F7-P2` should not finalize its stability map until `E2-F2` documents the
  per-box environment shape that later GPU condensation work must honor.
- `E2-F7-P3` and `E2-F7-P4` should stage behind `E2-F6` because the published
  recommendation must stay inside the accepted precision envelope.

## Internal Code Dependencies

- `particula/gpu/kernels/condensation.py` for the current explicit GPU path.
- `particula/gpu/warp_types.py` and `particula/gpu/conversion.py` for fixed
  shape GPU container setup.
- `particula/dynamics/condensation/mass_transfer.py` and
  `mass_transfer_utils.py` for CPU reference equations and limiters.
- `particula/dynamics/condensation/condensation_strategies.py` for CPU
  simultaneous and staggered prior art.
- `particula/dynamics/particle_process.py` for existing CPU sub-step API
  expectations.

## External Dependencies

- Warp, already optional in the GPU test stack.
- NumPy and pytest for deterministic stress-case and metric tests.
- CUDA is optional; all fast validation should pass in Warp CPU mode.

## Downstream Consumers

- Later E2 GPU condensation implementation phases should consume the selected
  integration foundation.
- Autodiff work should consume the deterministic/fixed-loop guidance from this
  feature.
- Documentation and support-boundary tracks should consume the final stiffness
  map and recommendation.
