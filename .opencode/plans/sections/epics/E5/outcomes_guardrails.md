# Outcomes and Guardrails

- **Primary Outcome:** Extend particle-resolved `coagulation_step_gpu` from its
  Brownian baseline to validated charged, Brownian-plus-charged, SP2016
  sedimentation, ST1956 turbulent-shear, and supported additive combined
  mechanisms without regressing numerical correctness or ownership contracts.
- **Secondary Goals:**
  - Conserve species mass for every mechanism and conserve charge for every
    charge-bearing merge.
  - Preserve backward compatibility for Brownian calls, caller-owned persistent
    RNG state, fixed-shape buffers, and explicit CPU/Warp transfer boundaries.
  - Publish mechanism support, required inputs, tested devices, limitations,
    and a direct GPU coagulation example.
  - Close the E4 carry-forward with an independent CPU/Warp condensation
    walkthrough and explicit ownership for deferred capabilities.
- **Guardrails / Non-Goals:**
  - No DNS turbulence or unfinished sedimentation collision-efficiency model;
    initial sedimentation uses collision efficiency 1.
  - No binned or continuous-PDF GPU coagulation; support remains
    particle-resolved.
  - No dynamic slot allocation, hidden CPU fallback, hidden transfers, or
    general performance redesign.
  - No high-level `Aerosol`/`Runnable` integration, graph capture, broad
    autodiff, or adaptive stepping.
  - No exact CPU/Warp collision-pair replay requirement and no claims beyond
    the deterministic and bounded stochastic cases tested.
  - CUDA remains optional additive evidence; routine release validation must
    not require CUDA hardware.
