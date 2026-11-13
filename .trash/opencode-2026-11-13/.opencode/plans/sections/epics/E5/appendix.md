# Appendix

## Authoritative Intake

- GitHub issue #1320: `[Planner]: GPU Coagulation Physics Coverage`
- Epic plan: E5, **GPU Coagulation Physics Coverage**
- Feature tracks: E5-F1 through E5-F9, mapped one-to-one from issue tracks
  T1 through T9.
- Maintenance Tracks: none
- Research Tracks: none
- Classifier diagnostics: none

## Key References

- `docs/Features/Roadmap/index.md:61-128` - roadmap sequence and closeout
  artifact expectations.
- `docs/Features/Roadmap/data-oriented-gpu.md:989-1034` - mechanism scope and
  exit bar.
- `particula/gpu/kernels/coagulation.py:433-465,792-1022` - merge behavior and
  current Brownian orchestration boundary.
- `particula/gpu/warp_types.py:24-78` - existing particle charge storage.
- `particula/dynamics/coagulation/charged_kernel_strategy.py:28-305` - CPU
  charged variants.
- `particula/dynamics/coagulation/sedimentation_kernel.py:28-179` - SP2016
  reference and unfinished efficiency helper.
- `particula/dynamics/coagulation/turbulent_shear_kernel.py:30-138` - ST1956
  reference and required inputs.
- `particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py:63-154`
  - additive CPU combination semantics.
- `.opencode/guides/testing_guide.md:166-245` - device-aware deterministic and
  stochastic test policy.
- `.opencode/plans/sections/epics/E4/child_plans.md:18-27` - deferred
  condensation walkthrough.
- `particula/integration_tests/condensation_latent_heat_conservation_test.py:231-299`
  and `docs/Examples/gpu_direct_kernels_quick_start.py:172-327` - independent
  CPU and explicit Warp walkthrough inputs.

## Planning Notes

- The plan CLI reports that epic records do not support phase objects. The nine
  authoritative child feature tracks and dependency-ordered milestones are the
  epic-level work breakdown; detailed implementation phases belong in each
  feature plan and must co-locate tests with changed functions.
- The required nested codebase-researcher dispatch was unavailable because the
  drafter was already at the configured subagent-depth limit. Drafting therefore
  uses issue #1320's extensive codebase research and references without adding
  scope.

## Rejected Approaches

- Sequential stochastic passes per mechanism: rejected because combined CPU
  semantics add kernels before stepping and sequential passes can bias counts.
- Reusing the Brownian majorant without proof: rejected for charged and mixed
  mechanisms.
- Hidden CPU fallback or transfers: rejected because ownership and device
  support must remain explicit.
- Exact CPU/Warp collision replay: rejected in favor of pair parity, aggregate
  stochastic bounds, and conservation invariants.
