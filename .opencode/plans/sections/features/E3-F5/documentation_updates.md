# Documentation Updates

## Required Updates

- `.opencode/guides/testing_guide.md`
  - Add a device-aware Warp pytest policy under the NVIDIA Warp Tests section.
  - List marker names, default behavior, CUDA-if-available semantics, and skip
    expectations.
  - Record stochastic parity tolerance guidance and examples.
- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Mark E3-F5 policy as defined once implementation lands.
  - Record CUDA local/manual validation expectations and parity tolerance
    classes.

## P1 Shipped Status

- No external documentation files changed in `E3-F5-P1`.
- The shipped implementation limited documentation work to inline module
  docstrings/comments in `particula/conftest.py` describing shared marker
  registration and benchmark-only collection gating.
- Testing guide and roadmap updates remain queued for `E3-F5-P3` and `E3-F5-P5`.

## Conditional Updates

- `docs/Features/data-containers-and-gpu-foundations.md`
  - Update only if public user-facing GPU test/run guidance needs clarification.
- Contributor or release docs
  - Add focused release validation commands if a suitable document already
    exists.

## Documentation Content Requirements

- State that CUDA is optional and must skip cleanly when unavailable.
- State that Warp CPU is the default parity backend when Warp is installed.
- State that stochastic tests use aggregate statistical expectations, not exact
  per-seed equality.
- Include focused commands for Warp CPU and CUDA-if-available validation.
