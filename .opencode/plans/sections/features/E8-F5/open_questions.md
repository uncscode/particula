# Open Questions

- [x] Should captured full-loop validation remain E8-F5 or be reconciled to
  E8-F4 before issue generation?
  - Resolved 2026-08-30: Keep captured full-loop validation in E8-F5 and correct
    stale parent and sibling mappings before issue generation.
  - Rationale: The orchestrator assignment and canonical E8-F5 plan agree that
    E8-F4 owns replay while E8-F5 owns the reusable correctness gate.
  - Evidence:
    - `.opencode/plans/sections/features/E8-F5/overview.md:13` - E8-F5 explicitly
      supplies the three-way validation matrix.
    - `.opencode/plans/sections/epics/E8/child_plans.md:13` - The parent table
      contains the conflicting stale assignment.
  - Resolved by: plan-question-resolver

- [ ] Can the strict MkDocs gate run in the isolated #1579 worktree?
  - Blocked 2026-09-04: no worktree-scoped MkDocs runner is available. This is
    the sole blocker for P5 shipment; rerun `mkdocs build --strict` before
    changing E8-F5 status.

- [x] Which CPU process composition is sufficiently equivalent for the complete
  three-way fixture where a process has no exact high-level CPU twin?
  - Resolved 2026-08-30: Compose independent per-process NumPy equations and
    bounded CPU seams in the exact prepared-node order; label bounded oracles
    explicitly instead of claiming whole-API identity.
  - Rationale: Repository GPU policy requires independent references, while the
    existing high-level resident test uses no-op process adapters and is not a
    complete physics oracle.
  - Evidence:
    - `.opencode/guides/testing_guide.md:198` - GPU behavior must be checked
      against independent Python or NumPy references.
    - `particula/execution/tests/transport_loop_test.py:130` - Existing tests use
      independent state and amount equations for composed resident behavior.
  - Resolved by: plan-question-resolver

- [x] What qualified CUDA devices will provide literal captured evidence?
  - Resolved 2026-08-30: Probe every CUDA device visible to the active Warp
    runtime and record each device result independently. The current local
    development host has one NVIDIA GeForce RTX 5060; this is an initial evidence
    device, not a hardware allow-list. Future devices qualify through the same
    capture/replay and metadata contract.
  - Rationale: Runtime enumeration permits future hardware evidence without
    hiding device-specific failures behind the default `cuda` alias or treating
    driver visibility as successful capture.
  - Options:
    - [x] A. Qualify every available CUDA device at runtime and record literal metadata (Selected)
    - [ ] B. Require only the default `cuda` device alias to qualify
    - [ ] C. Name a fixed hardware model and block all other devices
  - Evidence:
    - `particula/gpu/tests/cuda_availability.py:17` - CUDA availability is probed
      from the active Warp runtime.
    - `particula/gpu/kernels/tests/condensation_graph_capture_test.py:186` - A
      qualified capture device must expose all required public capture calls.
    - Local development probe, 2026-08-30 - `nvidia-smi` reports one NVIDIA
      GeForce RTX 5060 with KMD/driver `610.57.04` and CUDA UMD `13.3`; actual
      qualification still requires the Warp capture/replay probe.
  - Resolved by: user decision

- [x] Should exact seed-by-seed CPU/CUDA stochastic replay be required?
  - Resolved 2026-08-30: No. Use aggregate stochastic bounds while checking RNG
    lifecycle and conservation separately.
  - Rationale: Repository policy separates stochastic evidence from deterministic
    parity and tight conservation checks.
  - Evidence:
    - `.opencode/guides/testing_guide.md:299` - Device-aware policy defines
      separate deterministic, conservation, and stochastic assertion classes.
  - Resolved by: plan-question-resolver
