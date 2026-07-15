# Open Questions

- [x] What `rtol`/`atol` applies to combined-physics parity?
  - Resolved 2026-07-13: target `rtol=1e-10` with a scale-derived `atol` for
    fp64 Warp CPU and CUDA. F6 records any empirically justified case/device
    relaxation; the stiffness-study `5e-2` bound is not accepted for production
    qualification.
- [x] How is complete scratch for graph capture validated?
  - Resolved 2026-07-13: enumerate every sidecar buffer, warm compilation, then
    assert subsequent calls allocate and synchronize zero times while retaining
    all identities and shapes. Missing or partial complete-scratch fields fail
    before mutation.
- [x] Which devices support graph capture and what skip policy applies?
  - Resolved 2026-07-15: exercise capture/replay on Warp CPU and, when CUDA is
    available, CUDA. Missing capture APIs or capture capability failures at
    begin, recording, end, or launch skip only that device with operation
    context; normal-launch and post-launch correctness failures remain failures.
- [x] Which smooth-interior slice is differentiated?
  - Resolved 2026-07-15 by #1311: the test differentiates only the
    out-of-place `condensation_mass_transfer_kernel` raw-rate proposal with
    positive inventory and executable interior-margin checks. Warp Tape is
    compared with a centered fp64 derivative with
    `rtol=2e-5`, `atol=1e-18`; P2 clamps, inventory scaling, and in-place mass
    updates remain forward-only non-claims.
- [x] Does strict conservation require an absolute tolerance?
  - Resolved 2026-07-13: yes. Use `rtol=1e-12` with
    `atol=max(1e-18, scale * eps)` so near-zero inventories remain meaningful.
- [x] Which issue #1272 document is canonical for the final evidence matrix?
  - Resolved 2026-07-13: `docs/Features/Roadmap/data-oriented-gpu.md` owns the
    final support/evidence matrix. `condensation-stiffness-study.md` remains the
    historical integration decision record.

Diagnostics are intentionally **none**; questions must not introduce a public
diagnostics surface merely to make tests convenient.
