# Documentation Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the latent equation,
  issue #1272 mapping, gates, and focused Warp CPU/CUDA commands.
- Update the relevant `docs/Features/` GPU guide with units, fp64 sidecar
  ownership, keyword-only usage, fallback, and no-host-transfer contract.
- Define diagnostics as signed `(n_boxes, n_species)` whole-four-substep totals
  computed from bounded applied mass.
- State that E4-F4 neither evolves temperature nor proves gas/system
  conservation; link to E4-F5/E4-F6.
- Update `AGENTS.md` only if the public low-level invocation changes materially.
- P1 shipped without public documentation changes; its private helpers and
  validation-only sidecars are intentionally deferred to the P4 documentation
  update. These plan sections record the delivered contract.
