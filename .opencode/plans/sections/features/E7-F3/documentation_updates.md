# Documentation Updates

## P2/P3/P4 Status

No user-facing documentation or exports changed in P2, P3, or P4. The adapter remains a
concrete-only adapter boundary, so this implementation intentionally adds no
README, feature-guide, API-reference, example, or public-export documentation.
Selected API documentation remains deferred to P6.

- Add backend-selected Brownian coagulation to the E7 backend-selection feature
  guide, including CPU and Warp setup, explicit backend/device requests, and
  capability errors.
- Update the coagulation feature/API documentation with the T3 support matrix:
  Brownian selection only, CPU reference behavior, Warp particle-resolved
  behavior, and explicit exclusions for other mechanisms/distributions.
- Document caller ownership and lifetime of particles, environment/volume,
  collision buffers, and `(n_boxes,)` persistent RNG state. Show seed once,
  reuse with no reset, and deliberate reset examples.
- State that selected execution performs no hidden upload, restore,
  synchronization, fallback, or per-step reseeding and that asynchronous Warp
  failures have no rollback guarantee after launch.
- Add or update a focused runnable example based on
  `docs/Examples/gpu_coagulation_direct.py` without misrepresenting T3 as the
  future E7-F4 resident session or E7-F5 scheduler.
- Update `docs/Features/Roadmap/data-oriented-gpu.md` and E7 plan status only
  when the implementation evidence ships; preserve Epic H/I deferrals.
- Add documentation regression coverage for supported imports, limitations,
  persistent RNG wording, and example execution; run `mkdocs build --strict`.
