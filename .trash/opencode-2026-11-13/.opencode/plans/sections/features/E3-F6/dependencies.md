## Dependencies

### Internal Dependencies

- Parent epic `E3` for the broader GPU/docs roadmap context.
- Existing CPU condensation strategy implementation in
  `particula/dynamics/condensation/condensation_strategies.py`.
- Existing latent heat builders/factories in `particula/gas/`.
- Existing `MassCondensation` runnable in `particula/dynamics/particle_process.py`.
- Existing docs examples under `docs/Examples/Dynamics/Condensation/` for style
  and notebook pairing conventions.
- Existing docs tooling in `.opencode/tools/validate_notebook.py` and
  `.opencode/tools/run_notebook.py` if paired notebooks are used.

### External Dependencies

- Standard project documentation dependencies already used by examples
  (`numpy`, optional `matplotlib`, Jupyter tooling for notebooks).
- No new runtime or development dependency is expected.

### Feature Dependencies

- Declared track dependencies: none.
- Sibling E3 features may update GPU docs or testing policy, but this feature
  can ship independently because it is CPU documentation/example work.
- `E3-F7` should consume this feature's final runnable scenario when selecting a
  long-lived CPU latent-heat baseline, but E3-F6 does not need to wait on E3-F7
  to land the example itself.

### Dependency Risks

- If docs tooling is unavailable locally, the script can still be validated, but
  notebook sync/execution must be completed before shipping a paired notebook.
- If public factory parameter names differ from assumptions, use existing tests
  and builders as the source of truth instead of reaching into private internals.

### Phase Ordering Notes

- P1 comes first because the runnable `.py` example is the source artifact for
  any paired notebook and for downstream CPU latent-heat cross-links.
- P2 follows P1 so notebook sync, index wiring, and optional feature links point
  at a stable example path and verified narrative.
- P3 closes the feature only after P1/P2 artifacts execute successfully and the
  CPU-only guidance matches the final published example.
