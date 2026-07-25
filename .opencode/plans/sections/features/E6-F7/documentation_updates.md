# Documentation Updates

- P1 shipped module and API docstrings in
  `particula/dynamics/nucleation/nucleation_strategies.py` and updated the
  user-facing theory page
  `docs/Theory/Technical/Dynamics/Nucleation_Equations.md`.
- The theory page now documents the bounded scalar potential-rate contract:
  mass-to-number concentration conversion, activation and kinetic SI
  equations, caller-supplied survival factor, closed concentration/temperature
  domains, asymmetric saturation gating, formation/injection metadata, and the
  explicit no-source/no-inventory-mutation boundary.
- P2 added concrete-module docstrings in
  `particula/dynamics/nucleation/particle_source.py`; it added no user-facing
  documentation, examples, or public exports. P7 should extend the theory page
   with the shipped source-finalization equations and implementation status.
- P3 added concrete-module documentation for the unexported
  `commit_particle_source` transaction in
  `particula/dynamics/nucleation/particle_source.py`. No user-facing docs,
  examples, or package exports were added; P7 remains responsible for those
   documentation decisions.
- P4 added module/API docstrings for strict builders, `NucleationFactory`, and
  immutable `NucleationSourceConfig`, plus architecture-boundary corrections in
  `.opencode/guides/architecture/architecture_outline.md`,
  `.opencode/guides/architecture_reference.md`, and
  `.opencode/guides/architecture/architecture_guide.md`. These describe the
  bounded public construction surface and preserve P2/P3 as concrete-only.
  No user-facing feature documentation, examples, or runnable was added; P7
   remains responsible for those decisions.
- P5 added public Google-style code docstrings for CPU-only single-box
  `Nucleation` and immutable `NucleationCommitConfig` in
  `particula/dynamics/particle_process.py`. Broader user and architecture
   documentation remains deferred to P7; no P2/P3 transaction API was exposed.
- P6 (issue #1435) changed only the three nucleation test modules. It added no
  production docstrings, user-facing documentation, examples, exports, or
  public API documentation; P7 remains responsible for documentation updates.
- Retain citations to Seinfeld & Pandis (2016), Kulmala et al. (2006, DOI
  `10.5194/acp-6-787-2006`), and Kerminen & Kulmala (2002, DOI
  `10.1016/S0021-8502(01)00194-X`). State that Vehkamäki et al. (2002, DOI
  `10.1029/2002JD002184`) is bounded-model context, not implemented here.
- Add/update `docs/Features/` with supported/deferred models, strategy and
  factory examples, diagnostics, mutation boundary, no-ops, and conservation.
- Replace or distinguish the illustrative custom workflow in
  `docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.py`
  with supported API usage; sync/execute its paired notebook if modified.
- Update `AGENTS.md` with imports, units, E6-F5/F6 interaction, conservation,
  failure behavior, and focused test commands.
- Cross-link E6-F5/F6, E6-F8/F9, and the data-oriented GPU roadmap.
- Update feature phases/status and parent E6 sections if final APIs or the
  scientific boundary change.

Documentation must not market empirical forms as universal predictions or
imply hidden survival correction, full Vehkamäki physics, GPU support, dynamic
slots, automatic scheduling, or performance proof.
