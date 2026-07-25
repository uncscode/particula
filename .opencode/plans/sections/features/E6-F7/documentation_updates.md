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
