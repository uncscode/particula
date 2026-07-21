# Documentation Updates

- Update `docs/Theory/Technical/Dynamics/Nucleation_Equations.md` with the exact
  strategy APIs, SI conversions, closed validity domains, source finalization
  equations, and implementation status.
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
