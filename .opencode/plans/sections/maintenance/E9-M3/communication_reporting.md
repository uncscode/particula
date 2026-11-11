# Communication and Reporting

- Preserve phase evidence in the repository records and the single M3 PR.
  Do not generate separate phase PRs or send issue messages as a side effect.
  The maintainer owns scientific acceptance and the M4 integration handoff.
- The PR records native call/configuration changes, assertion migration,
  units/species/normalization decisions, commands/results and temporary seams
  with M6 removal ownership. Review metrics at every phase gate.
- Escalate scientific drift, missing native paths, lost behavior, required
  failures or unavailable evidence immediately. Do not weaken tolerances or
  silently defer scientific implementation to M4.
- Wall-loss reports name BOTH suites and separate results. Recursive/full
  collection does not prove the adjacent suite ran. Report focused assertions
  separately from untargeted full-package coverage.
- P6 handoff includes validated revision, process configuration, units/native
  call contracts, identity/failure semantics, ledgers and reviewer approval.
  M4 begins only after explicit completed-and-validated acceptance. M6 receives
  a deletion inventory, not permission for early removal.
- Implementation evidence is separate from reviewed acceptance; publishing
  the PR does not authorize merging, releasing or starting M4.
