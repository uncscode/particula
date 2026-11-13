---
name: particula-guide
description: "Explain aerosol science and Particula APIs with authoritative sources; use for tutorials, feature questions, equations, units, or choosing a modeling approach."
---

# Explain Particula
1. Determine the scientific question and the user's level. Start with the physical process, then connect it to Particula objects and computations.
2. Consult current official docs or the selected version's source. Cite precise pages for nontrivial scientific/API claims. Distinguish library release behavior from main-branch documentation and planned features.
3. Explain assumptions, SI units, array shapes, radius versus diameter, and whether concentrations are per bin or densities per radius/log-radius.
4. Cover gas and particle construction, activity/equilibria, condensation/evaporation, coagulation, dilution, wall loss, and nucleation as relevant. Verify availability instead of asserting every feature exists in every release.
5. Use a minimal concrete example. If a calculation was not run, label it as illustrative. Do not claim scientific validation from a successful import or smoke run.
6. For GPU questions, consult the feature-specific contract and verify device, precision, transfers, synchronization and supported combinations. Do not infer a high-level GPU runnable or fallback from a direct kernel.
Return a concise explanation, a practical next step, and source links. Route implementation to particula-build and execution to particula-run.

## Shared operating rules
Use the selected interpreter’s `python -m particula_docs --json` to locate and search version-matched installed docs before relying on the live site; see `../../references/sources.md` for commands and older-version fallback.
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
