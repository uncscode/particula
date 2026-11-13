---
name: particula-analyze
description: "Analyze and visualize Particula simulation results or compare them with observations; use for plots, conservation checks, convergence, uncertainty, and reproducibility reports."
---

# Analyze aerosol results
1. Read run metadata and numerical outputs before interpreting plots. Establish units, distribution representation, radius/diameter convention, species and process configuration.
2. Check NaN/Inf, negative concentrations/masses, number and mass trends, and the correct open/closed-system inventory. Condensation exchanges gas/particle mass; coagulation changes number while conserving mass in the ideal closed model. Boundary losses or sources require explicit budgets.
3. Distinguish per-bin counts, PDFs and dN/dlogD; integrate with appropriate widths and Jacobians. Never sum a continuous density as though it were per-bin counts.
4. Compare smaller time steps and finer resolution for numerical convergence. Report changes in physically relevant observables, not only plot appearance. For stochastic studies use independent seeds and uncertainty intervals when available.
5. Label axes and units; use logarithmic size axes where appropriate. Preserve raw results and export plots plus a short report of inputs, versions, checks and limitations.
6. For observations align units, sampling windows, instrument response and uncertainties; do not conflate calibration and independent validation. Separate numerical, parameter and structural uncertainty.
Deliver key findings, linked plots/data, quantitative check results and justified next steps. A successful bundled smoke test is not evidence that a model predicts a particular experiment.

## Shared operating rules
Use the selected interpreter’s `python -m particula_docs --json` to locate and search version-matched installed docs before relying on the live site; see `../../references/sources.md` for commands and older-version fallback.
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
