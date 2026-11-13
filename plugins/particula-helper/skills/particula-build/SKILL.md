---
name: particula-build
description: "Design and write Particula aerosol simulations; use for scripts, notebooks, process selection, parameter sweeps, or translating a research question into a model."
---

# Build an aerosol study
1. Identify the objective, species, distribution representation, initial conditions with units, temperature, pressure, processes, duration and outputs. Separate supplied values from demonstration assumptions. Do not invent measured parameters.
2. Inspect installed signatures and authoritative examples. Use `../../scripts/run_example.py` as a tested CPU starter, not a universal scientific model. Copy it into the user's study only if editing is needed.
3. Keep configuration, model construction, stepping, diagnostics and plotting separate. Use explicit units and state shapes. Bound time steps and output sizes; seed stochastic algorithms where supported and record the seed.
4. Compose only processes supported by the installed version. Account for process ordering and operator splitting. For source/sink models define the inventory budget explicitly.
5. Save the script, parameters, dependencies and output instructions together. Write outputs into a fresh run directory; preserve earlier results.
6. Start small, then check finite/nonnegative state, relevant conservation and time-step or resolution sensitivity before increasing scale. Avoid clamping negative values to hide numerical errors.
Deliver executable code, assumptions, exact run command, expected artifacts, and validation criteria. If requested, continue with particula-run rather than stopping at code generation.

## Shared operating rules
Use the selected interpreter’s `python -m particula_docs --json` to locate and search version-matched installed docs before relying on the live site; see `../../references/sources.md` for commands and older-version fallback.
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
