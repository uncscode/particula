---
name: particula-run
description: "Execute local Particula simulations and parameter sweeps reproducibly; use when the user asks to run, simulate, compare scenarios, or save results."
---

# Run locally
1. Resolve the local interpreter and project directory; run `../../scripts/doctor.py`. Review the requested script and configuration for scope, inputs, outputs and resource demand.
2. For a starter, run `../../scripts/run_example.py --process condensation --output <new-directory>` or use `--process coagulation`. Defaults are illustrative CPU studies. Use `--help` for supported parameters; do not invent flags.
3. The first run should be small. Respect the user's resource budget and existing authorization; ask before moving to a materially expensive or long run. Never silently switch backends.
4. Preserve logs, exit status, version/parameter metadata and outputs. A failed run is not a completed simulation even if partial files exist. The example writes run.json only after its numerical checks pass.
5. Inspect time_series.csv, distribution.csv, result.png and run.json. Confirm relevant physical budgets and compare a smaller time step when drawing quantitative conclusions.
6. For sweeps use separate run directories and an explicit parameter matrix. Summarize success/failure per case and do not discard inconvenient outcomes.
Report what actually ran, scientific assumptions, numerical checks, output links and limitations. Use particula-analyze for interpretation. Never upload research data or start a hosted job unless requested.

## Shared operating rules
Use the selected interpreter’s `python -m particula_docs --json` to locate and search version-matched installed docs before relying on the live site; see `../../references/sources.md` for commands and older-version fallback.
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
