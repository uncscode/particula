---
name: particula-debug
description: "Diagnose Particula exceptions, incorrect results, version migrations or CPU/GPU mismatches; use for traceback analysis, API drift, conservation failures, or numerical instability."
---

# Diagnose and repair
1. Capture the exact interpreter, Particula version, complete relevant traceback, minimal inputs, shapes/dtypes/units and expected behavior. Run `../../scripts/doctor.py`; avoid dumping environment variables or credentials.
2. Classify the issue: environment/import, unavailable API, builder validation, representation/shape mismatch, units, numerical instability, physical assumptions, or backend capability.
3. Inspect the installed function signature/source and matching documentation. An online example may target a different release. Do not guess renamed methods or upgrade as the default fix.
4. Reduce to a small deterministic case. Change one cause at a time and preserve the original script. Re-run the failing case after a fix, then check scientific invariants as well as exit status.
5. If a finer time step changes the answer materially, report nonconvergence. Do not clip away failures. For GPU issues verify explicit device ownership, dtype, transfer and synchronization; report any fallback rather than masking it.
6. Explain the root cause only as strongly as the evidence supports. If not reproduced, state what remains uncertain and provide the next diagnostic action.
Deliver the minimal correction, before/after behavior, verification evidence and any unresolved limitations.

## Shared operating rules
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
