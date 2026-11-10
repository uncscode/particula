---
name: particula-setup
description: "Set up and inspect a local Particula Python environment; use for installation, dependency problems, environment selection, or first-run verification."
---

# Set up Particula
1. Establish the user's project directory and existing interpreter/environment. Check Python and installed Particula before installing anything. Use the bundled `../../scripts/doctor.py` with that interpreter; it does not install packages.
2. For a fresh CPU demonstration, use an isolated `.venv` and `../../requirements.txt`. See `../../references/local-setup.md` for commands. These requirements pin the tested Particula version, not the user's existing research environment.
3. Do not install GPU drivers or change system Python. Determine GPU needs separately; availability does not establish compatibility or correctness.
4. Re-run the doctor and report Python, Particula, NumPy, plotting availability, and required API checks. Missing optional components are different from broken core imports.
5. Run a small example into a new output directory. Confirm actual numerical outputs and files before saying setup works. Route failures to particula-debug.
Deliver the chosen interpreter, versions, a working command, and any remaining limitations.

## Shared operating rules
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
