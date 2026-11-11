# Compatibility and evidence

The helpers require Python 3.12 or newer. Always run `scripts/doctor.py` with
the user's selected interpreter before simulation. An editable checkout may
differ from the installed distribution metadata; inspect both.

## Numerical evidence

- Particula 0.2.14, Python 3.12.12, NumPy 2.3.4, SciPy 1.16.2,
  Matplotlib 3.10.7: a 10 s CPU condensation run with 1,000 steps passed
  finite/nonnegative, number, and total-mass checks. Maximum relative mass
  error was 1.9e-15. Halving the time step changed final particle mass by
  approximately 0.0011%. This is one illustrative case, not broad convergence.
- A 40-bin coagulation trial on that environment with 1e12 particles/m^3,
  600 s duration, and 600 steps failed the 1e-6 relative mass tolerance:
  error was about 1.3%. Its cause is not established. Preserve the failure;
  do not loosen conservation tolerance or call arbitrary coagulation runs
  validated. Use condensation for first-run instructions.
- `requirements.txt` pins the original CPU starter environment for Particula
  0.2.15. CI smoke tests verify the current source checkout separately.
- GPU execution is not qualified by this plugin's CPU smoke tests.

## Host boundaries

CLI command syntax was inspected with Codex 0.167.0, Claude Code 2.1.319,
and OpenCode 2.1.3. These are validation targets, not claimed minimum versions.
Local marketplace install, six-skill inventory, and uninstall were checked in
isolated configurations for Codex and Claude. OpenCode discovered all six
skills from an extracted npm package after empty-session initialization.
The installed-package tests also cover resource paths, repeatable transforms,
duplicate preservation, and disposal; they do not call a model.
The OpenCode adapter uses the OpenCode 2 API and pins SDK 2.1.3; OpenCode 1
users should use the standalone skills bundle instead.

Package validation, installed-package checks, and a Python smoke run do not
establish successful model-driven execution in every host. Desktop uploads,
Cowork Python availability, and access to a pre-existing local virtual
environment require verification in the actual host. A package never grants
filesystem access on its own.
