# Compatibility and evidence

The helpers require Python 3.12 or newer. Always run `scripts/doctor.py` with
the user's selected interpreter before simulation. An editable checkout may
differ from the installed distribution metadata; inspect both.

## Numerical evidence

The current starters use native Aerosol, ParticleData, GasData and
EnvironmentData. Use an installation with the 0.3.0 native APIs as described
in local-setup.md (a selected checkout when these APIs are not yet released); the older release evidence below is historical, not a claim
that this script still supports that facade API. Native default 1 s/100-step
smokes passed: condensation total-mass error 7.9e-16, coagulation particle-mass
error 2.5e-9, with the existing 1e-6 helper tolerance unchanged.

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
- `requirements.txt` pins plotting dependencies only. CI smoke tests verify
  the current native source checkout; no new release compatibility is claimed.
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

## Repository skills

Plugin 0.3.0 expands the catalog from six to eleven skills. The five repository
skills use the selected source checkout and its current instructions, not a
bundled copy of the library. The historical live-host checks above cover the
original six; package checks can verify the expanded catalog and resource paths
but do not establish live discovery or model-driven behavior in every host.

## Local documentation

Plugin 0.3.1 uses the selected interpreter's `particula_docs` locator to find
release-matched documentation, with explicit fallback for older installations.
Documentation packaging and locator checks do not expand the numerical or
live-host qualification above. Repository development still uses the selected
checkout, while reading a wheel's bundled documentation needs no checkout.
