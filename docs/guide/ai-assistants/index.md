# AI Assistants

Particula Helper provides six scientific workflows: environment setup, guidance,
simulation design, execution, analysis and debugging, plus five repository
development skills for planning, implementation, testing, documentation and review. It packages
instructions and Python examples; your assistant supplies the tools to read
files and execute Python. No hosted simulation service or MCP server is included.

## Choose your assistant

| Host | Installation | Execution environment |
| --- | --- | --- |
| Codex CLI / desktop | [Codex guide](codex.md) | Selected project environment when local execution is available |
| ChatGPT | [Codex and ChatGPT guide](codex.md#chatgpt-and-desktop-packages) | Host-provided environment; no automatic access to your computer |
| Claude Code | [Claude guide](claude.md) | Selected local project and Python interpreter |
| Claude Desktop / Cowork | [Claude guide](claude.md#desktop-and-cowork) | Verify folder access and Python availability in the actual host |
| OpenCode 2 | [OpenCode guide](opencode.md) | Selected local project and Python interpreter |

The canonical source is [plugins/particula-helper](https://github.com/uncscode/particula/tree/main/plugins/particula-helper).
Repository installation commands require a revision containing that directory
and its marketplace manifests. For development before merge, use the local
checkout instructions in each guide.

## Start with your existing environment

Installing the assistant **does not install the Particula Python library**.
Ask it to inspect your existing environment first:

> Use particula-setup to check this project's .venv. Report the Python and
> Particula versions and whether plotting works. Do not replace my environment.

Then try:

> Run a small CPU condensation example using that environment. Save the script,
> parameters, CSV results, and a plot in a fresh directory. Check total mass,
> particle number, and finite nonnegative state.

Other prompts:

- “Explain the units and assumptions in this condensation model.”
- “Compare the results after halving the time step.”
- “Debug this traceback against my installed Particula version.”

## Contribute to the library

Repository contributors can use `particula-plan`, `particula-develop`,
`particula-test`, `particula-docs`, and `particula-review`. The repository's
[agent entry point](../../../AGENTS.md) links directly to each skill and its
shared references; installation is optional when working from that checkout.
The [plugin source](https://github.com/uncscode/particula/tree/main/plugins/particula-helper)
packages the same guidance for all supported hosts.

These skills use the selected source checkout and direct Python/pytest/Ruff,
plugin packaging and Astro/Starlight commands. They replace the retired root
ADW/OpenCode development workflow. The standalone OpenCode host adapter remains
available. Archived plans and architecture records retain historical context,
not authority over current native APIs or qualification evidence.

## Manual condensation walkthrough

From a checkout or an extracted Codex/Claude plugin, locate its `scripts`
directory. These commands assume a repository checkout and an existing project
environment. Substitute your actual interpreter path:

```bash
.venv/bin/python plugins/particula-helper/scripts/doctor.py
.venv/bin/python plugins/particula-helper/scripts/run_example.py \
  --process condensation --output runs/condensation-001
```

On Windows, use `.venv\Scripts\python.exe`. If starting a new project, create
a virtual environment and install Particula plus Matplotlib there; the bundled
`requirements.txt` is an optional pinned starter, not an upgrade instruction for
an existing research environment.

The default illustrative CPU study runs 100 steps over one second, with 40 size
bins, a 100 nm mode radius, 1e8 particles/m³, 298.15 K, and 101325 Pa. It writes:

- `time_series.csv` and `distribution.csv`: numerical results.
- `result.png`: size distribution and bulk evolution.
- `run.json`: parameters, versions, source path/hash, and scientific checks.

Existing output directories are rejected. Finite-state and conservation failures
raise an error; do not interpret a failed run as a completed simulation. These
are illustrative parameters, not measured or calibrated research results.

## Compatibility and limitations

| Evidence | Status / boundary |
| --- | --- |
| Codex CLI 0.167.0; Claude Code 2.1.319 | Local marketplace/package installation tested in isolated configurations; see the PR for recorded commands |
| OpenCode 2.1.3 / SDK 2.1.3 | Original six-skill inventory discovered in a live isolated host from an extracted npm package; installed-package tests cover paths, reload, duplicate preservation, and cleanup |
| Python 3.12 / Particula 0.2.14 | Condensation numerical smoke and time-step comparison passed for the documented bounded case |
| Current repository Python package | CI tests the installed checkout with explicit numerical dependency versions |
| Desktop import / Cowork execution | Not verified by CLI/package tests; verify in the target app |
| Model-driven end-to-end conversations | Separate from installation tests; not claimed by the automated checks |
| GPU | Not qualified by CPU smoke tests |

The five repository skills are new package content; the historical host checks
above do not establish live discovery or model-driven execution of the expanded
catalog. Validate the updated package in the target host after installation.

These CLI versions are tested targets, not inferred minimum supported versions.
If an install command is absent, consult your host's `--help` and update it or
use its documented local-package flow.

A coagulation trial on Particula 0.2.14 with 1e12 particles/m³ over 600 seconds
and 600 steps failed mass conservation by approximately 1.3%. The cause is not
established. Condensation is therefore the introductory example. Do not loosen
the checks to make other workloads pass. See the
[compatibility record](https://github.com/uncscode/particula/blob/main/plugins/particula-helper/references/compatibility.md).

## Versions and downloads

[Particula releases](https://github.com/uncscode/particula/releases) carry
`particula-helper-codex.zip`, `particula-helper-claude.zip`,
`particula-helper-opencode.zip`, and `SHA256SUMS` starting with releases that
include this distribution workflow. Earlier releases will not have these files.
The OpenCode ZIP includes both a native package and manual-install skills.

Plugin versions are independent of Python library versions. Archives include
`BUILD.json` with the source commit, dirty-tree indicator, resource hashes,
and a pointer to compatibility evidence. Verify downloads with
`sha256sum -c SHA256SUMS` after downloading all three ZIPs.
Branch-based installs can change when refreshed. Use a published release tag
or full commit for a fixed revision; examples in the host guides use
`RELEASE_TAG` as a placeholder for an actual release containing these files.

## Troubleshooting

- **Import fails:** run the doctor with the same interpreter the simulation uses.
  Activating a terminal environment does not necessarily change an assistant's
  selected interpreter.
- **Skill missing:** check the host's plugin inventory and start a new session.
  In OpenCode, a project skill with the same ID takes precedence over the adapter.
- **Wrong version:** compare installed metadata with the source path/version
  reported by the doctor, especially for editable checkouts.
- **No local execution:** the assistant can explain or produce code, but must not
  claim it ran your local simulation. Desktop installation alone does not grant
  access to a local `.venv`.
- **Numerical check fails:** retain logs and parameters, and use particula-debug.
  Successful import or plotting does not establish scientific validity.
