---
name: particula-test
description: "Write or run Particula repository tests, numerical parity and conservation checks, lint/type checks, or release validation; use to choose the correct package, example, infrastructure, or GPU suite."
---

# Validate repository behavior

Read `../../references/repository-conventions.md` for the selected checkout and
`../../references/repository-validation.md` for suite boundaries, commands,
warning policy, scientific assertions and release checks.

Determine the behavior and evidence requested before selecting checks. Start
with relevant assertions, then run the broader suites warranted by the change
or requested gate. Distinguish source tests, installed-wheel checks, external
conda builds, documentation builds and performance qualification.

Use independent physical expectations and keep deterministic agreement,
conservation and stochastic criteria separate. Do not weaken tolerances,
warning filters or fixture workloads to obtain a pass. Preserve caller-owned
state and validate rejected/no-op paths when the contract requires it.

Use direct pytest/Ruff/mypy commands. Focused runs do not establish package-wide
coverage; the standard full coverage gate is 80%. CUDA skips are unavailable
CUDA evidence. Native-capture requirements cannot be met with CPU fallback.

Report exact commands and outcomes, meaningful skips and any blocked checks.
Do not describe a documented or planned command as executed. For a read-only
validation request, use Ruff `--no-fix` and do not rewrite source, saved example
outputs or golden fixtures to make tests pass.
