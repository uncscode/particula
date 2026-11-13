---
name: particula-review
description: "Review Particula repository changes or prepare requested commits and PR descriptions, prioritizing scientific correctness, native API contracts, validation, and evidence. Use for review/publication preparation rather than routine simulation analysis."
---

# Review a repository change

Read `../../references/repository-conventions.md` for checkout and delivery
rules, `../../references/scientific-boundaries.md` for scientific/API boundaries,
and `../../references/repository-validation.md` for applicable evidence.

Inspect the actual diff and owning source, including callers, affected tests,
exports and documentation. Prioritize concrete regressions: incorrect equations
or units, count/volume/species normalization, broken state ownership, invalid
no-op/rejection behavior, RNG continuity, and lifecycle races. Check geometry
and distribution paths for wall loss and separate inventory checks for coupled
particle/gas updates. Examine complexity and memory only where relevant; preserve
intentional sequential stepping.

Verify tests use independent expectations and meaningful tolerances; large
aggregate reservoirs must not hide component errors. Distinguish Warp CPU,
optional CUDA, native-capture evidence and recorded historical measurements.
Review changed example code and saved results together. Prose-only edits need
not trigger every scientific workload.

Use read-only checks for a review request: Ruff `--no-fix` and format `--check`.
Do not silently repair the branch or refresh evidence unless fixes are requested.
Return actionable findings with affected file/line, trigger, consequence and
supporting evidence. State when no actionable issue was found and identify
material validation limits. Do not manufacture findings to fill a checklist.

When the user requests a commit or PR, inspect the final intended diff, use
Conventional Commit wording and the repository PR template, and state the
concrete problem/result and actual validation. Do not automatically push,
post review comments, mark external threads resolved, merge, or publish solely
because this skill was selected.
