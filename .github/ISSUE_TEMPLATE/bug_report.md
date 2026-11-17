---
name: Bug or scientific concern
about: Report a suspected error or unexpected scientific result in Particula
title: ''
labels: ''
assignees: ''
---

A proven bug or fix is not required. Use a concise symptom-based title. Mark
unknown details explicitly and include only small, sanitized examples without
credentials, private research data or identifying local paths.

## Summary and impact

Describe the intended calculation, discrepancy and effect on the research.

## Reproduction status

State: reproduced, intermittent, or not yet reproduced. Say who reproduced it
and under which conditions; do not imply the agent ran a user-reported result.

## Environment

Record Particula and Python versions, installation source or commit, operating
system and relevant dependency versions. Include CPU/GPU backend, device, dtype
and driver/runtime details when relevant. Mark unavailable details as unknown.

## Minimal reproduction

Provide small runnable code, shareable or synthetic inputs, and the execution
command. Include relevant units, array shapes, process order, time steps and
random seeds. If reduction is incomplete, state the missing input or step.

## Expected behavior and basis

Explain the expected outcome and its basis: version-matched documentation,
equation, reference, independent calculation or explicit scientific assumption.
Distinguish an expectation from an established implementation contract.

## Observed behavior

Include actual output, the relevant traceback or a quantified discrepancy.
Label researcher-supplied evidence separately from agent-executed results.

## Investigation

List inspected source paths/symbols and version/commit links, relevant tests,
checks actually performed and their results. Link related issues or fixes and
state their applicability to the installed version. If duplicate search or
execution was unavailable, say not checked or not run.

## Scope and uncertainty

State known affected/unaffected cases, any suspected cause explicitly labeled
as a hypothesis, workarounds and remaining questions. Do not generalize beyond
the tested cases. Identify the next useful diagnostic step when unresolved.
