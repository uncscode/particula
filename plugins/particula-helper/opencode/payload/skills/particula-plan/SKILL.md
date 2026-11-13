---
name: particula-plan
description: "Plan Particula repository features, maintenance, or research when a development plan or issue scope is requested; define code boundaries, scientific acceptance criteria, dependencies, and evidence. Not for designing an aerosol simulation study."
---

# Plan a repository change

Read `../../references/repository-conventions.md` to select the checkout and
its current instructions. Read `../../references/scientific-boundaries.md`
when the proposal touches physical state, process semantics or GPU execution.

Establish the requested outcome, current behavior, in/out of scope, relevant
constraints and scientific unknowns. Inspect the owning symbols, adjacent tests,
maintained guides and existing issue/plan before recommending new architecture.
Separate observed facts from assumptions and research questions.

Produce a plan proportional to the change. For substantial work, identify
concrete code boundaries, dependencies and independently reviewable steps. Give
each step an observable completion criterion and appropriate checks from
`../../references/repository-validation.md`. Split by coherent behavior or
uncertainty, not fixed line counts, mandatory phase counts or obsolete phase IDs.
Research steps should identify the question, evidence to obtain and decision
that evidence enables; never invent a literature result or numerical oracle.

Include native data units, species/volume semantics, public versus concrete-only
imports, failure boundaries, CPU/GPU coverage, documentation and release effects
where they actually matter. Preserve numerical and qualification gates.

Return the plan in the requested place or chat. If a repository artifact is
requested without a destination, use a descriptive file under `plans/`; public
roadmap entries contain pending outcomes and durable scientific evidence belongs
in `docs/contribute/validation/`. Do not require a workflow engine, generate
agent-state files, assign automation labels, post issues or launch implementation
unless requested. When planning is part of an implementation request, continue
within the user's authorized scope rather than stopping at the plan.
