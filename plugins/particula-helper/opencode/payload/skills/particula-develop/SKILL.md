---
name: particula-develop
description: "Implement or refactor the Particula Python library, native data model, CPU processes, direct Warp kernels, or resident execution. Use for repository source changes, not for authoring a user simulation script."
---

# Develop the library

Read `../../references/repository-conventions.md` for checkout selection,
style, module boundaries and delivery. Before changing numerical behavior,
state ownership or execution, read `../../references/scientific-boundaries.md`
and the relevant maintained guide in the selected checkout.

Inspect the affected symbols and call paths, existing tests and public exports.
Identify the scientific invariant, current behavior and smallest coherent fix.
Preserve native population/species semantics and explicit transfer, RNG and
failure boundaries. Do not restore removed facades, introduce hidden fallback
or promote concrete-only APIs incidentally.

Use existing physical formulas, constants and validation patterns. Retain
sequential inventory coupling where required; vectorization must preserve the
model. Document units, shapes, assumptions and primary citations using
`../../references/docstrings.md` when adding or changing an API.

Validate behavior using `../../references/repository-validation.md`, with an
independent oracle and regression coverage where useful. Update public guides
and execute affected scientific examples in full when behavior changes. Report
what changed, the checks actually run, device skips and remaining limits.
Commit, push and publication follow the user's requested scope, not an implied
workflow phase.
