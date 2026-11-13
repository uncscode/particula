# Working on the Particula repository

Use this reference for library contributions, not for running a user's aerosol
study. Paths below resolve inside the user's selected Particula checkout.
Bundled plugin files are guidance, not an implicit checkout or output folder.

## Authority and environment

Read that checkout's `AGENTS.md`, `pyproject.toml`, and relevant CI configuration.
Current user instructions and checkout-specific instructions take precedence
over this portable guidance. Inspect the actual owning implementation when a
guide and code disagree; report the discrepancy rather than silently changing
physics to fit a stale description. Historical records are evidence for their
recorded revision, not current requirements.

Use the selected interpreter; prefer an existing `.venv` when appropriate.
Development installation is `python -m pip install -e '.[dev,extra]'` when setup
is requested. Never replace an existing research environment to satisfy this
skill. Check the active branch and worktree and preserve unrelated changes.
Use CodeGraph first for code navigation when the checkout provides it; otherwise
use ordinary source search. No retired agent workflow tools are required.

The 0.3.0 source tree requires Python 3.12+. `particula/__init__.py` owns
`__version__`; Flit reads it dynamically. Plugin versions are independent.
Neither changing a version nor passing tests publishes a release.

For library development, read the selected checkout's maintained `docs/` tree.
For questions about an installed release, locate its matching bundle with that
interpreter's `python -m particula_docs --json`, then search the returned path;
see [source discovery](sources.md) for commands and older-release fallback.
An editable installation resolves directly to repository docs. Never edit the
generated installed bundle or use it in place of the checkout's build inputs.

## Package boundaries

- `particles/`: native particle data, distributions, slots, properties.
- `gas/`: gas/environment data, vapor pressure and thermodynamic strategies.
- `dynamics/`: physical process strategies, builders and runnables.
- `activity/`, `equilibria/`: activity and phase partitioning.
- `execution/`: execution metadata and concrete resident lifecycle.
- `gpu/`: Warp mirrors, explicit transfers and direct kernels.
- `util/`: constants, units and input validation.

Reuse the established strategy/builder/factory/runnable pattern when the
owning process uses it; do not introduce a layer just to match a pattern.
Separate physical formulas, state ownership and orchestration. Inspect
`__init__.py` exports before adding public imports. Concrete-only records and
helpers are intentional boundaries, including most resident lifecycle APIs.

Before changing physical state or execution, read
[scientific boundaries](scientific-boundaries.md). The checkout's maintained
`docs/guide/data-model/`, `docs/guide/processes/`, and `docs/guide/gpu/` contain
the detailed contracts; `docs/contribute/architecture/` explains design choices.

## Scientific Python style

Use four spaces, double quotes, 80-character lines and absolute package imports.
Order imports standard library, third party, then local; let Ruff format them.
Use `snake_case`, `PascalCase`, and `UPPER_CASE` for functions/variables, classes,
and constants. Prefer meaningful physical names over ambiguous abbreviations.
Use modern Python annotations and `NDArray[np.float64]` where dtype matters;
use `TYPE_CHECKING` to avoid unnecessary runtime dependencies and import cycles.
For Warp runtime types, follow the owning module's established annotations and
validation; avoid using `isinstance(value, wp.array)` as a generic type check.

Use `particula.util.constants` rather than duplicating physical constants.
Validate public numerical domains with existing helpers when suitable. Handle
finite values, shapes, units and scalar/array behavior deliberately. Vectorize
large array operations where the algorithm allows it; sequential staggered
Gauss-Seidel updates must retain their inventory coupling.

Explain equations and assumptions with primary citations. Comments should
explain physical meaning, units or why an operation is necessary. Keep public
Google-style docstrings synchronized with actual behavior; use
[docstring guidance](docstrings.md) for function/class edits.

## Scope and delivery

Implement the smallest coherent change. Do not weaken numerical assertions,
add hidden fallback, or broaden an API to make a test/example pass. Validate
using [repository validation](repository-validation.md), scaling checks to the
change. Update supported usage and example outputs when behavior changes.

For requested commits use a concise Conventional Commit title:
`<type>(<optional scope>): <imperative summary>`. Common types are `feat`, `fix`,
`test`, `docs`, `refactor`, `perf`, and `chore`. Commit only intended files and
inspect hook edits. Do not add workflow IDs or mandatory generated trailers.
For requested PRs, use the repository template and describe the concrete
problem, resulting behavior, validation and remaining limitations. A review or
planning request alone does not authorize posting comments, opening issues,
pushing, merging or publishing.
