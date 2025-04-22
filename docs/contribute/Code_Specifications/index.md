# Code Specifications

This document provides a concise overview of repository-wide coding and
documentation standards. It serves as the single entry-point for newcomers and
quick reference for experienced contributors.

## Folder Structure Quick Reference

- `particula/` — main source code
  - `particula/...tests/` — tests are located as close to the relevant code as possible, in a `tests/` folder.
- `docs/` — documentation and specifications

A stable directory layout keeps import paths constant across refactors,
reducing churn in downstream notebooks and published papers.

## High-Level Principles

- Prioritize readability and maintainability.
- Ensure testability and type-safety.
- Write docstrings and examples that are LLM-friendly (clear, explicit, consistent).
- Prefer code explicitness over cleverness.

### Why these principles?

Particula’s goal is to **enable rapid, verifiable scientific iteration**.  
Readable code invites review; maintainable code survives graduate‑student
turnover; testable and type‑safe code lets us refactor with confidence.  
Finally, being LLM‑friendly acknowledges that many contributors—including the
project itself—will leverage AI tooling.  Clear, explicit patterns dramatically
improve the quality of the machine suggestions we receive.

### The WARMED Principles

| Letter | Focus | One-line guideline |
|--------|-------|--------------------|
| **W** | **Writing** | Write code that is direct, minimal, and fits the problem. |
| **A** | **Agreeing** | Discuss and settle on _how_ a feature is implemented **before** merging. |
| **R** | **Reading** | Code and variable names must explain themselves; comments/docstring fill the gaps, not the voids. |
| **M** | **Modifying** | Any competent dev should be able to extend or swap a component in minutes. |
| **E** | **Executing** | Favor vectorization and avoid hidden `for` loops. |
| **D** | **Debugging** | Fail fast with helpful messages and provide deterministic tests. |

These six commitments underpin every rule that follows.  Each subsequent
section calls out its relevant **WARMED** letter(s) so readers can see
how an individual guideline maps back to the overall developer experience.

## Naming Conventions
_Focus: **R / M** — descriptive names ease reading and future extension._

- **Functions:**  
  Use `get_<quantity>` or, for functions with that use system state [e.g., standard temperature and pressure], use 
  `get_<quantity>[_via_system_state]`.
- **Classes:**  
  Use `<Descriptor><PatternName>`, e.g., `TurbulentShearCoagulationStrategy`.
- **Constants:**  
  Use `ALL_CAPS`.
- **Private:**  
  Use `_leading_underscore` for private members.


_Rationale:_ Descriptive, structured names act as **self‑healing
documentation**.  Prefixes like `get_` signal a side‑effect‑free accessor,
while the `<Descriptor><PatternName>` template exposes the design pattern in
play (e.g., *Strategy*, *Builder*).  These conventions help human reviewers,
 and language models infer intent without digging into the
implementation.

## Docstring Style
_Focus: **R** — rich docstrings turn code into readable, searchable documentation._

- Use the templates specifications for:
  - Function docstrings: [Function_docstring_format](Details/Function_docstring_format.md).
  - Class docstrings: [Class_docstring_format](Details/Class_docstring_format.md).

- Emphasize:
    - Unicode equations for mathematical expressions.
    - Parameter list format: `- parameter : Description`.
    - Include **Examples** and **References** sections.
    - Keep line length ≤ 79 characters.

These templates are more than bureaucracy—they power the
auto‑documentation pipeline (code → Mkdocs website) and give language
models deterministic anchors when summarizing or refactoring code.

The templates are also used by LLMs in a AI-developer-workflow to generate/revise the docstrings.

## Code Style Guidelines
_Focus: **W / E** — minimal, consistent style and vector‑friendly patterns improve writing and execution._

- When not otherwise specified in templates, default to the
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Use `flake8` and `black` for linting and formatting.

Why automated style?  Tools such as `flake8` and `black` eliminate “formatting
debate” noise from reviews, keep diffs minimal, and help first‑time
contributors pass CI without memorizing an idiosyncratic style.

## Git Repository

The git repository follows a linear history model. For a clear and clean history.
See more at [Linear Git Repository](Details/Linear_Git_Repository.md)