# Code Specifications

## 1. Title and Purpose

This document provides a concise overview of repository-wide coding and
documentation standards. It serves as the single entry-point for newcomers.
Older, more detailed specifications remain in `docs/contribute/Code_Specifications/`.

## 2. High-Level Principles

- Prioritize readability and maintainability.
- Ensure testability and type-safety.
- Write code and docs that are LLM-friendly (clear, explicit, consistent).
- Prefer explicitness over cleverness.

## 3. Naming Conventions

- **Functions:**  
  Use `get_<quantity>` or, for functions with internal state,  
  `get_<quantity>[_via_system_state]`.
- **Classes:**  
  Use `<Descriptor><PatternName>`, e.g., `TurbulentShearCoagulationStrategy`.
- **Constants:**  
  Use `ALL_CAPS`.
- **Private:**  
  Use `_leading_underscore` for private members.
- See [Naming Conventions Spec](docs/contribute/Code_Specifications/index.md)
  for details.

## 4. Style Guidelines

- When not otherwise specified, default to the
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Use `ruff` and `black` for linting and formatting if configured in the repo.

## 5. Docstring & Documentation Generation

- Use the templates in
  `docs/contribute/Code_Specifications/Function_docstring_format.md` and
  `Class_docstring_format.md`.
- Emphasize:
    - Unicode equations for mathematical expressions.
    - Parameter list format: `- parameter : Description`.
    - Include **Examples** and **References** sections.
    - Keep line length ≤ 79 characters.

## 6. AI Developer Workflow

### 6.1 Roles

- **Human Maintainer:**  
  Owns product vision, reviews code, and makes final decisions.
- **AI Assistant:**  
  Rapidly scaffolds code, refactors, and enriches docstrings.

### 6.2 Typical Loop

1. Human opens an issue or feature request.
2. AI proposes a patch (never edits read-only specs).
3. Human reviews and merges.
4. CI runs tests and doc build, then deploys.

### 6.3 Using Specification Files in LLM Prompts

- Always provide `Function_docstring_format.md` and
  `Class_docstring_format.md` when asking AI to write or modify docstrings.
- Ensure new code respects naming conventions so downstream LLM tools
  (auto-doc, code-review bots) can reason about function purpose from name alone.

### 6.4 Commit Message Template for AI

- Header line (≤ 50 chars).
- Body explaining WHAT/WHY, referencing spec section numbers.
- Footer (`#skip-ci`, etc.) if needed.

## 7. Human–AI Collaboration Tips

- Prefer descriptive variable names over comments.
- Let the AI generate the first draft; humans polish edge-cases.
- Ping the AI with failing test output to suggest fixes.

## 8. Folder Structure Quick Reference

- `particula/` — main source code
- `docs/` — documentation and specifications
- `tests/` — test suite
- (add other folders as needed)

## 9. Contribution Checklist

- [ ] Code passes `pytest`.
- [ ] `ruff --fix` run.
- [ ] Docstrings follow templates.
- [ ] Added/updated references.
- [ ] Updated `CHANGELOG.md` when applicable.

## 10. License & Credits

See `LICENSE` for terms.  
Credits to all contributors, human and AI.
