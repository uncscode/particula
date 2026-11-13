# Build documentation

`build_docs` with `action=build` prepares all Markdown/Python pages, builds
Starlight, indexes search and verifies routes, fragments, assets and API pages.
`action=check` checks source ownership, registry coverage and navigation.
Pass `cwd` to target a Particula worktree within the current workspace. Start
the agent in an external worktree to authorize that workspace instead. Empty,
external (including symlink escapes), or invalid roots fail
before launching a subprocess. Neither action executes scientific examples or updates saved results.

Install dependencies as described in `website/README.md`. Run from the
repository root. Command-line equivalents are `npm --prefix website run build`
and `python scripts/check_docs_structure.py`.
