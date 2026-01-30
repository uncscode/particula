# Maintenance M3: Jupytext Notebook Sync Migration (Pilot)

**Status**: Planning
**Priority**: P2
**Owners**: ADW / Maintainers
**Target Date**: 2026-Q1
**Last Updated**: 2026-01-30
**Size**: Small (4 notebooks, 4 phases)

## Vision

**Pilot migration** of `Activity/` and `Gas_Phase/` notebooks to validate the
Jupytext paired sync workflow before migrating remaining directories.

Benefits of paired sync:

1. **LLM-friendly editing**: Edit plain `.py` files instead of JSON notebooks
2. **Linting support**: Run ruff/mypy on example code
3. **Clean diffs**: Readable PR diffs instead of JSON noise
4. **Merge conflict resolution**: Standard git merge on `.py` files
5. **Type checking**: Enable mypy on documentation examples

The paired sync approach keeps both `.ipynb` (for users/MkDocs) and `.py:percent`
(for development/linting) in sync, with the newer file winning on sync.

## Current State Analysis (Pilot Scope)

| Directory | Notebooks | Already Paired | To Convert |
|-----------|-----------|----------------|------------|
| Activity | 1 | 0 | 1 |
| Gas_Phase | 3 | 0 | 3 |
| **Total** | **4** | **0** | **4** |

### Notebooks to Convert

```
docs/Examples/
├── Activity/
│   └── activity_tutorial.ipynb          # To convert
└── Gas_Phase/Notebooks/
    ├── AtmosphereTutorial.ipynb         # To convert
    ├── Gas_Species.ipynb                # To convert
    └── Vapor_Pressure.ipynb             # To convert
```

### Note: Orphaned `.py` Files in Activity/

Activity/ has 4 `.py` files without matching `.ipynb`:
- `bat_activity_example.py`
- `equilibria_example.py`
- `ideal_activity_example.py`
- `kappa_activity_example.py`

These appear to be standalone scripts, not Jupytext pairs. They will be linted
but not synced.

## Scope

### In Scope (Pilot)

- `docs/Examples/Activity/activity_tutorial.ipynb`
- `docs/Examples/Gas_Phase/Notebooks/*.ipynb` (3 notebooks)
- Configuration files: `jupytext.toml`, `pyproject.toml`, `mkdocs.yml`, `.gitignore`
- Documentation updates for the workflow

### Out of Scope (Future M4)

- All other notebooks in `docs/Examples/` (~35 remaining)
- Pre-commit hook implementation
- CI/CD pipeline changes
- Notebook execution validation

## Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| `validate_notebook` tool | Ready | Already supports `--sync`, `--convert-to-py`, `--check-sync` |
| `run_notebook` tool | Ready | Executes notebooks for validation |
| `jupytext` package | Available | Used by ADW tools internally |

## Phase Checklist

### Phase 1: Configuration Files (`M3-P1`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

- [ ] **M3-P1-1:** Create `jupytext.toml` with pairing rules for `docs/Examples/`
- [ ] **M3-P1-2:** Update `pyproject.toml` - add `docs/Examples/**/*.py` to ruff config
- [ ] **M3-P1-3:** Update `mkdocs.yml` to exclude `.py` files from documentation build
- [ ] **M3-P1-4:** Add `*.ipynb.bak` to `.gitignore`

**Acceptance Criteria:**
- `jupytext.toml` exists with correct pairing rules
- `ruff check docs/Examples/` can lint `.py` files
- `mkdocs build` excludes `.py` files from navigation
- Backup files are git-ignored

### Phase 2: Convert Pilot Notebooks (`M3-P2`)

**Issue:** TBD | **Size:** S | **Status:** Not Started

- [ ] **M3-P2-1:** Convert `docs/Examples/Activity/activity_tutorial.ipynb` to paired format
- [ ] **M3-P2-2:** Convert `docs/Examples/Gas_Phase/Notebooks/*.ipynb` (3 notebooks)
- [ ] **M3-P2-3:** Run `ruff check` on new `.py` files and fix linting issues
- [ ] **M3-P2-4:** Validate all conversions with `--check-sync`

**Acceptance Criteria:**
- All 4 pilot notebooks have corresponding `.py:percent` files
- `validate_notebook docs/Examples/Activity --recursive --check-sync` passes
- `validate_notebook docs/Examples/Gas_Phase --recursive --check-sync` passes
- `ruff check docs/Examples/Activity docs/Examples/Gas_Phase` passes

### Phase 3: Workflow Validation (`M3-P3`)

**Issue:** TBD | **Size:** XS | **Status:** Not Started

- [ ] **M3-P3-1:** Test edit workflow: edit `.py` → sync → validate notebook runs
- [ ] **M3-P3-2:** Execute converted notebooks to verify they still work
- [ ] **M3-P3-3:** Document any issues or lessons learned

**Acceptance Criteria:**
- End-to-end workflow tested on at least one notebook
- All 4 converted notebooks execute successfully via `run_notebook`
- Issues documented for future phases

### Phase 4: Documentation Update (`M3-P4`)

**Issue:** TBD | **Size:** XS | **Status:** Not Started

- [ ] **M3-P4-1:** Update `adw-docs/documentation_guide.md` with Jupytext workflow
- [ ] **M3-P4-2:** Update `AGENTS.md` with notebook editing guidance
- [ ] **M3-P4-3:** Update this plan with completion notes and lessons learned
- [ ] **M3-P4-4:** Create follow-up issue M4 for remaining ~35 notebooks

**Acceptance Criteria:**
- Documentation guide includes complete LLM editing workflow
- ADW agents have clear guidance on notebook editing
- Plan status updated to Shipped
- M4 issue created for remaining notebook migration

## Critical Testing Requirements

- **No Coverage Modifications**: This maintenance task doesn't affect code coverage
- **Notebook Validation**: Phase 2 validates with `--check-sync`
- **Execution Testing**: Phase 3 executes all 4 notebooks via `run_notebook`
- **Linting Compliance**: Phase 2 runs `ruff check` and fixes issues before completion

## Testing Strategy

### Validation Commands (Pilot Scope)

```bash
# Convert notebook to .py:percent
validate_notebook docs/Examples/Activity/activity_tutorial.ipynb --convert-to-py
validate_notebook docs/Examples/Gas_Phase/Notebooks --recursive --convert-to-py

# Check sync status (read-only, CI-friendly)
validate_notebook docs/Examples/Activity --recursive --check-sync
validate_notebook docs/Examples/Gas_Phase --recursive --check-sync

# Lint example Python files
ruff check docs/Examples/Activity docs/Examples/Gas_Phase --fix
ruff format docs/Examples/Activity docs/Examples/Gas_Phase

# Execute notebooks to verify they still work
run_notebook docs/Examples/Activity/activity_tutorial.ipynb
run_notebook docs/Examples/Gas_Phase/Notebooks --recursive
```

### ADW Tool Usage

```python
# Convert to .py:percent
validate_notebook({
    "notebookPath": "docs/Examples/Activity/activity_tutorial.ipynb",
    "convertToPy": True
})

# Check sync status
validate_notebook({
    "notebookPath": "docs/Examples/Activity",
    "recursive": True,
    "checkSync": True
})

# Execute notebook
run_notebook({
    "notebookPath": "docs/Examples/Activity/activity_tutorial.ipynb"
})
```

## Configuration Details

### jupytext.toml (to be created)

```toml
# Jupytext configuration for particula
# Pairs notebooks in docs/Examples with percent-format Python scripts

[formats]
"docs/Examples/**/*.ipynb" = "ipynb,py:percent"
```

### pyproject.toml Updates

```toml
[tool.ruff]
# Current config (line 60-64):
# src = ["particula"]
# include = ["particula/**/*.py"]
# extend-exclude = ["**/*.ipynb"]

# UPDATE: Add docs/Examples to src for ruff to lint
src = ["particula", "docs/Examples"]
include = ["particula/**/*.py", "docs/Examples/**/*.py"]

# Keep .ipynb exclusion (unchanged)
extend-exclude = [
  "**/*.ipynb",    # ignore every .ipynb anywhere in the project
]

# ADD: Per-file ignores for example scripts (less strict than library code)
[tool.ruff.lint.per-file-ignores]
"*_test.py" = ["S101", "E721", "B008"]
"docs/Examples/**/*.py" = ["D100", "D103", "INP001"]  # Allow missing docstrings in examples
```

**Note:** The `INP001` ignore allows example scripts without `__init__.py` (implicit namespace packages).

### mkdocs.yml Updates

```yaml
exclude_docs: |
  .assets/
  Examples/**/*.py  # Hide synced .py files from docs build
```

### .gitignore Updates

```gitignore
# Jupyter notebook backups (created by run_notebook tool)
*.ipynb.bak
```

Add this after the existing Jupyter Notebook section (line 102-103).

## LLM Development Workflow

After migration, LLMs should follow this workflow for notebook edits:

```
1. Edit .py file (percent format)
   - Plain text, lintable, easy diffs

2. Lint the .py file
   - ruff check docs/Examples/path/to/file.py --fix
   - ruff format docs/Examples/path/to/file.py

3. Sync to regenerate .ipynb
   - validate_notebook({notebookPath: '...ipynb', sync: true})

4. Execute .ipynb to validate and generate outputs
   - run_notebook({notebookPath: '...ipynb'})

5. Commit both files (.py and .ipynb)
```

**Critical**: Sync before execute! Otherwise you're testing the OLD code.

## Future Work

See [M4: Jupytext Full Migration](M4-jupytext-full-migration.md) for:
- Converting remaining ~35 notebooks
- Pre-commit hook for automatic sync+execute
- CI validation for sync status
- Long-running simulation notebook handling

M4 is blocked until this pilot (M3) completes successfully.

## Related Documents

- [Documentation Guide](../../documentation_guide.md) - Notebook guidelines
- [Linting Guide](../../linting_guide.md) - Ruff configuration
- [AGENTS.md](../../../AGENTS.md) - ADW agent reference

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-30 | Initial plan created | ADW |
| 2026-01-30 | Added notebook execution workflow and pre-commit hook design | ADW |
| 2026-01-30 | Clarified workflow order: lint → sync → execute (critical for correctness) | ADW |
| 2026-01-30 | Split into 8 smaller phases with size estimates and ID prefixes | ADW |
| 2026-01-30 | Fixed pyproject.toml config, added .gitignore entry, added final docs phase | ADW |
| 2026-01-30 | Reduced scope to pilot: Activity + Gas_Phase only (4 notebooks, 4 phases) | ADW |
