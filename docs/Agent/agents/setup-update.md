# Setup/Update Agent - Usage Guide

## Overview

The `setup_update` agent automates ADW adoption and version updates by comprehensively analyzing repository conventions (testing, linting, code style, docs, architecture, commits, PRs), filling template guides in `docs/Agent/`, and presenting findings to users for confirmation before writing changes.

This agent bridges the gap between ADW's generic template system and repository-specific conventions, enabling ADW to work across Python, JavaScript, Rust, C++, and other language ecosystems.

## When to Use

- **Initial ADW Setup**: You're setting up ADW in a repository that has never used ADW before
  - Example: "Set up ADW in this new Python project"
  
- **ADW Version Updates**: You're updating existing `docs/Agent/` guides to match new ADW template versions
  - Example: "Update ADW guides from v2.0 to v2.1"
  
- **Convention Changes**: Repository conventions changed (e.g., switched from pytest to jest) and guides need updating
  - Example: "We migrated from eslint to biome, update the linting guide"
  
- **Guide Validation**: You want to detect inconsistencies between repository code and existing ADW guides
  - Example: "Check if our guides match the actual conventions in the codebase"

## Permissions

- **Mode**: `write`
- **Read Access**: 
  - All source code files (`.py`, `.js`, `.ts`, `.rs`, `.cpp`, etc.)
  - Configuration files (`pyproject.toml`, `package.json`, `Cargo.toml`, `.pre-commit-config.yaml`, etc.)
  - CI/CD workflows (`.github/workflows/`, etc.)
  - Existing documentation (`docs/Agent/`, `README.md`, `CONTRIBUTING.md`)
  - ADW templates (`adw/templates/Agent/`)
- **Write Access**: 
  - `docs/Agent/` - All guide files
  - Configuration files (`.pre-commit-config.yaml`, `pyproject.toml`, etc.) - **only after user confirmation**
  - `.env.example` - ADW environment variables template
- **Restrictions**: 
  - **Cannot write** to main source code files
  - **Cannot write** to test files
  - **Cannot write** to Git configuration or virtual environments

## Required Context Files

The agent reads from:
- **ADW Templates**: `adw/templates/Agent/` - Canonical guide templates with `{{PLACEHOLDER}}` syntax
- **Existing Guides** (Update mode): `docs/Agent/` - Current repository-specific guides
- **Repository Context**: Source files, configs, CI/CD workflows for convention detection

The agent writes to:
- **Active Guides**: `docs/Agent/` - Repository-specific guides filled with detected conventions

## Recommended Tools

- **File Reading Tools**: To analyze repository structure and conventions
- **Pattern Matching**: To detect conventions across 50+ source files per category
- **Diff Tools**: To show differences between existing guides and new templates (Update mode)
- **File Writing Tools**: To create/update guides and configuration files

## Usage Examples

### Example 1: Initial Setup in Python Repository

**Context**: New Python project using pytest, ruff, mypy, and Google-style docstrings. Need to set up ADW for the first time.

**Command**:
```
Set up ADW in this repository
```

**Expected Behavior**:

1. **Mode Detection**: Agent detects no `docs/Agent/` directory → Setup mode
2. **Repository Analysis** (20-30 minutes):
   - Analyzes 1,234 `.py` files
   - Detects pytest from `pyproject.toml`
   - Detects ruff + mypy from config and CI
   - Samples 50+ files for naming conventions
   - Detects Google-style docstrings
   - Analyzes 100 recent commits for format
3. **Present Findings**:
   ```
   ADW REPOSITORY ANALYSIS COMPLETE
   ════════════════════════════════════════════════════════
   Primary Language: Python
   Testing: pytest 7.4+ (HIGH confidence)
   Linting: ruff, mypy (HIGH confidence)
   Code Style: snake_case, PascalCase, UPPER_SNAKE_CASE (HIGH confidence)
   Docstrings: Google-style (HIGH confidence)
   Commits: Conventional Commits (HIGH confidence)
   
   No conflicts detected ✓
   ```
4. **Confirmation Questions**:
   ```
   Q1: Confirm pytest as test framework? [Y/n]: Y
   Q2: Confirm ruff + mypy as linters? [Y/n]: Y
   Q3: Confirm Google-style docstrings? [Y/n]: Y
   ...
   ```
5. **Create Guides**:
   - Copies templates from `adw/templates/Agent/` to `docs/Agent/`
   - Replaces `{{PACKAGE_NAME}}`, `{{TEST_FRAMEWORK}}`, etc. with detected values
   - Saves each guide incrementally:
     ```
     ✓ Created docs/Agent/testing_guide.md
     ✓ Created docs/Agent/linting_guide.md
     ✓ Created docs/Agent/code_style.md
     ...
     ```
6. **Update Configuration Files**:
   ```
   Create .pre-commit-config.yaml with ruff hooks? [Y/n]: Y
   ✓ Created .pre-commit-config.yaml
   
   Update pyproject.toml with ADW settings? [Y/n]: Y
   ✓ Updated pyproject.toml
   ```
7. **Validation**:
   ```
   VALIDATION COMPLETE
   ✓ All 9 required guides present
   ✓ No placeholders remaining
   ✓ All cross-references valid
   ```
8. **Summary**:
   ```
   ADW SETUP COMPLETE 🎉
   Guides Created: 9
   Config Files Updated: 3
   
   Next Steps:
   1. Review docs/Agent/ guides
   2. Commit changes
   3. Install pre-commit hooks
   4. Test ADW commands: /test, /lint, /commit
   ```

**Files Written**:
- `docs/Agent/testing_guide.md`
- `docs/Agent/linting_guide.md`
- `docs/Agent/code_style.md`
- `docs/Agent/docstring_guide.md`
- `docs/Agent/architecture_reference.md`
- `docs/Agent/review_guide.md`
- `docs/Agent/documentation_guide.md`
- `docs/Agent/commit_conventions.md`
- `docs/Agent/pr_conventions.md`
- `.pre-commit-config.yaml` (created)
- `pyproject.toml` (updated)
- `.env.example` (created)

### Example 2: Update Existing Guides to New ADW Version

**Context**: Repository currently uses ADW v2.0. Upgrading to ADW v2.1 which has updated template guides with new features and best practices.

**Command**:
```
Update ADW guides to version 2.1
```

**Expected Behavior**:

1. **Mode Detection**: Agent detects existing `docs/Agent/` directory → Update mode
2. **Repository Analysis**:
   - Re-analyzes repository conventions
   - Detects conventions unchanged from v2.0
   - Identifies version mismatch (current guides reference v2.0, templates are v2.1)
3. **Compare Guides**:
   - For each guide, shows unified diff:
     ```
     Differences in testing_guide.md:
     ════════════════════════════════════════════════════════
     --- Current: docs/Agent/testing_guide.md
     +++ Template: adw/templates/Agent/testing_guide.md
     
     @@ -10,7 +10,7 @@
      ## Test Framework
     
     -adw uses **pytest 7.4+** as the primary testing framework.
     +adw uses **pytest 8.0+** as the primary testing framework.
     
     @@ -150,6 +150,15 @@
      ## Coverage Requirements
     +
     +### New in v2.1: Coverage Enforcement
     +
     +ADW now enforces coverage thresholds in CI:
     +```bash
     +pytest --cov=adw --cov-fail-under=50
     +```
     
     ════════════════════════════════════════════════════════
     Apply changes to testing_guide.md? [Y/n/e]: _
     ```
4. **User Decisions**:
   - User reviews each diff
   - Applies 7 guides, skips 2 (custom sections user wants to preserve)
5. **Update Config Files**:
   ```
   Update .pre-commit-config.yaml to v2.1 hook versions? [Y/n]: Y
   ✓ Updated .pre-commit-config.yaml
   ```
6. **Summary**:
   ```
   ADW UPDATE COMPLETE 🎉
   Guides Updated: 7
   Guides Skipped: 2 (user preserved custom content)
   Config Files Updated: 1
   ```

**Files Updated**:
- 7 guides in `docs/Agent/` (updated to v2.1 templates)
- 2 guides unchanged (user skipped to preserve custom sections)
- `.pre-commit-config.yaml` (updated hook versions)

### Example 3: Repository with Conflicts (Multiple Test Frameworks)

**Context**: Python repository has both pytest configuration and unittest imports. Need to resolve which framework is primary.

**Command**:
```
Set up ADW in this repository
```

**Expected Behavior**:

1. **Mode Detection**: Setup mode
2. **Repository Analysis**:
   - Detects pytest in `pyproject.toml`
   - Detects unittest imports in 12 test files
   - **Flags conflict**: Multiple test frameworks
3. **Present Findings with Conflict**:
   ```
   CONFLICTS DETECTED
   ════════════════════════════════════════════════════════
   ⚠️ CONFLICT 1: Multiple test frameworks
      Evidence:
        - pytest found: pyproject.toml [tool.pytest.ini_options]
        - unittest found: 12 files import unittest
      
      Files using unittest:
        - tests/legacy/test_old_module.py
        - tests/legacy/test_deprecated.py
        ...
   
      → ACTION REQUIRED: Which framework is primary?
   ```
4. **Resolve Conflict**:
   ```
   Q: Multiple test frameworks detected
   
   Option A: pytest (recommended)
     - Configured in pyproject.toml
     - Modern features (fixtures, parametrize, plugins)
     - Matches ADW best practices
   
   Option B: unittest
     - Used in 12 test files (likely legacy)
     - Standard library, no external dependencies
   
   Option C: Both (multi-framework setup)
     - Support both frameworks
     - Document transition plan in guide
   
   Select option [A/B/C]: A
   ```
5. **User Selects Option A** (pytest)
6. **Create Guides**:
   - Uses pytest as primary framework
   - Notes unittest usage in testing_guide.md:
     ```markdown
     ## Legacy Tests
     
     Some test files still use unittest (legacy):
     - tests/legacy/test_old_module.py
     - tests/legacy/test_deprecated.py
     
     These should be migrated to pytest over time.
     ```
7. **Summary**: Setup complete with pytest as primary, unittest noted as legacy

**Result**: Conflict resolved by user input, guides accurately reflect repository state with migration note.

### Example 4: JavaScript Repository with TypeScript

**Context**: JavaScript/TypeScript project using jest, eslint, prettier, and JSDoc. Setting up ADW.

**Command**:
```
Set up ADW in this repository
```

**Expected Behavior**:

1. **Language Detection**:
   ```
   Language Analysis:
   ══════════════════════════════════════════════════════
   Primary Language: JavaScript/TypeScript (845 .ts/.js files)
   Secondary Languages: None
   
   Using PRIMARY LANGUAGE (JavaScript/TypeScript) for guides.
   ```
2. **Repository Analysis**:
   - Detects jest from `package.json` devDependencies
   - Detects eslint + prettier from configs
   - Detects JSDoc style from `/** ... */` comments
   - Detects Conventional Commits from git history
3. **Confirmation** (all HIGH confidence):
   ```
   Q1: Confirm jest as test framework? [Y/n]: Y
   Q2: Confirm eslint + prettier as linters? [Y/n]: Y
   Q3: Confirm JSDoc documentation style? [Y/n]: Y
   Q4: Confirm Conventional Commits format? [Y/n]: Y
   ```
4. **Create Guides**:
   - Fills templates with JavaScript-specific values:
     - `{{TEST_FRAMEWORK}}` → jest
     - `{{TEST_COMMAND}}` → npm test
     - `{{TEST_FILE_PATTERN}}` → *.test.ts
     - `{{LINTER_TOOLS}}` → eslint, prettier
     - `{{DOCSTRING_STYLE}}` → JSDoc
   - Removes Python/Rust/C++ examples from templates
5. **Config Updates**:
   ```
   Create .pre-commit-config.yaml with eslint/prettier hooks? [Y/n]: Y
   ✓ Created .pre-commit-config.yaml
   
   Update package.json with ADW scripts? [Y/n]: Y
   ✓ Updated package.json
   ```
6. **Summary**: 9 guides created for JavaScript/TypeScript project

**Files Written**:
- `docs/Agent/` guides filled with JavaScript/TypeScript conventions
- `.pre-commit-config.yaml` with eslint/prettier hooks
- `package.json` updated with ADW scripts
- `.env.example` created

## Best Practices

1. **Trust the Analysis**: The agent samples 50+ files per category for accurate detection. High confidence detections are usually correct.

2. **Review Diffs Carefully** (Update mode): When updating guides, review diffs to ensure custom sections aren't lost. Use `e` (edit) option for complex changes.

3. **Resolve Conflicts Thoughtfully**: When conflicts are detected, consider:
   - Which convention is more modern/widely used?
   - Which aligns with team preferences?
   - Can legacy conventions be migrated over time?

4. **Confirm Config Changes**: Always review configuration file changes before approving. Config changes affect the entire team.

5. **Run Validation**: After setup/update, run validation to catch any missed placeholders or broken references.

6. **Commit Incrementally**: If setup is interrupted, partial progress is saved (incremental saves per guide). You can resume later.

7. **Test ADW Commands**: After setup, test that ADW commands work correctly:
   ```bash
   /test     # Should run detected test framework
   /lint     # Should run detected linters
   /commit   # Should use detected commit format
   ```

8. **Iterate if Needed**: If conventions change later, re-run the agent in Update mode to sync guides.

## Limitations

- **Primary Language Only**: For multi-language repositories, the agent focuses on the primary language (most files). Secondary languages are ignored.

- **Manual Validation**: The `/validate-adw` command doesn't exist yet. Validation is currently manual (checking for placeholders, cross-references).

- **Convention Detection Accuracy**: While the agent samples 50+ files, edge cases or uncommon conventions might be misdetected. Always review findings.

- **Custom Sections Lost in Updates**: In Update mode, custom sections in guides may be lost if not marked with special comments. Use `e` (edit) option to preserve custom content.

- **Config File Safety**: Config file updates are conservative and always ask for confirmation, but review changes carefully to avoid breaking builds.

- **No Rollback**: Once guides are written, there's no automatic rollback. Use git to revert changes if needed.

## Integration with Other Agents

- **Validation Command** (future): After running `setup_update`, run `/validate-adw` to verify guides are complete and valid.

- **Sync Command** (`/sync`): While `setup_update` fills `docs/Agent/` guides, the `/sync` command keeps `.opencode/command/` command files in sync with `adw/commands/` templates. Run both for complete ADW setup.

- **Implementor Agent**: After setup, the implementor agent uses `docs/Agent/` guides to understand repository conventions when implementing features.

- **Review Agent**: Review agent references `docs/Agent/review_guide.md` filled by `setup_update` to enforce quality standards.

## Troubleshooting

### Issue: Conflicting Conventions Detected

**Symptom**: Agent reports multiple frameworks or inconsistent conventions.

**Cause**: Repository uses multiple tools (e.g., both pytest and unittest) or has inconsistent patterns.

**Solution**:
- Review the evidence presented (file paths, config snippets)
- Select the preferred convention using provided options (A/B/C)
- Add migration notes in guides if transitioning from legacy tools

**Example**:
```
⚠️ CONFLICT: Multiple linters detected
   - ruff found in pyproject.toml
   - flake8 found in .flake8 config
   → Select primary linter: [A: ruff / B: flake8]: A
```

### Issue: Low Confidence Detections

**Symptom**: Agent marks detections as LOW confidence or can't detect conventions.

**Cause**: Repository lacks configuration files, uses uncommon patterns, or has inconsistent code.

**Solution**:
- Agent will ask detailed questions for LOW confidence detections
- Provide answers based on team conventions
- Consider adding configuration files to make conventions explicit (e.g., `pyproject.toml` for Python)

**Example**:
```
Q: Docstring style (LOW confidence - inconsistent patterns)
   Detected: Mix of Google-style (45%) and NumPy-style (35%)
   Preferred docstring style: [Google/NumPy/reST/Custom]: Google
```

### Issue: Template Placeholders Remain After Fill

**Symptom**: Validation reports `{{PLACEHOLDER}}` syntax still exists in guides.

**Cause**: Placeholder wasn't replaced during fill, or custom placeholder was added manually.

**Solution**:
- Re-run the agent to fill remaining placeholders
- Or manually replace placeholders:
  ```bash
  grep -r "{{" docs/Agent/
  # Find remaining placeholders
  # Edit files to replace with appropriate values
  ```

**Example**:
```
VALIDATION FAILED
✗ Placeholders remaining:
   - docs/Agent/architecture_reference.md:15 - {{ARCHITECTURE_DOC_LINK}}
   
→ Please provide value for {{ARCHITECTURE_DOC_LINK}}: docs/design/system-architecture.md
```

### Issue: User Rejects All Changes

**Symptom**: User says "no" to all confirmation questions.

**Cause**: Detected conventions are incorrect, or user doesn't want automated setup.

**Solution**:
- Agent asks if you want to abort: "Abort setup? [Y/n]"
- If abort: Exit cleanly without writing files
- If continue: Agent asks user to manually specify conventions
- Or: Re-run agent with hints (manually edit configs first to guide detection)

**Example**:
```
All detections rejected. 
Abort setup? [Y/n]: n

Manual specification mode activated.
Please provide values for each guide section...

Q1: Test framework: jest
Q2: Test command: npm test
...
```

### Issue: Diff Shows Too Many Changes (Update Mode)

**Symptom**: In Update mode, diffs for guides show extensive changes, hard to review.

**Cause**: New template version has significant restructuring or additions.

**Solution**:
- Use `e` (edit) option to see full content and manually merge changes
- Or: Skip update for that guide, manually update later
- Or: Review diff in external tool:
  ```bash
  diff docs/Agent/testing_guide.md adw/templates/Agent/testing_guide.md
  ```

**Example**:
```
Differences in testing_guide.md:
════════════════════════════════════════════════════════
[300 lines of diff...]

Apply changes? [Y/n/e]: e

[Opens full file in editor for manual merging]
```

### Issue: Configuration File Update Breaks Build

**Symptom**: After config update, CI fails or tests don't run.

**Cause**: Config change incompatible with existing setup or introduced syntax error.

**Solution**:
- Revert config change: `git checkout HEAD -- <config-file>`
- Review the diff that was applied
- Manually apply changes incrementally and test after each
- Or: Skip config updates, manually update later after testing

**Example**:
```
Updated .pre-commit-config.yaml
→ CI fails with "unknown hook: ruff-pre-commit"

Solution:
git checkout HEAD -- .pre-commit-config.yaml
# Review diff, identify issue (outdated hook repo)
# Manually fix and test
```

## See Also

- **ADW Template System**: `adw/templates/Agent/` - Canonical guide templates with placeholders
- **Template Customization Guide**: `adw/templates/Agent/README.md` - How templates work, placeholder reference
- **Guide Documentation**: `docs/Agent/README.md` - Overview of all guides (created by setup_update agent)
- **ADW README**: `README.md` - Complete ADW system documentation including adoption guide
- **Sync Command**: `.opencode/command/sync.md` - Command for syncing ADW commands (similar pattern)
- **Validation Command**: Future feature for validating guide completeness

## Feedback and Issues

If you encounter issues with the `setup_update` agent or have suggestions for improving convention detection:

1. Check troubleshooting section above
2. Review detected evidence (file paths, config snippets) for accuracy
3. Open an issue with:
   - Repository language/framework
   - Detected conventions (from analysis report)
   - Expected conventions
   - Evidence files (configs, sample source files)
4. Consider contributing detection improvements for uncommon frameworks

The `setup_update` agent is designed to handle common patterns in Python, JavaScript, Rust, and C++ projects. For other languages or unconventional setups, manual guide customization may be needed.
