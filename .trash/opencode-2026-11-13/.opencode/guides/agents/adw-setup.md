# ADW Setup Agent - Usage Guide

> **DEPRECATED**: This agent has been split into two focused agents:
> - **[setup-adw](setup-adw.md)**: For setting up ADW in target repositories
> - **[adw-setup-maintainer](adw-setup-maintainer.md)**: For maintaining templates in the ADW repository
>
> This documentation is retained for reference. Use the specialized agents above.

## Overview

The `adw-setup` agent automates ADW configuration in repositories where ADW is installed as a Python package/CLI tool. It handles both fresh setups and updates after ADW package upgrades, customizing `.opencode/` configuration and `adw-docs/` guides for the specific repository's language, tooling, and conventions.

**Note**: This agent used deprecated `adw sync` commands. The new agents use:
- `adw setup env` - Interactive environment wizard
- `adw setup template init/apply/extract/validate` - Template management

## Migration Guide

| Old Command | New Command |
|-------------|-------------|
| `adw sync all` | `adw setup template apply` |
| `adw sync all --dry-run` | `adw setup template apply --dry-run` |
| `adw sync check` | `adw setup template validate` |

| Old Agent | New Agent | Use Case |
|-----------|-----------|----------|
| `adw-setup` | `setup-adw` | Setting up ADW in target repos |
| `adw-setup` | `adw-setup-maintainer` | Maintaining templates in ADW repo |

## When to Use (Historical)

- **Fresh ADW Setup**: Setting up ADW in a repository for the first time
  - **Now use**: `setup-adw`
  
- **ADW Package Updates**: Checking for configuration updates after upgrading
  - **Now use**: `setup-adw`
  
- **Template Maintenance**: Extracting docs changes to templates
  - **Now use**: `adw-setup-maintainer`

## Permissions

- **Mode**: `all` (full access)
- **Read Access**: 
  - All source code files for language detection
  - Configuration files (`pyproject.toml`, `Cargo.toml`, `package.json`, etc.)
  - CI/CD workflows (`.github/workflows/`)
  - Existing `.opencode/` and `adw-docs/` files
  - ADW templates (via `adw sync` command)
  
- **Write Access**: 
  - `.opencode/` - All agent, command, tool, workflow, and plugin files
  - `adw-docs/` - All guide files
  - `AGENTS.md` - Root agent reference file
  
- **Tool Access**:
  - `adw` CLI - sync, health, status, workflow commands
  - `bash` - File operations, git commands, ripgrep searches
  - `get_version` - Version detection
  - `get_datetime` - Timestamp generation (UTC or America/Denver via `localtime`)

- **Restrictions**: 
  - Does NOT modify source code files
  - Does NOT modify test files
  - Asks for confirmation before overwriting existing customized files

## Required Context Files

The agent reads from:
- **ADW Templates**: Accessed via `adw sync all` command
- **Project Configs**: `pyproject.toml`, `Cargo.toml`, `package.json`, etc.
- **Existing Configuration**: `.opencode/`, `adw-docs/` if present
- **Git Information**: Remote URL, branch names

The agent writes to:
- **OpenCode Config**: `.opencode/agent/`, `.opencode/command/`, `.opencode/workflow/`, etc.
- **Documentation**: `adw-docs/*.md` guide files
- **Root Reference**: `AGENTS.md`

## Recommended Tools

- **`adw` CLI**: Primary tool for syncing templates and validation
  - `adw sync all` - Copy templates to `.opencode/`
  - `adw sync all --dry-run` - Check for template changes without applying
  - `adw health` - Validate ADW configuration
  - `adw workflow list` - Verify workflows load correctly
  
- **`rg` (ripgrep)**: Search for placeholders and path references
  - Find unfilled `{{PLACEHOLDER}}` values
  - Locate path references that need updating
  
- **`get_version`**: Extract version from project config files
- **`get_datetime`**: Generate current date for `{{LAST_UPDATED}}` placeholders (UTC default; use `localtime` for America/Denver)

## Usage Examples

### Example 1: Fresh Python Repository Setup

**Context**: New Python project using pytest, ruff, and mypy. ADW installed via pip, need to configure for first use.

**Invocation**:
```
Help me set up ADW in this repository
```

**Expected Behavior**:

1. **Analysis Phase**:
   ```
   Analyzing repository...
   
   Detected:
   - Language: Python (245 .py files)
   - Package Manager: pip (pyproject.toml found)
   - Source Directory: src/mypackage/
   - Test Framework: pytest (configured in pyproject.toml)
   - Linting: ruff, mypy
   - Package Name: mypackage
   ```

2. **Confirmation**:
   ```
   Configuration looks good:
   - Source directory: src/mypackage/ [Y/n]: Y
   - Test command: pytest src/ [Y/n]: Y  
   - Lint command: ruff check src/ && mypy src/ [Y/n]: Y
   ```

3. **Sync and Customize**:
   ```
   Running `adw sync all`...
   ✓ Synced 47 template files to .opencode/
   
   Customizing configuration...
   ✓ Replaced {{PACKAGE_NAME}} with 'mypackage' (23 files)
   ✓ Replaced {{REPO_URL}} with 'https://github.com/user/repo' (15 files)
   ✓ Updated test paths: adw/ → src/mypackage/ (8 files)
   ✓ Updated lint paths: adw/ → src/mypackage/ (5 files)
   ```

4. **Validation**:
   ```
   Validating configuration...
   ✓ adw health: PASSED
   ✓ adw workflow list: 10 workflows available
   ✓ No remaining placeholders
   ```

5. **Summary**:
   ```
   ADW CONFIGURATION COMPLETE
   
   Files Created/Updated:
   - .opencode/agent/*.md (22 files)
   - .opencode/workflow/*.json (10 files)
   - adw-docs/*.md (9 files)
   
   Next Steps:
   1. Review configuration: cat .opencode/opencode.json
   2. Try a workflow: adw workflow complete <issue-number>
    3. Commit changes: git add .opencode adw-docs AGENTS.md
   ```

### Example 2: Update After ADW Package Upgrade

**Context**: Repository already configured with ADW v2.2, just upgraded to v2.3. Need to check for template changes.

**Invocation**:
```
I updated the ADW package, help me check if anything needs updating
```

**Expected Behavior**:

1. **Dry-Run Check**:
   ```
   Running `adw sync all --dry-run`...
   
   Template changes detected:
   
   NEW FILES:
   - .opencode/agent/workflow-builder.md
   - .opencode/workflow/documentation-update.json
   
   UPDATED FILES:
   - .opencode/workflow/complete.json (new 'document' step added)
   - .opencode/agent/implementor.md (improved task list format)
   
   UNCHANGED: 45 files
   ```

2. **Review and Apply**:
   ```
   Apply these changes?
   - New files will be added automatically
   - Updated files will overwrite existing versions
   
   [Y/n]: Y
   
   Running `adw sync all`...
   ✓ Added 2 new files
   ✓ Updated 2 existing files
   ```

3. **Re-customize**:
   ```
   Checking new files for placeholders...
   
   Customizing .opencode/agent/workflow-builder.md:
   ✓ Replaced {{PACKAGE_NAME}} with 'mypackage'
   ✓ Replaced {{REPO_URL}} with 'https://github.com/user/repo'
   ```

4. **Validation**:
   ```
   ✓ adw health: PASSED
   ✓ adw workflow list: 11 workflows available (1 new)
   
   Update complete!
   ```

### Example 3: Rust Project Configuration

**Context**: Rust project using cargo, clippy, and cargo-llvm-cov. Need to configure ADW.

**Invocation**:
```
Set up ADW for this Rust project
```

**Expected Behavior**:

1. **Detection**:
   ```
   Analyzing repository...
   
   Detected:
   - Language: Rust (Cargo.toml found)
   - Package Name: my_crate (from Cargo.toml)
   - Test Framework: cargo test (built-in)
   - Linting: clippy
   - Coverage: cargo-llvm-cov
   ```

2. **Customization**:
   ```
   Syncing templates...
   
   Customizing for Rust:
   - Test command: cargo test
   - Lint command: cargo clippy -- -D warnings
   - Coverage: cargo llvm-cov
   
   Updating adw-docs/ guides:
   - testing_guide.md: Using Rust/cargo test sections
   - linting_guide.md: Configuring clippy
   - code_style.md: Rust naming conventions
   ```

3. **Path Updates**:
   ```
   Updating path references:
   - pytest adw/ → cargo test (5 occurrences)
   - ruff check adw → cargo clippy (3 occurrences)
   - mypy adw/ → (removed, not applicable) (2 occurrences)
   ```

### Example 4: Multi-Language Repository

**Context**: Repository with Python backend and TypeScript frontend.

**Invocation**:
```
Help me set up ADW in this repository
```

**Expected Behavior**:

1. **Detection**:
   ```
   Multiple languages detected:
   
   Python:
   - 180 .py files in backend/
   - pytest configured
   - ruff, mypy for linting
   
   TypeScript:
   - 420 .ts files in frontend/
   - jest configured
   - eslint, prettier for linting
   
   Which should be the PRIMARY language for ADW configuration?
   (ADW focuses on one primary language for workflows)
   
   1. Python (backend/)
   2. TypeScript (frontend/)
   
   Select [1/2]: 
   ```

2. **User selects Python**:
   ```
   Configuring ADW for Python (backend/) as primary...
   
   Note: TypeScript tooling will be documented but not primary workflow target.
   
   Source directory: backend/
   Test command: pytest backend/
   Lint command: ruff check backend/ && mypy backend/
   ```

## Best Practices

1. **Run Dry-Run First**: For updates, always check with `adw sync all --dry-run` before applying changes to see what will change.

2. **Review Path Updates**: When the agent updates paths from `adw/` to your source directory, verify the replacements make sense for your project structure.

3. **Preserve Customizations**: If you've customized agent files, note which ones before updating. The agent will ask before overwriting.

4. **Validate After Changes**: Always run `adw health` and `adw workflow list` after setup/updates to verify everything works.

5. **Commit Incrementally**: Commit ADW configuration changes separately from code changes for easier rollback if needed.

6. **Check for Placeholders**: After setup, search for remaining `{{PLACEHOLDER}}` values that might have been missed.

7. **Test a Workflow**: After setup, try running a simple workflow to verify end-to-end functionality.

## Limitations

- **Single Primary Language**: For multi-language repos, ADW focuses on one primary language. Other languages can be documented but won't be the main workflow target.

- **Template-Based**: Configuration is based on ADW templates. Highly custom setups may need manual adjustment.

- **Path Detection**: Auto-detection of source directories works best with standard project layouts. Unusual structures may need manual specification.

- **No Rollback**: Once changes are made, there's no automatic rollback. Use git to revert if needed: `git restore .opencode/ adw-docs/`

- **Validation Scope**: The agent validates ADW loads correctly but doesn't verify that test/lint commands actually work (depends on project setup).

## Integration with Other Agents

- **Implementor Agent**: After setup, uses `adw-docs/` guides to understand repository conventions when implementing features.

- **Tester Agent**: References `testing_guide.md` filled by adw-setup for test framework and commands.

- **Linter Agent**: Uses `linting_guide.md` configuration for lint commands and tools.

- **Plan-Work Agent**: Reads `architecture_reference.md` for understanding project structure.

## Troubleshooting

### Issue: ADW CLI Not Found

**Symptom**: Agent reports "adw: command not found"

**Cause**: ADW package not installed or not in PATH

**Solution**:
```bash
# Install ADW
pip install adw
# or
uv pip install adw

# Verify installation
adw --version
```

### Issue: Sync Fails with Template Error

**Symptom**: `adw sync all` fails with "template not found"

**Cause**: ADW package installed but templates directory missing

**Solution**:
```bash
# Reinstall ADW
pip uninstall adw
pip install adw

# Or check installation location
python -c "import adw; print(adw.__file__)"
```

### Issue: Placeholders Remain After Setup

**Symptom**: Files still contain `{{PLACEHOLDER}}` values

**Cause**: Placeholder wasn't detected or replacement failed

**Solution**:
```bash
# Find remaining placeholders
rg '\{\{[A-Z_]+\}\}' .opencode/ adw-docs/

# Manually replace or re-run agent with specific values
```

### Issue: Wrong Source Directory Detected

**Symptom**: Paths reference wrong directory (e.g., `adw/` instead of `src/`)

**Cause**: Auto-detection picked wrong directory

**Solution**:
- Re-run agent and specify correct source directory when asked
- Or manually search and replace:
  ```bash
  rg 'adw/' .opencode/ adw-docs/ --files-with-matches
  # Then edit files to fix paths
  ```

### Issue: Validation Fails After Update

**Symptom**: `adw health` or `adw workflow list` fails after update

**Cause**: Incompatible configuration changes or missing dependencies

**Solution**:
```bash
# Check specific error from adw health
adw health

# If workflow parsing fails, check JSON syntax
python -m json.tool .opencode/workflow/complete.json

# Rollback if needed
git restore .opencode/
```

### Issue: Custom Agent Files Overwritten

**Symptom**: Custom modifications to agent files lost after sync

**Cause**: `adw sync all` overwrites files with templates

**Solution**:
- Before updates, save custom files: `cp .opencode/agent/custom.md .opencode/agent/custom.md.bak`
- After sync, re-apply customizations
- Consider keeping highly custom agents in a separate location

## See Also

- **ADW CLI Reference**: `adw --help`, `adw sync --help`
- **Template Structure**: `adw/templates/opencode_config/` in ADW package
- **Sync Command**: `.opencode/command/sync.md` - Manual sync command
- **ADW README**: Root README.md for complete ADW documentation
- **Setup/Update Agent**: `adw-docs/agents/setup-update.md` - Alternative setup agent for adw-docs/ guides
- **Architecture Guide**: `adw-docs/architecture/architecture_guide.md` - ADW architecture reference

## Feedback and Issues

If you encounter issues with the `adw-setup` agent:

1. Check the troubleshooting section above
2. Verify ADW package is correctly installed: `adw --version`
3. Check for template/config conflicts: `adw sync all --dry-run`
4. Open an issue with:
   - ADW version (`adw --version`)
   - Repository language/framework
   - Error messages or unexpected behavior
   - Steps to reproduce
