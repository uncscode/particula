# Agent Permission Reference

Use this reference when defining permissions for primary agents and subagents
in `.opencode/agent/*.md`. OpenCode uses the same `permission:` format for both
agent roles.

## Core Model

Agent `mode` controls how an agent is invoked:

- `primary`: selectable as a main conversational agent
- `subagent`: invoked through delegation or an `@` mention
- `all`: available in both roles

Permissions independently control what the agent may do. Each permission
resolves to `allow`, `ask`, or `deny`. This repository uses a deny-by-default
baseline and explicitly grants only required capabilities:

```yaml
permission:
  "*": deny
  read: allow
  find_files: allow
  search_content: allow
  adw_spec_messages: allow
  feedback_log: allow
```

Do not use the deprecated `tools:` frontmatter for new or updated agents. See
the [tools-to-permission migration guide](../../tools/guide_for_permission_migration.md)
when converting an existing definition.

## Restrict File Edits

The built-in `edit` permission gates OpenCode's `edit`, `write`, and
`apply_patch` tools. Use an ordered path-rule map to restrict those tools to
specific repository files or directories:

```yaml
permission:
  "*": deny
  read: allow
  edit:
    "*": deny
    "docs/**": allow
  adw_spec_messages: allow
  feedback_log: allow
```

OpenCode evaluates every matching pattern and applies the last match. Put the
broad catch-all first and narrower exceptions after it.

Common patterns include:

```yaml
# One file only
edit:
  "*": deny
  "docs/index.md": allow

# Several documentation locations with an exclusion
edit:
  "*": deny
  "README.md": allow
  "docs/**": allow
  ".opencode/guides/**": allow
  "docs/API/**": deny

# Tests only
edit:
  "*": deny
  "adw/tests/**/*_test.py": allow
  ".opencode/tools/__tests__/**/*.test.ts": allow
```

Permission patterns use simple wildcard matching: `*` matches zero or more
characters and `?` matches one character. Test rules against the actual paths
passed to OpenCode tools rather than treating them as regular expressions.

## Read-Only Agents

For review, research, and analysis agents, deny all tools first and allow only
the required read surfaces:

```yaml
---
description: Reviews code without modifying the repository.
mode: subagent
permission:
  "*": deny
  read: allow
  find_files: allow
  search_content: allow
  git_diff: allow
  adw_spec_messages: allow
  feedback_log: allow
---
```

`mode: subagent` does not make an agent read-only. Read-only behavior comes
from its permission map.

## Close Alternate Write Paths

An `edit` path restriction applies only to the built-in file-modification
tools it gates. It is not a complete filesystem sandbox. A restricted agent
must not receive another capability that can mutate files outside that scope.

Review these paths separately:

- `bash`: deny it or allow only narrowly selected read-only commands.
- Custom tools: grant exact split-wrapper names, not broad families such as
  `git_*` or `adw_*`.
- MCP tools: deny unneeded server tool patterns, especially mutating tools.
- `external_directory`: deny access unless the agent has a documented need to
  touch paths outside the worktree.
- Delegated agents: ensure children have their own narrow permission maps.

Example for a documentation-only subagent:

```yaml
---
description: Updates user documentation without modifying source code.
mode: subagent
permission:
  "*": deny
  read: allow
  edit:
    "*": deny
    "README.md": allow
    "docs/**": allow
  find_files: allow
  search_content: allow
  bash:
    "*": deny
    "git diff*": allow
    "git status*": allow
  external_directory: deny
  adw_spec_messages: allow
  feedback_log: allow
---
```

OpenCode permission patterns for `bash` match parsed command text. Prefer this
repository's narrow custom wrappers over shell access whenever a suitable
wrapper exists.

## Restrict Delegation

The `task` permission controls which subagent types an agent may launch through
the Task tool:

```yaml
permission:
  "*": deny
  task:
    "*": deny
    "adw-review-*": allow
    "codebase-researcher": allow
  adw_spec_messages: allow
  feedback_log: allow
```

The last matching task rule wins. A denied subagent is removed from the Task
tool description for that caller. This does not prevent a user from directly
invoking a visible subagent with an `@` mention; the invoked subagent's own
permission map remains the capability boundary.

## External Directories

`external_directory` is an additional boundary for paths outside the project
worktree. Allowing an external directory does not grant a specific operation;
the corresponding `read`, `edit`, or other tool permission must also allow it.

Keep external access denied unless it is required:

```yaml
permission:
  "*": deny
  read: allow
  external_directory: deny
```

If external access is necessary, use a narrow path rule and separately deny
edits when the external location should remain read-only.

## Repository Policy

- Start every active agent with `"*": deny`.
- Grant exact split wrappers required by the agent instead of wildcard tool
  families.
- Keep `adw_spec_messages: allow` and `feedback_log: allow` in active agent
  definitions as required by repository policy.
- Treat `declared_scope` metadata as auditable intent only. It does not enforce
  filesystem access and does not replace path permissions or path-safe tools.
- Use `ask` only where an intentional interactive approval boundary exists.
  Explicit `deny` rules remain enforced when OpenCode auto-approval is enabled.

## Validation

Validate agent changes with the repository checks:

```bash
pytest adw/tests/agent_permission_validation_test.py -v
pytest adw/tests/agent_reference_validation_test.py -v
scripts/validate_agent_references.sh
```

Restart OpenCode after changing agent definitions or OpenCode configuration;
the running session retains the configuration loaded at startup.

## Upstream References

- [OpenCode permissions](https://opencode.ai/docs/permissions/)
- [OpenCode agents](https://opencode.ai/docs/agents/)
- [OpenCode tools](https://opencode.ai/docs/tools/)
- [OpenCode configuration schema](https://opencode.ai/config.json)
