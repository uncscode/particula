# OpenCode Configuration Templates

This directory contains the complete OpenCode configuration for ADW workflows.
Copy this directory to `.opencode/` in your project to enable ADW functionality.

## Directory Structure

```
opencode_config/
├── agent/              # Agent definitions (22 agents)
│   ├── adw_default.md
│   ├── execute-plan.md
│   ├── plan-work.md
│   ├── tester.md
│   ├── linter.md
│   ├── shipper.md
│   ├── documentation.md
│   └── ...
├── command/            # Slash commands (14 commands)
│   ├── prime.md
│   ├── commit.md
│   ├── pull_request.md
│   └── ...
├── plugin/             # OpenCode plugins (8 plugins)
│   ├── _index.ts
│   ├── notification.ts
│   └── ...
├── tool/               # Custom tools (13 tools)
│   ├── adw.ts
│   ├── run_pytest.ts
│   ├── run_linters.ts
│   └── ...
├── utils/              # Shared utilities
│   └── index.ts
├── workflow/           # JSON workflow definitions (10 workflows)
│   ├── complete.json
│   ├── patch.json
│   ├── plan-work.json
│   └── examples/
├── opencode.json       # OpenCode configuration
├── package.json        # Node dependencies for plugins
└── utils.ts            # Root utilities
```

## Installation

1. Copy the entire directory to your project:
   ```bash
   cp -r adw/templates/opencode_config/ .opencode/
   ```

2. Install plugin dependencies:
   ```bash
   cd .opencode && bun install
   ```

3. Customize as needed for your project.

## Key Components

### Workflows

JSON-based workflow definitions that orchestrate multi-step development tasks:

| Workflow | Description |
|----------|-------------|
| `complete.json` | Full workflow: plan → execute → test → document → ship |
| `patch.json` | Quick fixes: plan → execute → test → ship |
| `plan-work.json` | Planning only |
| `execute-plan.json` | Implementation only |
| `test.json` | Testing only |
| `document.json` | Documentation workflow |
| `ship.json` | PR creation |

### Agents

Specialized agents for different workflow phases:

| Agent | Purpose |
|-------|---------|
| `plan-work` | Creates implementation plans |
| `execute-plan` | Implements planned changes |
| `tester` | Runs tests and fixes failures |
| `linter` | Code quality validation |
| `documentation` | Generates documentation |
| `shipper` | Creates PRs |

### Commands

Utility slash commands for manual operations:

- `/prime` - Understand codebase
- `/commit` - Create commits
- `/pull_request` - Create PRs
- `/sync` - Sync ADW state
- `/cleanup_worktrees` - Clean up worktrees

### Tools

Custom tools available to agents:

- `adw` - ADW CLI operations
- `run_pytest` - Run tests with coverage
- `run_linters` - Run linting tools
- `create_workspace` - Create isolated worktrees
- `workflow_builder` - Build custom workflows

## Customization

1. **Agents**: Modify agent prompts in `agent/` for project-specific behavior
2. **Commands**: Add project-specific commands in `command/`
3. **Workflows**: Create custom workflows in `workflow/`
4. **Tools**: Add custom tools in `tool/`

## Sync with ADW

To update templates from the active `.opencode/` directory:

```bash
# From ADW repository root
cp -r .opencode/* adw/templates/opencode_config/
```

Note: Do not copy `node_modules/`, `bun.lock`, or `.gitignore`.
