---
description: "Sync Commands from Template"
---

# Sync Commands from Template

Synchronize command files from the Agent template directory to the local `.claude/commands/` directory.

## Purpose

This command copies command files from `templates/claude_config/commands/` (or from an Agent submodule) to your repository's `.claude/commands/` directory. This is a one-way sync operation that ensures you have the latest command definitions.

## Variables

```
mode: $1 (optional: all, list, help)
```

## Instructions

### Step 1: Determine Source Path

Detect where the template commands are located:

1. **If in Agent repo**: Use `templates/claude_config/commands/`
2. **If Agent is a submodule**: Search for Agent submodule and use its template directory
3. **Common submodule paths to check**:
   - `Agent/templates/claude_config/commands/`
   - `agent/templates/claude_config/commands/`
   - `tools/Agent/templates/claude_config/commands/`
   - `tools/agent/templates/claude_config/commands/`
   - `.agent/templates/claude_config/commands/`

Use bash commands to detect:
```bash
# Check if we're in Agent repo
if [[ -d "templates/claude_config/commands" ]]; then
    SOURCE_DIR="templates/claude_config/commands"
# Check .gitmodules for Agent submodule path
elif [[ -f ".gitmodules" ]]; then
    AGENT_PATH=$(git config --file .gitmodules --get-regexp path | grep -i "agent" | awk '{print $2}' | head -1)
    if [[ -d "$AGENT_PATH/templates/claude_config/commands" ]]; then
        SOURCE_DIR="$AGENT_PATH/templates/claude_config/commands"
    fi
fi

# Check common submodule locations if not found yet
if [[ -z "$SOURCE_DIR" ]]; then
    for path in "Agent" "agent" "tools/Agent" "tools/agent" ".agent"; do
        if [[ -d "$path/templates/claude_config/commands" ]]; then
            SOURCE_DIR="$path/templates/claude_config/commands"
            break
        fi
    done
fi
```

If no source directory is found, show error and exit:
```
Error: Cannot find Agent template commands directory.
Please ensure:
1. You have Agent as a submodule, or
2. You're running from the Agent repository

To add Agent as a submodule:
  git submodule add <agent-repo-url> Agent
  git submodule update --init
```

### Step 2: Validate Destination

The destination is always `.claude/commands/`

Create the destination directory if it doesn't exist:
```bash
mkdir -p .claude/commands
```

### Step 3: Handle Mode

#### Mode: help (or no arguments)

Show usage:
```
Sync Commands - Copy command files from Agent template

Usage: /sync [mode]

Modes:
  (no args)  - Show this help message
  list       - List command files that would be synced
  all        - Copy all command files from template to .claude/commands/

Examples:
  /sync list    # Preview what would be copied
  /sync all     # Perform the sync

Source: templates/claude_config/commands/
Destination: .claude/commands/
```

#### Mode: list

Show which files would be synced using bash:
```bash
echo "Command files in template:"
echo "=========================="
echo ""
echo "Source: $SOURCE_DIR"
echo "Destination: .claude/commands/"
echo ""

# List all .md files in source
for file in "$SOURCE_DIR"/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")
        source_size=$(du -h "$file" | cut -f1)

        if [[ -f ".claude/commands/$filename" ]]; then
            dest_size=$(du -h ".claude/commands/$filename" | cut -f1)
            # Compare files
            if cmp -s "$file" ".claude/commands/$filename"; then
                echo "✓ $filename [identical] $source_size"
            else
                echo "✗ $filename [different] $source_size (local: $dest_size)"
            fi
        else
            echo "+ $filename [new] $source_size"
        fi
    fi
done

echo ""
echo "Legend:"
echo "  ✓ = identical (no changes needed)"
echo "  ✗ = different (will be updated)"
echo "  + = new (will be created)"
echo ""
echo "Run '/sync all' to perform the sync"
```

#### Mode: all

Copy all command files from source to destination:

**Step 3a: Show what will be synced**
```bash
echo "Syncing commands from template..."
echo "================================"
echo ""
echo "Source: $SOURCE_DIR"
echo "Destination: .claude/commands/"
echo ""
```

**Step 3b: Count files**
```bash
file_count=0
for file in "$SOURCE_DIR"/*.md; do
    if [[ -f "$file" ]]; then
        ((file_count++))
    fi
done

echo "Found $file_count command files to sync"
echo ""
```

**Step 3c: Ask for confirmation**
```bash
read -p "This will overwrite existing files in .claude/commands/. Continue? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Sync cancelled"
    exit 0
fi
```

**Step 3d: Perform the copy**
```bash
echo "Copying files..."
copied=0
updated=0
new=0

for file in "$SOURCE_DIR"/*.md; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file")

        if [[ -f ".claude/commands/$filename" ]]; then
            if cmp -s "$file" ".claude/commands/$filename"; then
                # Files are identical, skip
                continue
            else
                # File exists but is different
                cp "$file" ".claude/commands/$filename"
                echo "  Updated: $filename"
                ((updated++))
            fi
        else
            # New file
            cp "$file" ".claude/commands/$filename"
            echo "  Created: $filename"
            ((new++))
        fi
        ((copied++))
    fi
done

echo ""
echo "Sync complete!"
echo "=============="
echo "  Created: $new files"
echo "  Updated: $updated files"
echo "  Total: $copied files copied"
```

### Step 4: Success Message

After successful sync, show:
```
Commands are now in sync with template.

To undo this operation:
  git restore .claude/commands/

To view changes:
  git diff .claude/commands/
```

## Usage Examples

### Preview what would be synced
```
/sync list
```

### Sync all commands
```
/sync all
```

### Get help
```
/sync
```

## Notes

- This is a one-way sync from `templates/claude_config/commands/` to `.claude/commands/`
- Existing files will be overwritten (with confirmation)
- The sync.md file itself is included in the copy
- Use git to track and revert changes if needed
- For custom commands, consider keeping them in a separate directory or using a different naming convention

## Recovery

If you need to undo a sync:

```bash
# View changes
git diff .claude/commands/

# Restore a specific file
git restore .claude/commands/<filename>

# Restore all command files
git restore .claude/commands/

# Or restore from a specific commit
git restore --source=HEAD~1 .claude/commands/
```
