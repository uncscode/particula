# Branch-Sync Agent Permission Tests

## Required tools present (true)
- [x] read
- [x] edit
- [x] write
- [x] move
- [x] list
- [x] ripgrep
- [x] git_operations
- [x] platform_operations
- [x] todoread
- [x] todowrite
- [x] task
- [x] adw_spec

## Forbidden tools (false)
- [x] bash
- [x] webfetch
- [x] websearch
- [x] codesearch
- [x] create_workspace
- [x] workflow_builder
- [x] run_pytest
- [x] run_linters
- [x] adw

## Capabilities & commands
- [x] Mentions conflict read/edit guidance
- [x] References git_operations commands: fetch, sync, merge, rebase, reset, abort, push-force-with-lease

## Safety rules
- [x] Protected branch guard (block force push main/master)
- [x] Force-with-lease only (no --force)
- [x] Confirmation before destructive operations (reset --hard, force push)
- [x] Default abort guidance on conflict
- [x] No bash access

## Examples
- [x] Sync flow
- [x] Rebase flow
- [x] Conflict resolution flow
- [x] Recovery/rollback flow
