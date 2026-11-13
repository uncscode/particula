# git_commit

Execute the narrow ADW Git commit command through `git_commit`. This is separate
from the native runtime protected-git surface; it does not grant arbitrary Git
commands, shell access, auto-push, or background Git operations.

## Arguments

- `summary`: required nonblank commit summary.
- `description`: optional commit body.
- `adw_id`: optional workflow ID.
- `worktree_path`: optional worktree path; cannot begin with `-`.
- `stage_all`: stage all changes when true.
- `max_retries`: integer from 0 through 10, default 3, forwarded to ADW for
  hook-modified commit retries.
- `no_verify`: bypass hooks only when true.

```json
{ "summary": "Fix commit output preservation", "max_retries": 3 }
```

## Output preservation

Short responses retain their existing format, including the `Git Commit Command`
success prefix, commit status/SHA output, and deterministic `ERROR:` envelope,
exit code, diagnostic selection, and hints on failure.

If any raw failure stderr, stdout, or fallback message exceeds **500 characters**,
the concise error response is retained and a log saves all three original streams
in labeled `stderr:`, `stdout:`, and `message:` sections with `exit_code:`. Line
breaks and tabs are preserved before any diagnostic normalization or truncation.

Successful output exceeding **8,000 characters** is saved exactly. The response
contains the first **4,000 characters**, a `... [truncated]` marker, and the last
**2,000 characters**, beneath the existing success prefix.

Both long-output responses append:

```text
full_output_path: /absolute/repository/adforge_local/opencode/tmp/git-commit-<unique>/output.log
Read this temporary log for complete output before retrying.
```

Read `full_output_path` before retrying. This repository requires repo-local
runtime files, so logs live under `adforge_local/opencode/tmp/` in the checkout
containing the wrapper, rather than `~/.local/share/opencode/tool-output/`.
Node's `mkdtemp` creates unique private directories (0700), and log files are
created with mode 0600. The commit agent in `.opencode/agent/adw-commit.md` grants
`read: allow`; these repo-local logs need no external-directory permission grant.

If directory creation or log writing fails, the response reports
`output_log_error` and includes the **complete original output inline**, retaining
the success prefix or failure envelope as appropriate. Logging never reruns the
commit or changes its result. The wrapper invokes the ADW command once; ADW's
existing `max_retries` behavior remains unchanged.

## Retention

Before creating each new log directory, best-effort cleanup removes only direct
`git-commit-*` directories under the log root whose directory modification time
is **older than 7 days**. Symlinks, matching regular files, unrelated entries,
and matching directories nested beneath unrelated directories are left alone.
Cleanup errors do not prevent saving the new log or change the commit result.
Cleanup runs only when new long-output logs are written, **not on a background
schedule**; logs may therefore remain longer than seven days between invocations.

## Focused validation

```bash
cd .opencode/tools
bun test __tests__/git_commit.test.ts
```
