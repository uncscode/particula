# git_diff tool

`git_diff` is the narrow OpenCode wrapper for **read-only**, local Git
diagnostics. It supports `status`, `diff`, `log`, and `show`; it is not an
arbitrary Git command surface.

## Local adapter boundary

After admitting an explicitly supplied trusted local worktree, the TypeScript wrapper invokes the
wrapper-local `read_only_git_diagnostics.py` adapter once with one JSON-lines
request. The adapter reads exactly one bounded UTF-8 JSON object from standard
input and writes exactly one bounded, newline-terminated JSON response to
standard output.

Every adapter-dispatched core envelope for `status`, `diff`, `log`, and `show`
has the exact top-level `evidence_identity` object
`{"contract":"e37-m2-validation-git","version":1}`. Adapter admission and
unavailable failures are identity-free. The core adds the identity to both
successful and terminal validation, ref-verification, and execution-failure
read-only envelopes; protected mutating Git envelopes remain outside this
contract. The adapter returns a dispatched core mapping unchanged and never
mints an identity. Rendered TypeScript text remains a display channel, not a
structured evidence consumer channel.

The identity is a bounded compatibility marker, not proof that an inspection
executed or authority for state promotion, delegation, Git writes, shell use,
network access, or lifecycle mutation. Consumers requiring structured evidence
must use the adapter/core JSON envelope rather than rendered text.

The adapter constructs an explicit repository-scoped `ProjectContext` for the
admitted canonical worktree, denies local internet access, and delegates once
to `adforge_core.runtime.git_tools.execute_git_tool(...)`. It does not infer
authority from the ambient current directory.

This boundary does **not** import or dispatch the historical workflow CLI, the
core CLI module, or either CLI root. It has no CLI-root help route, no legacy
presentation fields, and no debug-log or other persisted diagnostic artifacts.

## Operation argument matrix

Only the fields in the selected operation's row are admitted. Unsupported
fields, including legacy `porcelain`, `stat`, `oneline`, and `help` fields, are
an `invalid_request` failure rather than compatibility aliases.

| Command | Admitted fields | Additional rules |
| --- | --- | --- |
| `status` | `worktree_path` | No revision or path arguments. |
| `diff` | `worktree_path`, `base`, `target`, `path` | No revisions compares the working tree; `base` compares that revision to the working tree; `base` and `target` compare two revisions. `target` requires `base`. |
| `log` | `worktree_path`, `ref`, `max_count`, `path` | `max_count` is an integer from `1` through `1000`; its default is `50`. |
| `show` | `worktree_path`, `ref`, `path` | `ref` is required. A path requires a compatible resolved tree-ish ref. |

The caller must supply `worktree_path` explicitly. The wrapper canonicalizes an
existing absolute directory and supplies that path in its adapter request; it
does not infer authority from its current directory. The adapter then admits
only its primary checkout or a linked worktree registered under that checkout's
canonical Git common directory. Admission requires a direct
`.git/worktrees/<id>` metadata directory, matching `commondir`, and an exact
back-pointer to the selected worktree's `.git` file. Ordinary sibling
repositories, foreign worktrees, malformed pointers, and symlinked metadata are
rejected before core dispatch.

## Local-only validation and inspection

The core runtime, not the TypeScript wrapper, owns the operation matrix,
argument validation, revision verification, path confinement, and fixed Git
argv. Before final inspection it rejects invalid types, unexpected fields,
blank or option-like revisions, unsafe revision syntax, and invalid revision
combinations. Supplied revisions are resolved with fixed local verification
commands before they can reach a final inspection command.

`path` is an optional non-empty literal repository-relative path. It cannot be
absolute, `.`/`./`, traversal, NUL-containing, option-like, or Git pathspec
magic; resolution must remain beneath the admitted worktree, including through
symlink checks. Final fixed commands place an admitted path after `--`.

All Git activity is local-only and non-network. The inspection commands do not
fetch, contact remotes, run arbitrary Git verbs, invoke a shell, mutate Git or
worktree state, or persist raw process output. Local upstream divergence uses
only local refs: it never fetches to refresh them.

## Bounded diagnostic outcomes

Successful responses preserve the `ok`, `operation`, and `data` envelope.
`data` contains only a capped/redacted command summary, relative worktree
identifier, optional normalized path, bounded stdout/stderr, and truncation
flags. It does not expose absolute worktree paths or raw argv.

The available successful classifications are:

| Command | Classification fields |
| --- | --- |
| `status` | `status`: `clean` or `changed`; `divergence`: `ahead`, `behind`, `diverged`, `in_sync`, `not_configured`, or `unavailable` |
| `diff` | `diff`: `no_diff` or `diff_present` |
| `log`, `show` | No additional diagnostic classification |

Failures preserve the same outer envelope with `ok: false` and a bounded
`error`. Read-only diagnostic error types are `invalid_request`, `unavailable`,
`execution_failed`, and `execution_timeout`. Adapter transport or preflight
failures are rendered as stable bounded wrapper errors; child stdout, stderr,
tracebacks, raw command argv, tokens, and absolute paths are not rendered.
