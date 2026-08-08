# run_linters

Compatibility wrapper for repository Ruff and mypy validation. New targeted
Ruff requests use the explicit `mode` contract below.

## Explicit Ruff modes

Explicit modes are Ruff-only and select exactly one operation:

| `mode` | Ruff invocation | Mutation | Confirmation |
| --- | --- | --- | --- |
| `check` | `ruff check -- <targets>` | Never edits files | No |
| `format-check` | `ruff format --check -- <targets>` | Never edits files | No |
| `format` | `ruff format -- <targets>` | May edit files | Required |

`check` and `format-check` are read-only. They never select `--fix` and never
run mutating `ruff format`. `format` is the only explicit mode that may edit
files, and it must receive the normal tool confirmation before dispatch.

### Targets and transport

Provide `targetPaths` only with an explicit `mode`. It is a non-empty, ordered
array of canonical repository-relative file or directory paths. Directories are
passed to Ruff as supplied; the wrapper does not enumerate them. Omitting
`targetPaths` with an explicit mode selects `.`.

The wrapper preserves target order and transports it internally as one compact
JSON array through `--target-paths-json`; callers must provide the array field,
not a space-separated target string. Limits are 64 entries, 1,024 UTF-8 bytes
per entry, and 8,192 UTF-8 bytes for the compact JSON array. Paths must be
nonblank, unique, canonical, non-option, confined repository-relative paths.

## Legacy compatibility path

`autoFix` and `targetDir` are legacy selectors and apply only when `mode` is
absent. `autoFix: false` is validation-only and does not modify files.
`autoFix: true` runs the mutating Ruff fix/format/final-check flow before mypy
and requires confirmation. `targetDir`, when supplied, is the legacy scalar
selector; an omitted value defers to project configuration.

Defaults remain entrypoint-specific when `mode` is absent: the direct
TypeScript wrapper defaults `autoFix` to `true`, while the runtime adapter
normalizes it to `false`. Do not treat an omitted `mode` as a new read-only
request.

## Selector and linter compatibility

New and legacy selectors cannot be combined. The wrapper rejects `mode` with
`autoFix`, `mode` with `targetDir`, `targetPaths` with `targetDir`, and
`targetPaths` without `mode`; no field takes precedence. Explicit modes accept
only omitted `linters` (normalized to Ruff) or `linters=ruff`. They reject
`linters=mypy` and `linters=ruff,mypy`. Legacy requests retain the
`ruff,mypy` composition.

## Direct fields

- `autoFix` (legacy only)
- `targetDir` (legacy only)
- `mode`: `check`, `format-check`, or `format`
- `targetPaths`: ordered explicit-mode targets
- `ruffTimeout`
- `mypyTimeout`

## Bounded `options` tokens

- `output=<summary|full|json>`
- `linters=<ruff|mypy comma-list>`

## Examples

Legacy validation-only request:

```json
{"autoFix":false}
```

Explicit read-only checks over ordered targets:

```json
{"mode":"check","targetPaths":["adforge_core/tools","adforge_core/runtime"]}
```

```json
{"mode":"format-check","targetPaths":["adforge_core/runtime"]}
```

The following mutating request requires confirmation:

```json
{"mode":"format","targetPaths":["adforge_core/runtime"]}
```

## Validation failures

Invalid requests fail before confirmation, dispatch, or subprocess execution.
Errors use deterministic `ERROR:` envelopes that identify the invalid field
(`mode`, `targetPaths`, `targetDir`, `autoFix`, `linters`, or `options`). A
selector conflict identifies both involved field names. Validation messages do
not echo unbounded caller input.
