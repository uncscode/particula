# Custom Tool Authoring Guide

Reference for building OpenCode custom tools in `.opencode/tools/`.

---

## File Structure

Each tool is a self-contained `.ts` (or `.js`) file that exports one or more `tool()` definitions.

```
.opencode/tools/
  my_tool.ts          # tool definition (required)
  my_tool.md          # extended description (optional)
  my_tool.py          # backend script (optional, if the tool shells out)
```

- The `.ts` filename becomes the tool name (e.g., `my_tool.ts` registers as `my_tool`).
- An optional `.md` file with the same base name provides an extended
  description injected into the tool's system prompt context.
- Backend scripts (`.py`, `.sh`, etc.) can live alongside and be invoked via
  `Bun.$` or `Bun.spawnSync`.

---

## Minimal Tool Template

```typescript
import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Short description of what this tool does",
  args: {
    name: tool.schema.string().describe("A required string argument"),
    count: tool.schema.number().optional().describe("An optional number"),
    verbose: tool.schema.boolean().optional()
      .describe("Enable verbose output. Omit unless explicitly requested."),
    mode: tool.schema.enum(["fast", "slow"]).optional()
      .describe("Execution mode. Omit to use the tool default."),
  },
  async execute(args) {
    // args is typed from the schema above
    return `Hello ${args.name}, count=${args.count ?? 0}`;
  },
});
```

---

## Multiple Tools Per File

You can export multiple tools from a single file. Each named export becomes a
separate tool named `<filename>_<exportname>`:

```typescript
// math.ts  -->  registers "math_add" and "math_multiply"
import { tool } from "@opencode-ai/plugin";

export const add = tool({
  description: "Add two numbers",
  args: {
    a: tool.schema.number().describe("First number"),
    b: tool.schema.number().describe("Second number"),
  },
  async execute(args) {
    return String(args.a + args.b);
  },
});

export const multiply = tool({
  description: "Multiply two numbers",
  args: {
    a: tool.schema.number().describe("First number"),
    b: tool.schema.number().describe("Second number"),
  },
  async execute(args) {
    return String(args.a * args.b);
  },
});
```

A `default` export registers under the bare filename. Named exports register as
`<filename>_<exportname>`. You can mix both in one file.

**Important:** Every export must be a valid `tool()` definition. Exporting
plain functions, constants, or types alongside tool definitions will cause the
loader to crash:

```
Object.entries requires that input parameter not be null or undefined
```

If you need local helpers, keep them as **non-exported** functions:

```typescript
// Good -- helper is local, only tool definitions are exported
function formatResult(n: number): string {
  return `Result: ${n}`;
}

export default tool({
  description: "...",
  args: { n: tool.schema.number() },
  async execute(args) {
    return formatResult(args.n);
  },
});
```

---

## Tool With a Python Backend

```typescript
import { tool } from "@opencode-ai/plugin";
import path from "path";

export default tool({
  description: "Run a Python script and return its output",
  args: {
    input: tool.schema.string().describe("Input value to process"),
  },
  async execute(args, context) {
    const scriptPath = path.join(context.worktree, ".opencode/tools/my_tool.py");
    try {
      const result = await Bun.$`python3 ${scriptPath} ${args.input}`.text();
      return result.trim() || "Script completed with no output.";
    } catch (error: any) {
      const stderr = error?.stderr?.toString?.() || "";
      const message = error?.message || "Unknown error";
      return `ERROR: ${stderr || message}`;
    }
  },
});
```

Use `context.worktree` to resolve paths relative to the project root.

---

## Schema Types

`tool.schema` is [Zod](https://zod.dev). You can also `import { z } from "zod"`
directly -- they are the same library.

| Method | Example |
|---|---|
| `.string()` | `tool.schema.string()` |
| `.number()` | `tool.schema.number()` |
| `.boolean()` | `tool.schema.boolean()` |
| `.enum([...])` | `tool.schema.enum(["a", "b", "c"])` |
| `.array(schema)` | `tool.schema.array(tool.schema.string())` |

Chain `.optional()` to make an argument optional, and `.describe("...")` to add
a description.

### Required vs Optional Parameters

- **Required**: Omit `.optional()`. The field appears in the JSON Schema `required` array.
- **Optional**: Chain `.optional()`. The field is omitted from `required`.

```typescript
args: {
  // Required -- LLM must always provide this
  command: tool.schema.enum(["start", "stop"]),

  // Optional -- LLM can omit this entirely
  timeout: tool.schema.number().optional().describe("Timeout in seconds"),
}
```

### Always Use `.describe()` on Optional Parameters

LLMs tend to send every parameter with a "default" value (`false`, `0`, `""`)
rather than omitting them. Adding `.describe()` with clear guidance significantly
reduces this behavior:

```typescript
// Bad -- LLM doesn't know when to omit, sends false/0/"" by default
verbose: tool.schema.boolean().optional(),
count: tool.schema.number().optional(),

// Good -- LLM understands when to omit
verbose: tool.schema.boolean().optional()
  .describe("Enable verbose output. Omit unless explicitly requested."),
count: tool.schema.number().optional()
  .describe("Number of results to return. Omit to use the default (10)."),
```

### Schema Budget for Tool Arguments

Counted direct fields are the top-level entries in `args`. Tool wrappers should
stay at four or fewer counted direct fields.

The counted-field budget is four direct fields; narrow exempt `options` and
usage-only `help` may exist separately when they follow the rules below.

`command` counts toward that budget even when it is only an enum router for
multiple subcommands.

Before relying on exceptions or broader schemas, prefer narrower focused or
split wrappers that expose only the fields each workflow actually needs.
Compatibility wrappers may still exist when repository policy allows them, but
they should not drive new broad top-level schemas.

Compliant example (four counted direct fields: `command`, `path`, `format`,
and `limit`):

```typescript
args: {
  command: tool.schema.enum(["status", "diff"]),
  path: tool.schema.string().optional()
    .describe("Path to inspect. Omit to use the current worktree root."),
  format: tool.schema.enum(["summary", "json"]).optional()
    .describe("Output format. Omit to use the wrapper default."),
  limit: tool.schema.number().optional()
    .describe("Maximum number of rows to return. Omit to use the default."),
}
```

This stays within budget because it exposes exactly four counted top-level
fields.

Over-budget example (five counted direct fields: `command`, `path`, `format`,
`limit`, and `timeout`):

```typescript
args: {
  command: tool.schema.enum(["status", "diff"]),
  path: tool.schema.string().optional()
    .describe("Path to inspect. Omit to use the current worktree root."),
  format: tool.schema.enum(["summary", "json"]).optional()
    .describe("Output format. Omit to use the wrapper default."),
  limit: tool.schema.number().optional()
    .describe("Maximum number of rows to return. Omit to use the default."),
  timeout: tool.schema.number().optional()
    .describe("Timeout in seconds. Omit to use the wrapper default."),
}
```

This is over budget. Redesign or split the tool instead of widening the direct
schema further.

Only these narrow cases are exempt from the counted-field budget:

- Bounded, documented, deterministically parsed `options`.
- Usage-only `help`, where `help: true` bypasses normal validation, returns
  concise usage guidance, and points to the companion `.md` path for extended
  documentation when relevant.

These exemptions are not broad escape hatches. Optional fields still need clear
descriptions, runtime parsing/validation should stay deterministic, and
wrappers should continue to omit noisy defaults instead of encoding wider
top-level schemas.

Bounded exemption example (`command` and `path` count; `options` and `help` do
not):

```typescript
args: {
  command: tool.schema.enum(["show"]),
  path: tool.schema.string().optional()
    .describe("Path to inspect. Omit to use the current worktree root."),
  options: tool.schema.string().optional().describe(
    "Bounded option tokens only (for example: 'json' or 'limit=10'). Omit " +
    "unless a documented token is required.",
  ),
  help: tool.schema.boolean().optional().describe(
    "Show usage guidance for this wrapper. Omit unless the caller explicitly " +
    "requests help.",
  ),
}
```

`options` is exempt only when it is bounded to known tokens and is
deterministically parsed and validated before execution. `help` is exempt only
for usage output.

Example help output:

```text
Usage: my_tool_show({ command: "show" })
Optional: path, options
Use help: true to show this guidance without running normal validation.
Docs: .opencode/tools/my_tool_show.md
```

Keep `help: true` narrow. It may bypass normal validation only to return usage
and companion-doc guidance; it must not become a broader runtime branching path
or a free-form passthrough surface, and the companion `.md` file remains
informational rather than changing runtime behavior.

---

## Execute Function

### Return Types

The `execute` function can return a plain string or a structured result object:

```typescript
// Simple -- return a string directly
async execute(args) {
  return `Result: ${args.name}`;
}

// Structured -- return title, output, and optional metadata/attachments
async execute(args) {
  return {
    title: "Query result",
    output: `Found ${rows.length} rows`,
    metadata: { rowCount: rows.length },
  };
}
```

Both forms are supported. The `output` field is the text shown to the agent.

### Context

The second argument to `execute` provides session context:

```typescript
async execute(args, context) {
  const {
    agent,       // current agent name
    sessionID,   // session identifier
    messageID,   // message identifier
    directory,   // session working directory
    worktree,    // git worktree root
    abort,       // AbortSignal for cancellation
  } = context;

  return `Agent: ${agent}, Dir: ${directory}`;
}
```

Use `context.directory` for the session working directory.
Use `context.worktree` for the git worktree root when resolving project paths.

When a wrapper accepts repository-relative paths, keep resolution root-confined:

- Reject absolute paths unless the wrapper contract explicitly allows them.
- Reject traversal attempts such as `..` segments before resolution.
- Canonicalize resolved paths (`realpath`) before use so symlink escapes are
  detected.
- Fail closed if the canonical target resolves outside `context.worktree`.

---

## Hard Constraints

These are enforced by the OpenCode tool loader and will cause build failures or
runtime crashes if violated.

### 1. Self-contained wrapper entrypoint

Each `.ts` tool wrapper must keep its exported `tool()` definition
self-contained and must not rely on importing another wrapper entrypoint as its
primary implementation path. In particular, you cannot:

- Import from `./other_tool`
- Import one wrapper from another wrapper directory path as the main
  execution path
- Turn a focused wrapper into a thin proxy around another wrapper file

The tool loader bundles each `.ts` file independently. Unsupported or
unresolved local imports produce a build error:

```
Cannot find module './lib/my_shared' from '.opencode/tools/my_tool.ts'
```

**Allowed imports:**
- `@opencode-ai/plugin` (the tool API)
- `zod` (same as `tool.schema`)
- Node/Bun built-ins: `fs`, `path`, `node:fs`, `node:path`, etc.
- Narrow local shared helper modules when repository policy explicitly permits
  them and the helper is not itself another wrapper

**If you need shared code**, prefer a narrow local shared helper module when
that pattern is already allowed by repository policy; otherwise inline the
logic. Do not import one tool wrapper from another as a shortcut.
Helper-module imports are fine only when the imported file is a non-wrapper
helper and the wrapper still owns its own schema, validation, and execution
boundary.

### 2. Every export must be a tool definition

Every export in the file must be a valid `tool()` return value. Exporting plain
functions, constants, or types alongside tool definitions crashes the loader.

See [Multiple Tools Per File](#multiple-tools-per-file) for the correct pattern.

### 3. The execute function must return a string or result object

The `execute` function must return `string`, `Promise<string>`, or a result
object with an `output: string` field. This is the text shown to the agent.

---

## Design Patterns

### Normalizing Optional Parameters

LLMs frequently send "noisy defaults" for optional parameters (`false`, `0`,
`""`, `[]`) instead of omitting them. Normalize these in `execute()` before use.

#### Lightweight (for simple tools)

Use `??` and `||` for inline defaults:

```typescript
async execute(args) {
  const mode = args.mode || "summary";       // falsy -> default
  const count = args.count ?? 10;            // nullish -> default
  const verbose = args.verbose === true;     // only explicit true
  const items = args.items?.length ? args.items : undefined;
  // ...
}
```

#### Per-field helpers (for medium-complexity tools)

Small local functions that normalize one field at a time:

```typescript
function normalizeOptional(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function normalizePositiveInt(value: unknown): number | undefined {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 && Number.isInteger(n) ? n : undefined;
}

async execute(args) {
  const path = normalizeOptional(args.path);
  const limit = normalizePositiveInt(args.limit) ?? 100;
  // ...
}
```

#### Bulk normalizer (for tools with many optional params)

Strip all noisy defaults in one pass before processing:

```typescript
const OPTIONAL_KEYS = [
  "path", "timeout", "verbose", "format", "maxResults",
] as const;

function stripDefaults(
  raw: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (!(OPTIONAL_KEYS as readonly string[]).includes(key)) {
      result[key] = value;  // keep required keys unconditionally
      continue;
    }
    // Strip noisy defaults
    if (
      value === undefined ||
      value === null ||
      value === "" ||
      value === false ||
      value === 0 ||
      (Array.isArray(value) && value.length === 0)
    ) {
      continue;  // omit -- treat as not provided
    }
    result[key] = value;
  }
  return result;
}

async execute(rawArgs) {
  const args = stripDefaults(rawArgs as Record<string, unknown>);
  // args.path is now truly undefined if the LLM sent ""
  // ...
}
```

### Conditionally Required Parameters

Declare parameters as `.optional()` in the schema, then enforce requirements
per-command in `execute()`:

```typescript
args: {
  command: tool.schema.enum(["get", "set"]),
  key: tool.schema.string().describe("Configuration key"),
  value: tool.schema.string().optional()
    .describe("New value. Required for 'set', omit for 'get'."),
},
async execute(args) {
  if (args.command === "set" && !args.value) {
    return "ERROR: 'value' is required for the 'set' command.";
  }
  // ...
}
```

### Error Envelopes

Return deterministic error strings prefixed with `ERROR:` so agents can
distinguish success from failure:

```typescript
async execute(args) {
  if (!args.required_field) {
    return "ERROR: 'required_field' is required.";
  }
  try {
    const result = await Bun.$`some-command ${args.required_field}`.text();
    return result || "Command completed with no output.";
  } catch (error: any) {
    const stderr = error?.stderr?.toString?.() || "";
    return `ERROR: Command failed.\n${stderr || error?.message || "Unknown error"}`;
  }
}
```

### Input Safety

Reject option-injection in path and ref arguments, and keep filesystem access
confined to the intended repository root:

```typescript
if (typeof args.path === "string" && args.path.startsWith("-")) {
  return "ERROR: path must not start with '-'.";
}

if (typeof args.path === "string" && path.isAbsolute(args.path)) {
  return "ERROR: absolute paths are not allowed.";
}

const candidate = path.resolve(context.worktree, args.path);
const canonical = await fs.promises.realpath(candidate);
const canonicalRoot = await fs.promises.realpath(context.worktree);
if (
  !canonical.startsWith(`${canonicalRoot}${path.sep}`) &&
  canonical !== canonicalRoot
) {
  return "ERROR: path resolves outside repository root.";
}
```

Also reject traversal-oriented inputs such as `..` segments when your wrapper
contract expects only repo-relative paths.

### Delegator / Compatibility Wrappers

When two tools need the same low-level implementation logic (for example, a
focused wrapper and a compatibility wrapper around the same CLI), keep the
wrapper entrypoints separate and **do not import one tool from another**.
Repository policy may allow a narrow shared helper module for the common logic;
otherwise inline the logic in both files.

```typescript
// my_tool_shared.ts - helper only, not a tool wrapper export file
export function executeSharedLogic(args: Record<string, unknown>): string {
  // ... shared helper logic
}

// my_tool_read.ts - wrapper entrypoint with a read-only gate
import { tool } from "@opencode-ai/plugin";
import { executeSharedLogic } from "./my_tool_shared";

const READ_COMMANDS = ["list", "show"] as const;

export default tool({
  description: "Read-only wrapper",
  args: { command: tool.schema.enum([...READ_COMMANDS]) },
  async execute(args) {
    return executeSharedLogic(args as Record<string, unknown>);
  },
});
```

### Spawning Subprocesses

Two options depending on whether you need streaming or sync execution:

```typescript
// Async with Bun shell (captures stdout as text)
const result = await Bun.$`uv run my-command --flag ${value}`.text();

// Sync with explicit pipe handling
const proc = Bun.spawnSync({
  cmd: ["uv", "run", "my-command", "--flag", value],
  stdout: "pipe",
  stderr: "pipe",
  timeout: 120_000,
});
const decoder = new TextDecoder();
const stdout = proc.stdout ? decoder.decode(proc.stdout) : "";
const stderr = proc.stderr ? decoder.decode(proc.stderr) : "";
```

---

## Companion Markdown Files

A `.md` file with the same base name as the tool (e.g., `my_tool.md` for
`my_tool.ts`) is injected as extended context for the agent when the tool is
available. Use it for:

- Detailed usage examples
- Routing guidance (when to use this tool vs. alternatives)
- Contract documentation (success/failure markers)

The `.md` file does not affect the tool's schema or behavior -- it is purely
informational context for the LLM. `help: true` output may point callers to
that companion file for extended usage guidance, but the `.md` content still
does not alter runtime behavior.

---

## Validation

Always test your tool with:

```bash
opencode debug agent build
```

This loads all tools and prints the agent configuration. If your tool appears in
the `"tools"` object, it loaded successfully. Any errors will show the failing
module and reason.

If your edit changes headings, anchors, wrapper names, or referenced guide/tool
paths, run the relevant docs/reference validation as part of the same change.

---

## Summary of Rules

- Keep counted direct args at four or fewer (excluding bounded `options` and
  usage-only `help`). Violation consequence: over-broad schema; split into
  focused wrappers instead.
- Do not import one tool wrapper from another as the primary implementation
  path. Violation consequence: build/policy failure or incorrect wrapper
  boundaries.
- Every export must be a `tool()` definition. Violation consequence:
  `Object.entries requires that input parameter not be null or undefined`.
- `execute` must return a string or `{ output: string }`. Violation
  consequence: type error at runtime.
- The file must use `@opencode-ai/plugin` or `zod`. Violation consequence:
  tool not recognized.

Keep each wrapper entrypoint self-contained, validate with
`opencode debug agent build`, and use only policy-approved narrow shared
helpers instead of wrapper-to-wrapper delegation.
