import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";

type DirectoryResolution = { path?: string; rejected?: string };

const REPOSITORY_ROOT = realpathSync(resolve(import.meta.dir, "..", ".."));
const MAX_REJECTED_PATH = 500;

const canonicalDirectory = (value: unknown): DirectoryResolution => {
  if (value !== undefined && value !== null && typeof value !== "string") {
    return { rejected: "<non-string>" };
  }
  const supplied = typeof value === "string" ? value.trim() : "";
  if (supplied.startsWith("-")) return { rejected: supplied };
  const candidate = supplied
    ? isAbsolute(supplied)
      ? supplied
      : resolve(REPOSITORY_ROOT, supplied)
    : REPOSITORY_ROOT;
  try {
    if (!existsSync(candidate) || !statSync(candidate).isDirectory()) {
      return { rejected: candidate };
    }
    return { path: realpathSync(candidate) };
  } catch {
    return { rejected: candidate };
  }
};

const MAX_RENDERED_STDOUT = 12_000;
const MAX_ADAPTER_RESPONSE = 131_072;
const MAX_ERROR_FIELD = 500;

const render = (response: unknown, command: string): string => {
  if (!response || typeof response !== "object") return "ERROR: Git diagnostics adapter returned malformed output";
  const value = response as { ok?: unknown; data?: Record<string, unknown>; error?: { type?: unknown; message?: unknown } };
  if (value.ok !== true) {
    if (typeof value.error?.type !== "string" || typeof value.error?.message !== "string") {
      return "ERROR: Git diagnostics adapter returned malformed output";
    }
    return `ERROR: ${value.error.type.slice(0, MAX_ERROR_FIELD)}: ${value.error.message.slice(0, MAX_ERROR_FIELD)}`;
  }
  const data = value.data;
  if (!data || typeof data.stdout !== "string") return "ERROR: Git diagnostics adapter returned malformed output";
  const classification = data.status ?? data.diff ?? "ok";
  if (typeof classification !== "string") return "ERROR: Git diagnostics adapter returned malformed output";
  return `Git Command: ${command} (${classification})\n\n${data.stdout.slice(0, MAX_RENDERED_STDOUT)}`;
};

export default tool({
  description: "Execute read-only Git status, diff, log, and show diagnostics through the local adapter.",
  args: {
    command: tool.schema.enum(["status", "diff", "log", "show"]),
    worktree_path: tool.schema.string().optional(),
    base: tool.schema.string().optional(),
    target: tool.schema.string().optional(),
    ref: tool.schema.string().optional(),
    path: tool.schema.string().optional(),
    max_count: tool.schema.number().optional(),
  },
  async execute(rawArgs) {
    const resolution = canonicalDirectory(rawArgs.worktree_path);
    if (!resolution.path) {
      const rejected = JSON.stringify(
        (resolution.rejected ?? "<unresolved>").slice(0, MAX_REJECTED_PATH),
      );
      return `ERROR: invalid or untrusted worktree_path: ${rejected}`;
    }
    const worktree_path = resolution.path;
    const request: Record<string, unknown> = { command: rawArgs.command, worktree_path };
    for (const key of ["base", "target", "ref", "path", "max_count"]) {
      if (rawArgs[key] !== undefined && rawArgs[key] !== null) request[key] = rawArgs[key];
    }
    const adapter = join(import.meta.dir, "read_only_git_diagnostics.py");
    if (!existsSync(adapter)) return "ERROR: Git diagnostics adapter unavailable";
    try {
      const result = Bun.spawnSync({
        cmd: ["python3", adapter],
        // Bun's synchronous process API requires byte input for a child stdin.
        // Supplying a string causes a launch exception in real Bun, while mocks
        // accept it, so encode the bounded JSON request explicitly.
        stdin: new TextEncoder().encode(JSON.stringify(request)),
        stdout: "pipe",
        stderr: "pipe",
        timeout: 30_000,
      });
      if (result.exitCode !== 0) return "ERROR: Git diagnostics adapter failed";
      try {
        const stdout = Buffer.from(result.stdout);
        if (stdout.byteLength > MAX_ADAPTER_RESPONSE) return "ERROR: Git diagnostics adapter returned oversized output";
        return render(JSON.parse(stdout.toString("utf8")), String(rawArgs.command));
      } catch {
        return "ERROR: Git diagnostics adapter returned malformed output";
      }
    } catch {
      return "ERROR: Git diagnostics adapter unavailable";
    }
  },
});
