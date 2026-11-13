import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/env_utils ---

function sanitizedEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    env[key] = value;
  }
  return env;
}

// --- Inlined from lib/adw_execute_shared ---

const decoder = new TextDecoder();
const ERROR_SNIPPET_LIMIT = 2000;
const SENSITIVE_PATTERNS: RegExp[] = [
  /\b(ghp_[A-Za-z0-9]{20,})\b/g,
  /\b(glpat-[A-Za-z0-9\-_]{20,})\b/g,
  /\b(sk-[A-Za-z0-9]{20,})\b/g,
  /(token\s*[=:]\s*)([^\s,;]+)/gi,
  /(password\s*[=:]\s*)([^\s,;]+)/gi,
  /(secret\s*[=:]\s*)([^\s,;]+)/gi,
  /(authorization\s*:\s*bearer\s+)([^\s]+)/gi,
];
function redactSensitive(value: string): string {
  let redacted = value;
  for (const pattern of SENSITIVE_PATTERNS) {
    redacted = redacted.replace(pattern, (_, prefix: string) => {
      if (prefix && (prefix.endsWith("=") || prefix.endsWith(":") || /bearer\s+$/i.test(prefix))) {
        return `${prefix}[REDACTED]`;
      }
      return "[REDACTED]";
    });
  }
  return redacted;
}
function sanitizeSnippet(value: string, limit: number = ERROR_SNIPPET_LIMIT): string {
  if (!value) return "";
  const normalized = redactSensitive(value.replace(/\r\n?/g, "\n")).trim();
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, limit).trimEnd()}... [truncated]`;
}
function executeAdwCommand(command: string, cmdParts: string[], timeoutMs: number): string {
  try {
    const result = Bun.spawnSync({
      cmd: cmdParts,
      stdout: "pipe",
      stderr: "pipe",
      timeout: timeoutMs,
      env: sanitizedEnv(),
    });
    const stdoutRaw = result.stdout ? decoder.decode(result.stdout) : "";
    const stderrRaw = result.stderr ? decoder.decode(result.stderr) : "";
    const timedOut = Boolean((result as { timedOut?: boolean }).timedOut);
    if (timedOut) {
      const output = sanitizeSnippet(stderrRaw || stdoutRaw || `Command timed out after ${timeoutMs}ms`);
      return `ERROR: Failed to execute 'adw ${command}' (timeout after ${timeoutMs}ms).\n${output}`;
    }
    if (result.exitCode !== 0) {
      const safeStderr = sanitizeSnippet(stderrRaw);
      const safeStdout = sanitizeSnippet(stdoutRaw);
      const output = safeStderr || safeStdout || `Exit code ${result.exitCode}`;
      return `ERROR: Failed to execute 'adw ${command}' (exit ${result.exitCode}).\n${output}`;
    }
    const output = sanitizeSnippet(stdoutRaw);
    if (!output) {
      return `ADW Command: ${command}\n\nadw ${command} completed with no output.`;
    }
    return `ADW Command: ${command}\n\n${output}`;
  } catch (error: any) {
    const stderr = error?.stderr instanceof Uint8Array
      ? decoder.decode(error.stderr)
      : typeof error?.stderr === "string" ? error.stderr : "";
    const stdout = error?.stdout instanceof Uint8Array
      ? decoder.decode(error.stdout)
      : typeof error?.stdout === "string" ? error.stdout : "";
    const message = error?.message ? String(error.message) : "";
    const safeStderr = sanitizeSnippet(stderr);
    const safeStdout = sanitizeSnippet(stdout);
    const safeMessage = sanitizeSnippet(message);
    const fallback = safeStderr || safeStdout || safeMessage || "Unknown execution error";
    return `ERROR: Failed to execute 'adw ${command}' (execution error).\n${fallback}`;
  }
}

// --- Inlined from lib/adw_shared ---

const DEFAULT_TIMEOUT_MS = 120_000;
const WORKFLOW_TIMEOUT_MS = 600_000;

const WORKFLOW_COMMAND_SET = new Set([
  "complete", "patch", "plan", "build", "test", "review", "document", "ship",
]);

function getCommandTimeout(command: string): number {
  return WORKFLOW_COMMAND_SET.has(command) ? WORKFLOW_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;
}

export default tool({
  description: `Manage ADW service lifecycle commands (launch/stop).

SIMPLE EXAMPLES (copy these patterns):

Launch service:        { command: "launch" }
Launch local mode:     { command: "launch", mode: "local" }
Launch in background:  { command: "launch", background: true }
Stop service:          { command: "stop" }
Force stop:            { command: "stop", force: true }
Get help:              { command: "launch", help: true }

RULES:
- Wrapper supports only service commands: launch, stop.
- launch mode must be local or remote when provided.
- background/force must be booleans when provided.
- Set help: true to show command help.`,
  args: {
    command: tool.schema.enum(["launch", "stop"]),
    mode: tool.schema.enum(["local", "remote"]).optional(),
    background: tool.schema.boolean().optional(),
    force: tool.schema.boolean().optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(input) {
    const { command, mode, background, force, help } = input;

    const cmdParts = ["uv", "run", "--active", "adw", command];

    if (help) {
      cmdParts.push("--help");
      return executeAdwCommand(command, cmdParts, getCommandTimeout(command));
    }

    if (command === "launch" && force !== undefined) {
      return "ERROR: 'force' is only supported for command 'stop'.";
    }
    if (mode !== undefined && mode !== "local" && mode !== "remote") {
      return "ERROR: 'mode' must be 'local' or 'remote' when provided.";
    }
    if (command === "stop" && mode !== undefined) {
      return "ERROR: 'mode' is only supported for command 'launch'.";
    }
    if (command === "stop" && background !== undefined) {
      return "ERROR: 'background' is only supported for command 'launch'.";
    }

    if (command === "launch") {
      if (mode) {
        cmdParts.push("--mode", mode);
      }
      if (background === true) {
        cmdParts.push("--background");
      }
    }

    if (command === "stop" && force === true) {
      cmdParts.push("--force");
    }

    return executeAdwCommand(command, cmdParts, getCommandTimeout(command));
  },
});
