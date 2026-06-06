import { tool } from "@opencode-ai/plugin";

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

const PROTECTED_FLAGS = new Set([
  "--adw-id", "--model", "--help", "--title", "--body", "--text", "--source-issue",
]);

const PROTECTED_FLAG_ALIASES = new Map<string, string>([
  ["--adw_id", "--adw-id"],
  ["--source_issue", "--source-issue"],
]);

function normalizeFlagToken(arg: string): { original: string; key: string; canonicalKey: string } {
  const trimmed = arg.trim();
  const equalsIndex = trimmed.indexOf("=");
  const rawKey = equalsIndex >= 0 ? trimmed.slice(0, equalsIndex) : trimmed;
  const loweredKey = rawKey.toLowerCase();
  const canonicalKey = PROTECTED_FLAG_ALIASES.get(loweredKey) ?? loweredKey;
  return { original: trimmed, key: rawKey, canonicalKey };
}

function getCommandTimeout(command: string): number {
  return WORKFLOW_COMMAND_SET.has(command) ? WORKFLOW_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;
}

function validateAdditionalArgs(rawAdditionalArgs: unknown):
  | { ok: true; args: string[] }
  | { ok: false; error: string } {
  if (rawAdditionalArgs === undefined) return { ok: true, args: [] };
  if (!Array.isArray(rawAdditionalArgs)) {
    return { ok: false, error: "ERROR: Invalid 'args': expected an array of strings." };
  }
  const invalidElement = rawAdditionalArgs.find(
    (value) => typeof value !== "string" || value.trim().length === 0,
  );
  if (invalidElement !== undefined) {
    return { ok: false, error: "ERROR: Invalid 'args': all entries must be non-empty strings." };
  }
  const parsedArgs = rawAdditionalArgs.map((value) => (value as string).trim());
  if (rawAdditionalArgs.length > 0 && parsedArgs.length === 0) {
    return { ok: false, error: "ERROR: Invalid 'args': no usable arguments after validation." };
  }
  const protectedFlag = parsedArgs
    .map((arg) => normalizeFlagToken(arg))
    .find((token) => PROTECTED_FLAGS.has(token.canonicalKey));
  if (protectedFlag) {
    const matchedFlag = protectedFlag.canonicalKey;
    return {
      ok: false,
      error: `ERROR: Protected flag '${matchedFlag}' is not allowed in 'args'. Use top-level tool arguments instead.`,
    };
  }
  return { ok: true, args: parsedArgs };
}

export default tool({
  description: `Execute ADW read-only status commands.

SIMPLE EXAMPLES (copy these patterns):

Check status: { command: "status" }
Health check: { command: "health" }

RULES:
- issue_number is not required.
- Omit optional fields entirely -- blank strings are treated as omitted.
- Set help: true to see CLI usage for any command.`,
  args: {
    command: tool.schema.enum(["status", "health"]),
    adw_id: tool.schema.string().optional(),
    args: tool.schema.any().optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(args) {
    const { command, args: rawAdditionalArgs, help } = args;

    const additionalArgsValidation = help
      ? { ok: true as const, args: [] }
      : validateAdditionalArgs(rawAdditionalArgs);
    if (!additionalArgsValidation.ok) {
      return additionalArgsValidation.error;
    }
    const additionalArgs = additionalArgsValidation.args;

    const cmdParts = ["uv", "run", "adw", command];
    if (help) {
      cmdParts.push("--help");
    }

    if (additionalArgs.length > 0) {
      cmdParts.push(...additionalArgs);
    }

    const commandTimeout = getCommandTimeout(command);
    return executeAdwCommand(command, cmdParts, commandTimeout);
  },
});
