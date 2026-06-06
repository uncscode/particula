import { tool } from "@opencode-ai/plugin";

// --- Inlined from lib/adw_id ---

const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;

function normalizeAdwId(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  if (!ADW_ID_PATTERN.test(trimmed)) return null;
  return trimmed.toLowerCase();
}

function adwIdValidationMessage(): string {
  return "Invalid 'adw_id': must be an 8-character hexadecimal string (e.g. 'a1b2c3d4').";
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

function normalizeOptionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  const trimmed = String(value).trim();
  return trimmed.length > 0 ? trimmed : undefined;
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
  description: `Execute ADW workflow commands (complete/patch/plan/build/test/review/document/ship).

SIMPLE EXAMPLES (copy these patterns):

Full workflow:  { command: "complete", issue_number: 123, model: "base" }
Quick patch:    { command: "patch", issue_number: 456 }
Resume work:    { command: "build", issue_number: 123, adw_id: "a1b2c3d4" }

RULES:
- Commands require issue_number unless help: true.
- Omit optional fields entirely -- blank strings are treated as omitted.
- Set help: true to see CLI usage for any command.`,
  args: {
    command: tool.schema.enum(["complete", "patch", "plan", "build", "test", "review", "document", "ship"]),
    issue_number: tool.schema.number().optional(),
    adw_id: tool.schema.string().optional(),
    model: tool.schema.enum(["light", "base", "heavy"]).optional(),
    args: tool.schema.any().optional(),
    help: tool.schema.boolean().optional(),
  },
  async execute(args) {
    const { command, issue_number, model, args: rawAdditionalArgs, help } = args;
    const adw_id = normalizeOptionalString(args.adw_id);

    const additionalArgsValidation = help
      ? { ok: true as const, args: [] }
      : validateAdditionalArgs(rawAdditionalArgs);
    if (!additionalArgsValidation.ok) {
      return additionalArgsValidation.error;
    }
    const additionalArgs = additionalArgsValidation.args;

    if (!help) {
      if (issue_number === undefined || issue_number === null) {
        return `ERROR: Command '${command}' requires 'issue_number' argument.\n\nUsage: adw ${command} <issue_number> [--adw-id <id>] [--model <light|base|heavy>]`;
      }
      if (!Number.isInteger(issue_number) || issue_number <= 0) {
        return `ERROR: Command '${command}' requires a positive integer 'issue_number'.`;
      }
    }

    const cmdParts = ["uv", "run", "adw", command];

    if (help) {
      cmdParts.push("--help");
    }

    if (issue_number !== undefined && issue_number !== null && !help) {
      cmdParts.push(issue_number.toString());
    }

    if (adw_id !== undefined) {
      const normalizedAdwId = normalizeAdwId(adw_id);
      if (!normalizedAdwId) {
        return `ERROR: ${adwIdValidationMessage()}`;
      }
      cmdParts.push("--adw-id", normalizedAdwId);
    }

    if (model) {
      cmdParts.push("--model", model);
    }

    if (additionalArgs.length > 0) {
      cmdParts.push(...additionalArgs);
    }

    const commandTimeout = getCommandTimeout(command);
    return executeAdwCommand(command, cmdParts, commandTimeout);
  },
});
