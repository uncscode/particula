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

const SETUP_COMMANDS = new Set([
  "env",
  "validate",
  "check",
  "labels",
  "docs",
  "pull-opencode",
  "pull-plans",
]);

const ARGS_ALLOWED_COMMANDS = new Set(["docs", "pull-opencode", "pull-plans"]);
const EXTRA_PROTECTED_ARGS_FLAGS = new Set([
  "--help",
  "--with-templates",
  "--skip-templates",
  "--format",
  "--dry-run",
]);

function validateSetupArgs(args: string[]): string | undefined {
  const flagged = args.find((arg) =>
    [...EXTRA_PROTECTED_ARGS_FLAGS].some((flag) => arg === flag || arg.startsWith(`${flag}=`))
  );
  if (!flagged) {
    return undefined;
  }

  const matched = [...EXTRA_PROTECTED_ARGS_FLAGS].find(
    (flag) => flagged === flag || flagged.startsWith(`${flag}=`),
  );
  return `ERROR: Protected flag '${matched}' is not allowed in 'args'. Use top-level tool arguments instead.`;
}

export default tool({
  description: `Manage ADW setup subcommands via a bounded wrapper. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

Wizard:            { wizard: true }
Group help:        { help: true }
Subcommand help:   { command: "env", help: true }
Env setup:         { command: "env", with_templates: true }
Validate:          { command: "validate", format: "json" }
Labels dry-run:    { command: "labels", dry_run: true }
Docs passthrough:  { command: "docs", args: ["--strict"] }

RULES:
- Use wizard: true for bare 'adw setup' (no subcommand).
- wizard and command are mutually exclusive.
- args are only allowed for docs/pull-opencode/pull-plans.
- 'template' setup subcommands are intentionally unsupported in this wrapper.`,
  args: {
    command: tool.schema
      .enum(["env", "validate", "check", "labels", "docs", "pull-opencode", "pull-plans"])
      .optional(),
    help: tool.schema.boolean().optional(),
    wizard: tool.schema.boolean().optional(),
    with_templates: tool.schema.boolean().optional(),
    skip_templates: tool.schema.boolean().optional(),
    format: tool.schema.enum(["panel", "table", "json"]).optional(),
    dry_run: tool.schema.boolean().optional(),
    args: tool.schema.any().optional(),
  },
  async execute(input) {
    const {
      command,
      help,
      wizard,
      with_templates,
      skip_templates,
      format,
      dry_run,
      args: rawAdditionalArgs,
    } = input;

    if (help) {
      const cmdParts = ["uv", "run", "adw", "setup"];
      if (command) {
        cmdParts.push(command, "--help");
      } else {
        cmdParts.push("--help");
      }
      return executeAdwCommand("setup", cmdParts, getCommandTimeout("setup"));
    }

    const argsValidation = validateAdditionalArgs(rawAdditionalArgs);
    if (!argsValidation.ok) {
      return argsValidation.error;
    }
    const additionalArgs = argsValidation.args;

    if (wizard && command) {
      return "ERROR: 'wizard' cannot be combined with 'command'.";
    }
    if (!wizard && !command) {
      return "ERROR: setup wrapper requires 'command' unless 'help' or 'wizard' is set.";
    }
    if (command && !SETUP_COMMANDS.has(command)) {
      return `ERROR: Unsupported setup command '${command}'.`;
    }

    if (with_templates && command !== "env") {
      return "ERROR: 'with_templates' is only supported for command 'env'.";
    }
    if (skip_templates && command !== "env") {
      return "ERROR: 'skip_templates' is only supported for command 'env'.";
    }
    if (with_templates && skip_templates) {
      return "ERROR: 'with_templates' and 'skip_templates' cannot be combined.";
    }

    if (format && command !== "validate") {
      return "ERROR: 'format' is only supported for command 'validate'.";
    }
    if (dry_run && command !== "labels") {
      return "ERROR: 'dry_run' is only supported for command 'labels'.";
    }

    if (additionalArgs.length > 0 && (!command || !ARGS_ALLOWED_COMMANDS.has(command))) {
      return "ERROR: 'args' is only supported for commands 'docs', 'pull-opencode', and 'pull-plans'.";
    }
    const protectedArgsError = validateSetupArgs(additionalArgs);
    if (protectedArgsError) {
      return protectedArgsError;
    }

    const cmdParts = ["uv", "run", "adw", "setup"];

    if (wizard) {
      // Intentionally no subcommand.
    } else if (command) {
      cmdParts.push(command);

      if (command === "env") {
        if (with_templates) cmdParts.push("--with-templates");
        if (skip_templates) cmdParts.push("--skip-templates");
      }
      if (command === "validate" && format) {
        cmdParts.push("--format", format);
      }
      if (command === "labels" && dry_run) {
        cmdParts.push("--dry-run");
      }
      if (ARGS_ALLOWED_COMMANDS.has(command) && additionalArgs.length > 0) {
        cmdParts.push(...additionalArgs);
      }
    }

    return executeAdwCommand("setup", cmdParts, getCommandTimeout("setup"));
  },
});
