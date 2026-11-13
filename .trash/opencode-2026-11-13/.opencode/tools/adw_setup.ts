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
  "template",
]);

const ARGS_ALLOWED_COMMANDS = new Set(["docs", "pull-opencode", "template"]);
const EXTRA_PROTECTED_ARGS_FLAGS = new Set([
  "--help",
  "--with-templates",
  "--skip-templates",
  "--format",
]);

const SETUP_DOCS_SUBCOMMANDS = new Set(["scaffold", "apply", "token"]);
const SETUP_DOCS_TOKEN_SUBCOMMANDS = new Set(["list", "set", "remove"]);
const SETUP_DOCS_LANGUAGE_VALUES = new Set(["python", "cpp", "typescript", "minimal"]);
const SETUP_PULL_FLAG_RULES = new Map<string, { takesValue: boolean }>([
  ["--source-repo", { takesValue: true }],
  ["--source-path", { takesValue: true }],
  ["--dest", { takesValue: true }],
  ["--ref", { takesValue: true }],
  ["--dry-run", { takesValue: false }],
  ["--yes", { takesValue: false }],
  ["-y", { takesValue: false }],
  ["--preserve-manifest", { takesValue: true }],
  ["--preserve", { takesValue: false }],
  ["--no-preserve", { takesValue: false }],
]);

const SETUP_TEMPLATE_SUBCOMMANDS = new Set(["init", "apply", "extract", "validate", "token"]);
const SETUP_TEMPLATE_GITIGNORE_MODE_VALUES = new Set(["active", "commented"]);
const SETUP_TEMPLATE_FORMAT_VALUES = new Set(["json"]);
const SETUP_TEMPLATE_TOKEN_SUBCOMMANDS = new Set(["list", "add", "remove"]);

function splitFlagValue(arg: string): { flag: string; inlineValue?: string } {
  const equalsIndex = arg.indexOf("=");
  if (equalsIndex < 0) {
    return { flag: arg };
  }
  return {
    flag: arg.slice(0, equalsIndex),
    inlineValue: arg.slice(equalsIndex + 1),
  };
}

function validateDocsArgs(args: string[]): string | undefined {
  if (args.length === 0) {
    return "ERROR: 'args' for command 'docs' must include a supported subcommand.";
  }

  const [subcommand, ...rest] = args;
  if (!SETUP_DOCS_SUBCOMMANDS.has(subcommand)) {
    return `ERROR: Unsupported docs subcommand '${subcommand}'. Allowed subcommands: scaffold, apply, token.`;
  }

  if (subcommand === "scaffold") {
    let index = 0;
    while (index < rest.length) {
      const token = rest[index];
      const { flag, inlineValue } = splitFlagValue(token);
      if (flag === "--language") {
        const languageValue = inlineValue ?? rest[index + 1];
        if (!languageValue || !languageValue.trim() || languageValue.startsWith("-")) {
          return "ERROR: '--language' requires a non-empty value.";
        }
        if (!SETUP_DOCS_LANGUAGE_VALUES.has(languageValue.trim())) {
          return "ERROR: '--language' must be one of: python, cpp, typescript, minimal.";
        }
        if (inlineValue === undefined) {
          index += 1;
        }
      } else if (flag !== "--force" && flag !== "--no-detect") {
        return `ERROR: Unsupported docs arg '${token}' for subcommand 'scaffold'.`;
      }
      index += 1;
    }
    return undefined;
  }

  if (subcommand === "apply") {
    for (const token of rest) {
      if (token !== "--check" && token !== "--dry-run") {
        return `ERROR: Unsupported docs arg '${token}' for subcommand 'apply'.`;
      }
    }
    return undefined;
  }

  if (rest.length === 0) {
    return "ERROR: 'token' requires a supported nested subcommand (list, set, remove).";
  }

  const [tokenSubcommand, ...tokenRest] = rest;
  if (!SETUP_DOCS_TOKEN_SUBCOMMANDS.has(tokenSubcommand)) {
    return `ERROR: Unsupported docs token subcommand '${tokenSubcommand}'. Allowed values: list, set, remove.`;
  }
  if (tokenSubcommand === "list" && tokenRest.length !== 0) {
    return "ERROR: 'docs token list' does not accept additional args.";
  }
  if (tokenSubcommand === "set" && tokenRest.length !== 2) {
    return "ERROR: 'docs token set' requires exactly two positional args: <key> <value>.";
  }
  if (tokenSubcommand === "remove" && tokenRest.length !== 1) {
    return "ERROR: 'docs token remove' requires exactly one positional arg: <key>.";
  }
  if (tokenRest.some((value) => value.startsWith("-"))) {
    return "ERROR: 'docs token' positional args must not start with '-'.";
  }

  return undefined;
}

function validatePullArgs(command: string, args: string[]): string | undefined {
  let index = 0;
  while (index < args.length) {
    const token = args[index];
    if (!token.startsWith("-")) {
      return `ERROR: Unsupported positional arg '${token}' for command '${command}'.`;
    }

    const { flag, inlineValue } = splitFlagValue(token);
    const rule = SETUP_PULL_FLAG_RULES.get(flag);
    if (!rule) {
      return `ERROR: Unsupported passthrough flag '${flag}' for command '${command}'.`;
    }
    if (rule.takesValue) {
      if (inlineValue !== undefined) {
        if (!inlineValue.trim()) {
          return `ERROR: Flag '${flag}' requires a non-empty value.`;
        }
      } else {
        const next = args[index + 1];
        if (!next || next.startsWith("-")) {
          return `ERROR: Flag '${flag}' requires a non-empty value.`;
        }
        index += 1;
      }
    } else if (inlineValue !== undefined) {
      return `ERROR: Flag '${flag}' does not accept an '=value' suffix.`;
    }

    index += 1;
  }

  return undefined;
}

function validateTemplateArgs(args: string[]): string | undefined {
  if (args.length === 0) {
    return "ERROR: 'args' for command 'template' must include a supported subcommand.";
  }

  const [subcommand, ...rest] = args;
  if (!SETUP_TEMPLATE_SUBCOMMANDS.has(subcommand)) {
    return `ERROR: Unsupported template subcommand '${subcommand}'. Allowed subcommands: init, apply, extract, validate, token.`;
  }

  if (subcommand === "init") {
    let index = 0;
    while (index < rest.length) {
      const token = rest[index];
      const { flag, inlineValue } = splitFlagValue(token);
      if (flag === "--yes" || flag === "-y") {
        if (inlineValue !== undefined) {
          return `ERROR: Flag '${flag}' does not accept an '=value' suffix.`;
        }
      } else if (flag === "--gitignore-mode") {
        const modeValue = inlineValue ?? rest[index + 1];
        if (!modeValue || !modeValue.trim() || modeValue.startsWith("-")) {
          return "ERROR: '--gitignore-mode' requires a non-empty value.";
        }
        if (!SETUP_TEMPLATE_GITIGNORE_MODE_VALUES.has(modeValue.trim())) {
          return "ERROR: '--gitignore-mode' must be one of: active, commented.";
        }
        if (inlineValue === undefined) {
          index += 1;
        }
      } else {
        return `ERROR: Unsupported template arg '${token}' for subcommand 'init'.`;
      }
      index += 1;
    }
    return undefined;
  }

  if (subcommand === "apply" || subcommand === "extract") {
    const allowed = subcommand === "apply"
      ? new Set(["--check", "--dry-run", "--yes", "-y"])
      : new Set(["--diff", "--dry-run", "--yes", "-y"]);
    for (const token of rest) {
      if (!allowed.has(token)) {
        return `ERROR: Unsupported template arg '${token}' for subcommand '${subcommand}'.`;
      }
    }
    return undefined;
  }

  if (subcommand === "validate") {
    if (rest.length === 0) {
      return undefined;
    }
    if (rest.length !== 2) {
      return "ERROR: 'template validate' accepts only '--format <json>'.";
    }
    const [flag, value] = rest;
    if (flag !== "--format") {
      return `ERROR: Unsupported template arg '${flag}' for subcommand 'validate'.`;
    }
    if (!SETUP_TEMPLATE_FORMAT_VALUES.has(value)) {
      return "ERROR: '--format' must be one of: json.";
    }
    return undefined;
  }

  if (rest.length === 0) {
    return "ERROR: 'template token' requires a supported nested subcommand (list, add, remove).";
  }

  const [tokenSubcommand, ...tokenRest] = rest;
  if (!SETUP_TEMPLATE_TOKEN_SUBCOMMANDS.has(tokenSubcommand)) {
    return `ERROR: Unsupported template token subcommand '${tokenSubcommand}'. Allowed values: list, add, remove.`;
  }
  if (tokenSubcommand === "list") {
    if (tokenRest.length !== 0) {
      return "ERROR: 'template token list' does not accept additional args.";
    }
    return undefined;
  }
  if (tokenSubcommand === "remove") {
    if (tokenRest.length < 1 || tokenRest.length > 2) {
      return "ERROR: 'template token remove' requires '<key>' and optional '--yes'.";
    }
    const [key, maybeYes] = tokenRest;
    if (key.startsWith("-")) {
      return "ERROR: 'template token remove' key must not start with '-'.";
    }
    if (maybeYes !== undefined && maybeYes !== "--yes") {
      return `ERROR: Unsupported template arg '${maybeYes}' for subcommand 'token remove'.`;
    }
    return undefined;
  }

  if (tokenRest.length < 5) {
    return "ERROR: 'template token add' requires '<key> --default <value> --description <value>' and optional '--force'.";
  }

  const [key, ...flagRest] = tokenRest;
  if (key.startsWith("-")) {
    return "ERROR: 'template token add' key must not start with '-'.";
  }

  let sawDefault = false;
  let sawDescription = false;
  let index = 0;
  while (index < flagRest.length) {
    const token = flagRest[index];
    if (token === "--force") {
      index += 1;
      continue;
    }
    if (token !== "--default" && token !== "--description") {
      return `ERROR: Unsupported template arg '${token}' for subcommand 'token add'.`;
    }
    const value = flagRest[index + 1];
    if (!value || value.startsWith("-")) {
      return `ERROR: '${token}' requires a non-empty value.`;
    }
    if (token === "--default") {
      if (sawDefault) {
        return "ERROR: duplicate '--default' is not allowed.";
      }
      sawDefault = true;
    } else {
      if (sawDescription) {
        return "ERROR: duplicate '--description' is not allowed.";
      }
      sawDescription = true;
    }
    index += 2;
  }

  if (!sawDefault || !sawDescription) {
    return "ERROR: 'template token add' requires both '--default <value>' and '--description <value>'.";
  }

  return undefined;
}

function validateCommandScopedArgs(command: string, args: string[]): string | undefined {
  if (command === "docs") {
    return validateDocsArgs(args);
  }
  if (command === "pull-opencode") {
    return validatePullArgs(command, args);
  }
  if (command === "template") {
    return validateTemplateArgs(args);
  }
  return undefined;
}

function validateSetupArgs(command: string | undefined, args: string[]): string | undefined {
  const flagged = args.find((arg) =>
    [...EXTRA_PROTECTED_ARGS_FLAGS].some((flag) => {
      if (command === "template" && flag === "--format") {
        return false;
      }
      return arg === flag || arg.startsWith(`${flag}=`);
    })
  );
  if (!flagged) {
    return undefined;
  }

  const matched = [...EXTRA_PROTECTED_ARGS_FLAGS].find(
    (flag) => {
      if (command === "template" && flag === "--format") {
        return false;
      }
      return flagged === flag || flagged.startsWith(`${flag}=`);
    },
  );
  return `ERROR: Protected flag '${matched}' is not allowed in 'args'. Use top-level tool arguments instead.`;
}

const BOUNDED_OPTION_NAME_PATTERN = /^[a-z][a-z0-9-]*$/;
const SETUP_FORMAT_VALUES = new Set(["panel", "table", "json"]);

function parseSetupOptions(command: string | undefined, rawOptions: unknown):
  | {
    ok: true;
    withTemplates?: boolean;
    skipTemplates?: boolean;
    format?: "panel" | "table" | "json";
    dryRun?: boolean;
  }
  | { ok: false; error: string } {
  if (rawOptions === undefined || rawOptions === null) {
    return { ok: true };
  }
  if (typeof rawOptions !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalizedOptions = rawOptions.trim();
  if (!normalizedOptions) {
    return { ok: true };
  }
  if (!command) {
    return { ok: false, error: "ERROR: 'options' requires a setup 'command'." };
  }

  const values: {
    withTemplates?: boolean;
    skipTemplates?: boolean;
    format?: "panel" | "table" | "json";
    dryRun?: boolean;
  } = {};

  for (const token of normalizedOptions.split(/\s+/)) {
    const separatorCount = token.split("=").length - 1;
    if (separatorCount > 1) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }

    if (separatorCount === 0) {
      if (!BOUNDED_OPTION_NAME_PATTERN.test(token)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': token names must use lowercase-kebab-case.` };
      }

      if (token === "with-templates") {
        if (command !== "env") {
          return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for command '${command}'.` };
        }
        if (values.withTemplates) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'with-templates' token is not allowed.` };
        }
        if (values.skipTemplates) {
          return { ok: false, error: "ERROR: 'with-templates' and 'skip-templates' cannot be combined." };
        }
        values.withTemplates = true;
        continue;
      }

      if (token === "skip-templates") {
        if (command !== "env") {
          return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for command '${command}'.` };
        }
        if (values.skipTemplates) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'skip-templates' token is not allowed.` };
        }
        if (values.withTemplates) {
          return { ok: false, error: "ERROR: 'with-templates' and 'skip-templates' cannot be combined." };
        }
        values.skipTemplates = true;
        continue;
      }

      if (token === "dry-run") {
        if (command !== "labels") {
          return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for command '${command}'.` };
        }
        if (values.dryRun) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'dry-run' token is not allowed.` };
        }
        values.dryRun = true;
        continue;
      }

      return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for command '${command}'.` };
    }

    const [tokenName, rawValue] = token.split("=");
    if (!BOUNDED_OPTION_NAME_PATTERN.test(tokenName)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token names must use lowercase-kebab-case.` };
    }
    if (!rawValue) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token value must not be empty.` };
    }
    if (tokenName !== "format") {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for command '${command}'.` };
    }
    if (command !== "validate") {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token is not allowed for command '${command}'.` };
    }
    if (!SETUP_FORMAT_VALUES.has(rawValue)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': format values must be one of: panel, table, json.` };
    }
    if (values.format !== undefined) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate 'format' token is not allowed.` };
    }
    values.format = rawValue as "panel" | "table" | "json";
  }

  return { ok: true, ...values };
}

export default tool({
  description: `Manage ADW setup subcommands via a bounded wrapper. Only include parameters you need.

SIMPLE EXAMPLES (copy these patterns):

Wizard:            { wizard: true }
Group help:        { help: true }
Subcommand help:   { command: "env", help: true }
Env setup:         { command: "env", options: "with-templates" }
Validate:          { command: "validate", options: "format=json" }
Labels dry-run:    { command: "labels", options: "dry-run" }
Docs passthrough:  { command: "docs", args: ["apply", "--check"] }
Template check:    { command: "template", args: ["validate", "--format", "json"] }

RULES:
- Use wizard: true for bare 'adw setup' (no subcommand).
- wizard and command are mutually exclusive.
- args are only allowed for docs/pull-opencode/template.
- template commands are validated through bounded subcommand allowlists.`,
  args: {
    command: tool.schema
      .enum(["env", "validate", "check", "labels", "docs", "pull-opencode", "template"])
      .optional(),
    help: tool.schema.boolean().optional(),
    wizard: tool.schema.boolean().optional(),
    options: tool.schema.string().optional(),
    args: tool.schema.any().optional(),
  },
  async execute(input) {
    const {
      command,
      help,
      wizard,
      args: rawAdditionalArgs,
    } = input;

    if (wizard && command) {
      return "ERROR: 'wizard' cannot be combined with 'command'.";
    }

    if (help) {
      const cmdParts = ["uv", "run", "--active", "adw", "setup"];
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

    if (!wizard && !command) {
      return "ERROR: setup wrapper requires 'command' unless 'help' or 'wizard' is set.";
    }
    if (command && !SETUP_COMMANDS.has(command)) {
      return `ERROR: Unsupported setup command '${command}'.`;
    }

    const parsedOptions = parseSetupOptions(command, input.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    if (additionalArgs.length > 0 && (!command || !ARGS_ALLOWED_COMMANDS.has(command))) {
      return "ERROR: 'args' is only supported for commands 'docs', 'pull-opencode', and 'template'.";
    }
    const protectedArgsError = validateSetupArgs(command, additionalArgs);
    if (protectedArgsError) {
      return protectedArgsError;
    }
    const commandScopedArgsError = command ? validateCommandScopedArgs(command, additionalArgs) : undefined;
    if (commandScopedArgsError) {
      return commandScopedArgsError;
    }

    const cmdParts = ["uv", "run", "--active", "adw", "setup"];

    if (wizard) {
      // Intentionally no subcommand.
    } else if (command) {
      cmdParts.push(command);

      if (command === "env") {
        if (parsedOptions.withTemplates) cmdParts.push("--with-templates");
        if (parsedOptions.skipTemplates) cmdParts.push("--skip-templates");
      }
      if (command === "validate" && parsedOptions.format) {
        cmdParts.push("--format", parsedOptions.format);
      }
      if (command === "labels" && parsedOptions.dryRun) {
        cmdParts.push("--dry-run");
      }
      if (ARGS_ALLOWED_COMMANDS.has(command) && additionalArgs.length > 0) {
        cmdParts.push(...additionalArgs);
      }
    }

    return executeAdwCommand("setup", cmdParts, getCommandTimeout("setup"));
  },
});
