import { tool } from "@opencode-ai/plugin";
import { validateCwdWithinRepo, validatePathWithinRepo } from "./lib/path_validation";
import { validatePytestTimeoutSeconds } from "./lib/pytest_validation";

// --- Inlined from lib/run_pytest_shared.ts ---

type OutputMode = "summary" | "full" | "json";

type ParsedBasicOptions = {
  outputMode?: OutputMode;
  failFast?: true;
  testFilter?: string;
};

type ParsedBasicOptionsResult =
  | { ok: true; options: ParsedBasicOptions }
  | { ok: false; error: string };

const BASIC_OPTION_RULES = new Set(["output", "fail-fast", "test-filter"]);
const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);

const ADVANCED_PYTEST_KEYS = new Set([
  "coverage",
  "coverageSource",
  "coverageThreshold",
  "covReport",
  "durations",
  "durationsMin",
  "overrideIni",
  "pytestArgs",
]);

const LEGACY_DIRECT_KEYS = new Set(["outputMode", "failFast", "testFilter"]);

const hasExplicitAdvancedKey = (args: Record<string, unknown>): string | undefined => {
  for (const key of ADVANCED_PYTEST_KEYS) {
    if (Object.hasOwn(args, key)) {
      return key;
    }
  }

  return undefined;
};

const hasLegacyDirectKey = (args: Record<string, unknown>): string | undefined => {
  for (const key of LEGACY_DIRECT_KEYS) {
    if (Object.hasOwn(args, key)) {
      return key;
    }
  }

  return undefined;
};

const tokenizeOptions = (options: string): { ok: true; tokens: string[] } | { ok: false; error: string } => {
  const tokens: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;

  for (let index = 0; index < options.length; index += 1) {
    const char = options[index];
    if (quote) {
      current += char;
      if (char === quote) {
        quote = undefined;
      }
      continue;
    }

    if (char === "'" || char === '"') {
      quote = char;
      current += char;
      continue;
    }

    if (/\s/.test(char)) {
      if (current) {
        tokens.push(current);
        current = "";
      }
      continue;
    }

    current += char;
  }

  if (quote) {
    return { ok: false, error: "ERROR: Invalid options string: unterminated quoted value." };
  }
  if (current) {
    tokens.push(current);
  }

  return { ok: true, tokens };
};

const stripOptionalQuotes = (value: string): string => {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === '"' || first === "'") && last === first) {
      return value.slice(1, -1);
    }
  }
  return value;
};

const parseBasicOptions = (options: unknown): ParsedBasicOptionsResult => {
  if (options === undefined || options === null) {
    return { ok: true, options: {} };
  }
  if (typeof options !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalized = options.trim();
  if (!normalized) {
    return { ok: true, options: {} };
  }

  const tokenized = tokenizeOptions(normalized);
  if (!tokenized.ok) {
    return tokenized;
  }

  const parsed: ParsedBasicOptions = {};
  for (const token of tokenized.tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }

    if (separatorIndex === -1) {
      if (!BASIC_OPTION_RULES.has(token)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': token is not supported.` };
      }
      if (token !== "fail-fast") {
        return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
      }
      if (parsed.failFast) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.failFast = true;
      continue;
    }

    const name = token.slice(0, separatorIndex);
    const rawValue = token.slice(separatorIndex + 1);
    if (!BASIC_OPTION_RULES.has(name)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token is not supported.` };
    }
    if (!rawValue) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }
    if (name === "fail-fast") {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token does not accept a value.` };
    }

    const value = stripOptionalQuotes(rawValue).trim();
    if (!value) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.` };
    }

    if (name === "output") {
      if (!OUTPUT_MODES.has(value as OutputMode)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': output must be one of summary, full, json.` };
      }
      if (parsed.outputMode !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.outputMode = value as OutputMode;
      continue;
    }

    if (name === "test-filter") {
      if (parsed.testFilter !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.testFilter = value;
      continue;
    }
  }

  return { ok: true, options: parsed };
};

const validatePositiveFiniteNumber = (name: string, value: unknown): string | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return `ERROR: ${name} must be a positive finite number.`;
  }
  return undefined;
};

const validateTestPathWithinRepo = (
  testPath: string | undefined,
  cwd: string | undefined,
): string | undefined => {
  return validatePathWithinRepo(testPath, "testPath", cwd);
};

const getRoutineArgs = (args: Record<string, unknown>, optionArgs: ParsedBasicOptions = {}) => {
  const outputMode = optionArgs.outputMode || (args.outputMode as OutputMode | undefined) || "summary";
  const minTests = (args.minTests as number | undefined) ?? 1;
  const timeout = (args.timeout as number | undefined) ?? 600;
  const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
  const failFast = optionArgs.failFast === true || args.failFast === true;

  return { outputMode, minTests, timeout, cwd, failFast };
};

const buildBasePytestCommand = (
  args: Record<string, unknown>,
  optionArgs: ParsedBasicOptions = {},
): (string | number)[] => {
  const { outputMode, minTests, timeout, cwd, failFast } = getRoutineArgs(args, optionArgs);
  const cmdParts: (string | number)[] = [
    "python3",
    `${import.meta.dir}/run_pytest.py`,
    `--output=${outputMode}`,
    `--min-tests=${minTests}`,
    `--timeout=${timeout}`,
  ];

  if (cwd) {
    cmdParts.push(`--cwd=${cwd}`);
  }
  if (failFast) {
    cmdParts.push("--fail-fast");
  }

  return cmdParts;
};

const appendRoutineTargeting = (
  cmdParts: (string | number)[],
  args: Record<string, unknown>,
  optionArgs: ParsedBasicOptions = {},
): string | undefined => {
  const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
  const testFilter = optionArgs.testFilter
    ?? (typeof args.testFilter === "string" ? args.testFilter.trim() : undefined);
  const testPath = typeof args.testPath === "string" ? args.testPath.trim() : undefined;

  if (testFilter) {
    cmdParts.push("-k", testFilter);
  }
  if (testPath) {
    if (testPath.startsWith("-")) {
      return "ERROR: testPath must not start with '-' (potential option injection).";
    }
    const testPathError = validateTestPathWithinRepo(testPath, cwd);
    if (testPathError) {
      return testPathError;
    }
    cmdParts.push(testPath);
  }

  return undefined;
};

const executePytestCommand = async (cmdParts: (string | number)[]): Promise<string> => {
  try {
    const result = await Bun.$`${cmdParts}`.text();
    return result || "Pytest completed but returned no output.";
  } catch (error: any) {
    const stdout = error.stdout?.toString?.() || "";
    const stderr = error.stderr?.toString?.() || "";
    const message = error.message || "Unknown error";

    if (stdout.trim()) {
      return stdout;
    }
    if (stderr.trim()) {
      return `ERROR: Failed to run pytest: ${stderr}`;
    }
    return `ERROR: Failed to run pytest: ${message}`;
  }
};

// --- Tool definition ---

export default tool({
  description: "Run routine pytest checks only (basic wrapper). Rejects advanced coverage/duration/passthrough controls.",
  args: {
    minTests: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
    cwd: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
    testPath: tool.schema.string().optional(),
  },
  async execute(args) {
    const legacyDirectKey = hasLegacyDirectKey(args as Record<string, unknown>);
    if (legacyDirectKey) {
      return `ERROR: run_pytest_basic does not accept direct field '${legacyDirectKey}'. Use 'options' instead.`;
    }

    const forbiddenKey = hasExplicitAdvancedKey(args as Record<string, unknown>);
    if (forbiddenKey) {
      return `ERROR: run_pytest_basic does not accept advanced option '${forbiddenKey}'. Use run_pytest_advanced.`;
    }

    const parsedOptions = parseBasicOptions((args as Record<string, unknown>).options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const { minTests, timeout } = getRoutineArgs(args as Record<string, unknown>, parsedOptions.options);
    const minTestsError = validatePositiveFiniteNumber("minTests", minTests);
    if (minTestsError) {
      return minTestsError;
    }
    const timeoutError = validatePytestTimeoutSeconds(timeout);
    if (timeoutError) {
      return timeoutError;
    }

    const cwdError = validateCwdWithinRepo(
      getRoutineArgs(args as Record<string, unknown>, parsedOptions.options).cwd,
    );
    if (cwdError) {
      return cwdError;
    }

    const cmdParts = buildBasePytestCommand(args as Record<string, unknown>, parsedOptions.options);
    const targetingError = appendRoutineTargeting(
      cmdParts,
      args as Record<string, unknown>,
      parsedOptions.options,
    );
    if (targetingError) {
      return targetingError;
    }

    return executePytestCommand(cmdParts);
  },
});
