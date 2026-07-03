import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync } from "node:fs";
import path from "node:path";
import { validateCwdWithinRepo, validatePathWithinRepo } from "./lib/path_validation";
import { validatePytestTimeoutSeconds } from "./lib/pytest_validation";

// --- Inlined from lib/run_pytest_shared.ts ---

type OutputMode = "summary" | "full" | "json";

type ParsedAdvancedOptions = {
  outputMode?: OutputMode;
  failFast?: true;
  testFilter?: string;
  covReport?: string[];
  durations?: number;
  durationsMin?: number;
};

type ParsedAdvancedOptionsResult =
  | { ok: true; options: ParsedAdvancedOptions }
  | { ok: false; error: string };

const ADVANCED_OPTION_RULES = new Set([
  "output",
  "fail-fast",
  "test-filter",
  "cov-report",
  "durations",
  "durations-min",
]);
const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);
const LEGACY_DIRECT_KEYS = new Set([
  "outputMode",
  "failFast",
  "testFilter",
  "covReport",
  "durations",
  "durationsMin",
]);

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

const parseAdvancedOptions = (options: unknown): ParsedAdvancedOptionsResult => {
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

  const parsed: ParsedAdvancedOptions = {};
  for (const token of tokenized.tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }

    if (separatorIndex === -1) {
      if (!ADVANCED_OPTION_RULES.has(token)) {
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
    if (!ADVANCED_OPTION_RULES.has(name)) {
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

    if (name === "cov-report") {
      if (parsed.covReport !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      const entries = value.split(",").map((entry) => entry.trim()).filter(Boolean);
      if (entries.length === 0) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': cov-report requires at least one non-empty report value.` };
      }
      parsed.covReport = entries;
      continue;
    }

    if (name === "durations" || name === "durations-min") {
      const numericValue = Number(value);
      if (!Number.isFinite(numericValue)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': ${name} must be a finite number.` };
      }
      if (name === "durations") {
        if (parsed.durations !== undefined) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
        }
        parsed.durations = numericValue;
      } else {
        if (parsed.durationsMin !== undefined) {
          return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
        }
        parsed.durationsMin = numericValue;
      }
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

const validateNonNegativeFiniteNumber = (
  name: string,
  value: unknown,
): string | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return `ERROR: ${name} must be a non-negative finite number.`;
  }
  return undefined;
};

const validateTestPathWithinRepo = (
  testPath: string | undefined,
  cwd: string | undefined,
): string | undefined => {
  return validatePathWithinRepo(testPath, "testPath", cwd);
};

const COVERAGE_PYTEST_ARG_PATTERN = /^(--cov(?:=|\b)|--cov-report(?:=|\b)|--cov-fail-under(?:=|\b)|--cov-config(?:=|\b)|--cov-context(?:=|\b))/;

const getCoverageRepoRoot = (cwd: string | undefined): string => {
  if (cwd) {
    return realpathSync(cwd);
  }

  let current = realpathSync(process.cwd());
  while (true) {
    if (existsSync(path.join(current, "pyproject.toml")) || existsSync(path.join(current, ".git"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return realpathSync(process.cwd());
    }
    current = parent;
  }
};

const validateCoverageSourceEntry = (
  source: string,
  cwd: string | undefined,
): string | undefined => {
  if (!source.includes("/") && !source.includes("\\") && !source.startsWith(".") && !source.endsWith(".py")) {
    return undefined;
  }

  const repoRoot = getCoverageRepoRoot(cwd);
  const resolved = path.resolve(repoRoot, source);
  const rel = path.relative(repoRoot, resolved);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    return `ERROR: coverageSource must stay within the repository/worktree root: ${source}`;
  }
  return undefined;
};

const getRoutineArgs = (args: Record<string, unknown>, optionArgs: ParsedAdvancedOptions = {}) => {
  const outputMode = optionArgs.outputMode || (args.outputMode as OutputMode | undefined) || "summary";
  const minTests = (args.minTests as number | undefined) ?? 1;
  const timeout = (args.timeout as number | undefined) ?? 600;
  const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
  const failFast = optionArgs.failFast === true || args.failFast === true;

  return { outputMode, minTests, timeout, cwd, failFast };
};

const buildBasePytestCommand = (
  args: Record<string, unknown>,
  optionArgs: ParsedAdvancedOptions = {},
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
  optionArgs: ParsedAdvancedOptions = {},
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
      try {
        const payload = JSON.parse(stdout);
        if (
          payload
          && typeof payload === "object"
          && (
            ("success" in payload && payload.success === false)
            || ("ok" in payload && payload.ok === false)
          )
        ) {
          return stdout;
        }
      } catch {
        // Fall through to deterministic marker handling below.
      }
      if (stdout.includes("VALIDATION: FAILED") || stdout.trimStart().startsWith("ERROR:")) {
        return stdout;
      }
      return `ERROR: Failed to run pytest: subprocess exited unexpectedly and stdout did not report failure semantics. ${stdout}`;
    }
    if (stderr.trim()) {
      return `ERROR: Failed to run pytest: ${stderr}`;
    }
    return `ERROR: Failed to run pytest: ${message}`;
  }
};

const parseCoverageSources = (
  coverageSource: string,
  cwd: string | undefined,
): { ok: true; sources: string[] } | { ok: false; error: string } => {
  const rawEntries = coverageSource.split(",");
  const sources: string[] = [];

  for (const entry of rawEntries) {
    const trimmed = entry.trim();
    if (!trimmed) {
      return {
        ok: false,
        error: "ERROR: coverageSource must not contain empty comma-separated entries.",
      };
    }
    if (path.isAbsolute(trimmed)) {
      return {
        ok: false,
        error: `ERROR: coverageSource must not contain absolute paths: ${trimmed}`,
      };
    }
    const scopeError = validateCoverageSourceEntry(trimmed, cwd);
    if (scopeError) {
      return { ok: false, error: scopeError };
    }
    sources.push(trimmed);
  }

  if (sources.some((source) => source.toLowerCase() === "all")) {
    return { ok: true, sources: [] };
  }

  return { ok: true, sources };
};

// --- Tool definition ---

export default tool({
  description: "Run pytest with advanced controls (coverage, durations, override-ini, passthrough args).",
  args: {
    minTests: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
    cwd: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
    testPath: tool.schema.string().optional(),
    pytestArgs: tool.schema.array(tool.schema.string()).optional(),
    coverage: tool.schema.boolean().optional(),
    coverageSource: tool.schema.string().optional(),
    coverageThreshold: tool.schema.number().optional(),
    overrideIni: tool.schema.array(tool.schema.string()).optional(),
  },
  async execute(args) {
    const legacyDirectKey = hasLegacyDirectKey(args as Record<string, unknown>);
    if (legacyDirectKey) {
      return `ERROR: run_pytest_advanced does not accept direct field '${legacyDirectKey}'. Use 'options' instead.`;
    }

    const parsedOptions = parseAdvancedOptions((args as Record<string, unknown>).options);
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

    const coverage = args.coverage !== false;
    const cwd = getRoutineArgs(args as Record<string, unknown>, parsedOptions.options).cwd;
    const coverageSource = typeof args.coverageSource === "string" ? args.coverageSource.trim() : "";
    const coverageThreshold = args.coverageThreshold;
    const covReport = parsedOptions.options.covReport ?? (Array.isArray(args.covReport)
      ? (args.covReport as string[]).map((entry) => entry.trim()).filter(Boolean)
      : []);
    const durations = parsedOptions.options.durations ?? args.durations;
    const durationsMin = parsedOptions.options.durationsMin ?? args.durationsMin;
    const overrideIni = Array.isArray(args.overrideIni)
      ? (args.overrideIni as string[]).map((entry) => entry.trim()).filter(Boolean)
      : [];
    const pytestArgs = Array.isArray(args.pytestArgs)
      ? (args.pytestArgs as string[]).map((entry) => String(entry).trim()).filter(Boolean)
      : [];

    if (!coverage && pytestArgs.some((entry) => COVERAGE_PYTEST_ARG_PATTERN.test(entry))) {
      return "ERROR: coverage-related pytest arguments are not allowed when coverage is disabled";
    }

    if (coverage) {
      cmdParts.push("--coverage");
      if (coverageSource && coverageSource !== "all") {
        const parsedCoverageSources = parseCoverageSources(coverageSource, cwd);
        if (!parsedCoverageSources.ok) {
          return parsedCoverageSources.error;
        }
        const sources = parsedCoverageSources.sources;
        for (const source of sources) {
          cmdParts.push(`--coverage-source=${source}`);
        }
      }
      if (covReport.length > 0) {
        cmdParts.push(`--cov-report=${covReport.join(",")}`);
      }
    } else {
      cmdParts.push("--no-coverage");
    }

    if (coverageThreshold !== undefined) {
      const thresholdError = validateNonNegativeFiniteNumber("coverageThreshold", coverageThreshold);
      if (thresholdError) {
        return thresholdError;
      }
      cmdParts.push(`--coverage-threshold=${coverageThreshold}`);
    }

    if (durations !== undefined) {
      const durationsError = validateNonNegativeFiniteNumber("durations", durations);
      if (durationsError) {
        return durationsError;
      }
      cmdParts.push(`--durations=${durations}`);
      if (durationsMin !== undefined) {
        const durationsMinError = validateNonNegativeFiniteNumber("durationsMin", durationsMin);
        if (durationsMinError) {
          return durationsMinError;
        }
        cmdParts.push(`--durations-min=${durationsMin}`);
      }
    }

    for (const entry of overrideIni) {
      cmdParts.push(`--override-ini=${entry}`);
    }

    if (pytestArgs.length > 0) {
      cmdParts.push(...pytestArgs);
    }

    return executePytestCommand(cmdParts);
  },
});
