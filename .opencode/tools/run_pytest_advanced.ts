import { tool } from "@opencode-ai/plugin";
import { existsSync, lstatSync, realpathSync } from "node:fs";
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

const COVERAGE_SOURCE_INFO =
  "INFO: coverageSource supports only 'all' or existing repository-relative directories; dotted modules, file targets, and other non-directory entries are ignored.";

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
const PYTEST_VALUE_OPTIONS = new Set(["-k", "-m"]);
const PYTEST_STANDALONE_OPTIONS = new Set(["--collect-only", "-q", "-v", "--verbose"]);
const PYTEST_RESERVED_PREFIXES = [
  "--output", "--min-tests", "--timeout", "--cwd", "--test-path", "--test-filter",
  "--coverage", "--no-coverage", "--coverage-source", "--coverage-threshold", "--cov-report",
  "--fail-fast", "--durations", "--durations-min", "--pytest-argv-json", "--override-ini-json",
  "--override-ini", "--test-paths-json",
];
const MAX_TEST_PATHS = 7;

const validatePytestArgs = (value: unknown, cwd: string | undefined): { ok: true; value: string[] } | { ok: false; error: string } => {
  if (value === undefined) return { ok: true, value: [] };
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string")) {
    return { ok: false, error: "ERROR: pytestArgs must be an array of strings." };
  }
  for (let index = 0; index < value.length; index += 1) {
    const token = value[index] as string;
    if (token === "--" || COVERAGE_PYTEST_ARG_PATTERN.test(token) || PYTEST_RESERVED_PREFIXES.some((prefix) => token.startsWith(prefix))) {
      return { ok: false, error: `ERROR: pytestArgs token '${token}' is not permitted.` };
    }
    if (PYTEST_STANDALONE_OPTIONS.has(token) || (/^--tb=(short|long|line|native|no)$/).test(token)) continue;
    if (token.startsWith("--override-ini=") || token === "-o") {
      return { ok: false, error: `ERROR: pytestArgs token '${token}' is not permitted.` };
    }
    if (PYTEST_VALUE_OPTIONS.has(token)) {
      const option = value[index + 1];
      if (typeof option !== "string" || option.startsWith("-") || !option) return { ok: false, error: `ERROR: pytestArgs token '${token}' has an invalid value.` };
      index += 1;
      continue;
    }
    if (token.startsWith("-") || path.isAbsolute(token) || validatePathWithinRepo(token, "pytestArgs", cwd)) return { ok: false, error: `ERROR: pytestArgs token '${token}' is not permitted.` };
  }
  return { ok: true, value: value as string[] };
};

const validateStringArray = (value: unknown, name: string): { ok: true; value: string[] } | { ok: false; error: string } => {
  if (value === undefined) return { ok: true, value: [] };
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string")) return { ok: false, error: `ERROR: ${name} must be an array of strings.` };
  if (value.length > 0) {
    return { ok: false, error: "ERROR: overrideIni controls are not permitted." };
  }
  return { ok: true, value: value as string[] };
};

const validateTestPaths = (
  value: unknown,
  cwd: string | undefined,
): { ok: true; value: string[] } | { ok: false; error: string } => {
  if (!Array.isArray(value) || value.length === 0) return { ok: false, error: "ERROR: testPaths must be a non-empty array of strings." };
  if (value.length > MAX_TEST_PATHS) return { ok: false, error: `ERROR: testPaths must contain at most ${MAX_TEST_PATHS} entries.` };
  for (let index = 0; index < value.length; index += 1) {
    const target = value[index];
    const label = `testPaths[${index}]`;
    if (typeof target !== "string" || !target || target.startsWith("-")) return { ok: false, error: `ERROR: ${label} must be a non-empty repository-relative target.` };
    const pathPart = target.split("::", 2)[0];
    if (pathPart.includes("\\") || path.isAbsolute(pathPart) || pathPart.split("/").some((part) => !part || part === "." || part === "..")) return { ok: false, error: `ERROR: ${label} must be a canonical relative POSIX target.` };
    const pathError = validatePathWithinRepo(pathPart, label, cwd);
    if (pathError) return { ok: false, error: pathError };
  }
  return { ok: true, value: value as string[] };
};

const getCoverageRepoRoot = (cwd: string | undefined): string => {
  const requestedRoot = realpathSync(cwd ?? process.cwd());
  let current = requestedRoot;
  while (true) {
    if (existsSync(path.join(current, "pyproject.toml")) || existsSync(path.join(current, ".git"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return requestedRoot;
    }
    current = parent;
  }
};

const classifyCoverageSourceEntry = (
  source: string,
  cwd: string | undefined,
): { kind: "valid" } | { kind: "unsupported" } | { kind: "error"; error: string } => {
  if (source.includes("\\")) {
    return { kind: "error", error: `ERROR: coverageSource must be a relative POSIX path: ${source}` };
  }
  if (path.isAbsolute(source)) {
    return { kind: "error", error: `ERROR: coverageSource must be a relative POSIX path: ${source}` };
  }

  const repoRoot = getCoverageRepoRoot(cwd);
  const resolved = path.resolve(repoRoot, source);
  const rel = path.relative(repoRoot, resolved);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    return {
      kind: "error",
      error: `ERROR: coverageSource must stay within the repository/worktree root: ${source}`,
    };
  }
  if (source.split("/").some((part) => !part || part === "." || part === "..")) {
    return {
      kind: "error",
      error: `ERROR: coverageSource has noncanonical path components: ${source}`,
    };
  }
  if (!existsSync(resolved)) return { kind: "unsupported" };
  const target = lstatSync(resolved);
  if (target.isSymbolicLink()) {
    return {
      kind: "error",
      error: `ERROR: coverageSource must be an existing safe directory: ${source}`,
    };
  }
  if (!target.isDirectory()) return { kind: "unsupported" };
  return { kind: "valid" };
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
    cmdParts.push(`--test-filter=${testFilter}`);
  }
  if (testPath) {
    if (testPath.startsWith("-")) {
      return "ERROR: testPath must not start with '-' (potential option injection).";
    }
    const testPathError = validateTestPathWithinRepo(testPath, cwd);
    if (testPathError) {
      return testPathError;
    }
    cmdParts.push(`--test-path=${testPath}`);
  }

  return undefined;
};

const executePytestCommand = async (cmdParts: (string | number)[], outputMode: OutputMode): Promise<string> => {
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
      return outputMode === "json"
        ? JSON.stringify({ success: false, outcome: { classification: "runner", reason: "pytest runner returned an invalid failure envelope", phase: "pre_spawn" } })
        : "ERROR: Failed to run pytest: subprocess exited unexpectedly and stdout did not report failure semantics.";
    }
    if (stderr.trim()) {
      return outputMode === "json"
        ? JSON.stringify({ success: false, outcome: { classification: "runner", reason: "pytest runner failed without a valid failure envelope", phase: "pre_spawn" } })
        : `ERROR: Failed to run pytest: ${stderr}`;
    }
    return outputMode === "json"
      ? JSON.stringify({ success: false, outcome: { classification: "runner", reason: "pytest runner failed without a valid failure envelope", phase: "pre_spawn" } })
      : "ERROR: Failed to run pytest.";
  }
};

const parseCoverageSources = (
  coverageSource: string,
  cwd: string | undefined,
): { ok: true; sources: string[]; ignoredUnsupported: boolean } | { ok: false; error: string } => {
  const rawEntries = coverageSource.split(",");
  const sources: string[] = [];
  let ignoredUnsupported = false;

  if (rawEntries.some((entry) => entry.trim().toLowerCase() === "all") && rawEntries.length !== 1) {
    return { ok: false, error: "ERROR: coverageSource 'all' must be the sole source." };
  }

  for (const entry of rawEntries) {
    const trimmed = entry.trim();
    if (!trimmed) {
      return {
        ok: false,
        error: "ERROR: coverageSource must not contain empty comma-separated entries.",
      };
    }
    if (trimmed.toLowerCase() === "all") {
      return { ok: true, sources: [], ignoredUnsupported: false };
    }
    const classification = classifyCoverageSourceEntry(trimmed, cwd);
    if (classification.kind === "error") {
      return { ok: false, error: classification.error };
    }
    if (classification.kind === "unsupported") {
      ignoredUnsupported = true;
      continue;
    }
    sources.push(trimmed);
  }

  return { ok: true, sources, ignoredUnsupported };
};

const addCoverageSourceInfo = (output: string, outputMode: OutputMode): string => {
  if (outputMode !== "json") return `${COVERAGE_SOURCE_INFO}\n${output}`;
  try {
    const payload = JSON.parse(output);
    payload.info = [...(Array.isArray(payload.info) ? payload.info : []), COVERAGE_SOURCE_INFO];
    return JSON.stringify(payload, null, 2);
  } catch {
    return output;
  }
};

// --- Tool definition ---

export default tool({
  description: "Run pytest with advanced controls and a validated ordered pytest argv suffix.",
  args: {
    minTests: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
    cwd: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
    testPath: tool.schema.string().optional(),
    testPaths: tool.schema.array(tool.schema.string()).optional(),
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

    const rawArgs = args as Record<string, unknown>;
    if (Object.hasOwn(rawArgs, "testPath") && Object.hasOwn(rawArgs, "testPaths")) {
      return "ERROR: testPath and testPaths cannot be combined.";
    }
    const testPathsResult = Object.hasOwn(rawArgs, "testPaths")
      ? validateTestPaths(rawArgs.testPaths, getRoutineArgs(rawArgs, parsedOptions.options).cwd)
      : { ok: true as const, value: [] as string[] };
    if (!testPathsResult.ok) return testPathsResult.error;

    const cmdParts = buildBasePytestCommand(args as Record<string, unknown>, parsedOptions.options);
    const targetingError = appendRoutineTargeting(
      cmdParts,
      args as Record<string, unknown>,
      parsedOptions.options,
    );
    if (targetingError) {
      return targetingError;
    }
    if (testPathsResult.value.length > 0) cmdParts.push(`--test-paths-json=${JSON.stringify(testPathsResult.value)}`);

    const coverage = args.coverage !== false;
    const cwd = getRoutineArgs(args as Record<string, unknown>, parsedOptions.options).cwd;
    const coverageThreshold = args.coverageThreshold;
    let ignoredCoverageSources = false;
    const durations = parsedOptions.options.durations ?? args.durations;
    const durationsMin = parsedOptions.options.durationsMin ?? args.durationsMin;
    const overrideIniResult = validateStringArray(args.overrideIni, "overrideIni");
    if (!overrideIniResult.ok) return overrideIniResult.error;
    const pytestArgsResult = validatePytestArgs(args.pytestArgs, cwd);
    if (!pytestArgsResult.ok) return pytestArgsResult.error;
    const overrideIni = overrideIniResult.value;
    const pytestArgs = pytestArgsResult.value;

    if (pytestArgs.includes("--collect-only") && args.coverage !== false) {
      return "ERROR: --collect-only requires coverage: false.";
    }

    if (coverage) {
      const coverageSource = typeof args.coverageSource === "string" ? args.coverageSource.trim() : "";
      const covReport = parsedOptions.options.covReport ?? (Array.isArray(args.covReport)
        ? (args.covReport as string[]).map((entry) => entry.trim()).filter(Boolean)
        : []);

      cmdParts.push("--coverage");
      if (coverageSource && coverageSource !== "all") {
        const parsedCoverageSources = parseCoverageSources(coverageSource, cwd);
        if (!parsedCoverageSources.ok) {
          return parsedCoverageSources.error;
        }
        ignoredCoverageSources = parsedCoverageSources.ignoredUnsupported;
        const sources = parsedCoverageSources.sources;
        for (const source of sources) {
          cmdParts.push(`--coverage-source=${source}`);
        }
      }
      if (covReport.length > 0) {
        cmdParts.push(`--cov-report=${covReport.join(",")}`);
      }
    } else {
      if (
        typeof args.coverageSource === "string"
        || args.coverageThreshold !== undefined
        || parsedOptions.options.covReport !== undefined
        || (Array.isArray(args.covReport) && args.covReport.length > 0)
      ) {
        return "ERROR: coverage-specific controls are not allowed when coverage is disabled.";
      }
      cmdParts.push("--no-coverage");
    }

    if (coverage && coverageThreshold !== undefined) {
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

    if (overrideIni.length > 0) cmdParts.push(`--override-ini-json=${JSON.stringify(overrideIni)}`);
    if (pytestArgs.length > 0) cmdParts.push(`--pytest-argv-json=${JSON.stringify(pytestArgs)}`);

    const outputMode = getRoutineArgs(args as Record<string, unknown>, parsedOptions.options).outputMode;
    const output = await executePytestCommand(cmdParts, outputMode);
    return ignoredCoverageSources ? addCoverageSourceInfo(output, outputMode) : output;
  },
});
