/**
 * Bun Test Runner Tool
 *
 * Wraps the Python backing script run_bun_test.py to execute bun test with
 * filtering and structured outputs. Mirrors the run_ctest.ts pattern.
 */

import { tool } from "@opencode-ai/plugin";
import { validateCwdWithinRepo, validatePathWithinRepo } from "./lib/path_validation";

// --- Tool-local helpers ---

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_bun_test.py.";

type OutputMode = "summary" | "full" | "json";
type ParsedBunOptions = {
  outputMode?: OutputMode;
  testFilter?: string;
  failFast?: true;
};

type ParsedBunOptionsResult =
  | { ok: true; options: ParsedBunOptions }
  | { ok: false; error: string };

const BUN_OPTION_RULES = new Set(["output", "test-filter", "fail-fast"]);
const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);
const LEGACY_DIRECT_KEYS = new Set(["outputMode", "testFilter", "failFast"]);

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

const parseBunOptions = (options: unknown): ParsedBunOptionsResult => {
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

  const parsed: ParsedBunOptions = {};
  for (const token of tokenized.tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.` };
    }
    if (separatorIndex === -1) {
      if (!BUN_OPTION_RULES.has(token)) {
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
    if (!BUN_OPTION_RULES.has(name)) {
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
    }
  }

  return { ok: true, options: parsed };
};

const validatePositiveFiniteNumber = (name: string, value: unknown): string | undefined => {
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

// --- Tool definition ---

export default tool({
  description: `Run bun test for TypeScript test execution with parsing and validation.

EXAMPLES:
- Run all tests: run_bun_test({ testPath: '__tests__/' })
- Single test file: run_bun_test({ testPath: '__tests__/get_datetime.test.ts' })
- Filter by name: run_bun_test({ options: 'test-filter=datetime' })
- Fail fast: run_bun_test({ testPath: '__tests__/', options: 'fail-fast' })
- JSON output: run_bun_test({ options: 'output=json' })

IMPORTANT: Requires bun to be installed on the host system.
NOTE: Default working directory is .opencode/tools/ (where package.json lives).`,
  args: {
    testPath: tool.schema
      .string()
      .optional()
      .describe("Path to test file or directory. Example: '__tests__/', '__tests__/get_datetime.test.ts'."),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 300). Must be positive."),
    minTests: tool.schema
      .number()
      .optional()
      .describe("Minimum expected test count (default: 1). Must be positive."),
    cwd: tool.schema
      .string()
      .optional()
      .describe("Working directory override (default: .opencode/tools/)."),
    options: tool.schema
      .string()
      .optional()
      .describe("Bounded options string. Supported tokens: output=<summary|full|json>, test-filter=<value>, fail-fast."),
  },
  async execute(args) {
    const legacyDirectKey = hasLegacyDirectKey(args as Record<string, unknown>);
    if (legacyDirectKey) {
      return `ERROR: run_bun_test does not accept direct field '${legacyDirectKey}'. Use 'options' tokens instead.`;
    }

    const parsedOptions = parseBunOptions((args as Record<string, unknown>).options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const testPath = typeof args.testPath === "string" ? args.testPath.trim() : undefined;
    const testFilter = parsedOptions.options.testFilter;
    const timeout = args.timeout ?? 300;
    const minTests = args.minTests ?? 1;
    const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
    const failFast = parsedOptions.options.failFast === true;

    const timeoutError = validatePositiveFiniteNumber("timeout", timeout);
    if (timeoutError) {
      return timeoutError;
    }

    const minTestsError = validatePositiveFiniteNumber("minTests", minTests);
    if (minTestsError) {
      return minTestsError;
    }

    if (typeof args.testPath === "string" && !testPath) {
      return "ERROR: testPath must not be blank when provided.";
    }
    if (testPath?.startsWith("-")) {
      return "ERROR: testPath must not start with '-' (potential option injection).";
    }
    if (parsedOptions.options.testFilter !== undefined && !testFilter) {
      return "ERROR: testFilter must not be blank when provided.";
    }
    if (typeof args.cwd === "string" && !cwd) {
      return "ERROR: cwd must not be blank when provided.";
    }

    const cwdError = validateCwdWithinRepo(cwd);
    if (cwdError) {
      return cwdError;
    }
    const testPathError = validateTestPathWithinRepo(testPath, cwd);
    if (testPathError) {
      return testPathError;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_bun_test.py`,
      `--output=${outputMode}`,
      `--min-tests=${minTests}`,
      `--timeout=${timeout}`,
    ];

    if (testPath) {
      cmdParts.push(`--test-path=${testPath}`);
    }

    if (testFilter) {
      cmdParts.push(`--filter=${testFilter}`);
    }

    if (failFast) {
      cmdParts.push("--bail");
    }

    if (cwd) {
      cmdParts.push(`--cwd=${cwd}`);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "bun test completed but returned no output.";
    } catch (error: any) {
      const stdout = error?.stdout?.toString?.() || "";
      const stderr = error?.stderr?.toString?.() || "";
      const message = error?.message || "Unknown error";
      const combinedLower = `${stderr} ${message}`.toLowerCase();

      if (stdout.trim()) {
        return stdout;
      }

      if (stderr.trim()) {
        const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
        return `ERROR: Bun test failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run bun test: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run bun test: ${message}`;
    }
  },
});
