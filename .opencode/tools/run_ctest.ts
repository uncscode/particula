/**
 * CTest Runner Tool
 *
 * Wraps the Python backing script run_ctest.py to execute CTest with
 * filtering, parallelism, and structured outputs for OpenCode tools.
 */

import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync } from "node:fs";
import path from "node:path";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_ctest.py (dependency #1363).";

type OutputMode = "summary" | "full" | "json";

type ParsedCtestOptions = {
  outputMode?: OutputMode;
  testFilter?: string;
  excludeFilter?: string;
  parallel?: number;
};

const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);

function resolvePathForRepoCheck(value: string): string {
  const repoRoot = realpathSync(process.cwd());
  const candidate = path.isAbsolute(value) ? path.normalize(value) : path.resolve(repoRoot, value);
  let probe = candidate;
  const tailSegments: string[] = [];

  while (!existsSync(probe)) {
    const parent = path.dirname(probe);
    if (parent === probe) {
      throw new Error(`path does not exist: ${value}`);
    }
    tailSegments.unshift(path.basename(probe));
    probe = parent;
  }

  const resolvedProbe = realpathSync(probe);
  return tailSegments.length > 0 ? path.resolve(resolvedProbe, ...tailSegments) : resolvedProbe;
}

function validateBuildDirWithinRepoRoot(value: string): string | undefined {
  try {
    const repoRoot = realpathSync(process.cwd());
    const resolvedPath = resolvePathForRepoCheck(value);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: buildDir path resolves outside repository root: ${value} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid buildDir path: ${value} (${message})`;
  }

  return undefined;
}

function tokenizeOptions(options: string): { ok: true; tokens: string[] } | { ok: false; error: string } {
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
}

function stripOptionalQuotes(value: string): string {
  if (value.length >= 2) {
    const first = value[0];
    const last = value[value.length - 1];
    if ((first === '"' || first === "'") && last === first) {
      return value.slice(1, -1);
    }
  }
  return value;
}

function parseCtestOptions(rawOptions: unknown):
  | { ok: true; options: ParsedCtestOptions }
  | { ok: false; error: string } {
  if (rawOptions === undefined || rawOptions === null) {
    return { ok: true, options: {} };
  }
  if (typeof rawOptions !== "string") {
    return { ok: false, error: "ERROR: 'options' must be a string when provided." };
  }

  const normalized = rawOptions.trim();
  if (!normalized) {
    return { ok: true, options: {} };
  }

  const tokenized = tokenizeOptions(normalized);
  if (!tokenized.ok) {
    return tokenized;
  }

  const parsed: ParsedCtestOptions = {};
  for (const token of tokenized.tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.`,
      };
    }
    if (separatorIndex === -1) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    const name = token.slice(0, separatorIndex);
    const value = stripOptionalQuotes(token.slice(separatorIndex + 1)).trim();
    if (!value) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }

    if (name === "output") {
      if (!OUTPUT_MODES.has(value as OutputMode)) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': output must be one of summary, full, json.`,
        };
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

    if (name === "exclude-filter") {
      if (parsed.excludeFilter !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.excludeFilter = value;
      continue;
    }

    if (name === "parallel") {
      if (!/^\d+$/.test(value)) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': parallel must be a positive integer.`,
        };
      }
      if (parsed.parallel !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.parallel = Number(value);
      continue;
    }

    return {
      ok: false,
      error: `ERROR: Invalid options token '${token}': token is not supported.`,
    };
  }

  return { ok: true, options: parsed };
}

export default tool({
  description: `Run CTest for C++ test execution with parsing and validation.

EXAMPLES:
- Run all tests: run_ctest({ buildDir: 'build' })
- Filter tests: run_ctest({ buildDir: 'build', options: 'test-filter=test_add' })
- Exclude tests: run_ctest({ buildDir: 'build', options: 'exclude-filter=slow' })
- Parallel execution: run_ctest({ buildDir: 'build', options: 'parallel=4' })
- With minimum test count: run_ctest({ buildDir: 'build', minTests: 5 })
- JSON output: run_ctest({ buildDir: 'build', options: 'output=json' })

  IMPORTANT: buildDir must point to a CMake build directory containing CTestTestfile.cmake.
  NOTE: Use testFilter for -R pattern (include), excludeFilter for -E pattern (exclude).`,
  args: {
    buildDir: tool.schema
      .string()
      .describe(
        "CMake build directory containing CTestTestfile.cmake (required). Example: 'build', 'example_cpp_dev/build'.",
      ),
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 300 = 5 minutes). Must be positive."),
    minTests: tool.schema
      .number()
      .optional()
      .describe(
        "Minimum expected test count for validation (default: 1). Set higher for full suite validation. Must be positive.",
      ),
    options: tool.schema
      .string()
      .optional()
      .describe("Bounded options: output=<summary|full|json>, test-filter=<regex>, exclude-filter=<regex>, parallel=<positive-int>."),
  },
  async execute(args) {
    const parsedOptions = parseCtestOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : undefined;
    const testFilter = parsedOptions.options.testFilter;
    const excludeFilter = parsedOptions.options.excludeFilter;
    const parallel = parsedOptions.options.parallel;
    const timeout = args.timeout ?? 300;
    const minTests = args.minTests ?? 1;

    if (!buildDir) {
      return "ERROR: buildDir is required. Provide the CMake build directory containing CTestTestfile.cmake.";
    }

    const buildDirError = validateBuildDirWithinRepoRoot(buildDir);
    if (buildDirError) {
      return buildDirError;
    }

    if (!Number.isFinite(timeout) || timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    if (!Number.isFinite(minTests) || minTests <= 0) {
      return `ERROR: minTests must be positive (received ${minTests}).`;
    }

    if (parallel !== undefined && (!Number.isFinite(parallel) || parallel <= 0)) {
      return `ERROR: parallel must be positive (received ${parallel}).`;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_ctest.py`,
      `--build-dir=${buildDir}`,
      `--output=${outputMode}`,
      `--min-tests=${minTests}`,
      `--timeout=${timeout}`,
    ];

    if (testFilter) {
      cmdParts.push("-R", testFilter);
    }

    if (excludeFilter) {
      cmdParts.push("-E", excludeFilter);
    }

    if (parallel !== undefined) {
      cmdParts.push("-j", parallel);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "CTest completed but returned no output.";
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
        return `ERROR: CTest failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run CTest: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run CTest: ${message}`;
    }
  },
});
