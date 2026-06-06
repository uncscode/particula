import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

// --- Inlined from lib/cpp_lint_wrapper_shared.ts (isStatDirectory only) ---

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

// --- Inlined from lib/run_pytest_shared.ts ---

type OutputMode = "summary" | "full" | "json";

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

const validateCwdWithinRepo = (cwd: string | undefined): string | undefined => {
  if (!cwd) {
    return undefined;
  }

  try {
    if (!existsSync(cwd)) {
      return `ERROR: cwd path does not exist: ${cwd}`;
    }
    if (!isStatDirectory(statSync(cwd))) {
      return `ERROR: cwd path is not a directory: ${cwd}`;
    }

    const repoRoot = realpathSync(process.cwd());
    const resolvedCwd = realpathSync(cwd);
    const rel = path.relative(repoRoot, resolvedCwd);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: cwd path resolves outside repository root: ${cwd} (canonical: ${resolvedCwd})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid cwd path: ${cwd} (${message})`;
  }

  return undefined;
};

const getRoutineArgs = (args: Record<string, unknown>) => {
  const outputMode = (args.outputMode as OutputMode | undefined) || "summary";
  const minTests = (args.minTests as number | undefined) ?? 1;
  const timeout = (args.timeout as number | undefined) ?? 600;
  const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
  const failFast = args.failFast === true;

  return { outputMode, minTests, timeout, cwd, failFast };
};

const buildBasePytestCommand = (args: Record<string, unknown>): (string | number)[] => {
  const { outputMode, minTests, timeout, cwd, failFast } = getRoutineArgs(args);
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
): string | undefined => {
  const testFilter = typeof args.testFilter === "string" ? args.testFilter.trim() : undefined;
  const testPath = typeof args.testPath === "string" ? args.testPath.trim() : undefined;

  if (testFilter) {
    cmdParts.push("-k", testFilter);
  }
  if (testPath) {
    if (testPath.startsWith("-")) {
      return "ERROR: testPath must not start with '-' (potential option injection).";
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
      return `ERROR: Failed to run pytest: ${stdout}`;
    }
    if (stderr.trim()) {
      return `ERROR: Failed to run pytest: ${stderr}`;
    }
    return `ERROR: Failed to run pytest: ${message}`;
  }
};

// --- Tool definition ---

export default tool({
  description: "Run pytest with advanced controls (coverage, durations, override-ini, passthrough args).",
  args: {
    outputMode: tool.schema.enum(["summary", "full", "json"]).optional(),
    minTests: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
    cwd: tool.schema.string().optional(),
    failFast: tool.schema.boolean().optional(),
    testPath: tool.schema.string().optional(),
    testFilter: tool.schema.string().optional(),
    pytestArgs: tool.schema.array(tool.schema.string()).optional(),
    coverage: tool.schema.boolean().optional(),
    coverageSource: tool.schema.string().optional(),
    coverageThreshold: tool.schema.number().optional(),
    covReport: tool.schema.array(tool.schema.string()).optional(),
    durations: tool.schema.number().optional(),
    durationsMin: tool.schema.number().optional(),
    overrideIni: tool.schema.array(tool.schema.string()).optional(),
  },
  async execute(args) {
    const { minTests, timeout } = getRoutineArgs(args as Record<string, unknown>);
    const minTestsError = validatePositiveFiniteNumber("minTests", minTests);
    if (minTestsError) {
      return minTestsError;
    }
    const timeoutError = validatePositiveFiniteNumber("timeout", timeout);
    if (timeoutError) {
      return timeoutError;
    }

    const cwdError = validateCwdWithinRepo(getRoutineArgs(args as Record<string, unknown>).cwd);
    if (cwdError) {
      return cwdError;
    }

    const cmdParts = buildBasePytestCommand(args as Record<string, unknown>);
    const targetingError = appendRoutineTargeting(cmdParts, args as Record<string, unknown>);
    if (targetingError) {
      return targetingError;
    }

    const coverage = args.coverage !== false;
    const coverageSource = typeof args.coverageSource === "string" ? args.coverageSource.trim() : "";
    const coverageThreshold = args.coverageThreshold;
    const covReport = Array.isArray(args.covReport)
      ? (args.covReport as string[]).map((entry) => entry.trim()).filter(Boolean)
      : [];
    const durations = args.durations;
    const durationsMin = args.durationsMin;
    const overrideIni = Array.isArray(args.overrideIni)
      ? (args.overrideIni as string[]).map((entry) => entry.trim()).filter(Boolean)
      : [];
    const pytestArgs = Array.isArray(args.pytestArgs)
      ? (args.pytestArgs as string[]).map((entry) => String(entry).trim()).filter(Boolean)
      : [];

    if (coverage) {
      cmdParts.push("--coverage");
      if (coverageSource && coverageSource !== "all") {
        const sources = coverageSource.split(",").map((source) => source.trim()).filter(Boolean);
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
