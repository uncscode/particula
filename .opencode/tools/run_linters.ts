/**
 * Linter Runner Tool
 *
 * Runs configured linters (ruff, mypy) for the Agent repository.
 * Supports a mutating auto-fix path and a validation-only non-mutating path.
 */

import { tool } from "@opencode-ai/plugin";

type OutputMode = "summary" | "full" | "json";

type ParsedLinterOptions = {
  outputMode?: OutputMode;
  linters?: string[];
};

const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);
const SUPPORTED_LINTERS = new Set(["ruff", "mypy"]);

function parseLinterOptions(rawOptions: unknown):
  | { ok: true; options: ParsedLinterOptions }
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

  const parsed: ParsedLinterOptions = {};
  for (const token of normalized.split(/\s+/)) {
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
    const value = token.slice(separatorIndex + 1).trim();
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

    if (name === "linters") {
      if (parsed.linters !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      const linters = value
        .split(",")
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);
      if (linters.length === 0) {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': linters must contain at least one supported linter.`,
        };
      }
      for (const linter of linters) {
        if (!SUPPORTED_LINTERS.has(linter)) {
          return {
            ok: false,
            error: `ERROR: Invalid options token '${token}': unsupported linter '${linter}'.`,
          };
        }
      }
      parsed.linters = linters;
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
  description: "Run configured linters (ruff, mypy) in either mutating auto-fix mode or validation-only mode. Follows .github/workflows/lint.yml workflow and returns comprehensive linting results with pass/fail status.",
  args: {
    autoFix: tool.schema
      .boolean()
      .optional()
      .describe("Automatically fix issues where possible (default: true). When false, runs validation-only Ruff checking without formatting or fixes."),
    targetDir: tool.schema
      .string()
      .optional()
      .describe("Target directory to lint. If omitted, uses pyproject.toml config (lints from project root)."),
    ruffTimeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds for ruff commands (default: 120 = 2 minutes)"),
    mypyTimeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds for mypy command (default: 180 = 3 minutes)"),
    options: tool.schema
      .string()
      .optional()
      .describe("Bounded options: output=<summary|full|json>, linters=<ruff|mypy comma-list>."),
  },
  async execute(args) {
    const parsedOptions = parseLinterOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const autoFix = args.autoFix !== false; // Default to true
    const linters = parsedOptions.options.linters || ["ruff", "mypy"]; // Match CI workflow
    const targetDir = args.targetDir;
    const ruffTimeout = args.ruffTimeout ?? 120;
    const mypyTimeout = args.mypyTimeout ?? 180;

    if (!Number.isFinite(ruffTimeout) || ruffTimeout <= 0) {
      return `ERROR: ruffTimeout must be positive (received ${ruffTimeout}).`;
    }

    if (!Number.isFinite(mypyTimeout) || mypyTimeout <= 0) {
      return `ERROR: mypyTimeout must be positive (received ${mypyTimeout}).`;
    }

    // Build command
    const cmdParts = [
      "python3",
      `${import.meta.dir}/run_linters.py`,
      `--output=${outputMode}`,
      `--ruff-timeout=${ruffTimeout}`,
      `--mypy-timeout=${mypyTimeout}`,
    ];

    // Only pass --target-dir if explicitly provided
    // Otherwise let ruff/mypy use pyproject.toml config from project root
    if (targetDir) {
      cmdParts.push(`--target-dir=${targetDir}`);
    }

    if (autoFix) {
      cmdParts.push("--auto-fix");
    } else {
      cmdParts.push("--no-auto-fix");
    }

    if (linters.length > 0) {
      cmdParts.push(`--linters=${linters.join(",")}`);
    }

    try {
      // Execute the Python script
      const result = await Bun.$`${cmdParts}`.text();
      return result;
    } catch (error: any) {
      // Linter failed - return the output anyway
      // The Python script provides detailed error information
      const stdoutText = error.stdout?.toString() ?? "";
      if (stdoutText.length > 0) {
        return stdoutText;
      }
      const stderrText = error.stderr?.toString() ?? "";
      if (stderrText.length > 0) {
        return stderrText;
      }
      return `ERROR: Failed to run linters: ${error.message}`;
    }
  },
});
