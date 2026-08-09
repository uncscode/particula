/**
 * Linter Runner Tool
 *
 * Runs configured linters (ruff, mypy) for the Agent repository.
 * Supports a mutating auto-fix path and a validation-only non-mutating path.
 */

import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync } from "node:fs";
import { resolve, relative } from "node:path";
import { validateCwdWithinRepo } from "./lib/path_validation";

type OutputMode = "summary" | "full" | "json";
type LinterMode = "check" | "format-check" | "format";

type ParsedLinterOptions = {
  outputMode?: OutputMode;
  linters?: string[];
};

const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);
const SUPPORTED_LINTERS = new Set(["ruff", "mypy"]);
const LINTER_MODES = new Set<LinterMode>(["check", "format-check", "format"]);
// Keep these transport limits in parity with the trusted Python runner and runtime adapter.
const MAX_TARGET_PATHS = 64;
const MAX_TARGET_PATH_BYTES = 1024;
const MAX_TARGET_PATHS_JSON_BYTES = 8192;

function validateTargetPaths(
  value: unknown,
  cwd?: string,
): { ok: true; paths: string[] } | { ok: false; error: string } {
  if (!Array.isArray(value) || value.length === 0 || value.length > MAX_TARGET_PATHS) {
    return { ok: false, error: "ERROR: targetPaths must be a non-empty bounded array of strings." };
  }
  const root = realpathSync.native(resolve(cwd ?? process.cwd()));
  const seen = new Set<string>();
  for (const path of value) {
    if (typeof path !== "string" || !path || path.trim() !== path || path.startsWith("-") ||
      path.startsWith("/") || path.includes("\\") || path.split("/").includes("..") ||
      path.split("/").includes(".") || path.split("/").includes("") ||
      Buffer.byteLength(path, "utf8") > MAX_TARGET_PATH_BYTES || seen.has(path)) {
      return { ok: false, error: "ERROR: targetPaths contains an invalid repository-relative path." };
    }
    const resolved = resolve(root, path);
    if (relative(root, resolved).startsWith("..")) {
      return { ok: false, error: "ERROR: targetPaths must stay within the repository root." };
    }
    let prefix = root;
    for (const part of path.split("/")) {
      prefix = resolve(prefix, part);
      if (existsSync(prefix)) {
        prefix = realpathSync.native(prefix);
        if (relative(root, prefix).startsWith("..")) {
          return { ok: false, error: "ERROR: targetPaths must stay within the repository root." };
        }
      }
    }
    seen.add(path);
  }
  const encoded = JSON.stringify(value);
  if (Buffer.byteLength(encoded, "utf8") > MAX_TARGET_PATHS_JSON_BYTES) {
    return { ok: false, error: "ERROR: targetPaths compact JSON transport exceeds its byte limit." };
  }
  return { ok: true, paths: [...value] };
}

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
    confirmed: tool.schema
      .boolean()
      .optional()
      .describe("Explicit confirmation required before a mutating format or auto-fix request is dispatched."),
    targetDir: tool.schema
      .string()
      .optional()
      .describe("Target directory to lint. If omitted, uses pyproject.toml config (lints from project root)."),
    cwd: tool.schema
      .string()
      .optional()
      .describe("Working directory for lint execution. Repository-relative targets are resolved from this checkout or worktree."),
    mode: tool.schema.enum(["check", "format-check", "format"]).optional()
      .describe("Explicit Ruff-only mode: check, format-check, or confirmed format."),
    targetPaths: tool.schema.array(tool.schema.string()).optional()
      .describe("Ordered repository-relative Ruff targets for an explicit mode."),
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

    const mode = args.mode as LinterMode | undefined;
    if (mode !== undefined && !LINTER_MODES.has(mode)) return "ERROR: mode must be check, format-check, or format.";
    if (args.targetPaths !== undefined && mode === undefined) return "ERROR: targetPaths requires mode.";
    // A false autoFix value may be materialized by schema handling even when the
    // caller omitted the legacy selector. It is compatible with explicit modes;
    // true remains an unambiguous conflicting mutating legacy request.
    if (mode !== undefined && args.autoFix === true) return "ERROR: mode conflicts with autoFix.";
    if (mode !== undefined && args.targetDir !== undefined) return "ERROR: mode conflicts with targetDir.";
    if (args.targetPaths !== undefined && args.targetDir !== undefined) return "ERROR: targetPaths conflicts with targetDir.";
    const cwd = typeof args.cwd === "string" ? args.cwd.trim() || undefined : undefined;
    const cwdError = validateCwdWithinRepo(cwd);
    if (cwdError) return cwdError;
    const targetPaths = args.targetPaths === undefined ? undefined : validateTargetPaths(args.targetPaths, cwd);
    if (targetPaths && !targetPaths.ok) return targetPaths.error;
    const outputMode = parsedOptions.options.outputMode || "summary";
    const autoFix = args.autoFix !== false; // Default to true
    const linters = parsedOptions.options.linters || (mode ? ["ruff"] : ["ruff", "mypy"]);
    if (mode && (linters.length !== 1 || linters[0] !== "ruff")) {
      return "ERROR: mode requires linters=ruff or omitted linters.";
    }
    const targetDir = args.targetDir;
    const ruffTimeout = args.ruffTimeout ?? 120;
    const mypyTimeout = args.mypyTimeout ?? 180;

    if (!Number.isFinite(ruffTimeout) || ruffTimeout <= 0) {
      return `ERROR: ruffTimeout must be positive (received ${ruffTimeout}).`;
    }

    if (!Number.isFinite(mypyTimeout) || mypyTimeout <= 0) {
      return `ERROR: mypyTimeout must be positive (received ${mypyTimeout}).`;
    }
    if ((mode === "format" || (mode === undefined && autoFix)) && args.confirmed !== true) {
      return "ERROR: explicit confirmation is required before mutating lint execution.";
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
    if (cwd) {
      cmdParts.push(`--cwd=${cwd}`);
    }

    if (mode) {
      cmdParts.push(`--mode=${mode}`);
      if (targetPaths) cmdParts.push(`--target-paths-json=${JSON.stringify(targetPaths.paths)}`);
    } else if (autoFix) {
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
