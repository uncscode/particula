/**
 * MkDocs Build Tool
 *
 * Wraps the Python backing script build_mkdocs.py to run mkdocs build with
 * structured output options. Mirrors the run_bun_test.ts facade patterns.
 */

import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

const MISSING_SCRIPT_HINT =
  "Encountered an ENOENT error. Ensure python3 is installed and on your PATH, mkdocs is " +
  "installed, and the backing script .opencode/tools/build_mkdocs.py exists.";

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

type OutputMode = "summary" | "full" | "json";

type ParsedMkdocsOptions = {
  outputMode?: OutputMode;
  strict?: true;
  clean?: boolean;
};

const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);

function validatePathWithinRepoRoot(
  value: string | undefined,
  parameterName: "cwd" | "configFile",
  cwdValue?: string,
): string | undefined {
  if (!value) {
    return undefined;
  }

  try {
    if (parameterName === "cwd") {
      if (!existsSync(value)) {
        return `ERROR: cwd path does not exist: ${value}`;
      }
      if (!isStatDirectory(statSync(value))) {
        return `ERROR: cwd path is not a directory: ${value}`;
      }
    }

    const repoRoot = realpathSync(process.cwd());
    const resolvedPath = parameterName === "configFile" && cwdValue && !path.isAbsolute(value)
      ? realpathSync(path.resolve(realpathSync(cwdValue), value))
      : realpathSync(value);
    const rel = path.relative(repoRoot, resolvedPath);
    if (rel.startsWith("..") || path.isAbsolute(rel)) {
      return `ERROR: ${parameterName} path resolves outside repository root: ${value} (canonical: ${resolvedPath})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid ${parameterName} path: ${value} (${message})`;
  }

  return undefined;
}

function parseMkdocsOptions(rawOptions: unknown):
  | { ok: true; options: ParsedMkdocsOptions }
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

  const parsed: ParsedMkdocsOptions = {};
  for (const token of normalized.split(/\s+/)) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.`,
      };
    }

    if (separatorIndex === -1) {
      if (token !== "strict") {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
        };
      }
      if (parsed.strict) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      parsed.strict = true;
      continue;
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

    if (name === "clean") {
      if (parsed.clean !== undefined) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      if (value !== "true" && value !== "false") {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': clean must be true or false.`,
        };
      }
      parsed.clean = value === "true";
      continue;
    }

    if (name === "strict") {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token does not accept a value.`,
      };
    }

    return {
      ok: false,
      error: `ERROR: Invalid options token '${token}': token is not supported.`,
    };
  }

  return { ok: true, options: parsed };
}

export default tool({
  description: `Run mkdocs build with output handling and validation.

EXAMPLES:
- Default build: build_mkdocs({})
- JSON output: build_mkdocs({ options: 'output=json' })
- Strict + no clean: build_mkdocs({ options: 'strict clean=false' })
- Custom config: build_mkdocs({ configFile: 'docs/mkdocs.yml' })
- Validate only: build_mkdocs({ validateOnly: true })

IMPORTANT:
  - Requires mkdocs to be installed on the host system.
  - Uses python3 to run the backing script.
  - Default timeout is 120 seconds.`,
  args: {
    timeout: tool.schema
      .number()
      .optional()
      .describe("Timeout in seconds (default: 120). Must be positive."),
    cwd: tool.schema
      .string()
      .optional()
      .describe("Working directory override for mkdocs build."),
    configFile: tool.schema
      .string()
      .optional()
      .describe("Path to mkdocs configuration file (default: mkdocs.yml)."),
    validateOnly: tool.schema
      .boolean()
      .optional()
      .describe("Build to a temporary directory and discard output."),
    options: tool.schema
      .string()
      .optional()
      .describe("Bounded options: output=<summary|full|json>, strict, clean=<true|false>."),
  },
  async execute(args) {
    const parsedOptions = parseMkdocsOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const timeout = args.timeout ?? 120;
    const cwd = args.cwd as string | undefined;
    const strict = parsedOptions.options.strict === true;
    const clean = parsedOptions.options.clean ?? true;
    const configFile = args.configFile as string | undefined;
    const validateOnly = args.validateOnly === true;

    if (!Number.isFinite(timeout) || timeout <= 0) {
      return `ERROR: Timeout must be positive (received ${timeout}).`;
    }

    const cwdError = validatePathWithinRepoRoot(cwd, "cwd");
    if (cwdError) {
      return cwdError;
    }
    const configFileError = validatePathWithinRepoRoot(configFile, "configFile", cwd);
    if (configFileError) {
      return configFileError;
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/build_mkdocs.py`,
      `--output=${outputMode}`,
      `--timeout=${timeout}`,
    ];

    if (cwd) {
      cmdParts.push(`--cwd=${cwd}`);
    }

    if (strict) {
      cmdParts.push("--strict");
    }

    if (!clean) {
      cmdParts.push("--no-clean");
    }

    if (configFile) {
      cmdParts.push(`--config-file=${configFile}`);
    }

    if (validateOnly) {
      cmdParts.push("--validate-only");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "mkdocs build completed but returned no output.";
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
        return `ERROR: MkDocs build failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run mkdocs build: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run mkdocs build: ${message}`;
    }
  },
});
