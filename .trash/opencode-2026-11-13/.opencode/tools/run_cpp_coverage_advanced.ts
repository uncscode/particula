import { tool } from "@opencode-ai/plugin";

import {
  parseCppCoverageAdvancedOptions,
  rejectLegacyDirectFields,
  resolveDirectoryPathWithinRepo,
  resolveExistingDirectoryWithinRepo,
  sanitizeDiagnosticValue,
} from "./run_cpp_wrapper_shared";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/run_cpp_coverage.py.";
const MIN_TIMEOUT_SECONDS = 1;
const MAX_TIMEOUT_SECONDS = 3_600;
const MIN_THRESHOLD = 0;
const MAX_THRESHOLD = 100;

function validateThreshold(threshold: unknown): string | undefined {
  if (threshold === undefined) {
    return undefined;
  }
  if (typeof threshold !== "number" || !Number.isFinite(threshold)) {
    return "ERROR: threshold must be a finite number between 0 and 100.";
  }
  if (threshold < MIN_THRESHOLD || threshold > MAX_THRESHOLD) {
    return "ERROR: threshold must be between 0 and 100.";
  }
  return undefined;
}

function validateTimeout(timeout: unknown): string | undefined {
  if (timeout === undefined) {
    return undefined;
  }
  if (!Number.isInteger(timeout)) {
    return "ERROR: timeout must be an integer between 1 and 3600 seconds.";
  }
  if (timeout < MIN_TIMEOUT_SECONDS || timeout > MAX_TIMEOUT_SECONDS) {
    return "ERROR: timeout must be between 1 and 3600 seconds.";
  }
  return undefined;
}

export default tool({
  description: `Run advanced C++ coverage checks with extended controls.

Use this wrapper when you need tool/filter/html options in addition to routine buildDir/threshold/timeout controls.`,
  args: {
    buildDir: tool.schema.string(),
    threshold: tool.schema.number().optional(),
    timeout: tool.schema.number().optional(),
    filter: tool.schema.string().optional(),
    html: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
  },
  async execute(args) {
    const legacyFieldError = rejectLegacyDirectFields(args, "run_cpp_coverage_advanced", ["outputMode", "tool", "extraArgs"]);
    if (legacyFieldError) {
      return legacyFieldError;
    }

    const parsedOptions = parseCppCoverageAdvancedOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const buildDir = typeof args.buildDir === "string" ? args.buildDir.trim() : "";
    if (!buildDir) {
      return "ERROR: buildDir is required. Provide a build directory containing coverage artifacts.";
    }
    const buildDirResult = resolveExistingDirectoryWithinRepo(buildDir, "buildDir");
    if (!buildDirResult.ok) {
      return buildDirResult.error;
    }

    const thresholdError = validateThreshold(args.threshold);
    if (thresholdError) {
      return thresholdError;
    }
    const timeoutError = validateTimeout(args.timeout);
    if (timeoutError) {
      return timeoutError;
    }

    const coverageTool = parsedOptions.options.tool || "";
    const filter = typeof args.filter === "string" ? args.filter.trim() : "";
    const html = typeof args.html === "string" ? args.html.trim() : "";
    let resolvedHtml: string | undefined;
    if (html) {
      const htmlResult = resolveDirectoryPathWithinRepo(html, "html");
      if (!htmlResult.ok) {
        return htmlResult.error;
      }
      resolvedHtml = htmlResult.path;
    }

    const outputMode = parsedOptions.options.outputMode || "summary";
    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/run_cpp_coverage.py`,
      `--build-dir=${buildDirResult.path}`,
      `--output=${outputMode}`,
    ];

    if (args.threshold !== undefined) {
      cmdParts.push("--threshold", Number(args.threshold));
    }
    if (args.timeout !== undefined) {
      cmdParts.push("--timeout", Number(args.timeout));
    }
    if (coverageTool) {
      cmdParts.push(`--tool=${coverageTool}`);
    }
    if (filter) {
      cmdParts.push(`--filter=${filter}`);
    }
    if (resolvedHtml) {
      cmdParts.push(`--html=${resolvedHtml}`);
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "C++ coverage advanced run completed but returned no output.";
    } catch (error: any) {
      const stdout = sanitizeDiagnosticValue(error?.stdout?.toString?.() || "");
      const stderr = sanitizeDiagnosticValue(error?.stderr?.toString?.() || "");
      const message = sanitizeDiagnosticValue(error?.message || "Unknown error");
      const combinedLower = `${stderr} ${stdout} ${message}`.toLowerCase();

      if (stdout.trim()) {
        return stdout;
      }

      if (stderr.trim()) {
        const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
        return `ERROR: C++ coverage advanced run failed\n\n${stderr}${hint}`;
      }

      const hint = combinedLower.includes("enoent") ? `\n${MISSING_SCRIPT_HINT}` : "";
      return `ERROR: Failed to run C++ coverage advanced wrapper: ${message}${hint}`;
    }
  },
});
