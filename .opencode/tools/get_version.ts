/**
 * ADW Version Tool
 *
 * Returns version information from the project.
 * This tool definition invokes the Python script.
 */

import { tool } from "@opencode-ai/plugin";

function validateOptionalFileInput(value: unknown): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    return "ERROR: 'file' must be a string when provided.";
  }
  return undefined;
}

function normalizeOptionalFile(value: string | undefined): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function buildExecutionFailure(error: unknown): string {
  const err = error as {
    stdout?: { toString?: () => string } | string;
    stderr?: { toString?: () => string } | string;
    message?: string;
    code?: string | number;
    exitCode?: number;
  };

  const stdout =
    typeof err?.stdout === "string" ? err.stdout : err?.stdout?.toString?.() ?? "";
  const stderr =
    typeof err?.stderr === "string" ? err.stderr : err?.stderr?.toString?.() ?? "";
  const message = err?.message?.trim() || "Unknown error";

  if (stdout.trim()) {
    return stdout.trim();
  }

  const scriptPath = `${import.meta.dir}/get_version.py`;
  const haystack = `${stderr}\n${message}`.toLowerCase();
  const isEnoentCode = err?.code === "ENOENT" || err?.exitCode === 127;
  const missingPython =
    isEnoentCode ||
    haystack.includes("python3: not found") ||
    haystack.includes("spawn python3 enoent");
  const missingScript =
    haystack.includes("can't open file") ||
    haystack.includes("get_version.py") ||
    haystack.includes("backend script not found");

  let hint = "";
  if (missingPython) {
    hint = "\nHint: Ensure python3 is installed and available on PATH.";
  } else if (missingScript) {
    hint = `\nHint: Backend script not found at ${scriptPath}.`;
  } else if (isEnoentCode || haystack.includes("enoent") || haystack.includes("no such file")) {
    hint = "\nHint: Execution failed with ENOENT; check python3 availability and backend script path.";
  }

  if (stderr.trim()) {
    return `ERROR: get_version failed\n\n${stderr.trim()}${hint}`;
  }

  return `ERROR: get_version failed\n\n${message}${hint}`;
}

export default tool({
  description: "Get version information from pyproject.toml first, then package.json fallback",
  args: {
    file: tool.schema
      .string()
      .optional()
      .describe(
        "Path to the file to read version from. Defaults to pyproject.toml first, then package.json fallback in the current directory",
      ),
  },
  async execute(args) {
    const fileInputError = validateOptionalFileInput(args.file);
    if (fileInputError) {
      return fileInputError;
    }

    const file = normalizeOptionalFile(args.file as string | undefined);
    const scriptPath = `${import.meta.dir}/get_version.py`;

    try {
      const command = file
        ? await Bun.$`python3 ${scriptPath} ${file}`.text()
        : await Bun.$`python3 ${scriptPath}`.text();
      return command.trim();
    } catch (error) {
      return buildExecutionFailure(error);
    }
  },
});
