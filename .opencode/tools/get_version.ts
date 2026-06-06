/**
 * ADW Version Tool
 * 
 * Returns version information from the project.
 * This tool definition invokes the Python script.
 */

import { tool } from "@opencode-ai/plugin";

function normalizeOptionalFile(value: unknown): string | undefined {
  if (typeof value !== "string") {
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

  const stdout = typeof err?.stdout === "string" ? err.stdout : err?.stdout?.toString?.() ?? "";
  const stderr = typeof err?.stderr === "string" ? err.stderr : err?.stderr?.toString?.() ?? "";
  const message = err?.message?.trim() || "Unknown error";

  if (stdout.trim()) {
    return stdout.trim();
  }

  const scriptPath = `${import.meta.dir}/get_version.py`;
  const haystack = `${stderr}\n${message}`.toLowerCase();
  const missingScript = haystack.includes("enoent") || haystack.includes("no such file") || haystack.includes("can't open file");
  const missingPython = haystack.includes("python3") && haystack.includes("not found");

  let hint = "";
  if (missingPython) {
    hint = "\nHint: Ensure python3 is installed and available on PATH.";
  } else if (missingScript) {
    hint = `\nHint: Backend script not found at ${scriptPath}.`;
  }

  if (stderr.trim()) {
    return `ERROR: get_version failed\n\n${stderr.trim()}${hint}`;
  }

  return `ERROR: get_version failed\n\n${message}${hint}`;
}

export default tool({
  description: "Get version information from pyproject.toml or package.json",
  args: {
    file: tool.schema
      .string()
      .optional()
      .describe("Path to the file to read version from. Defaults to pyproject.toml in the current directory"),
  },
  async execute(args) {
    const file = normalizeOptionalFile(args.file);
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
