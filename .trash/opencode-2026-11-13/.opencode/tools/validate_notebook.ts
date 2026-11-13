/**
 * Notebook Validation Tool
 *
 * Validates Jupyter notebook structure and syntax without execution.
 * Mirrors run_notebook patterns with summary/full/json modes.
 *
 * Exit codes: 0=all valid, 1=invalid notebooks, 2=tool error.
 */

import { tool } from "@opencode-ai/plugin";

const MISSING_SCRIPT_HINT = "Missing backing script .opencode/tools/validate_notebook.py";

type OutputMode = "summary" | "full" | "json";
type ValidationMode = "fast" | "full";

type ParsedValidationOptions = {
  outputMode?: OutputMode;
  skipSyntax?: true;
  validationMode?: ValidationMode;
  fast?: true;
  full?: true;
  explicitTokens?: true;
};

type ParsedValidationOptionsResult =
  | { ok: true; options: ParsedValidationOptions }
  | { ok: false; error: string };

const VALIDATION_OPTION_RULES = new Set([
  "output",
  "skip-syntax",
  "validation-mode",
  "fast",
  "full",
]);

const OUTPUT_MODES = new Set<OutputMode>(["summary", "full", "json"]);
const VALIDATION_MODES = new Set<ValidationMode>(["fast", "full"]);

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

const parseValidationOptions = (options: unknown): ParsedValidationOptionsResult => {
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

  const parsed: ParsedValidationOptions = { explicitTokens: true };
  for (const token of tokenized.tokens) {
    const separatorIndex = token.indexOf("=");
    if (separatorIndex !== token.lastIndexOf("=")) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': tokens must contain at most one '=' separator.`,
      };
    }

    if (separatorIndex === -1) {
      if (!VALIDATION_OPTION_RULES.has(token)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': token is not supported.` };
      }
      if (token === "output" || token === "validation-mode") {
        return {
          ok: false,
          error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
        };
      }
      if ((token === "skip-syntax" && parsed.skipSyntax)
        || (token === "fast" && parsed.fast)
        || (token === "full" && parsed.full)) {
        return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
      }
      if (token === "skip-syntax") {
        parsed.skipSyntax = true;
      } else if (token === "fast") {
        parsed.fast = true;
      } else if (token === "full") {
        parsed.full = true;
      }
      continue;
    }

    const name = token.slice(0, separatorIndex);
    const rawValue = token.slice(separatorIndex + 1);
    if (!VALIDATION_OPTION_RULES.has(name)) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token is not supported.` };
    }
    if (!rawValue) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': token requires a non-empty '=value' suffix.`,
      };
    }
    if (name === "skip-syntax" || name === "fast" || name === "full") {
      return { ok: false, error: `ERROR: Invalid options token '${token}': token does not accept a value.` };
    }

    const value = stripOptionalQuotes(rawValue).trim();
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

    if (!VALIDATION_MODES.has(value as ValidationMode)) {
      return {
        ok: false,
        error: `ERROR: Invalid options token '${token}': validation-mode must be one of fast, full.`,
      };
    }
    if (parsed.validationMode !== undefined) {
      return { ok: false, error: `ERROR: Invalid options token '${token}': duplicate token.` };
    }
    parsed.validationMode = value as ValidationMode;
  }

  return { ok: true, options: parsed };
};

export default tool({
  description: `Validate, convert, or sync Jupyter notebooks using Jupytext.

EXAMPLES:
- Validation: validate_notebook({notebookPath: 'notebook.ipynb'})
- Validation with options: validate_notebook({notebookPath: 'notebook.ipynb', options: 'output=json fast'})
- Convert: validate_notebook({notebookPath: 'notebook.ipynb', convertToPy: true})
- Convert with output dir: validate_notebook({notebookPath: 'docs/Examples', recursive: true, convertToPy: true, outputDir: 'scripts'})
- Convert to notebook: validate_notebook({notebookPath: 'script.py', convertToIpynb: true})
- Sync: validate_notebook({notebookPath: 'notebook.ipynb', sync: true})
- Check-sync (CI): validate_notebook({notebookPath: 'docs/Examples', recursive: true, checkSync: true})

Exit codes: 0=success, 1=functional failure (invalid/out-of-sync/convert failure), 2=tool error`,
  args: {
    notebookPath: tool.schema
      .string()
      .describe("Path to notebook file (.ipynb) or directory"),
    recursive: tool.schema
      .boolean()
      .optional()
      .describe("Search recursively when notebookPath is a directory"),
    options: tool.schema
      .string()
      .optional()
      .describe(
        "Validation-only bounded options: output=<summary|full|json>, skip-syntax, validation-mode=<fast|full>, fast, full",
      ),
    convertToPy: tool.schema
      .boolean()
      .optional()
      .describe("Convert notebooks to .py:percent format"),
    convertToIpynb: tool.schema
      .boolean()
      .optional()
      .describe("Convert py:percent scripts to .ipynb notebooks"),
    sync: tool.schema
      .boolean()
      .optional()
      .describe("Bidirectional sync between notebook and script (newer wins)"),
    checkSync: tool.schema
      .boolean()
      .optional()
      .describe("Check notebook/script sync state (read-only; exit 1 when out of sync)"),
    outputDir: tool.schema
      .string()
      .optional()
      .describe("Output directory for converted files (only with convertToPy or convertToIpynb)"),
  },
  async execute(args) {
    const notebookPath = typeof args.notebookPath === "string" ? args.notebookPath.trim() : "";
    if (!notebookPath) {
      return "ERROR: notebookPath is required and must be non-empty.";
    }

    const parsedOptions = parseValidationOptions(args.options);
    if (!parsedOptions.ok) {
      return parsedOptions.error;
    }

    const recursive = Boolean(args.recursive);
    const outputMode = parsedOptions.options.outputMode || "summary";
    const skipSyntax = parsedOptions.options.skipSyntax === true;
    const validationMode = parsedOptions.options.validationMode;
    const fast = parsedOptions.options.fast === true;
    const full = parsedOptions.options.full === true;
    const convertToPy = Boolean(args.convertToPy);
    const convertToIpynb = Boolean(args.convertToIpynb);
    const sync = Boolean(args.sync);
    const checkSync = Boolean(args.checkSync);
    const outputDir = typeof args.outputDir === "string" ? args.outputDir.trim() : undefined;

    if (validationMode && (fast || full)) {
      return "ERROR: Conflicting validation options: use either 'validation-mode' or 'fast/full' aliases, not both.";
    }
    if (fast && full) {
      return "ERROR: Conflicting validation options: 'fast' and 'full' cannot both be true.";
    }

    const isValidationAction = !convertToPy && !convertToIpynb && !sync && !checkSync;
    const usesValidationOnlyFlags =
      parsedOptions.options.explicitTokens === true
      || outputMode !== "summary"
      || skipSyntax
      || Boolean(validationMode)
      || fast
      || full;

    if (!isValidationAction && usesValidationOnlyFlags) {
      return (
        "ERROR: Validation-only bounded options (options: output=..., skip-syntax, validation-mode=..., fast, full) " +
        "can only be used with validation (omit convert/sync/checkSync flags)."
      );
    }

    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/validate_notebook.py`,
      notebookPath,
    ];

    if (convertToPy) {
      cmdParts.push("--convert-to-py");
    }
    if (convertToIpynb) {
      cmdParts.push("--convert-to-ipynb");
    }
    if (sync) {
      cmdParts.push("--sync");
    }
    if (checkSync) {
      cmdParts.push("--check-sync");
    }
    if (outputDir) {
      cmdParts.push("--output-dir", outputDir);
    }

    if (isValidationAction) {
      cmdParts.push(`--output=${outputMode}`);
      if (skipSyntax) {
        cmdParts.push("--skip-syntax");
      }
      if (validationMode) {
        cmdParts.push("--validation-mode", validationMode);
      }
      if (fast) {
        cmdParts.push("--fast");
      }
      if (full) {
        cmdParts.push("--full");
      }
    }

    if (recursive) {
      cmdParts.push("--recursive");
    }

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Notebook tool completed but returned no output.";
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
        return `ERROR: Notebook tool failed\n\n${stderr}${hint}`;
      }

      if (combinedLower.includes("enoent")) {
        return `ERROR: Failed to run notebook tool: ${message}\n${MISSING_SCRIPT_HINT}`;
      }

      return `ERROR: Failed to run notebook tool: ${message}`;
    }
  },
});
