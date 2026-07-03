/**
 * ADW Notes Tool for OpenCode Integration.
 *
 * Wraps `adw notes` CLI commands for agent-facing workflow note operations.
 */

import { tool } from "@opencode-ai/plugin";

import {
  buildFieldArgs,
  normalizeRef,
  parseFieldEntries,
  parseShowOutput,
  runNotesCommand,
  validateRequiredAdwId,
} from "./adw_notes_shared";

// --- Tool implementation ---

const COMMANDS = ["write", "write-from-state", "show"] as const;

const USAGE = `Usage:
  { command: "write", ref: "HEAD", fields?: [["key", "value"], ...] }
  { command: "write-from-state", ref: "HEAD", adw_id: "abc12345", fields?: [["key", "value"], ...] }
  { command: "show", ref: "HEAD" }

Notes:
  - fields may be provided as an ordered array of [key, value] pairs (duplicates preserved),
    an array of { key, value } objects, a plain object, or a JSON string containing one of those forms.
  - write-from-state requires adw_id and forwards only --adw-id to CLI state resolution.`;

type NoteCommand = (typeof COMMANDS)[number];

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE}`;
}

function normalizeCommand(value: unknown): NoteCommand | null {
  if (typeof value !== "string") {
    return null;
  }
  return (COMMANDS as readonly string[]).includes(value) ? (value as NoteCommand) : null;
}

export default tool({
  description: `Write and show ADW workflow context notes via \`adw notes\`.

AVAILABLE COMMANDS:
• write: Write explicit note fields to a git ref
• write-from-state: Write note fields from ADW workflow state for an adw_id
• show: Read and parse note JSON from a git ref

Examples:
  adw_notes({ command: "write", ref: "HEAD", fields: [["plan_summary", "done"]] })
  adw_notes({ command: "write-from-state", ref: "HEAD", adw_id: "abc12345" })
  adw_notes({ command: "show", ref: "HEAD" })`,

  args: {
    command: tool.schema
      .enum([...COMMANDS])
      .describe("Notes command to execute: write, write-from-state, or show."),
    ref: tool.schema.string().describe("Git ref to target (e.g., HEAD, branch name, commit SHA)."),
    adw_id: tool.schema
      .string()
      .optional()
      .describe("ADW workflow ID required only for write-from-state."),
    fields: tool.schema
      .any()
      .optional()
      .describe(
        "Optional fields payload for write/write-from-state. Accepts ordered [key,value] entries, {key,value} objects, plain object, or JSON string of those forms.",
      ),
  },

  async execute(args) {
    const command = normalizeCommand(args.command);
    if (!command) {
      return buildError(`Invalid command \"${String(args.command)}\". Valid commands: ${COMMANDS.join(
        ", ",
      )}.`);
    }

    const ref = normalizeRef(args.ref);
    if (!ref) {
      return buildError(`'ref' is required for '${command}'.`);
    }

    const adwId = validateRequiredAdwId(command, args.adw_id);
    if (adwId.error) {
      return buildError(adwId.error);
    }

    let fields: [string, string][] = [];
    if (command !== "show") {
      const parsedFields = parseFieldEntries((args as Record<string, unknown>).fields);
      if (!parsedFields.ok) {
        return buildError(
          `'fields' payload is malformed: ${parsedFields.diagnostic}. Accepted forms: ordered [key, value] entries, { key, value } objects, plain object, or JSON string of those forms.`,
        );
      }
      fields = parsedFields.entries;
    }

    const fieldArgs = buildFieldArgs(fields);
    const cmdParts = ["uv", "run", "--active", "adw", "notes"];
    if (command === "write") {
      cmdParts.push("write", "--ref", ref, ...fieldArgs);
    } else if (command === "write-from-state") {
      cmdParts.push("write-from-state", "--ref", ref, "--adw-id", adwId.value as string, ...fieldArgs);
    } else {
      cmdParts.push("show", "--ref", ref);
    }

    const result = runNotesCommand(command, cmdParts);
    if (!result.ok) {
      return result.error;
    }

    if (command === "show") {
      return parseShowOutput(result.stdout);
    }

    return `ADW Notes Command: ${command}\n\n${result.stdout}`;
  },
});
