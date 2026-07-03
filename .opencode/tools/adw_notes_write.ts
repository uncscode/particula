import { tool } from "@opencode-ai/plugin";

import {
  buildFieldArgs,
  normalizeRef,
  parseFieldEntries,
  runNotesCommand,
  validateRequiredAdwId,
} from "./adw_notes_shared";

const COMMANDS = ["write", "write-from-state"] as const;
type WriteCommand = (typeof COMMANDS)[number];

const USAGE = `Usage:
  { command: "write", ref: "HEAD", fields?: [["key", "value"], ...] }
  { command: "write-from-state", ref: "HEAD", adw_id: "abc12345", fields?: [["key", "value"], ...] }

Notes:
  - fields may be provided as an ordered array of [key, value] pairs (duplicates preserved),
    an array of { key, value } objects, a plain object, or a JSON string containing one of those forms.
  - write-from-state requires adw_id and forwards only --adw-id to CLI state resolution.`;

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE}`;
}

function normalizeCommand(value: unknown): WriteCommand | null {
  if (typeof value !== "string") {
    return null;
  }
  return (COMMANDS as readonly string[]).includes(value) ? (value as WriteCommand) : null;
}

export default tool({
  description: `Write ADW workflow context notes via \`adw notes\`.

AVAILABLE COMMANDS:
• write: Write explicit note fields to a git ref
• write-from-state: Write note fields from ADW workflow state for an adw_id

Examples:
  adw_notes_write({ command: "write", ref: "HEAD", fields: [["plan_summary", "done"]] })
  adw_notes_write({ command: "write-from-state", ref: "HEAD", adw_id: "abc12345" })`,

  args: {
    command: tool.schema.enum([...COMMANDS]).describe("Write command to execute: write or write-from-state."),
    ref: tool.schema.string().describe("Git ref to target (e.g., HEAD, branch name, commit SHA)."),
    adw_id: tool.schema.string().optional().describe("ADW workflow ID required only for write-from-state."),
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
      return buildError(`Invalid command \"${String(args.command)}\". Valid commands: ${COMMANDS.join(", ")}.`);
    }

    const ref = normalizeRef(args.ref);
    if (!ref) {
      return buildError(`'ref' is required for '${command}'.`);
    }

    const normalizedAdwId = validateRequiredAdwId(command, args.adw_id);
    if (normalizedAdwId.error) {
      return buildError(normalizedAdwId.error);
    }

    const parsedFields = parseFieldEntries((args as Record<string, unknown>).fields);
    if (!parsedFields.ok) {
      return buildError(
        `'fields' payload is malformed: ${parsedFields.diagnostic}. Accepted forms: ordered [key, value] entries, { key, value } objects, plain object, or JSON string of those forms.`,
      );
    }

    const fieldArgs = buildFieldArgs(parsedFields.entries);
    const cmdParts = ["uv", "run", "--active", "adw", "notes"];
    if (command === "write") {
      cmdParts.push("write", "--ref", ref, ...fieldArgs);
    } else {
      cmdParts.push("write-from-state", "--ref", ref, "--adw-id", normalizedAdwId.value as string, ...fieldArgs);
    }

    const result = runNotesCommand(command, cmdParts);
    if (!result.ok) {
      return result.error;
    }

    return `ADW Notes Command: ${command}\n\n${result.stdout}`;
  },
});
