import { tool } from "@opencode-ai/plugin";

import { normalizeRef, parseShowOutput, runNotesCommand } from "./adw_notes_shared";

const COMMANDS = ["show"] as const;
type ReadCommand = (typeof COMMANDS)[number];

const USAGE = `Usage:
  { command: "show", ref: "HEAD" }`;

function buildError(message: string): string {
  return `ERROR: ${message}\n\n${USAGE}`;
}

function normalizeCommand(value: unknown): ReadCommand | null {
  if (typeof value !== "string") {
    return null;
  }
  return (COMMANDS as readonly string[]).includes(value) ? (value as ReadCommand) : null;
}

export default tool({
  description: `Read ADW workflow context notes via \`adw notes show\`.

AVAILABLE COMMANDS:
• show: Read and parse note JSON from a git ref

Examples:
  adw_notes_read({ command: "show", ref: "HEAD" })`,

  args: {
    command: tool.schema.enum([...COMMANDS]).describe("Read command to execute: show."),
    ref: tool.schema.string().describe("Git ref to target (e.g., HEAD, branch name, commit SHA)."),
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

    const result = runNotesCommand(command, ["uv", "run", "--active", "adw", "notes", "show", "--ref", ref]);
    if (!result.ok) {
      return result.error;
    }

    return parseShowOutput(result.stdout);
  },
});
