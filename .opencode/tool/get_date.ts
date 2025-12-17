/**
 * ADW Date Tool
 *
 * Returns the current date or datetime in various formats (UTC).
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Get the current date or datetime in various formats (UTC)",
  args: {
    format: tool.schema
      .enum(["date", "datetime", "human"])
      .optional()
      .describe("Format: 'date' (YYYY-MM-DD), 'datetime' (ISO 8601), 'human' (human-readable). Default: 'date'"),
  },
  async execute(args) {
    const format = args.format || "date";
    const now = new Date();

    switch (format) {
      case "datetime":
        // ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        return now.toISOString().replace(/\.\d{3}Z$/, "Z");
      case "human": {
        // Human-readable: "Thursday, December 11, 2025 at 13:19 UTC"
        const options: Intl.DateTimeFormatOptions = {
          weekday: "long",
          year: "numeric",
          month: "long",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          timeZone: "UTC",
          hour12: false,
        };
        const formatted = now.toLocaleDateString("en-US", options);
        return `${formatted} UTC`;
      }
      case "date":
      default:
        // Date only: YYYY-MM-DD
        return now.toISOString().split("T")[0];
    }
  },
});