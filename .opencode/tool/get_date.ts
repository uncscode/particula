/**
 * ADW Date Tool
 * 
 * Returns the current date or datetime in various formats.
 * This tool definition invokes the Python script.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Get the current date or datetime in various formats (UTC)",
  args: {
    format: tool.schema
      .enum(["date", "datetime", "human"])
      .optional()
      .describe("Format: 'date' (date only at midnight UTC), 'datetime' (current date and time), 'human' (human-readable format). Default: 'date'"),
  },
  async execute(args) {
    const format = args.format || "date";
    let flag = "-d"; // Default to date only
    
    switch (format) {
      case "datetime":
        flag = "-t";
        break;
      case "human":
        flag = "-r";
        break;
      case "date":
      default:
        flag = "-d";
        break;
    }
    
    // Execute the Python script with Bun's shell API
    const result = await Bun.$`python3 ${import.meta.dir}/get_date.py ${flag}`.text();
    return result.trim();
  },
});