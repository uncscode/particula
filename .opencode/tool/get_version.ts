/**
 * ADW Version Tool
 * 
 * Returns version information from the project.
 * This tool definition invokes the Python script.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Get version information from pyproject.toml or package.json",
  args: {
    file: tool.schema
      .string()
      .optional()
      .describe("Path to the file to read version from. Defaults to pyproject.toml in the current directory"),
  },
  async execute(args) {
    // Execute the Python script with Bun's shell API
    const command = args.file
      ? await Bun.$`python3 ${import.meta.dir}/get_version.py ${args.file}`.text()
      : await Bun.$`python3 ${import.meta.dir}/get_version.py`.text();
    
    return command.trim();
  },
});