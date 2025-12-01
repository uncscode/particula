/**
 * Workspace Creator Tool
 *
 * Creates isolated ADW workspace with all pre-LLM setup steps:
 * - Fetches GitHub issue details
 * - Generates deterministic branch name
 * - Creates isolated git worktree
 * - Initializes ADW state with auto-populated fields
 *
 * This tool performs all deterministic setup before any LLM/AI work begins.
 * No LLM calls are made. No GitHub status updates are posted.
 *
 * Use this tool to prepare a workspace before running planning or implementation phases.
 */

import { tool } from "@opencode-ai/plugin";

export default tool({
  description: `Create isolated ADW workspace for a GitHub issue. Performs all pre-LLM setup:
  1. Fetches GitHub issue details
  2. Generates deterministic branch name
  3. Creates isolated git worktree under trees/{adw_id}/
  4. Initializes ADW state with all auto-populated fields
  
  Returns workspace details including ADW ID, branch name, and worktree path.
  
  Examples:
  - create_workspace({issueNumber: "123"}) - Create workspace for issue #123 with default complete workflow
  - create_workspace({issueNumber: "456", workflowType: "patch"}) - Create workspace for quick patch
  - create_workspace({issueNumber: "789", adwId: "abc12345"}) - Resume existing workspace
  - create_workspace({issueNumber: "101", outputMode: "json"}) - Get structured JSON output`,
  
  args: {
    issueNumber: tool.schema
      .string()
      .describe('GitHub issue number to create workspace for (e.g., "123"). Required.'),
    
    workflowType: tool.schema
      .enum(["complete", "patch", "document", "generate"])
      .optional()
      .describe('Workflow type: "complete" (default, full validation), "patch" (quick fixes), "document" (docs only), "generate" (code generation)'),
    
    adwId: tool.schema
      .string()
      .optional()
      .describe("Optional existing ADW ID (8-char hex like 'abc12345'). If not provided, generates new one."),
    
    triggeredBy: tool.schema
      .string()
      .optional()
      .describe('Who/what triggered this workflow. Examples: "manual" (default), "cron", "webhook", "user@example.com"'),
    
    outputMode: tool.schema
      .enum(["summary", "full", "json"])
      .optional()
      .describe('Output mode: "summary" (default, key details), "full" (complete state dump), "json" (structured data)'),
  },
  
  async execute(args) {
    const issueNumber = args.issueNumber;
    const workflowType = args.workflowType || "complete";
    const triggeredBy = args.triggeredBy || "manual";
    const outputMode = args.outputMode || "summary";
    
    // Build command
    const cmdParts = [
      "python3",
      `${import.meta.dir}/create_workspace.py`,
      issueNumber,
      `--workflow-type=${workflowType}`,
      `--triggered-by=${triggeredBy}`,
      `--output=${outputMode}`,
    ];
    
    // Add optional adw-id if provided
    if (args.adwId) {
      cmdParts.push(`--adw-id=${args.adwId}`);
    }
    
    try {
      // Execute the Python script
      const result = await Bun.$`${cmdParts}`.text();
      return result;
    } catch (error: any) {
      // Workspace creation failed - return the output with error details
      if (error.stdout) {
        return error.stdout.toString();
      }
      return `ERROR: Failed to create workspace: ${error.message}`;
    }
  },
});
