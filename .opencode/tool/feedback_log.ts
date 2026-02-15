/**
 * Feedback Log Tool
 *
 * Logs structured feedback when agents encounter friction, bugs, or workflow gaps.
 */

import { tool } from "@opencode-ai/plugin";

const PROJECT_ROOT = `${import.meta.dir}/../..`;

const shouldInclude = (value?: string) => Boolean(value && value.trim().length > 0);

export default tool({
  description: `Log structured feedback when you encounter friction, bugs, or workflow gaps across tools, ADW workflows, agents, or any part of the system.

Use this tool reactively when you:
- A tool returned an error that was unclear or unhelpful
- Had to retry an operation 3+ times before success
- Encountered a workflow gap (no tool exists for a needed operation)
- Found a workaround for a limitation that should be fixed
- Observed unexpected behavior from a tool, ADW workflow, or agent
- A tool's documentation/description didn't match its actual behavior
- An ADW workflow step failed, stalled, or produced unexpected results
- A platform operation (GitHub/GitLab) behaved inconsistently
- Agent coordination or subagent handoff had issues

Feedback is fire-and-forget: it never blocks your workflow. Rate limited to 1 entry per 60s per agent.`,
  args: {
    category: tool.schema
      .enum(["bug", "feature", "friction", "performance"])
      .describe("Category: bug (broken), feature (missing), friction (hard to use), performance (slow)."),
    severity: tool.schema
      .enum(["low", "medium", "high", "critical"])
      .describe("Severity: low (minor), medium (noticeable), high (significant), critical (blocking)."),
    description: tool.schema
      .string()
      .describe("Clear description of the issue encountered."),
    suggestedFix: tool.schema
      .string()
      .optional()
      .describe("Proposed fix or improvement (optional)."),
    toolName: tool.schema
      .string()
      .optional()
      .describe("Name of the tool that triggered this feedback (optional)."),
    workflowStep: tool.schema
      .string()
      .optional()
      .describe("Current workflow step (e.g., 'build', 'test', 'review')."),
    agentType: tool.schema
      .string()
      .optional()
      .describe("Agent type submitting feedback (e.g., 'adw-build', 'adw-review')."),
    adwId: tool.schema
      .string()
      .optional()
      .describe("ADW workflow ID for traceability."),
    context: tool.schema
      .string()
      .optional()
      .describe("Additional context (issue number, retry count, error message, etc.)."),
  },
  async execute(args) {
    const cmdParts: (string | number)[] = [
      "python3",
      `${import.meta.dir}/feedback_log.py`,
      "--category",
      args.category,
      "--severity",
      args.severity,
      "--description",
      args.description,
    ];

    if (shouldInclude(args.suggestedFix)) {
      cmdParts.push("--suggested-fix", args.suggestedFix);
    }
    if (shouldInclude(args.toolName)) {
      cmdParts.push("--tool-name", args.toolName);
    }
    if (shouldInclude(args.workflowStep)) {
      cmdParts.push("--workflow-step", args.workflowStep);
    }
    if (shouldInclude(args.agentType)) {
      cmdParts.push("--agent-type", args.agentType);
    }
    if (shouldInclude(args.adwId)) {
      cmdParts.push("--adw-id", args.adwId);
    }
    if (shouldInclude(args.context)) {
      cmdParts.push("--context", args.context);
    }

    try {
      const result = await Bun.$`${cmdParts}`.cwd(PROJECT_ROOT).text();
      return result || "Feedback logged but returned no output.";
    } catch (error: any) {
      const stdout = error?.stdout?.toString?.() || "";
      const stderr = error?.stderr?.toString?.() || "";
      const message = error?.message || "Unknown error";
      const exitCode = error?.exitCode ?? error?.code;
      const exitInfo = exitCode !== undefined ? ` (exit ${exitCode})` : "";

      if (stdout.trim()) {
        return stdout;
      }

      if (stderr.trim()) {
        return `ERROR: Feedback logging failed${exitInfo}\n\n${stderr}`;
      }

      return `ERROR: Failed to log feedback${exitInfo}: ${message}`;
    }
  },
});
