/**
 * Feedback Log Tool
 *
 * Logs structured feedback when agents encounter friction, bugs, or workflow gaps.
 */

import { tool } from "@opencode-ai/plugin";
import { realpathSync } from "node:fs";
import path from "node:path";

const PROJECT_ROOT = realpathSync(path.resolve(import.meta.dir, "../.."));
const SCRIPT_PATH = `${import.meta.dir}/feedback_log.py`;

const VALID_CATEGORIES = new Set(["bug", "feature", "friction", "performance"]);
const VALID_SEVERITIES = new Set(["low", "medium", "high", "critical"]);

const shouldInclude = (value?: string) => Boolean(value && value.trim().length > 0);

const normalizeRequiredString = (value: unknown, fieldName: string, cliFlag: string) => {
  if (typeof value !== "string") {
    return {
      error: `ERROR: ${fieldName} must be a non-empty string (${cliFlag}).`,
      value: undefined,
    };
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return {
      error: `ERROR: ${fieldName} must be a non-empty string (${cliFlag}).`,
      value: undefined,
    };
  }

  return { error: undefined, value: trimmed };
};

const normalizeOptionalString = (value: unknown): string | undefined => {
  if (typeof value !== "string") {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed || undefined;
};

const validateEnum = (
  value: unknown,
  {
    fieldName,
    cliFlag,
    allowed,
  }: {
  fieldName: string;
  cliFlag: string;
  allowed: Set<string>;
  },
) => {
  const normalized = normalizeRequiredString(value, fieldName, cliFlag);
  if (normalized.error || normalized.value === undefined) {
    return normalized;
  }

  if (!allowed.has(normalized.value)) {
    return {
      error: `ERROR: Invalid ${fieldName} '${normalized.value}'.`,
      value: undefined,
    };
  }

  return normalized;
};

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
      .describe("Current workflow step (e.g., 'build', 'test', 'review')."),
    agentType: tool.schema
      .string()
      .describe("Agent type submitting feedback (e.g., 'adw-build', 'adw-review')."),
    adwId: tool.schema
      .string()
      .describe("ADW workflow ID for traceability."),
    context: tool.schema
      .string()
      .optional()
      .describe("Additional context (issue number, retry count, error message, etc.)."),
  },
  async execute(args) {
    const category = validateEnum(args.category, {
      fieldName: "category",
      cliFlag: "--category",
      allowed: VALID_CATEGORIES,
    });
    if (category.error || category.value === undefined) {
      return category.error;
    }

    const severity = validateEnum(args.severity, {
      fieldName: "severity",
      cliFlag: "--severity",
      allowed: VALID_SEVERITIES,
    });
    if (severity.error || severity.value === undefined) {
      return severity.error;
    }

    const description = normalizeRequiredString(args.description, "description", "--description");
    if (description.error || description.value === undefined) {
      return description.error;
    }

    const workflowStep = normalizeRequiredString(
      args.workflowStep,
      "workflowStep",
      "--workflow-step",
    );
    if (workflowStep.error || workflowStep.value === undefined) {
      return workflowStep.error;
    }

    const agentType = normalizeRequiredString(args.agentType, "agentType", "--agent-type");
    if (agentType.error || agentType.value === undefined) {
      return agentType.error;
    }

    const adwId = normalizeRequiredString(args.adwId, "adwId", "--adw-id");
    if (adwId.error || adwId.value === undefined) {
      return adwId.error;
    }

    const suggestedFix = normalizeOptionalString(args.suggestedFix);
    const toolName = normalizeOptionalString(args.toolName);
    const context = normalizeOptionalString(args.context);

    const cmdParts: (string | number)[] = [
      "python3",
      SCRIPT_PATH,
      "--category",
      category.value,
      "--severity",
      severity.value,
      "--description",
      description.value,
      "--workflow-step",
      workflowStep.value,
      "--agent-type",
      agentType.value,
      "--adw-id",
      adwId.value,
    ];

    if (shouldInclude(suggestedFix)) {
      cmdParts.push("--suggested-fix", suggestedFix);
    }
    if (shouldInclude(toolName)) {
      cmdParts.push("--tool-name", toolName);
    }
    if (shouldInclude(context)) {
      cmdParts.push("--context", context);
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
