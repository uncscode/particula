import assert from "node:assert/strict";
import { beforeEach, describe, it, mock } from "node:test";

const createSchema = (kind: string, extras: Record<string, unknown> = {}) => {
  return {
    kind,
    ...extras,
    optional() {
      return { ...this, optional: true };
    },
    describe(description: string) {
      return { ...this, description };
    },
  };
};

const schema = {
  enum: (values: string[]) => createSchema("enum", { values }),
  string: () => createSchema("string"),
};

const toolFn = (definition: unknown) => definition;
(toolFn as any).schema = schema;

mock.module("@opencode-ai/plugin", () => ({
  tool: toolFn,
}));

let executionBehavior: { kind: "success"; output?: string } | { kind: "error"; error: any } = {
  kind: "success",
};
let lastCommand: (string | number)[] = [];
let lastCwd: string | undefined;

const setExecutionSuccess = (output?: string) => {
  executionBehavior = { kind: "success", output };
};

const setExecutionError = (error: any) => {
  executionBehavior = { kind: "error", error };
};

(globalThis as any).Bun = {
  $: (_strings: TemplateStringsArray, ...values: unknown[]) => {
    const parts = (values[0] || []) as (string | number)[];
    lastCommand = parts;
    lastCwd = undefined;

    const runText = async () => {
      if (executionBehavior.kind === "error") {
        throw executionBehavior.error;
      }
      if (executionBehavior.output !== undefined) {
        return executionBehavior.output;
      }
      return parts.join(" ");
    };

    return {
      cwd: (cwd: string) => {
        lastCwd = cwd;
        return {
          text: runText,
        };
      },
      text: runText,
    };
  },
};

const feedbackLogTool = (await import("../feedback_log")).default as {
  description: string;
  args: Record<string, any>;
  execute: (args: Record<string, unknown>) => Promise<string>;
};

const REQUIRED_BULLETS = [
  "Tool returned an error that was unclear or unhelpful",
  "Had to retry an operation 3+ times before success",
  "Encountered a workflow gap (no tool exists for needed operation)",
  "Found a workaround for a limitation that should be fixed",
  "Observed unexpected behavior from another tool",
  "A tool's documentation/description didn't match its actual behavior",
];

const REQUIRED_ARGS = {
  category: "bug",
  severity: "medium",
  description: "Something went wrong.",
};

const OPTIONAL_ARGS = {
  suggestedFix: "Try restarting.",
  toolName: "run_pytest",
  workflowStep: "build",
  agentType: "adw-build",
  adwId: "abc12345",
  context: "Extra context.",
};

const EXPECTED_PROJECT_ROOT = `${import.meta.dir}/../..`;

const includesFlag = (flag: string) => lastCommand.some((part) => part === flag);

const includesFlagValue = (flag: string, value: string) => {
  const index = lastCommand.findIndex((part) => part === flag);
  return index >= 0 && lastCommand[index + 1] === value;
};

describe("feedback_log tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
    lastCwd = undefined;
  });

  it("includes reactive use-case examples in description", () => {
    REQUIRED_BULLETS.forEach((bullet) => {
      assert.ok(feedbackLogTool.description.includes(bullet));
    });
  });

  it("defines required args with enums and descriptions", () => {
    const { args } = feedbackLogTool;
    assert.ok(args.category);
    assert.deepEqual(args.category.values, ["bug", "feature", "friction", "performance"]);
    assert.ok(args.severity);
    assert.deepEqual(args.severity.values, ["low", "medium", "high", "critical"]);
    assert.ok(args.description);
    assert.ok(String(args.description.description || "").length > 0);
  });

  it("builds CLI command with required args", async () => {
    await feedbackLogTool.execute({ ...REQUIRED_ARGS });

    assert.ok(lastCommand.includes("python3"));
    assert.ok(lastCommand.some((part) => String(part).endsWith("/feedback_log.py")));
    assert.ok(includesFlagValue("--category", REQUIRED_ARGS.category));
    assert.ok(includesFlagValue("--severity", REQUIRED_ARGS.severity));
    assert.ok(includesFlagValue("--description", REQUIRED_ARGS.description));
    assert.equal(lastCwd, EXPECTED_PROJECT_ROOT);
  });

  it("omits optional args when undefined or empty", async () => {
    await feedbackLogTool.execute({
      ...REQUIRED_ARGS,
      suggestedFix: "",
      toolName: "   ",
    });

    assert.equal(includesFlag("--suggested-fix"), false);
    assert.equal(includesFlag("--tool-name"), false);
  });

  it("includes optional args when provided", async () => {
    await feedbackLogTool.execute({ ...REQUIRED_ARGS, ...OPTIONAL_ARGS });

    assert.ok(includesFlagValue("--suggested-fix", OPTIONAL_ARGS.suggestedFix));
    assert.ok(includesFlagValue("--tool-name", OPTIONAL_ARGS.toolName));
    assert.ok(includesFlagValue("--workflow-step", OPTIONAL_ARGS.workflowStep));
    assert.ok(includesFlagValue("--agent-type", OPTIONAL_ARGS.agentType));
    assert.ok(includesFlagValue("--adw-id", OPTIONAL_ARGS.adwId));
    assert.ok(includesFlagValue("--context", OPTIONAL_ARGS.context));
  });

  it("returns stdout on success", async () => {
    setExecutionSuccess("ok");
    const result = await feedbackLogTool.execute({ ...REQUIRED_ARGS });

    assert.equal(result, "ok");
  });

  it("returns helpful error message on failure", async () => {
    setExecutionError({ stdout: "", stderr: "Boom", message: "bad", exitCode: 2 });
    const result = await feedbackLogTool.execute({ ...REQUIRED_ARGS });

    assert.ok(result.includes("Feedback logging failed"));
    assert.ok(result.includes("exit 2"));
    assert.ok(result.includes("Boom"));
  });

  it("prefers stdout when error includes stdout", async () => {
    setExecutionError({ stdout: "from-stdout", stderr: "from-stderr", message: "msg" });
    const result = await feedbackLogTool.execute({ ...REQUIRED_ARGS });

    assert.equal(result, "from-stdout");
  });
});
