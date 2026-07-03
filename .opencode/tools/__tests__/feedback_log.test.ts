import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { readFileSync, realpathSync } from "node:fs";
import path from "node:path";

import { buildDollarFailure, buildSuccessOutput } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("feedback_log wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("runs python backend with required workflow context and omits blank optional values", async () => {
    setDollarText(buildSuccessOutput("Feedback logged, thank you. [bug/high] example"));
    const execute = await loadToolExecute("../../feedback_log.ts");

    const result = await execute({
      category: "bug",
      severity: "high",
      description: "  wrapper failure  ",
      workflowStep: " build ",
      agentType: " adw-build ",
      adwId: " 4ba84482 ",
      suggestedFix: "   ",
      toolName: " feedback_log ",
      context: "   ",
    });

    expect(result).toBe("Feedback logged, thank you. [bug/high] example");

    const cmd = getInvocations().at(-1)?.args ?? [];
    const cwd = getInvocations().at(-1)?.cwd;
    const expectedRepoRoot = realpathSync(path.resolve(import.meta.dir, "../../.."));
    expect(cmd).toContain("python3");
    expect(cmd.join(" ")).toContain("feedback_log.py");
    expect(cwd).toBeDefined();
    expect(cwd).toBe(expectedRepoRoot);
    expect(cmd).toEqual(
      expect.arrayContaining([
        "--category",
        "bug",
        "--severity",
        "high",
        "--description",
        "wrapper failure",
        "--workflow-step",
        "build",
        "--agent-type",
        "adw-build",
        "--adw-id",
        "4ba84482",
        "--tool-name",
        "feedback_log",
      ]),
    );
    expect(cmd).not.toContain("--suggested-fix");
    expect(cmd).not.toContain("--context");
  });

  it("rejects missing required workflow context before subprocess execution", async () => {
    const execute = await loadToolExecute("../../feedback_log.ts");

    const result = await execute({
      category: "bug",
      severity: "high",
      description: "desc",
      workflowStep: "",
      agentType: "adw-build",
      adwId: "4ba84482",
    });

    expect(result).toBe("ERROR: workflowStep must be a non-empty string (--workflow-step).");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects blank agentType and adwId values before subprocess execution", async () => {
    const execute = await loadToolExecute("../../feedback_log.ts");

    expect(
      await execute({
        category: "bug",
        severity: "high",
        description: "desc",
        workflowStep: "build",
        agentType: "   ",
        adwId: "4ba84482",
      }),
    ).toBe("ERROR: agentType must be a non-empty string (--agent-type).");

    expect(
      await execute({
        category: "bug",
        severity: "high",
        description: "desc",
        workflowStep: "build",
        agentType: "adw-build",
        adwId: "   ",
      }),
    ).toBe("ERROR: adwId must be a non-empty string (--adw-id).");

    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects blank and malformed required values deterministically", async () => {
    const execute = await loadToolExecute("../../feedback_log.ts");

    expect(
      await execute({
        category: "bug",
        severity: "high",
        description: "desc",
        workflowStep: "build",
        agentType: 17,
        adwId: "4ba84482",
      }),
    ).toBe("ERROR: agentType must be a non-empty string (--agent-type).");

    expect(
      await execute({
        category: "wrong",
        severity: "high",
        description: "desc",
        workflowStep: "build",
        agentType: "adw-build",
        adwId: "4ba84482",
      }),
    ).toBe("ERROR: Invalid category 'wrong'.");

    expect(getInvocations()).toHaveLength(0);
  });

  it("preserves stdout, stderr, then message diagnostic precedence", async () => {
    const execute = await loadToolExecute("../../feedback_log.ts");

    setDollarError(buildDollarFailure({ stdout: "stdout diagnostic", stderr: "stderr shadow" }));
    expect(await execute(validArgs())).toBe("stdout diagnostic");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "stderr diagnostic" }));
    expect(await execute(validArgs())).toBe("ERROR: Feedback logging failed\n\nstderr diagnostic");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "", message: "message diagnostic" }));
    expect(await execute(validArgs())).toBe(
      "ERROR: Failed to log feedback: message diagnostic",
    );
  });

  it("keeps wrapper and backend category/severity allowlists in parity", async () => {
    const toolSource = readFileSync(path.resolve(import.meta.dir, "../feedback_log.ts"), "utf8");
    const backendSource = readFileSync(path.resolve(import.meta.dir, "../feedback_log.py"), "utf8");

    expect(extractArray(toolSource, "VALID_CATEGORIES")).toEqual(
      extractArray(backendSource, "FALLBACK_FEEDBACK_CATEGORIES"),
    );
    expect(extractArray(toolSource, "VALID_SEVERITIES")).toEqual(
      extractArray(backendSource, "FALLBACK_FEEDBACK_SEVERITIES"),
    );
  });
});

function extractArray(source: string, constName: string): string[] {
  const match = source.match(new RegExp(`${constName}\\s*=\\s*(?:new Set\\()?(\\[[^\\]]+\\])`));
  if (!match) {
    throw new Error(`Unable to find ${constName}`);
  }

  return JSON.parse(match[1]) as string[];
}

function validArgs() {
  return {
    category: "bug",
    severity: "high",
    description: "desc",
    workflowStep: "build",
    agentType: "adw-build",
    adwId: "4ba84482",
  };
}
