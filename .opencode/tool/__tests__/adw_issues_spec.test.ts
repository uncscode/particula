import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";

let lastCommand: (string | number)[] = [];
let executionError: any = null;
let executionOutput: string | undefined;

const setExecutionSuccess = (output?: string) => {
  executionError = null;
  executionOutput = output;
};

const setExecutionError = (error: any) => {
  executionError = error;
};

const bunTarget = (globalThis as any).Bun ?? {};
const bunShell = (_strings: TemplateStringsArray, ...values: unknown[]) => {
  const parts = (values[0] || []) as (string | number)[];
  lastCommand = parts;
  return {
    text: async () => {
      if (executionError) {
        throw executionError;
      }
      if (executionOutput !== undefined) {
        return executionOutput;
      }
      return parts.join(" ");
    },
  };
};

try {
  (bunTarget as any).$ = bunShell;
} catch {
  Object.defineProperty(bunTarget, "$", {
    value: bunShell,
    configurable: true,
    writable: true,
  });
}

if (!(globalThis as any).Bun) {
  (globalThis as any).Bun = bunTarget;
}

const adwIssuesSpecTool = (await import("../adw_issues_spec")).default as {
  execute: (args: Record<string, unknown>) => Promise<string>;
};

describe("adw_issues_spec tool", () => {
  beforeEach(() => {
    setExecutionSuccess();
    lastCommand = [];
  });

  it("builds batch-init command with and without adw_id", async () => {
    await adwIssuesSpecTool.execute({
      command: "batch-init",
      total: "5",
      source: "doc.md",
    });

    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "init",
      "--total",
      "5",
      "--source",
      "doc.md",
    ]);

    await adwIssuesSpecTool.execute({
      command: "batch-init",
      total: "2",
      source: "doc.md",
      adw_id: "abc12345",
    });

    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "init",
      "--total",
      "2",
      "--source",
      "doc.md",
      "--adw-id",
      "abc12345",
    ]);
  });

  it("builds batch-read commands for all read modes", async () => {
    await adwIssuesSpecTool.execute({ command: "batch-read", adw_id: "abc12345" });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "read",
      "--adw-id",
      "abc12345",
    ]);

    await adwIssuesSpecTool.execute({ command: "batch-read", adw_id: "abc12345", issue: "1" });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "read",
      "--adw-id",
      "abc12345",
      "--issue",
      "1",
    ]);

    await adwIssuesSpecTool.execute({
      command: "batch-read",
      adw_id: "abc12345",
      section: "scope",
    });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "read",
      "--adw-id",
      "abc12345",
      "--section",
      "scope",
    ]);

    await adwIssuesSpecTool.execute({
      command: "batch-read",
      adw_id: "abc12345",
      issue: "2",
      section: "testing_strategy",
      raw: true,
    });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "read",
      "--adw-id",
      "abc12345",
      "--issue",
      "2",
      "--section",
      "testing_strategy",
      "--raw",
    ]);
  });

  it("builds batch-write commands with and without section", async () => {
    await adwIssuesSpecTool.execute({
      command: "batch-write",
      adw_id: "abc12345",
      issue: "3",
      content: "payload",
    });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "write",
      "--adw-id",
      "abc12345",
      "--issue",
      "3",
      "--content",
      "payload",
    ]);

    await adwIssuesSpecTool.execute({
      command: "batch-write",
      adw_id: "abc12345",
      issue: "3",
      section: "scope",
      content: "payload",
    });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "write",
      "--adw-id",
      "abc12345",
      "--issue",
      "3",
      "--content",
      "payload",
      "--section",
      "scope",
    ]);
  });

  it("builds batch-log commands for write and read modes", async () => {
    await adwIssuesSpecTool.execute({
      command: "batch-log",
      adw_id: "abc12345",
      issue: "4",
      reviewer: "testing",
      status: "PASS",
      note: "ok",
    });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "log",
      "--adw-id",
      "abc12345",
      "--issue",
      "4",
      "--reviewer",
      "testing",
      "--status",
      "PASS",
      "--note",
      "ok",
    ]);

    await adwIssuesSpecTool.execute({
      command: "batch-log",
      adw_id: "abc12345",
      issue: "4",
      read: true,
    });
    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "log",
      "--adw-id",
      "abc12345",
      "--issue",
      "4",
      "--read",
    ]);
  });

  it("builds batch-summary command", async () => {
    await adwIssuesSpecTool.execute({ command: "batch-summary", adw_id: "abc12345" });

    assert.deepEqual(lastCommand, [
      "uv",
      "run",
      "adw",
      "spec",
      "batch",
      "summary",
      "--adw-id",
      "abc12345",
    ]);
  });

  it("rejects missing adw_id for non-init commands", async () => {
    const result = await adwIssuesSpecTool.execute({ command: "batch-read" });

    assert.ok(result.includes("adw_id"));
    assert.ok(result.includes("Example usage"));
  });

  it("rejects invalid issue values", async () => {
    const result = await adwIssuesSpecTool.execute({
      command: "batch-read",
      adw_id: "abc12345",
      issue: "0",
    });

    assert.ok(result.includes("Invalid issue"));
  });

  it("rejects invalid section values", async () => {
    const result = await adwIssuesSpecTool.execute({
      command: "batch-read",
      adw_id: "abc12345",
      section: "bad_section",
    });

    assert.ok(result.includes("Invalid section"));
  });

  it("rejects invalid command values", async () => {
    const result = await adwIssuesSpecTool.execute({ command: "bad-command" });

    assert.ok(result.includes("Invalid command"));
  });

  it("rejects invalid totals", async () => {
    const tooLow = await adwIssuesSpecTool.execute({
      command: "batch-init",
      total: "0",
      source: "doc.md",
    });
    assert.ok(tooLow.includes("Invalid total"));

    const tooHigh = await adwIssuesSpecTool.execute({
      command: "batch-init",
      total: "51",
      source: "doc.md",
    });
    assert.ok(tooHigh.includes("Invalid total"));
  });

  it("rejects missing total or source for batch-init", async () => {
    const missingTotal = await adwIssuesSpecTool.execute({
      command: "batch-init",
      source: "doc.md",
    });
    assert.ok(missingTotal.includes("batch-init requires"));

    const missingSource = await adwIssuesSpecTool.execute({
      command: "batch-init",
      total: "1",
    });
    assert.ok(missingSource.includes("batch-init requires"));
  });

  it("rejects invalid status values", async () => {
    const result = await adwIssuesSpecTool.execute({
      command: "batch-log",
      adw_id: "abc12345",
      issue: "1",
      reviewer: "testing",
      status: "BAD",
    });

    assert.ok(result.includes("Invalid status"));
  });

  it("rejects missing content for batch-write", async () => {
    const result = await adwIssuesSpecTool.execute({
      command: "batch-write",
      adw_id: "abc12345",
      issue: "1",
    });

    assert.ok(result.includes("batch-write requires 'content'"));
  });

  it("rejects invalid issue for batch-write", async () => {
    const result = await adwIssuesSpecTool.execute({
      command: "batch-write",
      adw_id: "abc12345",
      issue: "-1",
      content: "payload",
    });

    assert.ok(result.includes("Invalid issue"));
  });

  it("rejects missing reviewer or status for batch-log write", async () => {
    const missingReviewer = await adwIssuesSpecTool.execute({
      command: "batch-log",
      adw_id: "abc12345",
      issue: "1",
      status: "PASS",
    });
    assert.ok(missingReviewer.includes("batch-log requires 'reviewer'"));

    const missingStatus = await adwIssuesSpecTool.execute({
      command: "batch-log",
      adw_id: "abc12345",
      issue: "1",
      reviewer: "testing",
    });
    assert.ok(missingStatus.includes("batch-log requires 'status'"));
  });

  it("wraps CLI error output", async () => {
    setExecutionError({ stdout: "Error: boom", stderr: "nope", message: "fail" });
    const result = await adwIssuesSpecTool.execute({
      command: "batch-summary",
      adw_id: "abc12345",
    });

    assert.ok(result.includes("ERROR: adw spec batch command failed"));
    assert.ok(result.includes("Error: boom"));
  });

  it("wraps error output returned by the CLI", async () => {
    setExecutionSuccess("Error: bad request");
    const result = await adwIssuesSpecTool.execute({
      command: "batch-summary",
      adw_id: "abc12345",
    });

    assert.ok(result.includes("ERROR: adw spec batch command failed"));
    assert.ok(result.includes("Error: bad request"));
  });

  it("returns a fallback message when CLI output is empty", async () => {
    setExecutionSuccess("");
    const result = await adwIssuesSpecTool.execute({
      command: "batch-summary",
      adw_id: "abc12345",
    });

    assert.ok(result.includes("returned no output"));
  });
});
