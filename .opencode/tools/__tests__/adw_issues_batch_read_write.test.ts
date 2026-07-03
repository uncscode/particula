import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_issues_batch_read/write wrappers", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("reads metadata section with raw option token", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_read.ts");

    const result = await execute({
      adw_id: "A1B2C3D4",
      issue: "1",
      section: "metadata",
      options: "raw",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw spec batch read --adw-id a1b2c3d4 --issue 1 --section metadata --raw",
    );
  });

  it("reads full batch state with adw_id only", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_read.ts");

    const result = await execute({ adw_id: "deadbeef" });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "read",
      "--adw-id",
      "deadbeef",
    ]);
  });

  it("reads regular section without metadata shortcuts", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_read.ts");

    const result = await execute({ adw_id: "deadbeef", issue: "2", section: "scope" });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "read",
      "--adw-id",
      "deadbeef",
      "--issue",
      "2",
      "--section",
      "scope",
    ]);
  });

  it("omits section on metadata write when optional", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_write.ts");

    const result = await execute({
      adw_id: "A1B2C3D4",
      issue: "1",
      content: '{"metadata":{"title":"Add feature X"}}',
      section: "   ",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "write",
      "--adw-id",
      "a1b2c3d4",
      "--issue",
      "1",
      "--content",
      '{"metadata":{"title":"Add feature X"}}',
    ]);
  });

  it("rejects invalid issue or section with deterministic error", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_write.ts");

    const invalidIssue = await execute({ adw_id: "deadbeef", issue: "zero", content: "body" });
    const invalidSection = await execute({
      adw_id: "deadbeef",
      issue: "1",
      content: "body",
      section: "bad\nsection",
    });

    expect(invalidIssue).toContain('ERROR: Invalid issue "zero".');
    expect(invalidSection).toContain('ERROR: Invalid section token "bad\nsection".');
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects malformed duplicate or command-invalid read options", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_read.ts");

    const malformed = await execute({ adw_id: "deadbeef", options: "raw=value" });
    const duplicate = await execute({ adw_id: "deadbeef", options: "raw raw" });
    const invalid = await execute({ adw_id: "deadbeef", options: "read" });

    expect(malformed).toContain(
      'ERROR: Invalid options token "raw=value" for \'batch-read\': token does not accept a value.',
    );
    expect(duplicate).toContain(
      'ERROR: Invalid options token "raw" for \'batch-read\': duplicate token.',
    );
    expect(invalid).toContain(
      'ERROR: Invalid options token "read" for \'batch-read\': token is not allowed for this command.',
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects blank required inputs before subprocess execution", async () => {
    const executeRead = await loadToolExecute("../../adw_issues_batch_read.ts");
    const executeWrite = await loadToolExecute("../../adw_issues_batch_write.ts");

    const blankAdwId = await executeRead({ adw_id: "   " });
    const blankIssue = await executeWrite({ adw_id: "deadbeef", issue: "   ", content: "body" });
    const blankContent = await executeWrite({ adw_id: "deadbeef", issue: "1", content: "   " });

    expect(blankAdwId).toContain("'adw_id' is required for all commands except batch-init.");
    expect(blankIssue).toContain("batch-write requires 'issue'.");
    expect(blankContent).toContain("batch-write requires 'content'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-string or malformed multi-separator options", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_read.ts");

    const nonString = await execute({ adw_id: "deadbeef", options: 7 });
    const malformed = await execute({ adw_id: "deadbeef", options: "raw=a=b" });

    expect(nonString).toContain("ERROR: 'options' must be a string when provided.");
    expect(malformed).toContain(
      'ERROR: Invalid options token "raw=a=b" for \'batch-read\': tokens must contain at most one \'=\' separator.',
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("sanitizes invalid token diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_read.ts");

    const result = await execute({ adw_id: "deadbeef", options: "bad\nread" });

    expect(result).toContain("ERROR: 'options' must not contain control characters.");
    expect(result).not.toContain("bad\nread");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-empty options for batch-write", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_write.ts");

    const result = await execute({
      adw_id: "deadbeef",
      issue: "1",
      content: "body",
      options: "raw",
    });

    expect(result).toContain(
      'ERROR: Invalid options token "raw" for \'batch-write\': token is not allowed for this command.',
    );
    expect(getInvocations()).toHaveLength(0);
  });
});
