import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_issues_batch_log/summary wrappers", () => {
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

  it("routes log read with read option token", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_log.ts");

    const result = await execute({ adw_id: "A1B2C3D4", issue: "3", options: "read" });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "log",
      "--adw-id",
      "a1b2c3d4",
      "--issue",
      "3",
      "--read",
    ]);
  });

  it("routes log write with required reviewer and status", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_log.ts");

    const result = await execute({
      adw_id: "A1B2C3D4",
      issue: "3",
      reviewer: "testing",
      status: "PASS",
      note: "looks good",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "log",
      "--adw-id",
      "a1b2c3d4",
      "--issue",
      "3",
      "--reviewer",
      "testing",
      "--status",
      "PASS",
      "--note",
      "looks good",
    ]);
  });

  it("handles optional note without argument drift", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_log.ts");

    const result = await execute({
      adw_id: "deadbeef",
      issue: "4",
      reviewer: "scope",
      status: "REVISED",
      note: "   ",
    });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "log",
      "--adw-id",
      "deadbeef",
      "--issue",
      "4",
      "--reviewer",
      "scope",
      "--status",
      "REVISED",
    ]);
  });

  it("dispatches summary command with expected args", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_summary.ts");

    const result = await execute({ adw_id: "A1B2C3D4" });

    expect(result).toBe("ok");
    expect(getInvocations().at(-1)?.args).toEqual([
      "uv",
      "run",
      "--active",
      "adw",
      "spec",
      "batch",
      "summary",
      "--adw-id",
      "a1b2c3d4",
    ]);
  });

  it("rejects malformed duplicate or command-invalid log options", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_log.ts");

    const malformed = await execute({ adw_id: "deadbeef", issue: "3", options: "read=true" });
    const duplicate = await execute({ adw_id: "deadbeef", issue: "3", options: "read read" });
    const invalid = await execute({ adw_id: "deadbeef", issue: "3", options: "raw" });

    expect(malformed).toContain(
      'ERROR: Invalid options token "read=true" for \'batch-log\': token does not accept a value.',
    );
    expect(duplicate).toContain(
      'ERROR: Invalid options token "read" for \'batch-log\': duplicate token.',
    );
    expect(invalid).toContain(
      'ERROR: Invalid options token "raw" for \'batch-log\': token is not allowed for this command.',
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects blank write-mode reviewer and status with current guidance", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_log.ts");

    const blankReviewer = await execute({
      adw_id: "deadbeef",
      issue: "3",
      reviewer: "   ",
      status: "PASS",
    });
    const blankStatus = await execute({
      adw_id: "deadbeef",
      issue: "3",
      reviewer: "testing",
      status: "   ",
    });

    expect(blankReviewer).toContain("batch-log requires 'reviewer' when options is not 'read'.");
    expect(blankStatus).toContain("batch-log requires 'status' when options is not 'read'.");
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-string or malformed multi-separator log options", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_log.ts");

    const nonString = await execute({ adw_id: "deadbeef", issue: "3", options: true });
    const malformed = await execute({ adw_id: "deadbeef", issue: "3", options: "read=true=false" });

    expect(nonString).toContain("ERROR: 'options' must be a string when provided.");
    expect(malformed).toContain(
      'ERROR: Invalid options token "read=true=false" for \'batch-log\': tokens must contain at most one \'=\' separator.',
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("rejects non-empty options for batch-summary", async () => {
    const execute = await loadToolExecute("../../adw_issues_batch_summary.ts");

    const result = await execute({ adw_id: "deadbeef", options: "read" });

    expect(result).toContain(
      'ERROR: Invalid options token "read" for \'batch-summary\': token is not allowed for this command.',
    );
    expect(getInvocations()).toHaveLength(0);
  });
});
