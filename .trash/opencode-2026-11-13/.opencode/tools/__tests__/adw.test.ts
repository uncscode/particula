import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnError,
  setSpawnResponse,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw compatibility wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("normalizes adw_id and delegates supported workflow commands", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    await execute({ command: "build", issue_number: 42, adw_id: "A1B2C3D4" });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw build 42 --adw-id a1b2c3d4");
  });

  it("omits blank optional adw_id values", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    await execute({ command: "build", issue_number: 42, adw_id: "   " });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw build 42");
  });

  it("rejects invalid adw_id before spawn", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    const invocationCount = getInvocations().length;

    const result = await execute({ command: "build", issue_number: 42, adw_id: "not-hex" });

    assertContains(String(result), "8-character hex string");
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects non-array and blank args before spawn", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "build", issue_number: 42, args: "--dry-run" }))).toContain(
      "expected an array of strings",
    );
    expect(String(await execute({ command: "build", issue_number: 42, args: ["   "] }))).toContain(
      "all entries must be non-empty strings",
    );

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects protected flags in exact and equals forms before spawn", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "build", issue_number: 42, args: ["--help"] }))).toContain(
      "Protected flag '--help'",
    );
    expect(
      String(await execute({ command: "build", issue_number: 42, args: ["--adw-id=abc12345"] })),
    ).toContain("Protected flag '--adw-id'");

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects protected flag aliases before spawn", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "build", issue_number: 42, args: ["--adw_id"] }))).toContain(
      "Protected flag '--adw-id'",
    );
    expect(String(await execute({ command: "build", issue_number: 42, args: ["--adw_id=abc12345"] }))).toContain(
      "Protected flag '--adw-id'",
    );
    expect(
      String(await execute({ command: "interpret-issue", args: ["--source_issue"], text: "demo" })),
    ).toContain("Protected flag '--source-issue'");
    expect(
      String(await execute({ command: "interpret-issue", args: ["--source_issue=42"], text: "demo" })),
    ).toContain("Protected flag '--source-issue'");

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("maps interpret-issue issue_number inputs to --source-issue", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    await execute({ command: "interpret-issue", issue_number: 42 });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw interpret-issue --source-issue 42",
    );
  });

  it("help bypasses normal validation and emits help", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    const result = await execute({
      command: "build",
      help: true,
      adw_id: "bad-id",
      args: ["--help=true"],
    });

    expect(String(result)).toContain("ok");
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw build --help");
  });

  it("accepts the retained light model tier on compatibility workflow commands", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    await execute({ command: "build", issue_number: 42, model: "light" });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw build 42 --model light");
  });

  it("does not forward unrelated top-level args on non-setup help requests", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    await execute({
      command: "create-issue",
      help: true,
      model: "heavy",
      adw_id: "A1B2C3D4",
      title: "Ignored",
      body: "Ignored body",
      args: ["--verbose"],
    });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw create-issue --help");
  });

  it("prefers stderr over stdout in failure envelopes", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });

    const result = await execute({ command: "build", issue_number: 42 });
    const text = String(result);

    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("returns compatibility failure envelope when stdout starts with ERROR", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    setSpawnResponse({ exitCode: 0, stdout: "ERROR: downstream reported a failure" });

    const result = await execute({ command: "status" });

    expect(String(result)).toBe("ADW Command Failed:\nERROR: downstream reported a failure");
  });

  it("prefers stderr over stdout for execution errors", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    setSpawnError({ stderr: "fatal execution stderr", stdout: "shadow stdout", message: "boom" });

    const result = await execute({ command: "status" });
    const text = String(result);

    assertContains(text, "fatal execution stderr");
    expect(text).not.toContain("shadow stdout");
  });

  it("rejects unsupported command-specific invalid input before spawn", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "build" }))).toContain("requires 'issue_number'");
    expect(String(await execute({ command: "setup" }))).toContain("requires 'args' for subcommands");

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects create-issue when title or body is missing before spawn", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "create-issue", title: "Only title" }))).toContain(
      "requires both 'title' and 'body' arguments",
    );
    expect(String(await execute({ command: "create-issue", body: "Only body" }))).toContain(
      "requires both 'title' and 'body' arguments",
    );

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects interpret-issue when source issue is missing or invalid", async () => {
    const execute = await loadToolExecute("../../adw.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "interpret-issue" }))).toContain(
      "requires either 'text' argument or 'issue_number' argument",
    );
    expect(String(await execute({ command: "interpret-issue", issue_number: 0 }))).toContain(
      "requires a positive integer 'issue_number' when 'text' is omitted",
    );

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("supports retained compatibility-only issue commands", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    await execute({ command: "create-issue", title: "Add feature", body: "Body text" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      'uv run --active adw create-issue --title Add feature --body Body text',
    );

    await execute({ command: "interpret-issue", text: "Add tests for auth" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      'uv run --active adw interpret-issue --text Add tests for auth',
    );
  });

  it("preserves free-form setup passthrough behavior", async () => {
    const execute = await loadToolExecute("../../adw.ts");

    await execute({ command: "setup", args: ["template", "validate"] });

    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup template validate");
  });

  it("does not support the removed init command", async () => {
    const sourceText = await Bun.file(new URL("../adw.ts", import.meta.url)).text();
    const commandEnumMatch = sourceText.match(/command:\s*tool\.schema[\s\S]*?\.enum\(\[([\s\S]*?)\]\)/);
    expect(commandEnumMatch).not.toBeNull();
    expect(commandEnumMatch?.[1]).not.toContain('"init"');
  });

  it("does not advertise the unsupported ship command", async () => {
    const sourceText = await Bun.file(new URL("../adw.ts", import.meta.url)).text();
    const commandEnumMatch = sourceText.match(/command:\s*tool\.schema[\s\S]*?\.enum\(\[([\s\S]*?)\]\)/);
    expect(commandEnumMatch).not.toBeNull();
    expect(commandEnumMatch?.[1]).not.toContain('"ship"');
    expect(sourceText).toContain("complete/patch/plan/build/test/review/document");
    expect(sourceText).not.toContain("complete/patch/plan/build/test/review/document/ship");
  });
});
