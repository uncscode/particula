import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains } from "./helpers/assert-error-envelope";
import { getInvocations, installSubprocessMocks, restoreSubprocessMocks, setSpawnResponse } from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("adw_setup wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });
  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("requires command unless wizard/help", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const result = await execute({});
    assertContains(String(result), "requires 'command'");
  });

  it("rejects wizard+command combination", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const result = await execute({ wizard: true, command: "env" });
    assertContains(String(result), "cannot be combined");
  });

  it("supports wizard and help flows", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");

    await execute({ wizard: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup");

    await execute({ help: true });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup --help");

    await execute({ command: "env", help: true, options: "format=json" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup env --help");
  });

  it("rejects wizard+command combination even on help path", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const invocationCount = getInvocations().length;
    const result = await execute({ wizard: true, command: "env", help: true });
    assertContains(String(result), "cannot be combined");
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("assembles bounded env, validate, and labels options", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");

    await execute({ command: "env", options: "with-templates" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup env --with-templates");

    await execute({ command: "validate", options: "format=json" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup validate --format json");

    await execute({ command: "labels", options: "dry-run" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup labels --dry-run");
  });

  it("omits whitespace-only options", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    await execute({ command: "check", options: "   " });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup check");
  });

  it("preserves args allowlist for passthrough commands", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    await execute({ command: "docs", args: ["apply", "--check"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup docs apply --check");

    await execute({ command: "pull-opencode", args: ["--ref", "main", "--yes"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup pull-opencode --ref main --yes");

    await execute({ command: "template", args: ["validate", "--format", "json"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup template validate --format json");

    const result = await execute({ command: "env", args: ["--strict"] });
    assertContains(String(result), "'args' is only supported");
  });

  it("supports bounded docs subcommand passthrough shapes", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");

    await execute({ command: "docs", args: ["scaffold", "--language", "python", "--force", "--no-detect"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw setup docs scaffold --language python --force --no-detect",
    );

    await execute({ command: "docs", args: ["token", "list"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup docs token list");

    await execute({ command: "docs", args: ["token", "set", "name", "value"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup docs token set name value");

    await execute({ command: "docs", args: ["token", "remove", "name"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw setup docs token remove name");

    await execute({ command: "docs", args: ["scaffold", "--language=cpp"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw setup docs scaffold --language=cpp",
    );
  });

  it("rejects unsupported bounded passthrough args before spawn", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "docs", args: ["--strict"] }))).toContain(
      "Unsupported docs subcommand '--strict'",
    );
    expect(String(await execute({ command: "docs", args: ["apply", "--verbose"] }))).toContain(
      "Unsupported docs arg '--verbose'",
    );
    expect(String(await execute({ command: "pull-opencode", args: ["positional"] }))).toContain(
      "Unsupported positional arg 'positional'",
    );
    expect(String(await execute({ command: "template", args: ["sync"] }))).toContain(
      "Unsupported template subcommand 'sync'",
    );

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects invalid docs nested args before spawn", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "docs", args: ["scaffold", "--language"] }))).toContain(
      "'--language' requires a non-empty value",
    );
    expect(String(await execute({ command: "docs", args: ["scaffold", "--language", "rust"] }))).toContain(
      "'--language' must be one of: python, cpp, typescript, minimal",
    );
    expect(String(await execute({ command: "docs", args: ["token"] }))).toContain(
      "requires a supported nested subcommand",
    );
    expect(String(await execute({ command: "docs", args: ["token", "set", "only-key"] }))).toContain(
      "requires exactly two positional args",
    );
    expect(String(await execute({ command: "docs", args: ["token", "remove", "--bad"] }))).toContain(
      "positional args must not start with '-'",
    );

    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("supports bounded pull passthrough flags and rejects malformed values before spawn", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");

    await execute({ command: "pull-opencode", args: ["--source-repo=https://example.com/repo", "--yes"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw setup pull-opencode --source-repo=https://example.com/repo --yes",
    );

    await execute({ command: "template", args: ["extract", "--diff", "--yes"] });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe(
      "uv run --active adw setup template extract --diff --yes",
    );

    const invocationCount = getInvocations().length;
    expect(String(await execute({ command: "pull-opencode", args: ["--source-repo"] }))).toContain(
      "Flag '--source-repo' requires a non-empty value",
    );
    expect(String(await execute({ command: "template", args: ["init", "--gitignore-mode", "bad"] }))).toContain(
      "'--gitignore-mode' must be one of",
    );
    expect(String(await execute({ command: "template", args: ["token", "add", "KEY"] }))).toContain(
      "requires '<key> --default <value> --description <value>'",
    );
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects non-string options payloads before spawn", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const invocationCount = getInvocations().length;
    const result = await execute({ command: "env", options: 123 as unknown as string });
    assertContains(String(result), "'options' must be a string");
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects protected passthrough flags before spawn", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const invocationCount = getInvocations().length;
    const result = await execute({ command: "docs", args: ["apply", "--format=json"] });
    assertContains(String(result), "Protected flag '--format'");
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects invalid or conflicting options before spawn", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const invocationCount = getInvocations().length;

    expect(String(await execute({ command: "env", options: "with-templates skip-templates" }))).toContain(
      "cannot be combined",
    );
    expect(String(await execute({ command: "validate", options: "format=yaml" }))).toContain(
      "format values must be one of",
    );
    expect(String(await execute({ command: "validate", options: "dry-run" }))).toContain(
      "token is not allowed for command 'validate'",
    );
    expect(String(await execute({ command: "docs", options: "with-templates" }))).toContain(
      "token is not allowed for command 'docs'",
    );
    expect(String(await execute({ command: "labels", options: "format=json" }))).toContain(
      "token is not allowed for command 'labels'",
    );
    expect(String(await execute({ command: "env", options: "format=json=bad" }))).toContain(
      "at most one '=' separator",
    );
    expect(String(await execute({ command: "env", options: "Format=json" }))).toContain(
      "lowercase-kebab-case",
    );
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("rejects options without a command before spawn", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    const invocationCount = getInvocations().length;
    const result = await execute({ wizard: true, options: "with-templates" });
    assertContains(String(result), "'options' requires a setup 'command'");
    expect(getInvocations()).toHaveLength(invocationCount);
  });

  it("prefers stderr over stdout in non-zero failure diagnostics", async () => {
    const execute = await loadToolExecute("../../adw_setup.ts");
    setSpawnResponse({ exitCode: 2, stderr: "fatal stderr", stdout: "shadow stdout" });
    const result = await execute({ command: "env" });
    const text = String(result);
    assertContains(text, "fatal stderr");
    expect(text).not.toContain("shadow stdout");
  });
});
