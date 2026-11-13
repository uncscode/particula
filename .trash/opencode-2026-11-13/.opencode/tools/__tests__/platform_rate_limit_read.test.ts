import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./helpers/assert-error-envelope";
import { buildDollarFailure } from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("platform_rate_limit_read wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetCapturedToolDefinition();
    setDollarText("ok");
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("assembles rate-limit command with optional flags", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    await execute({ command: "rate-limit", output_format: "json", prefer_scope: "upstream" });
    expect(getInvocations().at(-1)?.args.join(" ")).toContain(
      "uv run --active adw platform rate-limit --format json --prefer-scope upstream",
    );
  });

  it("omits blank optional values", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    await execute({ command: "rate-limit" });
    expect(getInvocations().at(-1)?.args.join(" ")).toBe("uv run --active adw platform rate-limit");
  });

  it("validates output_format and prefer_scope", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");

    const badFormat = await execute({ command: "rate-limit", output_format: "yaml" });
    assertContains(String(badFormat), "'output_format' must be either 'text' or 'json'");

    const badScope = await execute({ command: "rate-limit", prefer_scope: " " });
    assertContains(String(badScope), "'prefer_scope' must be either 'fork' or 'upstream'");
  });

  it("help mode bypasses validation", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    setDollarText("usage");
    const result = await execute({ command: "rate-limit", help: true });
    expect(String(result)).toContain("usage");
  });

  it("failure envelope prefers stderr then stdout", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    setDollarError(buildDollarFailure({ stderr: "fatal stderr", stdout: "shadow stdout" }));
    const result = await execute({ command: "rate-limit" });
    const text = String(result);

    assertErrorPrefix(text, "ERROR:");
    assertContains(text, "fatal stderr");
    expect(text.indexOf("fatal stderr")).toBeLessThan(text.indexOf("shadow stdout"));
  });

  it("returns sanitized structured stdout for json failures", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    setDollarError(
      buildDollarFailure({
        stdout: '{"ok":false,"token":"ghp_secretsecretsecret"}',
        stderr: "ignored",
      }),
    );
    const result = await execute({ command: "rate-limit", output_format: "json" });
    expect(JSON.parse(String(result))).toEqual({ ok: false, token: "[REDACTED]" });
  });

  it("redacts quoted json-style secrets in failure diagnostics", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    setDollarError(buildDollarFailure({ stderr: '{"password":"abc123"}' }));
    const result = await execute({ command: "rate-limit" });
    const text = String(result);

    assertContains(text, '"password":"[REDACTED]"');
    expect(text).not.toContain("abc123");
  });

  it("falls back to error envelope when json failure stdout is not structured json", async () => {
    const execute = await loadToolExecute("../../platform_rate_limit_read.ts");
    setDollarError(buildDollarFailure({ stdout: "not-json", stderr: "fatal stderr" }));
    const result = await execute({ command: "rate-limit", output_format: "json" });
    const text = String(result);
    assertContains(text, "Failed to execute 'adw platform rate-limit'");
    assertContains(text, "fatal stderr");
  });
});
