import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import path from "node:path";

import {
  buildAstgrepCommand,
  classifyAstgrepFailure,
  normalizeAstgrepArgs,
  openAstgrepApplyTarget,
  resolveAstgrepTarget,
  runAstgrepCommand,
  sanitizeDiagnostic,
  selectDiagnostic,
} from "../lib/refactor_astgrep_shared";
import {
  getKillCount,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnResponse,
} from "./helpers/mock-subprocess";

describe("refactor_astgrep shared helpers", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
  });

  afterEach(() => {
    restoreSubprocessMocks();
  });

  it("normalizes required arguments and rejects missing or unsupported values", () => {
    expect(normalizeAstgrepArgs({ pattern: " old() ", rewrite: " new() ", lang: "python" })).toEqual({
      pattern: "old()",
      rewrite: "new()",
      lang: "python",
      path: ".",
    });
    expect(normalizeAstgrepArgs({ rewrite: "new()", lang: "python" })).toContain("pattern is required");
    expect(normalizeAstgrepArgs({ pattern: "old()", rewrite: "new()", lang: "elixir" })).toContain("lang must be one of");
  });

  it("builds a delimiter-protected command and only adds update-all for apply", () => {
    const args = { pattern: "old()", rewrite: "new()", lang: "python" as const, path: "." };
    expect(buildAstgrepCommand(args, "/repo/target")).toEqual([
      "ast-grep", "run", "-p", "old()", "-r", "new()", "-l", "python", "--", "/repo/target",
    ]);
    expect(buildAstgrepCommand(args, "/repo/target", true)).toContain("--update-all");
  });

  it("redacts diagnostics, preserves precedence, and classifies failures", () => {
    expect(sanitizeDiagnostic("api_key=supersecret\n\u0000next")).toBe("api_key: [REDACTED] next");
    expect(selectDiagnostic({ stderr: "stderr", stdout: "stdout", message: "message" })).toBe("stderr");
    expect(
      selectDiagnostic({
        stderr: new Uint8Array(),
        stdout: new TextEncoder().encode("Authorization: Bearer bearer-secret-value"),
      }),
    ).toBe("Authorization: Bearer [REDACTED]");
    expect(classifyAstgrepFailure({ message: "ENOENT ast-grep" })).toBe("missing_binary");
    expect(classifyAstgrepFailure({ stderr: "invalid pattern parse error" })).toBe("parse_input");
    expect(classifyAstgrepFailure({ message: "permission denied" })).toBe("execution");
  });

  it("confines targets lexically and canonically before execution", async () => {
    const root = process.cwd();
    const accepted = await resolveAstgrepTarget(".", root);
    expect(accepted).toEqual({ ok: true, target: root });

    expect(await resolveAstgrepTarget("../", root)).toEqual({
      ok: false,
      diagnostic: "Path must remain within the repository.",
    });
    expect(await resolveAstgrepTarget("does-not-exist", root)).toEqual({
      ok: false,
      diagnostic: "Path must be an existing accessible regular file or directory.",
    });
  });

  it("keeps apply targets descriptor-confined until the caller closes them", async () => {
    const target = await openAstgrepApplyTarget(".", process.cwd());

    expect(target.ok).toBe(true);
    if (!target.ok) return;
    try {
      expect(target.target).toMatch(new RegExp(`^/proc/${process.pid}/fd/\\d+$`));
      expect(await fs.promises.realpath(target.target)).toBe(process.cwd());
    } finally {
      await target.close();
    }
  });

  it("kills and reaps a timed-out ast-grep command before rejecting", async () => {
    setSpawnResponse({ hangs: true });

    await expect(runAstgrepCommand(["ast-grep", "run"], 5)).rejects.toThrow(
      "ast-grep timed out after 5ms",
    );
    expect(getKillCount()).toBe(1);
  });
});
