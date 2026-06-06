import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import fs from "node:fs";

import {
  adwIdValidationMessage,
  normalizeAdwId,
  normalizeOptionalString,
  runAdwSpecCommand,
  sanitizeSnippet,
  selectDiagnostic,
  validateAndNormalizeAdwId,
  validateCanonicalInRepoPath,
} from "../adw_spec_shared";
import {
  getInvocations,
  installSubprocessMocks,
  restoreSubprocessMocks,
  setSpawnError,
  setSpawnResponse,
} from "./helpers/mock-subprocess";

describe("adw_spec_shared", () => {
  beforeEach(() => {
    installSubprocessMocks();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });
  });

  afterEach(() => {
    restoreSubprocessMocks();
  });

  it("normalizes optional strings and adw ids", () => {
    expect(normalizeOptionalString(" spec_content ")).toBe("spec_content");
    expect(normalizeOptionalString("   ")).toBeUndefined();
    expect(normalizeOptionalString(42)).toBeUndefined();

    expect(normalizeAdwId("A1B2C3D4")).toBe("a1b2c3d4");
    expect(normalizeAdwId("bad")).toBeNull();
    expect(adwIdValidationMessage()).toContain("8-character hex string");
  });

  it("validates and normalizes adw_id with deterministic required and malformed errors", () => {
    expect(validateAndNormalizeAdwId("read", "A1B2C3D4")).toEqual({
      ok: true,
      adwId: "a1b2c3d4",
    });

    const missing = validateAndNormalizeAdwId("read", "   ");
    expect(missing.ok).toBe(false);
    expect(missing).toEqual(
      expect.objectContaining({ error: expect.stringContaining("'adw_id' parameter is required") }),
    );

    const malformed = validateAndNormalizeAdwId("read", "bad");
    expect(malformed).toEqual({
      ok: false,
      error: `ERROR: ${adwIdValidationMessage()}`,
    });
  });

  it("sanitizes diagnostics by stripping control characters, redacting paths, and truncating", () => {
    const sanitized = sanitizeSnippet(`boom\u0000 /tmp/private.txt\n${"x".repeat(1400)}`);

    expect(sanitized.hasVisibleContent).toBe(true);
    expect(sanitized.text).toContain("<path>");
    expect(sanitized.text).not.toContain("/tmp/private.txt");
    expect(sanitized.text).toContain("...");
  });

  it("redacts token-like secrets while preserving diagnostic keys", () => {
    const sanitized = sanitizeSnippet(
      'Authorization: Bearer ghp_superSecret123 token="abc123" api_key=xyz789',
    );

    expect(sanitized.text).toContain("Authorization: Bearer <redacted-secret>");
    expect(sanitized.text).toContain('token="<redacted-secret>"');
    expect(sanitized.text).toContain("api_key=<redacted-secret>");
    expect(sanitized.text).not.toContain("ghp_superSecret123");
    expect(sanitized.text).not.toContain("abc123");
    expect(sanitized.text).not.toContain("xyz789");
  });

  it("selects stderr then stdout then message, ignoring whitespace-only higher-priority content", () => {
    expect(
      selectDiagnostic(
        { stderr: "stderr first", stdout: "stdout second", message: "message third" },
        "fallback",
      ),
    ).toBe("stderr first");

    expect(
      selectDiagnostic({ stderr: "   ", stdout: "stdout second", message: "message third" }, "fallback"),
    ).toBe("stdout second");

    expect(selectDiagnostic({ stderr: "", stdout: "", message: "" }, "fallback")).toBe("fallback");
  });

  it("validates repository-confined canonical file paths", () => {
    const fixturePath = `${process.cwd()}/../../README.md`;
    const inside = validateCanonicalInRepoPath(fixturePath);
    expect(inside).toEqual({ ok: true, canonicalPath: fs.realpathSync(fixturePath) });

    expect(validateCanonicalInRepoPath("   ")).toEqual({
      ok: false,
      error: "ERROR: '--file' path must be non-empty.",
    });
    expect(validateCanonicalInRepoPath("does-not-exist.md")).toEqual({
      ok: false,
      error: "ERROR: '--file' path does not exist.",
    });

    const outside = validateCanonicalInRepoPath("/etc/hosts");
    expect(outside).toEqual({
      ok: false,
      error: "ERROR: '--file' path resolves outside repository root.",
    });
  });

  it("returns success stdout and deterministic failure envelopes from spawned commands", () => {
    expect(runAdwSpecCommand("read", ["uv", "run", "adw", "spec", "read"], {})).toEqual({
      ok: true,
      stdout: "ok",
    });
    expect(getInvocations()).toHaveLength(1);

    setSpawnResponse({ stderr: "fatal /tmp/private.txt", stdout: "shadow stdout", exitCode: 2 });
    const failure = runAdwSpecCommand("list", ["uv", "run", "adw", "spec", "list"], {});
    expect(failure.ok).toBe(false);
    expect(failure).toEqual(
      expect.objectContaining({ error: expect.stringContaining("ERROR: adw spec list failed (exit 2)") }),
    );
    expect((failure as { error: string }).error).toContain("<path>");
    expect((failure as { error: string }).error).not.toContain("shadow stdout");
  });

  it("uses catch-path stderr-first diagnostics and falls back to thrown message", () => {
    setSpawnError({ stderr: "stderr /tmp/private.txt", stdout: "shadow stdout", message: "ignored" });
    const stderrFailure = runAdwSpecCommand("list", ["uv", "run", "adw", "spec", "list"], {});
    expect(stderrFailure.ok).toBe(false);
    expect((stderrFailure as { error: string }).error).toContain("<path>");
    expect((stderrFailure as { error: string }).error).not.toContain("shadow stdout");

    setSpawnError({ message: "message fallback diagnostic" });
    const messageFailure = runAdwSpecCommand("list", ["uv", "run", "adw", "spec", "list"], {});
    expect(messageFailure).toEqual(
      expect.objectContaining({ error: expect.stringContaining("message fallback diagnostic") }),
    );
  });
});
