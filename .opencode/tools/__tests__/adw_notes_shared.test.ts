import { describe, expect, it } from "bun:test";

import {
  normalizeAdwId,
  normalizeRef,
  parseFieldEntries,
  parseShowOutput,
  sanitizeSnippet,
} from "../adw_notes_shared";

describe("adw_notes_shared helpers", () => {
  it("filters nullish and blank-key field entries while preserving order", () => {
    expect(
      parseFieldEntries([[" first ", "one"], null, { key: "second", value: "two" }] as any),
    ).toEqual({
      ok: true,
      entries: [
        ["first", "one"],
        ["second", "two"],
      ],
    });
  });

  it("returns structured diagnostics for malformed field entries", () => {
    expect(parseFieldEntries([["ok", 1]] as any)).toEqual({
      ok: false,
      diagnostic: "invalid fields entry at index 0: value has wrong type number",
    });
  });

  it("accepts plain-object and JSON-string field payloads", () => {
    expect(parseFieldEntries({ first: "one", second: "two" } as any)).toEqual({
      ok: true,
      entries: [
        ["first", "one"],
        ["second", "two"],
      ],
    });

    expect(parseFieldEntries('[{"key":"first","value":"one"},["second","two"]]')).toEqual({
      ok: true,
      entries: [
        ["first", "one"],
        ["second", "two"],
      ],
    });
  });

  it("fails closed for malformed non-null tuple/object entries", () => {
    expect(parseFieldEntries([["ok", "one", "extra"]] as any)).toEqual({
      ok: false,
      diagnostic: "invalid fields entry at index 0: tuple must contain exactly [key, value]",
    });
    expect(parseFieldEntries([{ key: "ok" }] as any)).toEqual({
      ok: false,
      diagnostic: "invalid fields entry at index 0: value is missing",
    });
    expect(parseFieldEntries([{ key: "ok", value: null }] as any)).toEqual({
      ok: false,
      diagnostic: "invalid fields entry at index 0: value is null",
    });
  });

  it("produces deterministic parse failure output for empty show stdout", () => {
    expect(parseShowOutput("")).toContain("<empty stdout>");
  });

  it("normalizes valid identifiers and refs", () => {
    expect(normalizeAdwId(" A1B2C3D4 ")).toBe("a1b2c3d4");
    expect(normalizeAdwId("bad-id")).toBeNull();
    expect(normalizeRef(" HEAD ")).toBe("HEAD");
    expect(normalizeRef("   ")).toBeNull();
  });

  it("sanitizes and truncates snippets", () => {
    const snippet = sanitizeSnippet(`a\u0000b ${"x".repeat(500)}`);
    expect(snippet).toContain("a b");
    expect(snippet).toContain("...(truncated)");
  });

  it("redacts secret-like values and absolute paths in snippets", () => {
    const snippet = sanitizeSnippet(
      'token=ghp_supersecret12345678 failed at /home/kyle/private.txt authorization: Bearer abc123',
    );
    expect(snippet).toContain("token=<redacted-secret>");
    expect(snippet).toContain("authorization: Bearer <redacted-secret>");
    expect(snippet).toContain("<path>");
    expect(snippet).not.toContain("ghp_supersecret12345678");
    expect(snippet).not.toContain("/home/kyle/private.txt");
  });
});
