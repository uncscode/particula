import { describe, expect, it } from "bun:test";

import {
  normalizeAdwId,
  normalizeRef,
  parseFieldEntries,
  parseShowOutput,
  sanitizeSnippet,
} from "../adw_notes_shared";

describe("adw_notes_shared helpers", () => {
  it("accepts ordered duplicate entries and nullable allowlisted values", () => {
    expect(
      parseFieldEntries([
        ["plan_summary", "# First\n\nMarkdown"],
        { key: "architecture_notes", value: null },
        ["plan_summary", "second"],
        { key: "review_findings", value: null },
      ] as any),
    ).toEqual({
      ok: true,
      entries: [
        ["plan_summary", "# First\n\nMarkdown"],
        ["architecture_notes", null],
        ["plan_summary", "second"],
        ["review_findings", null],
      ],
    });
  });

  it("returns structured diagnostics for malformed field entries", () => {
    expect(parseFieldEntries([["plan_summary", 1]] as any)).toEqual({
      ok: false,
      diagnostic: "invalid fields entry at index 0: value has wrong type number",
    });
  });

  it("accepts plain-object and JSON-string field payloads", () => {
    expect(parseFieldEntries({ plan_summary: "one", discovered_context: "two" } as any)).toEqual({
      ok: true,
      entries: [
        ["plan_summary", "one"],
        ["discovered_context", "two"],
      ],
    });

    expect(parseFieldEntries('[{"key":"plan_summary","value":"one"},["review_findings",null]]')).toEqual({
      ok: true,
      entries: [
        ["plan_summary", "one"],
        ["review_findings", null],
      ],
    });
  });

  it("fails closed for malformed non-null tuple/object entries", () => {
    expect(parseFieldEntries([["plan_summary", "one", "extra"]] as any)).toEqual({
      ok: false,
      diagnostic: "invalid fields entry at index 0: tuple must contain exactly [key, value]",
    });
    expect(parseFieldEntries([{ key: "plan_summary" }] as any)).toEqual({
      ok: false,
      diagnostic: "invalid fields entry at index 0: value is missing",
    });
    expect(parseFieldEntries([{ key: "plan_summary", value: null }] as any)).toEqual({
    ok: false,
    diagnostic: 'invalid fields entry at index 0: key "plan_summary" does not allow null',
  });
  });

  it("rejects unallowlisted keys in all supported forms", () => {
    expect(parseFieldEntries([["unknown", "value"]] as any)).toEqual({
      ok: false,
      diagnostic: 'invalid fields entry at index 0: key "unknown" is not allowlisted',
    });
    expect(parseFieldEntries({ unknown: "value" } as any)).toEqual({
      ok: false,
      diagnostic: 'invalid fields object key "unknown": key "unknown" is not allowlisted',
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
