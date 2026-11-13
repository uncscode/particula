import { describe, expect, it } from "bun:test";

import {
  normalizeSafeRelativeSourcePath,
  parseBatchOptions,
} from "../adw_issues_spec_shared";

describe("adw_issues_spec_shared helpers", () => {
  it("rejects control-character batch options instead of coercing them", () => {
    const result = parseBatchOptions("batch-read", "raw\nread");

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toBe("ERROR: 'options' must not contain control characters.");
    }
  });

  it("rejects drive-qualified absolute source paths", () => {
    expect(normalizeSafeRelativeSourcePath("C:/tmp/spec.md")).toBeNull();
    expect(normalizeSafeRelativeSourcePath("C:\\tmp\\spec.md")).toBeNull();
  });
});
