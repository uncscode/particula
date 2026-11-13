import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertErrorPrefix } from "./helpers/assert-error-envelope";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";
import { DATE_TIME_FORMATS } from "../get_datetime";

describe("get_datetime wrapper", () => {
  beforeEach(() => {
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    resetCapturedToolDefinition();
  });

  it("returns_date_format_by_default_when_args_omitted", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({});
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("returns_utc_datetime_without_milliseconds", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "datetime" });

    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(result).not.toContain(".");
  });

  it("returns_local_datetime_with_explicit_offset_when_localtime_true", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "datetime", localtime: true });

    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$/);
  });

  it("returns_human_format_string", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "human" });

    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(10);
    expect(result.startsWith("ERROR:")).toBe(false);
  });

  it("supports_local_date_formatting_path", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "date", localtime: true });
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("rejects_invalid_format_deterministically", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "bad-format" });
    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("Invalid format");
  });

  it("rejects_non_boolean_localtime_deterministically", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ localtime: "true" });

    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("'localtime' must be a boolean");
  });

  it("accepts_all_canonical_format_enum_values", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");

    expect(DATE_TIME_FORMATS).toEqual(["date", "datetime", "human"]);

    for (const format of DATE_TIME_FORMATS) {
      const result = await execute({ format });
      expect(typeof result).toBe("string");
      expect(result.startsWith("ERROR:")).toBe(false);
    }
  });
});
