import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { assertErrorPrefix } from "./helpers/assert-error-envelope";
import { loadToolExecute, resetCapturedToolDefinition } from "./helpers/tool_harness";

describe("get_datetime wrapper", () => {
  beforeEach(() => {
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    resetCapturedToolDefinition();
  });

  it("returns YYYY-MM-DD for date format", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "date" });
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("supports datetime and human formats with stable shape", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const datetimeResult = await execute({ format: "datetime" });
    const humanResult = await execute({ format: "human" });

    expect(datetimeResult).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    expect(typeof humanResult).toBe("string");
    expect(humanResult.length).toBeGreaterThan(10);
  });

  it("returns local datetime with explicit UTC offset when localtime=true", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "datetime", localtime: true });

    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$/);
  });

  it("supports local date formatting path", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "date", localtime: true });
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("defaults to date format and UTC-compatible shape when args omitted", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({});
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });

  it("returns deterministic error for invalid format", async () => {
    const execute = await loadToolExecute("../../get_datetime.ts");
    const result = await execute({ format: "bad-format" });
    assertErrorPrefix(String(result), "ERROR:");
    expect(result).toContain("Invalid format");
  });
});
