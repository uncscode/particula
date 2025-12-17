import assert from "node:assert/strict";
import { describe, it, mock } from "node:test";

mock.module("@opencode-ai/plugin", () => ({
  tool: (definition: unknown) => definition,
}));

// Import after mocking the plugin to avoid pulling the real dependency.
const getDatetimeTool = (await import("../get_datetime")).default as {
  execute: (args: { format?: "date" | "datetime" | "human"; localtime?: boolean }) =>
    Promise<string>;
};

const RealDate = Date;

const withFixedDate = async (iso: string, fn: () => Promise<void> | void) => {
  const fixed = new RealDate(iso);

  class MockDate extends RealDate {
    constructor(...args: unknown[]) {
      if (args.length === 0) {
        // @ts-ignore
        return fixed;
      }
      // @ts-ignore
      return new RealDate(...(args as [string | number | Date]));
    }

    static now() {
      return fixed.getTime();
    }
  }

  // @ts-ignore
  globalThis.Date = MockDate as DateConstructor;

  try {
    await fn();
  } finally {
    globalThis.Date = RealDate;
  }
};

describe("get_datetime tool", () => {
  it("returns UTC date by default (no args)", async () => {
    await withFixedDate("2025-12-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({});
      assert.equal(result, "2025-12-16");
    });
  });

  it("returns UTC datetime with Z and no milliseconds", async () => {
    await withFixedDate("2025-12-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({ format: "datetime" });
      assert.equal(result, "2025-12-16T12:34:56Z");
    });
  });

  it("returns UTC human string with UTC suffix", async () => {
    await withFixedDate("2025-12-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({ format: "human" });
      assert.ok(result.includes("UTC"));
      assert.ok(result.includes("2025"));
    });
  });

  it("returns local date respecting America/Denver boundary", async () => {
    await withFixedDate("2025-12-16T01:30:00Z", async () => {
      const result = await getDatetimeTool.execute({ localtime: true });
      // 2025-12-15 18:30 local time (MST)
      assert.equal(result, "2025-12-15");
    });
  });

  it("returns local datetime with standard offset (-07:00, MST winter)", async () => {
    await withFixedDate("2025-12-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({ format: "datetime", localtime: true });
      assert.equal(result, "2025-12-16T05:34:56-07:00");
    });
  });

  it("returns local datetime with daylight offset (-06:00, MDT summer)", async () => {
    await withFixedDate("2025-06-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({ format: "datetime", localtime: true });
      assert.equal(result, "2025-06-16T06:34:56-06:00");
    });
  });

  it("returns local human string with MST abbreviation (winter)", async () => {
    await withFixedDate("2025-12-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({ format: "human", localtime: true });
      assert.ok(result.includes("MST"));
    });
  });

  it("returns local human string with MDT abbreviation (summer)", async () => {
    await withFixedDate("2025-06-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({ format: "human", localtime: true });
      assert.ok(result.includes("MDT"));
    });
  });

  it("defaults localtime to false when omitted", async () => {
    await withFixedDate("2025-12-16T12:34:56Z", async () => {
      const result = await getDatetimeTool.execute({ format: "date" });
      assert.equal(result, "2025-12-16");
    });
  });
});
