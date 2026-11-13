import { describe, expect, it } from "bun:test";

import { buildDollarFailure, buildJsonFixture, buildSuccessOutput } from "./fixture-builders";

describe("fixture-builders helper", () => {
  it("builds deterministic dollar-failure fixtures with defaults and overrides", () => {
    expect(buildDollarFailure()).toEqual({
      stderr: "",
      stdout: "",
      message: "mock failure",
      code: undefined,
      exitCode: undefined,
    });

    expect(
      buildDollarFailure({
        stderr: "stderr",
        stdout: "stdout",
        message: "message",
        code: 2,
        exitCode: 3,
      }),
    ).toEqual({
      stderr: "stderr",
      stdout: "stdout",
      message: "message",
      code: 2,
      exitCode: 3,
    });
  });

  it("returns passthrough success output and pretty-printed json fixtures", () => {
    expect(buildSuccessOutput("ok")).toBe("ok");
    expect(buildJsonFixture({ ok: true, nested: { value: 1 } })).toBe(
      JSON.stringify({ ok: true, nested: { value: 1 } }, null, 2),
    );
  });
});
