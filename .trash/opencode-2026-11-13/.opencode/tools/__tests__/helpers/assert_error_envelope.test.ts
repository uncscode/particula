import { describe, expect, it } from "bun:test";

import { assertContains, assertErrorPrefix } from "./assert-error-envelope";
import { buildDollarFailure, buildSuccessOutput } from "./fixture-builders";

describe("assert-error-envelope helpers", () => {
  it("accepts matching error prefix and content", () => {
    const value = "ERROR: deterministic failure";
    expect(() => assertErrorPrefix(value, "ERROR:")).not.toThrow();
    expect(() => assertContains(value, "deterministic")).not.toThrow();
  });

  it("throws deterministic message for mismatched prefix", () => {
    expect(() => assertErrorPrefix("INFO: ok", "ERROR:")).toThrow(
      "Expected error prefix 'ERROR:'",
    );
  });

  it("throws deterministic message when snippet missing", () => {
    expect(() => assertContains("alpha", "beta")).toThrow(
      "Expected output to contain 'beta' but got 'alpha'",
    );
  });
});

describe("fixture-builders", () => {
  it("builds default failure payload", () => {
    expect(buildDollarFailure()).toEqual({
      stderr: "",
      stdout: "",
      message: "mock failure",
    });
  });

  it("applies override payload values", () => {
    expect(buildDollarFailure({ stderr: "s", stdout: "o", message: "m" })).toEqual({
      stderr: "s",
      stdout: "o",
      message: "m",
    });
  });

  it("returns success output unchanged", () => {
    expect(buildSuccessOutput("ok")).toBe("ok");
  });
});
