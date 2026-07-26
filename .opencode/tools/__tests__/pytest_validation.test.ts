import { describe, expect, it } from "bun:test";

import {
  MAX_PYTEST_TIMEOUT_SECONDS,
  validatePytestTimeoutSeconds,
} from "../lib/pytest_validation";

const TIMEOUT_ERROR =
  "ERROR: timeout must be a positive finite number in seconds and must not exceed 1200 seconds (20 minutes).";

describe("pytest_validation helper", () => {
  it("exports the shared timeout cap constant", () => {
    expect(MAX_PYTEST_TIMEOUT_SECONDS).toBe(1200);
  });

  it("accepts undefined and the inclusive 1200-second boundary", () => {
    expect(validatePytestTimeoutSeconds(undefined)).toBeUndefined();
    expect(validatePytestTimeoutSeconds(MAX_PYTEST_TIMEOUT_SECONDS)).toBeUndefined();
  });

  it("rejects invalid timeout values with the shared deterministic message", () => {
    expect(validatePytestTimeoutSeconds(0)).toBe(TIMEOUT_ERROR);
    expect(validatePytestTimeoutSeconds(Number.NaN)).toBe(TIMEOUT_ERROR);
    expect(validatePytestTimeoutSeconds(Infinity)).toBe(TIMEOUT_ERROR);
    expect(validatePytestTimeoutSeconds(1200.001)).toBe(TIMEOUT_ERROR);
  });
});
