export const MAX_PYTEST_TIMEOUT_SECONDS = 1200;

export const validatePytestTimeoutSeconds = (value: unknown): string | undefined => {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return "ERROR: timeout must be a positive finite number in seconds and must not exceed 1200 seconds (20 minutes).";
  }
  if (value > MAX_PYTEST_TIMEOUT_SECONDS) {
    return "ERROR: timeout must be a positive finite number in seconds and must not exceed 1200 seconds (20 minutes).";
  }
  return undefined;
};
