export const buildDollarFailure = (overrides?: {
  stderr?: string;
  stdout?: string;
  message?: string;
  code?: string | number;
  exitCode?: number;
}): { stderr: string; stdout: string; message: string; code?: string | number; exitCode?: number } => ({
  stderr: overrides?.stderr ?? "",
  stdout: overrides?.stdout ?? "",
  message: overrides?.message ?? "mock failure",
  code: overrides?.code,
  exitCode: overrides?.exitCode,
});

export const buildSuccessOutput = (value: string): string => value;

export const buildJsonFixture = (value: unknown): string => JSON.stringify(value, null, 2);
