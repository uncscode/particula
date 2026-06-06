export const buildDollarFailure = (overrides?: {
  stderr?: string;
  stdout?: string;
  message?: string;
}): { stderr: string; stdout: string; message: string } => ({
  stderr: overrides?.stderr ?? "",
  stdout: overrides?.stdout ?? "",
  message: overrides?.message ?? "mock failure",
});

export const buildSuccessOutput = (value: string): string => value;
