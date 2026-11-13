import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  buildDollarFailure,
  buildSuccessOutput,
} from "./helpers/fixture-builders";
import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
} from "./helpers/mock-subprocess";
import {
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";

describe("get_version wrapper", () => {
  beforeEach(() => {
    installSubprocessMocks();
    resetSubprocessMocks();
    resetCapturedToolDefinition();
  });

  afterEach(() => {
    restoreSubprocessMocks();
    resetCapturedToolDefinition();
  });

  it("ships_real_backend_script_at_expected_path", () => {
    const scriptPath = fileURLToPath(new URL("../get_version.py", import.meta.url));

    expect(existsSync(scriptPath)).toBe(true);
  });

  it("uses_default_lookup_when_file_omitted", async () => {
    setDollarText(buildSuccessOutput("1.2.3"));
    const execute = await loadToolExecute("../../get_version.ts");

    const result = await execute({});
    expect(result).toBe("1.2.3");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("get_version.py");
    expect(cmd).not.toContain("package.json");
    expect(cmd).not.toContain("pyproject.toml");
  });

  it("forwards_file_when_meaningful", async () => {
    setDollarText(buildSuccessOutput("1.2.3"));
    const execute = await loadToolExecute("../../get_version.ts");

    const result = await execute({ file: "package.json" });
    expect(result).toBe("1.2.3");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("python3");
    expect(cmd).toContain("get_version.py");
    expect(cmd).toContain("package.json");
  });

  it("trims_meaningful_file_values_before_execution", async () => {
    setDollarText(buildSuccessOutput("1.2.3"));
    const execute = await loadToolExecute("../../get_version.ts");

    const result = await execute({ file: "  package.json  " });
    expect(result).toBe("1.2.3");

    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).toContain("package.json");
    expect(cmd).not.toContain("  package.json  ");
  });

  it("omits_blank_file_values_from_argv", async () => {
    setDollarText(buildSuccessOutput("0.0.1"));
    const execute = await loadToolExecute("../../get_version.ts");

    await execute({ file: "   " });
    const cmd = getInvocations().at(-1)?.args.join(" ") ?? "";
    expect(cmd).not.toContain("package.json");
    expect(cmd).not.toContain("pyproject.toml");
  });

  it("rejects_malformed_file_input_deterministically", async () => {
    const execute = await loadToolExecute("../../get_version.ts");

    expect(await execute({ file: { path: "package.json" } })).toBe(
      "ERROR: 'file' must be a string when provided.",
    );
    expect(getInvocations()).toHaveLength(0);
  });

  it("prefers_stdout_over_stderr_and_message", async () => {
    const execute = await loadToolExecute("../../get_version.ts");
    setDollarError(
      buildDollarFailure({
        stdout: "stdout failure",
        stderr: "stderr shadow",
        message: "message shadow",
      }),
    );

    expect(await execute({})).toBe("stdout failure");
  });

  it("falls_back_to_stderr_when_stdout_empty", async () => {
    const execute = await loadToolExecute("../../get_version.ts");

    setDollarError(buildDollarFailure({ stdout: "", stderr: "stderr failure" }));
    expect(await execute({})).toBe("ERROR: get_version failed\n\nstderr failure");
  });

  it("falls_back_to_message_when_stdout_and_stderr_empty", async () => {
    const execute = await loadToolExecute("../../get_version.ts");

    setDollarError(
      buildDollarFailure({ stdout: "", stderr: "", message: "Unknown crash" }),
    );
    expect(await execute({})).toBe("ERROR: get_version failed\n\nUnknown crash");
  });

  it("includes_missing_python3_hint", async () => {
    const execute = await loadToolExecute("../../get_version.ts");

    setDollarError(
      buildDollarFailure({ stdout: "", stderr: "python3: not found", code: "ENOENT" }),
    );
    expect(await execute({})).toContain(
      "Hint: Ensure python3 is installed and available on PATH.",
    );
  });

  it("includes_missing_backend_script_hint", async () => {
    const execute = await loadToolExecute("../../get_version.ts");

    setDollarError(
      buildDollarFailure({ stdout: "", stderr: "can't open file /x/get_version.py" }),
    );
    expect(await execute({})).toContain("Hint: Backend script not found");
  });

  it("falls_back_to_neutral_enoent_hint_for_ambiguous_message_only_failures", async () => {
    const execute = await loadToolExecute("../../get_version.ts");

    setDollarError(
      buildDollarFailure({
        stdout: "",
        stderr: "",
        message: "No such file or directory",
      }),
    );
    expect(await execute({})).toContain(
      "Hint: Execution failed with ENOENT; check python3 availability and backend script path.",
    );
  });

  it("returns_neutral_enoent_hint_when_source_is_ambiguous", async () => {
    const execute = await loadToolExecute("../../get_version.ts");

    setDollarError(
      buildDollarFailure({ stdout: "", stderr: "ENOENT", message: "No such file", exitCode: 1 }),
    );
    expect(await execute({})).toContain(
      "Hint: Execution failed with ENOENT; check python3 availability and backend script path.",
    );
  });
});
