import { afterEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
  setSpawnResponse,
} from "./helpers/mock-subprocess";

describe("mock-subprocess helper", () => {
  afterEach(() => {
    restoreSubprocessMocks();
  });

  it("captures Bun.$ invocations and returns configured text", async () => {
    installSubprocessMocks();
    setDollarText("hello from dollar");

    const result = await Bun.$`echo ${"hello"}`.text();

    expect(result).toBe("hello from dollar");
    expect(getInvocations()).toEqual([{ kind: "$", args: ["echo", "hello"] }]);
  });

  it("rejects Bun.$ text with configured error diagnostics", async () => {
    installSubprocessMocks();
    setDollarError({ message: "mock subprocess failure", stderr: "stderr output" });

    const textPromise = Bun.$`echo ${"boom"}`.text();

    await expect(textPromise).rejects.toMatchObject({
      message: "mock subprocess failure",
      stderr: Buffer.from("stderr output"),
    });
    expect(getInvocations()).toEqual([{ kind: "$", args: ["echo", "boom"] }]);
  });

  it("resets configured state and captured invocations while keeping mocks installed", () => {
    installSubprocessMocks();
    setSpawnResponse({ stdout: "ok", exitCode: 0 });

    Bun.spawnSync(["uv", "run", "adw"]);
    expect(getInvocations()).toHaveLength(1);

    resetSubprocessMocks();

    expect(getInvocations()).toHaveLength(0);
    Bun.spawnSync(["uv", "run", "adw", "plans"]);
    expect(getInvocations()).toEqual([{ kind: "spawnSync", args: ["uv", "run", "adw", "plans"] }]);
  });
});
