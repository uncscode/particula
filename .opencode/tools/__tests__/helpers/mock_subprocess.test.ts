import { afterEach, describe, expect, it } from "bun:test";

import {
  getInvocations,
  installSubprocessMocks,
  resetSubprocessMocks,
  restoreSubprocessMocks,
  setDollarError,
  setDollarText,
  setSpawnResponse,
} from "./mock-subprocess";

describe("mock-subprocess", () => {
  afterEach(() => {
    restoreSubprocessMocks();
  });

  it("resets captured calls and restores Bun hooks", async () => {
    installSubprocessMocks();
    setDollarText("ok");

    await Bun.$`${["uv", "run", "adw", "status"]}`.text();
    expect(getInvocations().length).toBe(1);

    resetSubprocessMocks();
    expect(getInvocations().length).toBe(0);
  });

  it("captures spawnSync invocations and returns configured output", () => {
    installSubprocessMocks();
    setSpawnResponse({ stdout: "spawn-ok", stderr: "", exitCode: 0 });

    const result = Bun.spawnSync(["uv", "run", "adw", "status"]);
    expect(result.exitCode).toBe(0);
    expect(Buffer.from(result.stdout).toString()).toBe("spawn-ok");
    expect(getInvocations().at(-1)).toEqual({
      kind: "spawnSync",
      args: ["uv", "run", "adw", "status"],
    });
  });

  it("propagates configured $ error payload through .text() rejection", async () => {
    installSubprocessMocks();
    setDollarError({ stderr: "stderr message", stdout: "stdout shadow", message: "boom" });

    await expect(Bun.$`uv run adw status`.text()).rejects.toThrow("boom");
    expect(getInvocations().at(-1)?.kind).toBe("$");
  });
});
