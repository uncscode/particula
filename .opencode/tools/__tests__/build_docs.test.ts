import { afterEach, expect, it, spyOn } from "bun:test";
import { loadToolExecute } from "./helpers/tool_harness";

let spawn: ReturnType<typeof spyOn> | undefined;
afterEach(() => spawn?.mockRestore());

for (const [action, command, code] of [
  [undefined, ["npm", "--prefix", "website", "run", "build"], 0],
  ["check", ["node", "website/scripts/python.mjs", "scripts/check_docs_structure.py"], 1],
] as const) {
  it(`routes ${action ?? "default build"} and reports the child exit status`, async () => {
    spawn = spyOn(Bun, "spawn").mockImplementation(() => ({
      stdout: new Blob(["report"]).stream(),
      stderr: new Blob(["diagnostic"]).stream(),
      exited: Promise.resolve(code),
    }) as any);
    const execute = await loadToolExecute("../../build_docs.ts");
    const result = await execute({ action });
    expect(spawn).toHaveBeenCalledWith([...command], {
      cwd: process.cwd(), stdout: "pipe", stderr: "pipe",
    });
    expect(result).toContain(code === 0 ? "PASS:" : "FAIL:");
    expect(result).toContain("report");
    expect(result).toContain("diagnostic");
  });
}

it("rejects an empty explicit worktree without starting a build", async () => {
  spawn = spyOn(Bun, "spawn");
  const execute = await loadToolExecute("../../build_docs.ts");
  expect(await execute({ cwd: " " })).toContain("FAIL:");
  expect(spawn).not.toHaveBeenCalled();
});
