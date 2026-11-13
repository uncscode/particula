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

it("rejects external marker directories and symlink escapes", async () => {
  const { mkdtemp, mkdir, writeFile, symlink, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const path = await import("node:path");
  const external = await mkdtemp(path.join(tmpdir(), "docs-external-"));
  const link = path.join(process.cwd(), `.docs-escape-${Date.now()}`);
  try {
    await mkdir(path.join(external, "website"));
    await mkdir(path.join(external, "scripts"));
    await writeFile(path.join(external, "website/astro.config.mjs"), "");
    await writeFile(path.join(external, "scripts/check_docs_structure.py"), "");
    await symlink(external, link);
    spawn = spyOn(Bun, "spawn");
    const execute = await loadToolExecute("../../build_docs.ts");
    for (const cwd of [external, link]) {
      expect(await execute({ cwd })).toContain("FAIL:");
    }
    expect(spawn).not.toHaveBeenCalled();
  } finally {
    await rm(link, { force: true });
    await rm(external, { recursive: true, force: true });
  }
});
