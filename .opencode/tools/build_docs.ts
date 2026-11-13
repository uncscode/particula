import { realpath, stat } from "node:fs/promises";
import path from "node:path";
import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Build and verify the Starlight documentation, or check source structure without running simulations.",
  args: {
    cwd: tool.schema.string().optional().describe("Explicit repository/worktree root; defaults to the current workspace."),
    action: tool.schema.enum(["build", "check"]).optional().describe("Defaults to build; check validates source structure only."),
  },
  async execute({ action, cwd }) {
    let root: string;
    try {
      if (cwd !== undefined && !cwd.trim()) throw new Error("cwd must not be empty");
      root = await realpath(cwd ?? process.cwd());
      for (const file of ["website/astro.config.mjs", "scripts/check_docs_structure.py"]) {
        if (!(await stat(path.join(root, file))).isFile()) throw new Error("not a documentation worktree");
      }
    } catch {
      return "FAIL: documentation requires an existing Particula worktree root";
    }
    const command = action === "check"
      ? ["node", "website/scripts/python.mjs", "scripts/check_docs_structure.py"]
      : ["npm", "--prefix", "website", "run", "build"];
    const child = Bun.spawn(command, { cwd: root, stdout: "pipe", stderr: "pipe" });
    const [stdout, stderr, code] = await Promise.all([
      new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited,
    ]);
    return `${code === 0 ? "PASS" : "FAIL"}: documentation ${action ?? "build"}\n${stdout}\n${stderr}`;
  },
});
