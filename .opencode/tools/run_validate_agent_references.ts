import { tool } from "@opencode-ai/plugin";
import { existsSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

const VALIDATOR_SCRIPT_RELATIVE_PATH = "scripts/validate_agent_references.py";
const MISSING_SCRIPT_HINT =
  "Ensure python3 is installed and on your PATH, and scripts/validate_agent_references.py exists.";
const VALIDATOR_TRUST_HINT =
  "run_validate_agent_references only runs the committed validator script; revert local edits to scripts/validate_agent_references.py before retrying.";

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

function validateCwdWithinRepo(cwd: string | undefined, repoRoot: string): string | undefined {
  if (cwd === undefined) {
    return undefined;
  }

  try {
    if (!existsSync(cwd)) {
      return `ERROR: cwd path does not exist: ${cwd}`;
    }
    if (!isStatDirectory(statSync(cwd))) {
      return `ERROR: cwd path is not a directory: ${cwd}`;
    }

    const resolvedCwd = realpathSync(cwd);
    if (resolvedCwd !== repoRoot) {
      const rel = path.relative(repoRoot, resolvedCwd);
      if (rel.startsWith("..") || path.isAbsolute(rel)) {
        return `ERROR: cwd path resolves outside repository root: ${cwd} (canonical: ${resolvedCwd})`;
      }
      return `ERROR: cwd must resolve to the current repository/worktree root: ${cwd} (canonical: ${resolvedCwd}, expected: ${repoRoot})`;
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return `ERROR: invalid cwd path: ${cwd} (${message})`;
  }

  return undefined;
}

function validateTrustedValidatorScript(repoRoot: string, scriptPath: string): string | undefined {
  if (!existsSync(scriptPath)) {
    return undefined;
  }

  const gitResult = Bun.spawnSync({
    cmd: ["git", "status", "--porcelain=v1", "--", VALIDATOR_SCRIPT_RELATIVE_PATH],
    cwd: repoRoot,
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = gitResult.stdout.toString();
  const stderr = gitResult.stderr.toString();

  if ((gitResult.exitCode ?? 0) !== 0) {
    const detail = stderr.trim() || stdout.trim() || "Unknown git status failure.";
    return `ERROR: Failed to verify validator script trust state\n\n${detail}`;
  }

  if (stdout.trim()) {
    return `ERROR: ${VALIDATOR_TRUST_HINT}`;
  }

  return undefined;
}

function buildMissingScriptHint(stderr: string, message: string): string {
  const messageLower = message.toLowerCase();
  const stderrLower = stderr.toLowerCase();
  const missingPythonRuntime =
    messageLower.includes("spawn python3") && messageLower.includes("enoent");
  const missingValidatorScript =
    stderrLower.includes("validate_agent_references.py") &&
    stderrLower.includes("can't open file") &&
    stderrLower.includes("no such file or directory");

  if (missingPythonRuntime || missingValidatorScript) {
    return `\n${MISSING_SCRIPT_HINT}`;
  }
  return "";
}

export default tool({
  description: `Run repository agent-reference validation through a fixed, validation-safe wrapper.

EXAMPLES:
- Validate current repository root: run_validate_agent_references({})
- Validate a specific worktree root: run_validate_agent_references({ cwd: '/path/to/worktree' })

IMPORTANT:
- This wrapper only runs scripts/validate_agent_references.py via python3.
- It does not allow arbitrary script paths or shell arguments.
- Optional cwd must resolve to the current repository/worktree root exactly.
- The wrapper refuses to run if scripts/validate_agent_references.py has local modifications.`,
  args: {
    cwd: tool.schema
      .string()
      .optional()
      .describe("Repository/worktree root to validate. Must resolve to the current repository/worktree root exactly."),
  },
  async execute(args) {
    const repoRoot = realpathSync(process.cwd());
    const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
    if (typeof args.cwd === "string" && !cwd) {
      return "ERROR: cwd must not be blank when provided.";
    }

    const cwdError = validateCwdWithinRepo(cwd, repoRoot);
    if (cwdError) {
      return cwdError;
    }

    const validationRoot = cwd ? realpathSync(cwd) : repoRoot;
    const scriptPath = path.join(repoRoot, VALIDATOR_SCRIPT_RELATIVE_PATH);
    const trustError = validateTrustedValidatorScript(repoRoot, scriptPath);
    if (trustError) {
      return trustError;
    }

    const cmdParts: string[] = ["python3", scriptPath, `--root=${validationRoot}`];

    try {
      const result = await Bun.$`${cmdParts}`.text();
      return result || "Agent reference validation completed but returned no output.";
    } catch (error: any) {
      const stdout = error?.stdout?.toString?.() || "";
      const stderr = error?.stderr?.toString?.() || "";
      const message = error?.message || "Unknown error";

      if (stdout.trim()) {
        return stdout;
      }

      if (stderr.trim()) {
        const hint = buildMissingScriptHint(stderr, message);
        return `ERROR: Agent reference validation failed\n\n${stderr}${hint}`;
      }

      const hint = buildMissingScriptHint("", message);
      return `ERROR: Failed to run agent reference validation: ${message}${hint}`;
    }
  },
});
