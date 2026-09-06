import { tool } from "@opencode-ai/plugin";
import { existsSync, lstatSync, readFileSync, realpathSync, statSync } from "node:fs";
import path from "node:path";

const S_IFMT = 0o170000;
const S_IFDIR = 0o040000;

const VALIDATOR_SCRIPT_RELATIVE_PATH = "scripts/validate_agent_references.py";
const BASELINE_GUIDES_ROOT_RELATIVE_PATH = ".opencode/guides";
const MISSING_SCRIPT_HINT =
  "Ensure python3 is installed and on your PATH, and scripts/validate_agent_references.py exists.";
const VALIDATOR_TRUST_HINT =
  "run_validate_agent_references only runs the committed validator script; revert local edits to scripts/validate_agent_references.py before retrying.";
const BASELINE_TRUST_HINT =
  "run_validate_agent_references only accepts baselinePath values that point to committed clean files under .opencode/guides/.";

const REPO_ROOT = realpathSync(path.resolve(import.meta.dir, "../.."));

function readGitDirPointer(dotGitPath: string): string | undefined {
  try {
    const stat = lstatSync(dotGitPath);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 4096) return undefined;
    const content = readFileSync(dotGitPath, "utf8").trim();
    const prefix = "gitdir:";
    if (!content.startsWith(prefix)) return undefined;
    const value = content.slice(prefix.length).trim();
    if (!value || value.includes("\0")) return undefined;
    return realpathSync(path.resolve(path.dirname(dotGitPath), value));
  } catch {
    return undefined;
  }
}

function resolveCommonGitDir(repoRoot: string): string | undefined {
  const dotGitPath = path.join(repoRoot, ".git");
  try {
    const stat = lstatSync(dotGitPath);
    if (stat.isDirectory() && !stat.isSymbolicLink()) return realpathSync(dotGitPath);
  } catch {
    return undefined;
  }
  const worktreeGitDir = readGitDirPointer(dotGitPath);
  if (!worktreeGitDir) return undefined;
  try {
    const commonDirValue = readFileSync(path.join(worktreeGitDir, "commondir"), "utf8").trim();
    if (!commonDirValue || commonDirValue.includes("\0")) return undefined;
    return realpathSync(path.resolve(worktreeGitDir, commonDirValue));
  } catch {
    return undefined;
  }
}

function isOwnedLinkedWorktree(resolvedCwd: string, repoRoot: string): boolean {
  const relativeCandidate = path.relative(repoRoot, resolvedCwd);
  const parts = relativeCandidate.split(path.sep);
  if (parts.length !== 2 || parts[0] !== "trees" || !parts[1]) return false;

  const candidateGitDir = readGitDirPointer(path.join(resolvedCwd, ".git"));
  const commonGitDir = resolveCommonGitDir(repoRoot);
  if (!candidateGitDir || !commonGitDir) return false;
  const worktreesRoot = path.join(commonGitDir, "worktrees");
  const relativeGitDir = path.relative(worktreesRoot, candidateGitDir);
  return Boolean(
    relativeGitDir &&
      !relativeGitDir.startsWith("..") &&
      !path.isAbsolute(relativeGitDir) &&
      !relativeGitDir.includes(path.sep),
  );
}

function isStatDirectory(s: ReturnType<typeof statSync>): boolean {
  if (typeof s.isDirectory === "function") return s.isDirectory();
  if (typeof s.isDirectory === "boolean") return s.isDirectory;
  return (((s as any).mode ?? 0) & S_IFMT) === S_IFDIR;
}

export function validateCwdWithinRepo(
  cwd: string | undefined,
  repoRoot: string,
): string | undefined {
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
    if (resolvedCwd !== repoRoot && !isOwnedLinkedWorktree(resolvedCwd, repoRoot)) {
      const rel = path.relative(repoRoot, resolvedCwd);
      if (rel.startsWith("..") || path.isAbsolute(rel)) {
        return `ERROR: cwd path resolves outside repository root: ${cwd} (canonical: ${resolvedCwd})`;
      }
      return `ERROR: cwd must resolve to the current repository root or an owned linked worktree under trees/: ${cwd} (canonical: ${resolvedCwd}, repository: ${repoRoot})`;
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

function validateBaselinePath(baselinePath: string | undefined, repoRoot: string): string | undefined {
  if (!baselinePath) {
    return undefined;
  }
  if (path.isAbsolute(baselinePath)) {
    return `ERROR: baselinePath must be repo-relative under ${BASELINE_GUIDES_ROOT_RELATIVE_PATH}: ${baselinePath}`;
  }

  const resolvedBaselinePath = path.resolve(repoRoot, baselinePath);
  const guidesRoot = path.resolve(repoRoot, BASELINE_GUIDES_ROOT_RELATIVE_PATH);
  const relativeToGuides = path.relative(guidesRoot, resolvedBaselinePath);
  if (relativeToGuides.startsWith("..") || path.isAbsolute(relativeToGuides)) {
    return `ERROR: baselinePath must resolve under ${BASELINE_GUIDES_ROOT_RELATIVE_PATH}: ${baselinePath} (canonical: ${resolvedBaselinePath})`;
  }

  return undefined;
}

function validateTrustedBaseline(repoRoot: string, baselinePath: string | undefined): string | undefined {
  if (!baselinePath) {
    return undefined;
  }

  const absoluteBaselinePath = path.resolve(repoRoot, baselinePath);
  if (!existsSync(absoluteBaselinePath)) {
    return undefined;
  }

  const gitResult = Bun.spawnSync({
    cmd: ["git", "status", "--porcelain=v1", "--", baselinePath],
    cwd: repoRoot,
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = gitResult.stdout.toString();
  const stderr = gitResult.stderr.toString();

  if ((gitResult.exitCode ?? 0) !== 0) {
    const detail = stderr.trim() || stdout.trim() || "Unknown git status failure.";
    return `ERROR: Failed to verify baselinePath trust state\n\n${detail}`;
  }

  const firstStatusLine = stdout.split(/\r?\n/).find((line) => line.trim());
  if (!firstStatusLine) {
    return undefined;
  }

  if (firstStatusLine.startsWith("??")) {
    return `ERROR: ${BASELINE_TRUST_HINT} Rejecting untracked baselinePath: ${baselinePath}`;
  }

  return `ERROR: ${BASELINE_TRUST_HINT} Rejecting baselinePath with local git status ${firstStatusLine.slice(0, 2)}: ${baselinePath}`;
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
- Validate with a committed baseline: run_validate_agent_references({ cwd: '/path/to/worktree', baselinePath: '.opencode/guides/agent-reference-validation-baseline.json' })

IMPORTANT:
- This wrapper only runs scripts/validate_agent_references.py via python3.
- It does not allow arbitrary script paths or shell arguments.
- Optional cwd must resolve to the current repository root or an owned linked worktree under trees/ exactly.
- Optional baselinePath must be repo-relative and stay under .opencode/guides/.
- Optional baselinePath must point to a committed clean file under .opencode/guides/.
- The wrapper refuses to run if scripts/validate_agent_references.py has local modifications.`,
  args: {
    cwd: tool.schema
      .string()
      .optional()
      .describe("Repository root or owned linked worktree under trees/ to validate."),
    baselinePath: tool.schema
      .string()
      .optional()
      .describe("Optional committed baseline JSON path under .opencode/guides/."),
  },
  async execute(args) {
    const cwd = typeof args.cwd === "string" ? args.cwd.trim() : undefined;
    if (typeof args.cwd === "string" && !cwd) {
      return "ERROR: cwd must not be blank when provided.";
    }
    const baselinePath = typeof args.baselinePath === "string" ? args.baselinePath.trim() || undefined : undefined;

    const cwdError = validateCwdWithinRepo(cwd, REPO_ROOT);
    if (cwdError) {
      return cwdError;
    }
    const validationRoot = cwd ? realpathSync(cwd) : REPO_ROOT;
    const baselineError = validateBaselinePath(baselinePath, validationRoot);
    if (baselineError) {
      return baselineError;
    }
    const baselineTrustError = validateTrustedBaseline(validationRoot, baselinePath);
    if (baselineTrustError) {
      return baselineTrustError;
    }

    const scriptPath = path.join(REPO_ROOT, VALIDATOR_SCRIPT_RELATIVE_PATH);
    const trustError = validateTrustedValidatorScript(REPO_ROOT, scriptPath);
    if (trustError) {
      return trustError;
    }

    const cmdParts: string[] = ["python3", scriptPath, `--root=${validationRoot}`];
    if (baselinePath) {
      cmdParts.push(`--baseline-path=${baselinePath}`);
    }

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
