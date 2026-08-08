import * as fs from "node:fs";
import * as path from "node:path";

export const ALLOWED_LANGUAGES = ["python", "typescript", "javascript", "cpp", "c", "go", "java", "rust", "csharp", "kotlin", "swift", "ruby", "php"] as const;
export type AstgrepLanguage = (typeof ALLOWED_LANGUAGES)[number];
export type AstgrepNormalizedArgs = { pattern: string; rewrite: string; lang: AstgrepLanguage; path: string };
export const MISSING_BINARY_HINT = "Install ast-grep-cli (pip install ast-grep-cli) and ensure ast-grep is on PATH.";
export const ASTGREP_TIMEOUT_MS = 30_000;
const MAX_ASTGREP_OUTPUT_BYTES = 64 * 1024;

const controls = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
function text(value: unknown): string { return value instanceof Uint8Array ? new TextDecoder().decode(value) : value == null ? "" : String(value); }
export function sanitizeDiagnostic(value: unknown, limit = 500): string {
  const cleaned = text(value)
    .replace(controls, " ")
    .replace(/\bgh[pousr]_[A-Za-z0-9]{10,}\b/g, "[REDACTED]")
    .replace(/\bAuthorization\s*:\s*Bearer\s+[^\s"']+/gi, "Authorization: Bearer [REDACTED]")
    .replace(
      /\b(token|api[_-]?key|secret|password)\s*[:=]\s*[^\s"']+/gi,
      (_match, name: string) => `${name}: [REDACTED]`,
    )
    .replace(/\s+/g, " ")
    .trim();
  return cleaned.length <= limit ? cleaned : `${cleaned.slice(0, Math.max(0, limit - 14))}... [truncated]`;
}

/** Sanitize bounded preview text without collapsing its returned line structure. */
export function sanitizePreviewOutput(value: unknown, limit = 16_000): string {
  const cleaned = text(value)
    .replace(controls, " ")
    .replace(/\bgh[pousr]_[A-Za-z0-9]{10,}\b/g, "[REDACTED]")
    .replace(/\bAuthorization\s*:\s*Bearer\s+[^\s"']+/gi, "Authorization: Bearer [REDACTED]")
    .replace(
      /\b(token|api[_-]?key|secret|password)\s*[:=]\s*[^\s"']+/gi,
      (_match, name: string) => `${name}: [REDACTED]`,
    )
    .replace(/\r\n/g, "\n");
  return cleaned.length <= limit
    ? cleaned
    : `${cleaned.slice(0, Math.max(0, limit - 14))}... [truncated]`;
}
export function selectDiagnostic(error: unknown): string {
  const candidate = error as { stderr?: unknown; stdout?: unknown; message?: unknown };
  for (const value of [candidate?.stderr, candidate?.stdout, candidate?.message]) {
    const diagnostic = sanitizeDiagnostic(value);
    if (diagnostic) return diagnostic;
  }
  return "No diagnostic details available.";
}
export function classifyAstgrepFailure(error: unknown): "missing_binary" | "parse_input" | "execution" {
  const value = `${(error as any)?.stderr ?? ""} ${(error as any)?.stdout ?? ""} ${(error as any)?.message ?? ""}`.toLowerCase();
  if (value.includes("enoent") || value.includes("not found")) return "missing_binary";
  return ["parse error", "failed to parse", "cannot parse", "invalid pattern", "invalid rewrite", "pattern parse", "rewrite parse"].some((signal) => value.includes(signal)) ? "parse_input" : "execution";
}
export function normalizeAstgrepArgs(args: Record<string, unknown>): AstgrepNormalizedArgs | string {
  const pattern = typeof args.pattern === "string" ? args.pattern.trim() : "";
  if (!pattern) return "ERROR: pattern is required. Provide the AST pattern to match.";
  const rewrite = typeof args.rewrite === "string" ? args.rewrite.trim() : "";
  if (!rewrite) return "ERROR: rewrite is required. Provide the replacement pattern.";
  const lang = typeof args.lang === "string" ? args.lang.trim() : "";
  if (!ALLOWED_LANGUAGES.includes(lang as AstgrepLanguage)) return `ERROR: lang must be one of ${ALLOWED_LANGUAGES.join(", ")}.`;
  const target = typeof args.path === "string" && args.path.trim() ? args.path.trim() : ".";
  return { pattern, rewrite, lang: lang as AstgrepLanguage, path: target };
}
function within(target: string, root: string): boolean { return target === root || target.startsWith(root + path.sep); }
/** Resolve an existing regular file/directory after lexical then canonical confinement. */
export async function resolveAstgrepTarget(input: string, root = process.cwd()): Promise<{ ok: true; target: string } | { ok: false; diagnostic: string }> {
  const rootAbsolute = path.resolve(root);
  const lexical = path.resolve(rootAbsolute, input);
  if (!within(lexical, rootAbsolute)) return { ok: false, diagnostic: "Path must remain within the repository." };
  try {
    const stat = await fs.promises.stat(lexical);
    if (!stat.isFile() && !stat.isDirectory()) return { ok: false, diagnostic: "Path must be an existing regular file or directory." };
    const [target, canonicalRoot] = await Promise.all([fs.promises.realpath(lexical), fs.promises.realpath(rootAbsolute)]);
    if (!within(target, canonicalRoot)) return { ok: false, diagnostic: "Path must remain within the repository." };
    return { ok: true, target };
  } catch { return { ok: false, diagnostic: "Path must be an existing accessible regular file or directory." }; }
}

/**
 * Open an apply target without following a terminal symlink and expose the
 * held descriptor through procfs. Keeping the descriptor open pins the target
 * across validation and the child invocation on supported Linux hosts.
 */
export async function openAstgrepApplyTarget(
  input: string,
  root = process.cwd(),
): Promise<{ ok: true; target: string; close: () => Promise<void> } | { ok: false; diagnostic: string }> {
  const resolved = await resolveAstgrepTarget(input, root);
  if (!resolved.ok) return resolved;
  if (process.platform !== "linux") {
    return { ok: false, diagnostic: "Safe apply target handling is unavailable on this platform." };
  }
  try {
    const handle = await fs.promises.open(resolved.target, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
    const descriptorPath = `/proc/${process.pid}/fd/${handle.fd}`;
    const [pinnedTarget, canonicalRoot] = await Promise.all([
      fs.promises.realpath(descriptorPath),
      fs.promises.realpath(path.resolve(root)),
    ]);
    if (!within(pinnedTarget, canonicalRoot)) {
      await handle.close();
      return { ok: false, diagnostic: "Path must remain within the repository." };
    }
    return { ok: true, target: descriptorPath, close: () => handle.close() };
  } catch {
    return { ok: false, diagnostic: "Path must be an existing accessible regular file or directory." };
  }
}

/** Run ast-grep with bounded output and deterministic timeout cleanup. */
export async function runAstgrepCommand(command: string[], timeoutMs = ASTGREP_TIMEOUT_MS): Promise<string> {
  const subprocess = Bun.spawn(command, { stdout: "pipe", stderr: "pipe" } as never);
  const collect = async (stream: ReadableStream<Uint8Array> | null | undefined): Promise<{ text: string; clipped: boolean }> => {
    if (!stream) return { text: "", clipped: false };
    const reader = stream.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    let clipped = false;
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = value ?? new Uint8Array();
        const remaining = MAX_ASTGREP_OUTPUT_BYTES - size;
        if (remaining <= 0 || chunk.length > remaining) {
          if (remaining > 0) chunks.push(chunk.subarray(0, remaining));
          clipped = true;
          subprocess.kill();
          break;
        }
        chunks.push(chunk);
        size += chunk.length;
      }
    } finally { reader.releaseLock(); }
    return { text: Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString(), clipped };
  };
  const execution = Promise.all([collect(subprocess.stdout), collect(subprocess.stderr), subprocess.exited]);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const outcome = await Promise.race([
    execution.then((value) => ({ kind: "complete" as const, value })),
    new Promise<{ kind: "timeout" }>((resolve) => {
      timer = setTimeout(() => { subprocess.kill(); resolve({ kind: "timeout" }); }, timeoutMs);
    }),
  ]).finally(() => { if (timer) clearTimeout(timer); });
  if (outcome.kind === "timeout") {
    await subprocess.exited.catch(() => undefined);
    throw new Error(`ast-grep timed out after ${timeoutMs}ms`);
  }
  const [stdout, stderr, exitCode] = outcome.value;
  if (stdout.clipped || stderr.clipped) throw new Error("ast-grep output exceeded the safety limit");
  if (Number(exitCode ?? 0) !== 0) {
    const error = new Error(stderr.text || stdout.text || `ast-grep exited with code ${exitCode}`) as Error & { stderr?: string; stdout?: string };
    error.stderr = stderr.text;
    error.stdout = stdout.text;
    throw error;
  }
  return stdout.text;
}
export function buildAstgrepCommand(args: AstgrepNormalizedArgs, target: string, apply = false): string[] {
  return ["ast-grep", "run", "-p", args.pattern, "-r", args.rewrite, "-l", args.lang, ...(apply ? ["--update-all"] : []), "--", target];
}
