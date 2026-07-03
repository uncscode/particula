const ADW_ID_PATTERN = /^[0-9a-f]{8}$/i;
export const SHOW_PARSE_SNIPPET_LIMIT = 160;
const ERROR_OUTPUT_SNIPPET_LIMIT = 400;
const SPAWN_TIMEOUT_MS = 30_000;
const CONTROL_CHAR_PATTERN = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const WINDOWS_ABSOLUTE_PATH_PATTERN = /[A-Za-z]:\\(?:[^\\\r\n"']+\\)*[^\\\r\n"']+/g;
const QUOTED_UNIX_ABSOLUTE_PATH_PATTERN = /(["'])(\/[^\r\n"']+)\1/g;
const UNIX_COLON_PATH_PATTERN = /(^|[\s(\[])(\/(?:[^\s:\r\n)\]"']|:(?!\s))+?)(?=:\s|:\d|$)/gm;
const UNIX_BARE_PATH_PATTERN = /(^|[\s(\[])(\/[^\s)\]"']+)/gm;
const REDACTED_SECRET = "<redacted-secret>";
const SECRET_ASSIGNMENT_PATTERNS = [
  /\b(authorization\s*:\s*bearer\s+)([^\s]+)/gi,
  /\b((?:token|secret|password|passwd|api(?:_|-)?key|access(?:_|-)?token|refresh(?:_|-)?token)\s*[:=]\s*)("?)([^\s",']+)("?)/gi,
  /\b(gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]+)\b/g,
];

export function normalizeAdwId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!ADW_ID_PATTERN.test(trimmed)) return null;
  return trimmed.toLowerCase();
}

export function sanitizeSnippet(value: string, maxLen = ERROR_OUTPUT_SNIPPET_LIMIT): string {
  const redactedPaths = value
    .replace(WINDOWS_ABSOLUTE_PATH_PATTERN, "<path>")
    .replace(QUOTED_UNIX_ABSOLUTE_PATH_PATTERN, "$1<path>$1")
    .replace(UNIX_COLON_PATH_PATTERN, (_, prefix) => `${prefix}<path>`)
    .replace(UNIX_BARE_PATH_PATTERN, (_, prefix) => `${prefix}<path>`);
  const redacted = SECRET_ASSIGNMENT_PATTERNS.reduce((output, pattern) => {
    if (pattern.global && pattern.source.includes("authorization")) {
      return output.replace(pattern, `$1${REDACTED_SECRET}`);
    }
    if (pattern.global && pattern.source.includes("token|secret|password")) {
      return output.replace(pattern, `$1$2${REDACTED_SECRET}$4`);
    }
    return output.replace(pattern, REDACTED_SECRET);
  }, redactedPaths);
  const collapsed = redacted
    .replace(CONTROL_CHAR_PATTERN, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (collapsed.length === 0) return "";
  return collapsed.length > maxLen ? `${collapsed.slice(0, maxLen)}...(truncated)` : collapsed;
}

export function normalizeRef(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

export type ParsedFieldEntriesResult =
  | { ok: true; entries: [string, string][] }
  | { ok: false; diagnostic: string };

function describeType(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function stringifyKey(value: string): string {
  return JSON.stringify(value);
}

function buildIndexDiagnostic(index: number, message: string): ParsedFieldEntriesResult {
  return { ok: false, diagnostic: `invalid fields entry at index ${index}: ${message}` };
}

function buildKeyDiagnostic(key: string, message: string): ParsedFieldEntriesResult {
  return { ok: false, diagnostic: `invalid fields object key ${stringifyKey(key)}: ${message}` };
}

export function parseFieldEntries(fields: unknown): ParsedFieldEntriesResult {
  if (fields === undefined || fields === null) return { ok: true, entries: [] };
  let parsed: unknown = fields;
  if (typeof parsed === "string") {
    const trimmed = parsed.trim();
    if (trimmed.length === 0) return { ok: true, entries: [] };
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      return { ok: false, diagnostic: "'fields' JSON string could not be parsed" };
    }
  }

  const normalized: [string, string][] = [];
  if (Array.isArray(parsed)) {
    for (const [index, entry] of parsed.entries()) {
      if (entry === null || entry === undefined) continue;
      if (Array.isArray(entry)) {
        if (entry.length !== 2) {
          return buildIndexDiagnostic(index, "tuple must contain exactly [key, value]");
        }
        const rawKey = entry[0];
        const rawValue = entry[1];
        if (rawKey === null || rawKey === undefined) {
          return buildIndexDiagnostic(index, "key is missing");
        }
        if (rawValue === null) {
          return buildIndexDiagnostic(index, "value is null");
        }
        if (rawValue === undefined) {
          return buildIndexDiagnostic(index, "value is missing");
        }
        if (typeof rawKey !== "string") {
          return buildIndexDiagnostic(index, `key has wrong type ${describeType(rawKey)}`);
        }
        if (typeof rawValue !== "string") {
          return buildIndexDiagnostic(index, `value has wrong type ${describeType(rawValue)}`);
        }
        const key = rawKey.trim();
        if (key.length === 0) {
          return buildIndexDiagnostic(index, `key ${stringifyKey(rawKey)} is blank`);
        }
        normalized.push([key, rawValue]);
        continue;
      }
      if (entry && typeof entry === "object") {
        const rawKey = (entry as Record<string, unknown>).key;
        const rawValue = (entry as Record<string, unknown>).value;
        if (rawKey === null || rawKey === undefined) {
          return buildIndexDiagnostic(index, "key is missing");
        }
        if (rawValue === null) {
          return buildIndexDiagnostic(index, "value is null");
        }
        if (rawValue === undefined) {
          return buildIndexDiagnostic(index, "value is missing");
        }
        if (typeof rawKey !== "string") {
          return buildIndexDiagnostic(index, `key has wrong type ${describeType(rawKey)}`);
        }
        if (typeof rawValue !== "string") {
          return buildIndexDiagnostic(index, `value has wrong type ${describeType(rawValue)}`);
        }
        const key = rawKey.trim();
        if (key.length === 0) {
          return buildIndexDiagnostic(index, `key ${stringifyKey(rawKey)} is blank`);
        }
        normalized.push([key, rawValue]);
        continue;
      }
      return buildIndexDiagnostic(index, `entry has wrong type ${describeType(entry)}`);
    }
    return { ok: true, entries: normalized };
  }

  if (parsed && typeof parsed === "object") {
    for (const [rawKey, value] of Object.entries(parsed as Record<string, unknown>)) {
      if (value === null) {
        return buildKeyDiagnostic(rawKey, "value is null");
      }
      if (value === undefined) {
        return buildKeyDiagnostic(rawKey, "value is missing");
      }
      if (typeof value !== "string") {
        return buildKeyDiagnostic(rawKey, `value has wrong type ${describeType(value)}`);
      }
      const key = rawKey.trim();
      if (key.length === 0) {
        return buildKeyDiagnostic(rawKey, "key is blank");
      }
      normalized.push([key, value]);
    }
    return { ok: true, entries: normalized };
  }

  return { ok: false, diagnostic: "'fields' must decode to an array or plain object" };
}

export function buildFieldArgs(fields: [string, string][]): string[] {
  return fields.flatMap(([key, value]) => ["--field", key, value]);
}

export function selectDiagnostic(...candidates: unknown[]): string {
  for (const candidate of candidates) {
    const snippet = sanitizeSnippet(candidate ? String(candidate) : "");
    if (snippet) return snippet;
  }
  return "";
}

export function decodeDiagnosticText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === undefined || value === null) return "";
  if (value instanceof Uint8Array) return Buffer.from(value).toString();
  if (ArrayBuffer.isView(value)) {
    return Buffer.from(value.buffer, value.byteOffset, value.byteLength).toString();
  }
  if (value instanceof ArrayBuffer) return Buffer.from(value).toString();
  return String(value);
}

export function validateRequiredAdwId(
  command: "write" | "write-from-state" | "show",
  adwId: unknown,
): { value?: string; error?: string } {
  if (command !== "write-from-state") {
    return {};
  }
  if (typeof adwId !== "string" || adwId.trim().length === 0) {
    return { error: `'adw_id' is required for '${command}'.` };
  }

  const normalized = normalizeAdwId(adwId);
  if (!normalized) {
    return { error: `'adw_id' must be an 8-character hex string (e.g., "abc12345").` };
  }
  return { value: normalized };
}

export function runNotesCommand(command: "write" | "write-from-state" | "show", cmdParts: string[]): {
  ok: true; stdout: string;
} | {
  ok: false; error: string;
} {
  try {
    const result = Bun.spawnSync({
      cmd: cmdParts,
      env: Object.fromEntries(
        Object.entries(process.env).filter(([, value]) => value !== undefined),
      ) as Record<string, string>,
      stdout: "pipe",
      stderr: "pipe",
      timeout: SPAWN_TIMEOUT_MS,
    });
    const decoder = new TextDecoder();
    const stdout = result.stdout ? decoder.decode(result.stdout) : "";
    const stderr = result.stderr ? decoder.decode(result.stderr) : "";
    if (result.exitCode !== 0) {
      const diagnostic = selectDiagnostic(stderr, stdout, (result as any)?.message);
      const fallback = diagnostic || `Exit code ${result.exitCode}`;
      return { ok: false, error: `ERROR: adw notes ${command} failed.\n${fallback}` };
    }
    return { ok: true, stdout };
  } catch (error: any) {
    const diagnostic =
      selectDiagnostic(
        decodeDiagnosticText(error?.stderr),
        decodeDiagnosticText(error?.stdout),
        error?.message,
      ) || "Unknown execution error";
    return { ok: false, error: `ERROR: Failed to execute adw notes ${command}. ${diagnostic}` };
  }
}

export function parseShowOutput(stdout: string): string {
  try {
    const parsed = JSON.parse(stdout);
    return JSON.stringify(parsed, null, 2);
  } catch {
    const snippet = sanitizeSnippet(stdout, SHOW_PARSE_SNIPPET_LIMIT);
    const safeSnippet = snippet.length > 0 ? snippet : "<empty stdout>";
    return `ERROR: Failed to parse JSON output from adw notes show. Snippet: ${safeSnippet}`;
  }
}
