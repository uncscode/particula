const CONTROL_CHARS_PATTERN = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g;
const WHITESPACE_COLLAPSE_PATTERN = /\s+/g;

const REDACTION_MARKER = "[REDACTED]";
const REDACTION_PATTERNS: RegExp[] = [
  /\bgh[pousr]_[A-Za-z0-9]{10,}\b/g,
  /\b(?:token|api[_-]?key|client[_-]?secret|secret|password)\s*[:=]\s*([^\s"']+)/gi,
  /\bAuthorization\s*:\s*Bearer\s+([^\s"']+)/gi,
  /"(?:token|api[_-]?key|client[_-]?secret|secret|password)"\s*:\s*"[^"]*"/gi,
];

function redactSensitiveFragments(value: string): string {
  let redacted = value;
  redacted = redacted.replace(REDACTION_PATTERNS[0], REDACTION_MARKER);
  redacted = redacted.replace(REDACTION_PATTERNS[1], (match) => {
    const splitIndex = match.indexOf(":") >= 0 ? match.indexOf(":") : match.indexOf("=");
    if (splitIndex < 0) return REDACTION_MARKER;
    return `${match.slice(0, splitIndex + 1)} ${REDACTION_MARKER}`;
  });
  redacted = redacted.replace(REDACTION_PATTERNS[2], `Authorization: Bearer ${REDACTION_MARKER}`);
  redacted = redacted.replace(REDACTION_PATTERNS[3], (match) => {
    const splitIndex = match.indexOf(":");
    if (splitIndex < 0) return `"${REDACTION_MARKER}"`;
    return `${match.slice(0, splitIndex + 1)}"${REDACTION_MARKER}"`;
  });
  return redacted;
}

function decodeUnknown(value: unknown): string {
  if (value instanceof Uint8Array) return new TextDecoder().decode(value);
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  return String(value);
}

function sanitizeDiagnosticText(value: string, maxChars: number): string {
  if (!value) return "";
  const noControlChars = value.replace(CONTROL_CHARS_PATTERN, " ");
  const redacted = redactSensitiveFragments(noControlChars);
  const normalized = redacted.replace(WHITESPACE_COLLAPSE_PATTERN, " ").trim();
  if (!normalized) return "";
  if (normalized.length <= maxChars) return normalized;
  return `${normalized.slice(0, maxChars)} ...(truncated)`;
}

function redactJsonValue(value: unknown): unknown {
  if (typeof value === "string") {
    return redactSensitiveFragments(value);
  }
  if (Array.isArray(value)) {
    return value.map(redactJsonValue);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [key, redactJsonValue(nestedValue)]),
    );
  }
  return value;
}

export function getStructuredJsonPayload(value: unknown): string | undefined {
  const decoded = decodeUnknown(value).trim();
  if (!decoded) return undefined;
  try {
    return JSON.stringify(redactJsonValue(JSON.parse(decoded)));
  } catch {
    return undefined;
  }
}

export function normalizeLabels(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  const token = String(value).trim();
  if (!token) {
    return undefined;
  }
  const labels = token
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean);
  return labels.length > 0 ? labels.join(",") : undefined;
}

export function sanitizeAndTruncate(value: unknown, maxChars = 2000): string {
  return sanitizeDiagnosticText(decodeUnknown(value), maxChars);
}
