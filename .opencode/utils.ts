/**
 * Shared utilities for OpenCode plugins
 */

import { join } from "path";

/**
 * Base directory for all logs
 * Default is 'logs' in the current working directory
 */
const LOG_BASE_DIR = process.env.OPENCODE_HOOKS_LOG_DIR || "logs";

/**
 * Get the log directory for a specific session
 */
export function getSessionLogDir(sessionId: string): string {
  return join(LOG_BASE_DIR, sessionId);
}

/**
 * Ensure the log directory for a session exists
 */
export async function ensureSessionLogDir(sessionId: string): Promise<string> {
  const logDir = getSessionLogDir(sessionId);
  await Bun.write(join(logDir, ".keep"), ""); // Creates directory if it doesn't exist
  return logDir;
}

/**
 * Log data to a JSON file in the session directory with automatic rotation
 * 
 * To prevent unbounded growth, logs are rotated when they exceed MAX_LOG_ENTRIES.
 * The oldest entries are removed to keep the file size manageable.
 */
const MAX_LOG_ENTRIES = 1000; // Maximum entries before rotation

export async function logToFile(
  sessionId: string,
  filename: string,
  data: any
): Promise<void> {
  try {
    const logDir = await ensureSessionLogDir(sessionId);
    const logPath = join(logDir, filename);

    // Read existing log data or initialize empty array
    let logData: any[] = [];
    const file = Bun.file(logPath);
    if (await file.exists()) {
      try {
        const content = await file.text();
        logData = JSON.parse(content);
        if (!Array.isArray(logData)) {
          logData = [];
        }
      } catch {
        logData = [];
      }
    }

    // Append new data
    logData.push(data);

    // Rotate if necessary: keep only the most recent MAX_LOG_ENTRIES
    if (logData.length > MAX_LOG_ENTRIES) {
      const entriesToRemove = logData.length - MAX_LOG_ENTRIES;
      logData = logData.slice(entriesToRemove);
      // Rotation happens silently - no terminal output
    }

    // Write back to file with formatting
    await Bun.write(logPath, JSON.stringify(logData, null, 2));
  } catch (error) {
    // Fail silently - logging shouldn't break the workflow
    console.error(`Failed to log to ${filename}:`, error);
  }
}

/**
 * Check if a bash command is a dangerous rm command
 */
export function isDangerousRmCommand(command: string): boolean {
  // Normalize command by removing extra spaces and converting to lowercase
  const normalized = command.toLowerCase().replace(/\s+/g, " ");

  // Pattern 1: Standard rm -rf variations
  const patterns = [
    /\brm\s+.*-[a-z]*r[a-z]*f/,  // rm -rf, rm -fr, rm -Rf, etc.
    /\brm\s+.*-[a-z]*f[a-z]*r/,  // rm -fr variations
    /\brm\s+--recursive\s+--force/,  // rm --recursive --force
    /\brm\s+--force\s+--recursive/,  // rm --force --recursive
    /\brm\s+-r\s+.*-f/,  // rm -r ... -f
    /\brm\s+-f\s+.*-r/,  // rm -f ... -r
  ];

  // Check for dangerous patterns
  for (const pattern of patterns) {
    if (pattern.test(normalized)) {
      return true;
    }
  }

  // Pattern 2: Check for rm with recursive flag targeting dangerous paths
  const dangerousPaths = [
    /\//,           // Root directory
    /\/\*/,         // Root with wildcard
    /~/,            // Home directory
    /~\//,          // Home directory path
    /\$HOME/,       // Home environment variable
    /\.\./,         // Parent directory references
    /\*/,           // Wildcards in general rm -rf context
    /\./,           // Current directory
    /\.\s*$/,       // Current directory at end of command
  ];

  if (/\brm\s+.*-[a-z]*r/.test(normalized)) {  // If rm has recursive flag
    for (const path of dangerousPaths) {
      if (path.test(normalized)) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Check if a tool is trying to access .env files
 */
export function isEnvFileAccess(toolName: string, toolInput: any): boolean {
  const fileAccessTools = ["Read", "Edit", "MultiEdit", "Write", "Bash"];

  if (!fileAccessTools.includes(toolName)) {
    return false;
  }

  // Check file paths for file-based tools
  if (["Read", "Edit", "MultiEdit", "Write"].includes(toolName)) {
    const filePath = toolInput?.file_path || "";
    if (filePath.includes(".env") && !filePath.endsWith(".env.sample")) {
      return true;
    }
  }

  // Check bash commands for .env file access
  if (toolName === "Bash") {
    const command = toolInput?.command || "";

    // Patterns to detect .env file access (but allow .env.sample)
    const envPatterns = [
      /\b\.env\b(?!\.sample)/,  // .env but not .env.sample
      /cat\s+.*\.env\b(?!\.sample)/,  // cat .env
      /echo\s+.*>\s*\.env\b(?!\.sample)/,  // echo > .env
      /touch\s+.*\.env\b(?!\.sample)/,  // touch .env
      /cp\s+.*\.env\b(?!\.sample)/,  // cp .env
      /mv\s+.*\.env\b(?!\.sample)/,  // mv .env
    ];

    for (const pattern of envPatterns) {
      if (pattern.test(command)) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Load existing tool use log data from file
 */
export async function loadToolUseLog(sessionId: string): Promise<any[]> {
  try {
    const logDir = getSessionLogDir(sessionId);
    const logPath = join(logDir, "pre_tool_use.json");
    const file = Bun.file(logPath);
    
    if (await file.exists()) {
      const content = await file.text();
      const logData = JSON.parse(content);
      return Array.isArray(logData) ? logData : [];
    }
  } catch {
    // Return empty array on any error
  }
  return [];
}

/**
 * Detect if the same tool with the same input is being called repeatedly
 * 
 * @param logData - Array of previous tool calls in this session
 * @param currentTool - Current tool name being called
 * @param currentInput - Current tool input parameters
 * @param maxRepeats - Maximum allowed identical consecutive calls (default: 10)
 * @returns True if infinite loop detected, False otherwise
 */
export function detectInfiniteLoop(
  logData: any[],
  currentTool: string,
  currentInput: any,
  maxRepeats: number = 10
): boolean {
  if (logData.length < maxRepeats) {
    return false;
  }

  // Get the last N tool calls
  const recentCalls = logData.slice(-maxRepeats);

  // Check if all recent calls match the current call
  for (const call of recentCalls) {
    if (call.tool_name !== currentTool) {
      return false;
    }
    
    // Deep comparison of tool inputs
    if (JSON.stringify(call.tool_input) !== JSON.stringify(currentInput)) {
      return false;
    }
  }

  // All recent calls match - this is likely an infinite loop
  return true;
}

/**
 * Extract ADW ID from prompt text using known patterns.
 * 
 * ADW workflows pass the ADW ID via prompt arguments in formats like:
 * - "Arguments: adw_id=abc12345"
 * - "adw_id=def67890 worktree_path=/trees/def67890"
 * 
 * @param prompt - The prompt text sent to the agent
 * @returns ADW ID if found, null otherwise
 * 
 * @example
 * extractAdwIdFromPrompt("Arguments: adw_id=abc12345") // "abc12345"
 * extractAdwIdFromPrompt("No adw id here") // null
 */
export function extractAdwIdFromPrompt(prompt: string): string | null {
  if (!prompt) return null;

  // Pattern: adw_id= or adw_id: followed by 8 alphanumeric chars
  const match = prompt.match(/adw_id[=:]\s*([a-f0-9]{8})/i);
  return match ? match[1] : null;
}

/**
 * Extract ADW ID from environment variables.
 * 
 * Checks standard ADW environment variables that may be set by
 * workflow orchestration:
 * - ADW_ID: Primary ADW identifier
 * - ADW_WORKFLOW_ID: Alternative identifier
 * 
 * @returns ADW ID if set in environment, null otherwise
 * 
 * @example
 * // If ADW_ID=abc12345 is set
 * extractAdwIdFromEnvironment() // "abc12345"
 */
export function extractAdwIdFromEnvironment(): string | null {
  return process.env.ADW_ID || process.env.ADW_WORKFLOW_ID || null;
}

/**
 * Extract ADW ID from OpenCode session context.
 * 
 * Searches session metadata for ADW ID in standard locations:
 * - session.metadata.adw_id
 * - session.context.adw_id
 * - session.config.adw_id
 * 
 * @param session - OpenCode session object
 * @returns ADW ID if found in session context, null otherwise
 * 
 * @example
 * extractAdwIdFromSession(session) // "abc12345" or null
 */
export function extractAdwIdFromSession(session: any): string | null {
  if (!session) return null;

  return (
    session?.metadata?.adw_id ||
    session?.context?.adw_id ||
    session?.config?.adw_id ||
    null
  );
}

/**
 * Unified ADW ID extraction from all available sources.
 * 
 * Tries sources in order of reliability:
 * 1. Session metadata (most reliable)
 * 2. Prompt text (common pattern)
 * 3. Environment variables (fallback)
 * 
 * This is the recommended function for plugins needing ADW context.
 * 
 * @param session - OpenCode session object
 * @param prompt - Optional prompt text
 * @returns ADW ID if found from any source, null otherwise
 * 
 * @example
 * // In a plugin
 * const adwId = getAdwId(event.session, event.prompt);
 * if (adwId) {
 *   console.log(`Operating in ADW workflow: ${adwId}`);
 * }
 */
export function getAdwId(session: any, prompt?: string): string | null {
  // Try session context first (most reliable)
  const fromSession = extractAdwIdFromSession(session);
  if (fromSession) return fromSession;

  // Try prompt text
  if (prompt) {
    const fromPrompt = extractAdwIdFromPrompt(prompt);
    if (fromPrompt) return fromPrompt;
  }

  // Fallback to environment
  return extractAdwIdFromEnvironment();
}
