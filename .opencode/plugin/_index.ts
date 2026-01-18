/**
 * ADW OpenCode Plugins
 *
 * Equivalent to Claude Code hooks, converted to OpenCode plugin format.
 * These plugins provide logging, security checks, and session management.
 * 
 * Note: Each plugin file should export a default plugin.
 * OpenCode will automatically load all .ts/.js files in the plugin directory.
 */

// Re-export plugins if needed for testing or manual loading
export { default as NotificationPlugin } from "./notification";
export { default as PostToolUsePlugin } from "./post_tool_use";
export { default as PreToolUsePlugin } from "./pre_tool_use";
export { default as PreCompactPlugin } from "./pre_compact";
export { default as SessionStopPlugin } from "./session_stop";
export { default as UserPromptSubmitPlugin } from "./user_prompt_submit";
export { default as StepLoopDetectionPlugin } from "./step_loop_detection";
