import { tool } from "@opencode-ai/plugin";
import {
  ADVANCED_ONLY_KEYS,
  ALLOWED_OUTPUT_MODES,
  ALLOWED_SANITIZERS,
  type SanitizerArgs,
  executeSanitizers,
} from "./run_sanitizers_shared";

// --- Tool definition ---

export default tool({
  description: `Run routine sanitizer checks (buildDir + executable + sanitizer only).

Use this routine wrapper for standard sanitizer runs.
Advanced controls (suppressions/options/normalDuration/extraArgs) are intentionally blocked; use run_sanitizers_advanced for those options.`,
  args: {
    outputMode: tool.schema.enum(ALLOWED_OUTPUT_MODES).optional(),
    buildDir: tool.schema.string(),
    executable: tool.schema.string(),
    sanitizer: tool.schema.enum(ALLOWED_SANITIZERS),
    timeout: tool.schema.number().optional(),
  },
  async execute(args) {
    for (const key of ADVANCED_ONLY_KEYS) {
      if (Object.hasOwn(args, key)) {
        return `ERROR: run_sanitizers_basic does not accept advanced option '${key}'. Use run_sanitizers_advanced.`;
      }
    }

    return executeSanitizers(args as SanitizerArgs);
  },
});
