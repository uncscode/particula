import { tool } from "@opencode-ai/plugin";
import {
  ALLOWED_OUTPUT_MODES,
  ALLOWED_SANITIZERS,
  type SanitizerArgs,
  executeSanitizers,
} from "./run_sanitizers_shared";

// --- Tool definition ---

export default tool({
  description: `Run sanitizer-enabled binaries (ASAN, TSAN, UBSAN) with parsing and validation.

Use this advanced wrapper when you need suppressions/options/normalDuration/extraArgs controls.`,
  args: {
    outputMode: tool.schema.enum(ALLOWED_OUTPUT_MODES).optional(),
    buildDir: tool.schema.string(),
    executable: tool.schema.string(),
    sanitizer: tool.schema.enum(ALLOWED_SANITIZERS),
    timeout: tool.schema.number().optional(),
    suppressions: tool.schema.string().optional(),
    options: tool.schema.string().optional(),
    normalDuration: tool.schema.number().optional(),
    extraArgs: tool.schema.array(tool.schema.string()).optional(),
  },
  async execute(args) {
    return executeSanitizers(args as SanitizerArgs);
  },
});
