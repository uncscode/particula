import { lstat, mkdir, readFile, readdir, realpath, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";

export type WrapperInventoryStatus = "ok" | "no_tool_schema" | "inspection_failed";
export type WrapperKind = "compatibility" | "split" | "unsplit";
export type WrapperClassification =
  | "already-compliant"
  | "split-needed"
  | "bounded-options-needed"
  | "compatibility-defer-with-owner";
export type WrapperOwnerPlan =
  | "E27-M3"
  | "E27-M4"
  | "E27-M5"
  | "E27-M6"
  | "E27-M7"
  | "E27-M8"
  | "E27-M9"
  | "E27-M10"
  | "E27-M11";
export type WrapperTargetShape =
  | "direct_args_object"
  | "no_tool_schema"
  | "split-wrapper-family"
  | "bounded-options"
  | "compatibility-window";

export type WrapperFieldRole =
  | "required-identifier"
  | "command-selector"
  | "bounded-option-candidate"
  | "payload-bearing"
  | "safety-field"
  | "explicit-exception";

export type WrapperFieldRoleAudit = {
  field: string;
  role: WrapperFieldRole;
  notes?: string;
  commands?: string[];
};

export type WrapperSchemaInventoryRow = {
  wrapper_path: string;
  companion_doc_path: string | null;
  wrapper_kind: WrapperKind;
  status: WrapperInventoryStatus;
  current_shape: "direct_args_object" | "no_tool_schema" | "inspection_failed";
  counted_fields: string[];
  exempt_fields: string[];
  current_count: number | null;
  classification: WrapperClassification;
  target_shape: WrapperTargetShape;
  target_count: number | null;
  owner_plan: WrapperOwnerPlan | null;
  nearest_test_path: string | null;
  diagnostic: string | null;
  field_role_audit?: WrapperFieldRoleAudit[];
  audit_notes?: string[];
};

export type WrapperSchemaInventoryArtifact = {
  audit_scope: ".opencode/tools/*.ts";
  counting_rule: "direct_args_fields_with_help_and_bounded_options_exempt";
  rows: WrapperSchemaInventoryRow[];
};

type InventoryContext = {
  repoRoot: string;
  toolsDir: string;
  docsByBasename: Map<string, string>;
  compatibilityWrappers: Set<string>;
  testPathsByBasename: Map<string, string>;
  metadataDiagnostics: string[];
};

type WrapperFamilySpec = {
  ownerPlan: WrapperOwnerPlan;
  matches: (basename: string) => boolean;
  oversizedClassification: Exclude<WrapperClassification, "already-compliant">;
  testCandidateBasenames?: string[];
  testBasenamePrefixes?: string[];
};

type WrapperAuditBaseline = {
  fieldRoleAudit: WrapperFieldRoleAudit[];
  auditNotes?: string[];
};

const TOOL_REGISTRATION_PATTERN = /export\s+default\s+tool\s*\(\s*\{/;
const ARGS_OBJECT_PATTERN = /\bargs\s*:\s*\{/g;
const COMPATIBILITY_GUIDE_PATH = ".opencode/guides/tool-wrapper-exceptions.json";
const INVENTORY_ARTIFACT_PATH = ".opencode/tools/wrapper_schema_inventory.json";
const TEMP_OUTPUT_DIR = "adforge_local/opencode/tmp";
const MAX_REPO_CONTROLLED_READ_BYTES = 1024 * 1024;
const COMPATIBILITY_WRAPPER_NAMES = new Set<string>([
  "adw",
  "adw_issues_spec",
  "adw_notes",
  "adw_plans",
  "adw_spec",
  "build_mkdocs",
  "clear_build",
  "move",
  "platform_operations",
  "run_cmake",
  "run_cpp_linters",
  "run_sanitizers",
  "validate_notebook",
  "workflow_builder",
]);
const SPLIT_WRAPPER_NAMES = new Set<string>([
  "adw_issues_batch_init",
  "adw_issues_batch_log",
  "adw_issues_batch_read",
  "adw_issues_batch_summary",
  "adw_issues_batch_write",
  "adw_notes_read",
  "adw_notes_write",
  "adw_plans_mutate",
  "adw_plans_read",
  "adw_service",
  "adw_setup",
  "adw_spec_messages",
  "adw_spec_read",
  "adw_spec_write",
  "adw_status_health",
  "build_mkdocs_build",
  "build_mkdocs_validate",
  "clear_build_delete",
  "clear_build_preview",
  "convert_notebook_to_py",
  "convert_py_to_notebook",
  "find_files",
  "git_branch",
  "git_commit",
  "git_diff",
  "git_merge",
  "git_stage",
  "git_worktree",
  "move_overwrite",
  "move_safe",
  "move_trash",
  "platform_comment_write",
  "platform_issue_read",
  "platform_issue_write",
  "platform_label_write",
  "platform_pr_read",
  "platform_pr_review_write",
  "platform_pr_write",
  "platform_rate_limit_read",
  "refactor_astgrep_apply",
  "refactor_astgrep_preview",
  "ripgrep_advanced",
  "run_cmake_build",
  "run_cmake_configure",
  "run_cpp_coverage_advanced",
  "run_cpp_coverage_summary",
  "run_cpp_lint_check",
  "run_cpp_lint_fix",
  "run_pytest_advanced",
  "run_pytest_basic",
  "run_sanitizers_advanced",
  "run_sanitizers_basic",
  "search_content",
  "sync_notebook_pair",
  "validate_notebook_readonly",
  "workflow_builder_mutate",
  "workflow_builder_read",
]);
const BOUNDED_OPTIONS_WRAPPERS = new Set<string>([
  "adw_issues_batch_init",
  "adw_issues_batch_log",
  "adw_issues_batch_read",
  "adw_issues_batch_summary",
  "adw_issues_batch_write",
  "adw_issues_spec",
  "adw_setup",
  "adw_plans",
  "adw_plans_mutate",
  "adw_plans_read",
  "adw_spec",
  "adw_spec_messages",
  "adw_spec_read",
  "adw_spec_write",
  "auto_mode_manifest",
  "build_mkdocs",
  "build_mkdocs_build",
  "build_mkdocs_validate",
  "find_files",
  "ripgrep_advanced",
  "run_cmake",
  "run_cmake_build",
  "run_cmake_configure",
  "run_ctest",
  "run_linters",
  "run_bun_test",
  "run_pytest_advanced",
  "run_pytest_basic",
  "search_content",
]);
const SPLIT_NEEDED_WRAPPERS = new Set<string>([
  "adw_issues_spec",
  "adw_notes",
  "adw_spec",
  "build_mkdocs",
  "clear_build",
  "move",
  "platform_operations",
  "validate_notebook",
  "workflow_builder",
]);
const WRAPPER_TEST_FALLBACK_CANDIDATES = new Map<string, string[]>([
  ["adw", ["adw"]],
  ["adw_issues_batch_log", ["adw_issues_batch_log_summary"]],
  ["adw_issues_batch_summary", ["adw_issues_batch_log_summary"]],
  ["adw_issues_batch_write", ["adw_issues_batch_read_write"]],
  ["adw_issues_spec", ["adw_issues_spec"]],
  ["adw_plans", ["adw_plans_compat_required_args"]],
  ["move", ["move_safe"]],
  ["platform_operations", ["platform_operations_compat_comment_pr_review"]],
  ["run_cmake_build", ["run_cmake"]],
  ["run_cmake_configure", ["run_cmake"]],
  ["validate_notebook", ["validate_notebook_readonly"]],
  ["build_mkdocs_build", ["build_mkdocs"]],
  ["build_mkdocs_validate", ["build_mkdocs"]],
  ["clear_build_delete", ["clear_build"]],
  ["clear_build_preview", ["clear_build"]],
]);
const WRAPPER_TEST_FALLBACK_PREFIXES = new Map<string, string[]>([
  ["adw_notes", ["adw_notes_"]],
  ["adw_plans", ["adw_plans_"]],
  ["adw_spec", ["adw_spec_"]],
  ["build_mkdocs", ["build_mkdocs_"]],
  ["clear_build", ["clear_build_"]],
  ["move", ["move_"]],
  ["validate_notebook", ["validate_notebook"]],
  ["workflow_builder", ["workflow_builder_"]],
]);
const AUDIT_ONLY_TEST_COVERED_WRAPPERS = new Set<string>([
  "build_mkdocs",
  "build_mkdocs_validate",
  "build_mkdocs_build",
  "clear_build",
  "clear_build_preview",
  "clear_build_delete",
  "create_workspace",
  "move_overwrite",
  "move_trash",
  "refactor_astgrep_preview",
  "run_cmake",
  "run_cmake_configure",
  "run_cmake_build",
  "run_ctest",
  "run_pytest_advanced",
]);
const WRAPPER_AUDIT_BASELINES = new Map<string, WrapperAuditBaseline>([
  [
    "feedback_log",
    {
      fieldRoleAudit: [
        { field: "category", role: "bounded-option-candidate" },
        { field: "severity", role: "bounded-option-candidate" },
        { field: "description", role: "payload-bearing" },
        { field: "suggestedFix", role: "payload-bearing" },
        { field: "toolName", role: "payload-bearing" },
        { field: "workflowStep", role: "payload-bearing" },
        { field: "agentType", role: "payload-bearing" },
        { field: "adwId", role: "payload-bearing" },
        { field: "context", role: "payload-bearing" },
      ],
      auditNotes: [
        "E27-M11 audit confirms the utility/telemetry wrapper family is exactly feedback_log, get_datetime, and get_version with no extra wrapper assigned by committed inventory artifacts.",
        "Python backend read-mode inputs command/page/page-size/severity-filter are backend-only surface area in .opencode/tools/feedback_log.py and are intentionally not modeled as direct TypeScript wrapper fields in this audit.",
        "feedback_log schema/behavior migration landed in E27-M11 P2 with dedicated wrapper coverage; broader compatibility-window enforcement cleanup remains deferred to E27-M12.",
        "E27-M11 P5 adds a compact companion wrapper doc for write-mode usage while leaving AGENTS.md as the canonical operator note for backend read-mode and fallback-policy details.",
      ],
    },
  ],
  [
    "find_files",
    {
      fieldRoleAudit: [
        { field: "pattern", role: "payload-bearing", notes: "Discovery-mode glob selector." },
        { field: "path", role: "safety-field", notes: "Repository-constrained search root." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for discovery-only optional controls." },
        { field: "fileType", role: "bounded-option-candidate" },
        { field: "excludeFileType", role: "bounded-option-candidate" },
        { field: "globCaseInsensitive", role: "bounded-option-candidate" },
        { field: "compactOutput", role: "bounded-option-candidate" },
        { field: "maxResults", role: "bounded-option-candidate" },
        { field: "contentPattern", role: "explicit-exception", notes: "Accepted only to fail closed with split-wrapper guidance." },
        { field: "filesWithMatches", role: "explicit-exception" },
        { field: "filesWithoutMatches", role: "explicit-exception" },
        { field: "contextLines", role: "explicit-exception" },
        { field: "beforeContext", role: "explicit-exception" },
        { field: "afterContext", role: "explicit-exception" },
        { field: "maxMatchesPerFile", role: "explicit-exception" },
      ],
    },
  ],
  [
    "search_content",
    {
      fieldRoleAudit: [
        { field: "contentPattern", role: "payload-bearing", notes: "Required content-search selector." },
        { field: "pattern", role: "bounded-option-candidate", notes: "Optional glob filter paired with content search." },
        { field: "path", role: "safety-field", notes: "Repository-constrained search root." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for simple-search optional controls." },
        { field: "fileType", role: "bounded-option-candidate" },
        { field: "excludeFileType", role: "bounded-option-candidate" },
        { field: "globCaseInsensitive", role: "bounded-option-candidate" },
        { field: "compactOutput", role: "bounded-option-candidate" },
        { field: "maxResults", role: "bounded-option-candidate" },
        { field: "maxMatchesPerFile", role: "bounded-option-candidate" },
        { field: "contextLines", role: "explicit-exception" },
        { field: "beforeContext", role: "explicit-exception" },
        { field: "afterContext", role: "explicit-exception" },
        { field: "filesWithMatches", role: "explicit-exception" },
        { field: "filesWithoutMatches", role: "explicit-exception" },
        { field: "unrestricted", role: "explicit-exception" },
        { field: "ignoreGitignore", role: "explicit-exception" },
        { field: "includeHidden", role: "explicit-exception" },
      ],
    },
  ],
  [
    "get_datetime",
    {
      fieldRoleAudit: [
        { field: "format", role: "bounded-option-candidate" },
        {
          field: "localtime",
          role: "safety-field",
          notes: "Behavior toggle switches between UTC and America/Denver output modes.",
        },
      ],
      auditNotes: [
        "E27-M11 audit keeps get_datetime as an already-compliant row with owner_plan null while recording family membership and no additional wrapper assignment beyond feedback_log/get_datetime/get_version.",
        "No exception-only migration is needed for this row in P1; wrapper-contract changes remain deferred to E27-M11 P3 if later schema work is required.",
        "E27-M11 P5 adds a compact companion doc that records the shipped format/localtime schema and output guarantees without reopening wrapper behavior.",
      ],
    },
  ],
  [
    "get_version",
    {
      fieldRoleAudit: [{ field: "file", role: "payload-bearing" }],
      auditNotes: [
        "E27-M11 audit keeps get_version as an already-compliant row with owner_plan null while recording family membership and no additional wrapper assignment beyond feedback_log/get_datetime/get_version.",
        "Sibling-family boundary note: broader search/refactor/move/build wrapper leftovers remain owned by E27-M8 through E27-M10; no extra small utility wrapper is reassigned into E27-M11 by committed artifacts in this audit.",
        "No exception-only migration is needed for this row in P1; wrapper-contract changes remain deferred to E27-M11 P3 if later schema work is required.",
        "E27-M11 P5 refreshes the companion doc so the shipped blank-input omission, default lookup order, and deterministic failure hints remain discoverable.",
      ],
    },
  ],
  [
    "ripgrep_advanced",
    {
      fieldRoleAudit: [
        { field: "contentPattern", role: "payload-bearing", notes: "Required advanced content-search selector." },
        { field: "pattern", role: "bounded-option-candidate" },
        { field: "path", role: "safety-field", notes: "Repository-constrained search root." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for advanced content-search controls." },
        { field: "fileType", role: "bounded-option-candidate" },
        { field: "excludeFileType", role: "bounded-option-candidate" },
        { field: "globCaseInsensitive", role: "bounded-option-candidate" },
        { field: "compactOutput", role: "bounded-option-candidate" },
        { field: "maxResults", role: "bounded-option-candidate" },
        { field: "maxMatchesPerFile", role: "bounded-option-candidate" },
        { field: "contextLines", role: "bounded-option-candidate" },
        { field: "beforeContext", role: "bounded-option-candidate" },
        { field: "afterContext", role: "bounded-option-candidate" },
        { field: "filesWithMatches", role: "bounded-option-candidate" },
        { field: "filesWithoutMatches", role: "bounded-option-candidate" },
        { field: "unrestricted", role: "bounded-option-candidate" },
        { field: "ignoreGitignore", role: "bounded-option-candidate" },
        { field: "includeHidden", role: "bounded-option-candidate" },
      ],
    },
  ],
  [
    "ripgrep",
    {
      fieldRoleAudit: [
        { field: "pattern", role: "payload-bearing", notes: "Compatibility discovery selector when contentPattern is absent." },
        { field: "contentPattern", role: "payload-bearing", notes: "Compatibility content-search selector." },
        { field: "filesWithMatches", role: "explicit-exception", notes: "Compatibility-only mixed-mode toggle retained until split migration is complete." },
        { field: "filesWithoutMatches", role: "explicit-exception", notes: "Compatibility-only mixed-mode toggle retained until split migration is complete." },
        { field: "path", role: "safety-field", notes: "Repository-constrained search root." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for compatibility-safe optional filters." },
        { field: "ignoreGitignore", role: "explicit-exception" },
        { field: "includeHidden", role: "explicit-exception" },
        { field: "unrestricted", role: "explicit-exception" },
        { field: "fileType", role: "bounded-option-candidate" },
        { field: "excludeFileType", role: "bounded-option-candidate" },
        { field: "globCaseInsensitive", role: "bounded-option-candidate" },
        { field: "compactOutput", role: "bounded-option-candidate" },
        { field: "maxResults", role: "bounded-option-candidate" },
        { field: "maxMatchesPerFile", role: "explicit-exception" },
        { field: "contextLines", role: "explicit-exception" },
        { field: "beforeContext", role: "explicit-exception" },
        { field: "afterContext", role: "explicit-exception" },
      ],
      auditNotes: [
        "Compatibility row intentionally preserves the broader mixed discovery/content surface instead of forcing split-wrapper symmetry in E27-M8-P1.",
      ],
    },
  ],
  [
    "refactor_astgrep",
    {
      fieldRoleAudit: [
        { field: "pattern", role: "payload-bearing" },
        { field: "rewrite", role: "payload-bearing" },
        { field: "lang", role: "payload-bearing" },
        { field: "path", role: "safety-field", notes: "Explicit repository-confined rewrite scope." },
        { field: "dryRun", role: "explicit-exception", notes: "Compatibility-only mode selector bridging preview/apply wrappers." },
      ],
    },
  ],
  [
    "refactor_astgrep_preview",
    {
      fieldRoleAudit: [
        { field: "pattern", role: "payload-bearing" },
        { field: "rewrite", role: "payload-bearing" },
        { field: "lang", role: "payload-bearing" },
        { field: "path", role: "safety-field", notes: "Explicit repository-confined preview scope." },
      ],
    },
  ],
  [
    "refactor_astgrep_apply",
    {
      fieldRoleAudit: [
        { field: "pattern", role: "payload-bearing" },
        { field: "rewrite", role: "payload-bearing" },
        { field: "lang", role: "payload-bearing" },
        { field: "path", role: "safety-field", notes: "Explicit repository-confined mutation scope." },
      ],
    },
  ],
  [
    "move",
    {
      fieldRoleAudit: [
        { field: "source", role: "payload-bearing", notes: "Repo-confined path operand for all move modes." },
        { field: "destination", role: "payload-bearing", notes: "Repo-confined destination operand for safe/overwrite modes." },
        { field: "overwrite", role: "explicit-exception", notes: "Compatibility-only delegator toggle; split wrappers keep this mutation intent explicit." },
        { field: "trash", role: "explicit-exception", notes: "Compatibility-only delegator toggle; split wrappers keep trash intent explicit." },
      ],
    },
  ],
  ["move_safe", { fieldRoleAudit: [{ field: "source", role: "payload-bearing" }, { field: "destination", role: "payload-bearing" }] }],
  ["move_overwrite", { fieldRoleAudit: [{ field: "source", role: "payload-bearing" }, { field: "destination", role: "payload-bearing" }] }],
  ["move_trash", { fieldRoleAudit: [{ field: "source", role: "payload-bearing", notes: "Repo-confined path moved into .trash/<relative-path>." }] }],
  [
    "workflow_builder",
    {
      fieldRoleAudit: [
        { field: "command", role: "command-selector" },
        {
          field: "workflow_name",
          role: "safety-field",
          commands: ["create", "add_step", "remove_step", "get", "update"],
          notes: "Workflow selector determines which repository workflow JSON file is read or mutated.",
        },
        { field: "description", role: "payload-bearing", commands: ["create"] },
        { field: "version", role: "bounded-option-candidate", commands: ["create"] },
        { field: "workflow_type", role: "bounded-option-candidate", commands: ["create"] },
        { field: "step_json", role: "payload-bearing", commands: ["add_step"] },
        { field: "step_index", role: "required-identifier", commands: ["remove_step"], notes: "Alternative remove_step selector; paired with step_name." },
        { field: "step_name", role: "required-identifier", commands: ["remove_step"], notes: "Alternative remove_step selector; paired with step_index." },
        { field: "position", role: "bounded-option-candidate", commands: ["add_step"] },
        { field: "workflow_json", role: "payload-bearing", commands: ["update", "validate"] },
        { field: "output", role: "bounded-option-candidate" },
      ],
      auditNotes: [
        "Command-scoped meanings are intentional: the same field name can switch between required identifier, payload, or inert sparse optional depending on the selected command.",
      ],
    },
  ],
  [
    "workflow_builder_read",
    {
      fieldRoleAudit: [
        { field: "command", role: "command-selector", notes: "Read-only gate accepts list/get/validate only." },
        {
          field: "workflow_name",
          role: "safety-field",
          commands: ["get"],
          notes: "Workflow selector determines which repository workflow JSON file is read.",
        },
        {
          field: "description",
          role: "explicit-exception",
          commands: ["create"],
          notes: "Schema retained for compatibility pass-through but rejected by the read-wrapper command gate.",
        },
        { field: "version", role: "explicit-exception", commands: ["create"] },
        { field: "workflow_type", role: "explicit-exception", commands: ["create"] },
        { field: "step_json", role: "explicit-exception", commands: ["add_step"] },
        { field: "step_index", role: "explicit-exception", commands: ["remove_step"] },
        { field: "step_name", role: "explicit-exception", commands: ["remove_step"] },
        { field: "position", role: "explicit-exception", commands: ["add_step"] },
        { field: "workflow_json", role: "payload-bearing", commands: ["validate"] },
        { field: "output", role: "bounded-option-candidate" },
      ],
      auditNotes: [
        "Split read wrapper intentionally keeps the broad schema but records mutate-only fields as explicit rejection semantics rather than supported read-surface inputs.",
      ],
    },
  ],
  [
    "workflow_builder_mutate",
    {
      fieldRoleAudit: [
        { field: "command", role: "command-selector", notes: "Mutating gate accepts create/add_step/remove_step/update only." },
        {
          field: "workflow_name",
          role: "safety-field",
          commands: ["create", "add_step", "remove_step", "update"],
          notes: "Workflow selector determines which repository workflow JSON file is mutated.",
        },
        { field: "description", role: "payload-bearing", commands: ["create"] },
        { field: "version", role: "bounded-option-candidate", commands: ["create"] },
        { field: "workflow_type", role: "bounded-option-candidate", commands: ["create"] },
        { field: "step_json", role: "payload-bearing", commands: ["add_step"] },
        { field: "step_index", role: "required-identifier", commands: ["remove_step"] },
        { field: "step_name", role: "required-identifier", commands: ["remove_step"] },
        { field: "position", role: "bounded-option-candidate", commands: ["add_step"] },
        { field: "workflow_json", role: "payload-bearing", commands: ["update"] },
        { field: "output", role: "bounded-option-candidate" },
      ],
      auditNotes: ["Split mutate wrapper intentionally keeps the broad schema but treats read-only command payloads as command-gated explicit rejections."],
    },
  ],
  [
    "create_workspace",
    {
      fieldRoleAudit: [
        { field: "issueNumber", role: "required-identifier" },
        { field: "workflowType", role: "bounded-option-candidate", notes: "Workflow-family selector rather than a broad command surface." },
        { field: "adwId", role: "safety-field", notes: "Resume identifier normalized to canonical lowercase hex before dispatch." },
        { field: "triggeredBy", role: "payload-bearing" },
        { field: "outputMode", role: "bounded-option-candidate" },
      ],
      auditNotes: [
        "Backend companion .opencode/tools/create_workspace.py mirrors the wrapper contract via positional issue_number plus workflow/output selectors.",
        "Ownership remains with E27-M5 for downstream workspace-schema migration; E27-M8-P1 records the baseline only.",
      ],
    },
  ],
  [
    "run_pytest_basic",
    {
      fieldRoleAudit: [
        { field: "minTests", role: "explicit-exception", notes: "Validation guard retained direct for scoped test-count enforcement." },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for timeout enforcement." },
        { field: "cwd", role: "safety-field", notes: "Repository-root-confined execution root for routine test runs." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for routine wrapper toggles." },
        { field: "testPath", role: "payload-bearing", notes: "Direct routine test target operand." },
      ],
      auditNotes: [
        "Split basic wrapper now exposes only compact routine fields in schema space.",
        "Legacy direct routine toggles and advanced-only keys now fail closed in execute() with deterministic migration guidance.",
      ],
    },
  ],
  [
    "run_pytest_advanced",
    {
      fieldRoleAudit: [
        { field: "minTests", role: "explicit-exception", notes: "Validation guard retained direct for scoped test-count enforcement." },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for timeout enforcement." },
        { field: "cwd", role: "safety-field", notes: "Repository-root-confined execution root for advanced test runs." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for advanced wrapper toggles." },
        { field: "testPath", role: "payload-bearing", notes: "Direct routine/advanced test target operand." },
        { field: "pytestArgs", role: "payload-bearing", notes: "Direct pytest passthrough payload." },
        { field: "coverage", role: "explicit-exception", notes: "Direct coverage mode toggle retained across the advanced execution path." },
        { field: "coverageSource", role: "payload-bearing" },
        { field: "coverageThreshold", role: "explicit-exception", notes: "Coverage validation guard retained direct." },
        { field: "overrideIni", role: "payload-bearing" },
      ],
      auditNotes: [
        "Advanced split wrapper now keeps legacy routine/report toggles off the published schema and requires them through bounded options.",
      ],
    },
  ],
  [
    "run_bun_test",
    {
      fieldRoleAudit: [
        { field: "testPath", role: "payload-bearing" },
        { field: "testFilter", role: "bounded-option-candidate" },
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for timeout enforcement." },
        { field: "minTests", role: "explicit-exception", notes: "Validation guard retained direct for test-count enforcement." },
        { field: "cwd", role: "safety-field", notes: "Repository-root-confined execution root for bun test runs." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for bun test wrapper toggles." },
        { field: "failFast", role: "bounded-option-candidate" },
      ],
      auditNotes: [
        "Bun wrapper keeps legacy output/filter/fail-fast toggles off the published schema and requires them through bounded options.",
      ],
    },
  ],
  [
    "run_ctest",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "buildDir", role: "required-identifier", notes: "Required build-directory selector for the target CTest context." },
        { field: "testFilter", role: "bounded-option-candidate" },
        { field: "excludeFilter", role: "bounded-option-candidate" },
        { field: "parallel", role: "bounded-option-candidate" },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for timeout enforcement." },
        { field: "minTests", role: "explicit-exception", notes: "Validation guard retained direct for minimum-count enforcement." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/filter/parallel toggles." },
      ],
    },
  ],
  [
    "run_cmake",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "preset", role: "payload-bearing", notes: "Compatibility configure/build context selector when preset-driven flows are used." },
        { field: "sourceDir", role: "payload-bearing" },
        { field: "buildDir", role: "payload-bearing" },
        { field: "ninja", role: "bounded-option-candidate" },
        { field: "build", role: "explicit-exception", notes: "Compatibility-only mode selector bridging configure-only and build-only split wrappers." },
        { field: "jobs", role: "bounded-option-candidate" },
        { field: "buildTimeout", role: "explicit-exception", notes: "Build-phase timeout guard retained direct on the compatibility wrapper." },
        { field: "timeout", role: "explicit-exception", notes: "Configure-phase timeout guard retained direct on the compatibility wrapper." },
        { field: "cmakeArgs", role: "payload-bearing" },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/ninja/jobs compatibility toggles." },
      ],
      auditNotes: [
        "P3 follow-up covers run_ctest, run_cmake*, run_linters, run_validate_agent_references, build_mkdocs*, and clear_build* schema-budget migration work; E27-M9 records the baseline only.",
      ],
    },
  ],
  [
    "run_cmake_configure",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "preset", role: "payload-bearing" },
        { field: "sourceDir", role: "payload-bearing" },
        { field: "buildDir", role: "payload-bearing" },
        { field: "ninja", role: "bounded-option-candidate" },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for timeout enforcement." },
        { field: "cmakeArgs", role: "payload-bearing" },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/ninja configure toggles." },
      ],
    },
  ],
  [
    "run_cmake_build",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "preset", role: "required-identifier", notes: "Alternative build-context selector paired with buildDir." },
        { field: "buildDir", role: "required-identifier", notes: "Alternative build-context selector paired with preset." },
        { field: "jobs", role: "bounded-option-candidate" },
        { field: "buildTimeout", role: "explicit-exception", notes: "Build timeout guard retained direct." },
        { field: "timeout", role: "explicit-exception", notes: "Wrapper execution timeout guard retained direct." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/jobs build toggles." },
      ],
    },
  ],
  [
    "run_linters",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "autoFix", role: "safety-field", notes: "Mutating vs validation-only mode boundary must remain explicit." },
        { field: "linters", role: "bounded-option-candidate" },
        { field: "targetDir", role: "safety-field", notes: "Direct path scope for linter execution." },
        { field: "ruffTimeout", role: "explicit-exception", notes: "Tool-specific timeout guard retained direct." },
        { field: "mypyTimeout", role: "explicit-exception", notes: "Tool-specific timeout guard retained direct." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/linters toggles." },
      ],
    },
  ],
  [
    "run_validate_agent_references",
    {
      fieldRoleAudit: [
        { field: "cwd", role: "safety-field", notes: "Exact-root validation target; wrapper fails closed unless cwd resolves to the current repository/worktree root exactly." },
        { field: "baselinePath", role: "safety-field", notes: "Repo-relative committed baseline path confined to .opencode/guides/ for auditable suppressions." },
      ],
    },
  ],
  [
    "build_mkdocs",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for mkdocs wrapper execution." },
        { field: "cwd", role: "safety-field", notes: "Repository-root-confined docs build root." },
        { field: "strict", role: "bounded-option-candidate" },
        { field: "clean", role: "bounded-option-candidate" },
        { field: "configFile", role: "safety-field", notes: "Repository-root-confined mkdocs config path." },
        { field: "validateOnly", role: "explicit-exception", notes: "Compatibility-only mode selector bridging validate-only and artifact-producing split wrappers." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/strict/clean toggles." },
      ],
    },
  ],
  [
    "build_mkdocs_validate",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for mkdocs wrapper execution." },
        { field: "cwd", role: "safety-field", notes: "Repository-root-confined docs validation root." },
        { field: "strict", role: "bounded-option-candidate" },
        { field: "clean", role: "bounded-option-candidate" },
        { field: "configFile", role: "safety-field", notes: "Repository-root-confined mkdocs config path." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/strict/clean toggles." },
      ],
    },
  ],
  [
    "build_mkdocs_build",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "timeout", role: "explicit-exception", notes: "Validation guard retained direct for mkdocs wrapper execution." },
        { field: "cwd", role: "safety-field", notes: "Repository-root-confined docs build root." },
        { field: "strict", role: "bounded-option-candidate" },
        { field: "clean", role: "bounded-option-candidate" },
        { field: "configFile", role: "safety-field", notes: "Repository-root-confined mkdocs config path." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output/strict/clean toggles." },
      ],
    },
  ],
  [
    "clear_build",
    {
      fieldRoleAudit: [
        { field: "buildDir", role: "safety-field", notes: "Repo-confined destructive target path; build cleanup scope must remain explicit even when preview/delete mode is selected elsewhere." },
        { field: "dryRun", role: "explicit-exception", notes: "Compatibility-only preview/delete selector retained direct to preserve the split-wrapper boundary." },
        { field: "force", role: "safety-field", notes: "Destructive authorization gate must remain explicit." },
      ],
    },
  ],
  [
    "clear_build_preview",
    {
      fieldRoleAudit: [{ field: "buildDir", role: "safety-field", notes: "Repo-confined cleanup target path for read-only preview scope." }],
    },
  ],
  [
    "clear_build_delete",
    {
      fieldRoleAudit: [
        { field: "buildDir", role: "safety-field", notes: "Repo-confined destructive target path for explicit delete mode." },
        { field: "force", role: "safety-field", notes: "Destructive authorization gate must remain explicit." },
      ],
    },
  ],
  [
    "validate_notebook",
    {
      fieldRoleAudit: [
        { field: "notebookPath", role: "payload-bearing", notes: "Primary notebook or directory operand across validation, conversion, and sync flows." },
        { field: "recursive", role: "safety-field", notes: "Directory traversal scope remains explicit for notebook tree operations." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for validation-only controls on the compatibility wrapper." },
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "skipSyntax", role: "bounded-option-candidate" },
        { field: "validationMode", role: "bounded-option-candidate" },
        { field: "fast", role: "bounded-option-candidate" },
        { field: "full", role: "bounded-option-candidate" },
        { field: "convertToPy", role: "explicit-exception", notes: "Compatibility-only direct mode selector bridging the dedicated conversion wrapper." },
        { field: "convertToIpynb", role: "explicit-exception", notes: "Compatibility-only direct mode selector bridging the dedicated conversion wrapper." },
        { field: "sync", role: "explicit-exception", notes: "Compatibility-only direct mode selector bridging the dedicated sync wrapper." },
        { field: "checkSync", role: "explicit-exception", notes: "Read-only sync-check selector remains explicit because it switches validation vs sync-state semantics." },
        { field: "outputDir", role: "explicit-exception", notes: "Compatibility conversion-output operand retained direct to preserve split-wrapper parity and rejection behavior elsewhere." },
      ],
      auditNotes: [
        "Compatibility notebook wrapper intentionally preserves convert/sync direct toggles as an explicit bridge to the split notebook family; E27-M10 records the audit baseline only.",
      ],
    },
  ],
  [
    "validate_notebook_readonly",
    {
      fieldRoleAudit: [
        { field: "notebookPath", role: "payload-bearing", notes: "Primary read-only notebook or directory operand." },
        { field: "recursive", role: "safety-field", notes: "Directory traversal scope remains explicit for read-only notebook validation." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for read-only validation-only controls." },
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "skipSyntax", role: "bounded-option-candidate" },
        { field: "validationMode", role: "bounded-option-candidate" },
        { field: "fast", role: "bounded-option-candidate" },
        { field: "full", role: "bounded-option-candidate" },
        { field: "checkSync", role: "explicit-exception", notes: "Explicit read-only mode selector with conflict checking distinct from validation-output flags." },
      ],
      auditNotes: [
        "Read-only split wrapper omits mutating keys from its schema and keeps check-sync semantics explicit instead of folding them into a generic bounded-options bucket.",
      ],
    },
  ],
  [
    "convert_notebook_to_py",
    {
      fieldRoleAudit: [
        { field: "notebookPath", role: "payload-bearing", notes: "Notebook or directory operand for py:percent conversion." },
        { field: "recursive", role: "safety-field", notes: "Directory traversal scope remains explicit for bulk conversion." },
        { field: "outputDir", role: "safety-field", notes: "Conversion output directory remains explicit because the backend resolves it under the repository root before writing scripts." },
      ],
    },
  ],
  [
    "convert_py_to_notebook",
    {
      fieldRoleAudit: [
        { field: "notebookPath", role: "payload-bearing", notes: "py:percent script or directory operand for notebook conversion." },
        { field: "recursive", role: "safety-field", notes: "Directory traversal scope remains explicit for bulk conversion." },
        { field: "outputDir", role: "safety-field", notes: "Conversion output directory remains explicit because the backend resolves it under the repository root before writing notebooks." },
      ],
    },
  ],
  [
    "sync_notebook_pair",
    {
      fieldRoleAudit: [
        { field: "notebookPath", role: "payload-bearing", notes: "Notebook/script operand for pair synchronization." },
        { field: "recursive", role: "safety-field", notes: "Directory traversal scope remains explicit for bulk sync operations." },
        { field: "outputDir", role: "explicit-exception", notes: "Schema key is retained only for deterministic rejection because sync does not support conversion output targets." },
      ],
      auditNotes: [
        "Split sync wrapper keeps the unsupported outputDir key only to fail closed with conversion-wrapper guidance.",
      ],
    },
  ],
  [
    "run_notebook",
    {
      fieldRoleAudit: [
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output-mode selection." },
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "notebookPath", role: "payload-bearing", notes: "Primary notebook/script/directory execution operand." },
        { field: "recursive", role: "safety-field", notes: "Directory traversal scope remains explicit for bulk execution." },
        { field: "script", role: "safety-field", notes: "Execution-mode selector remains explicit because it switches directory collection from notebooks to scripts." },
        { field: "timeout", role: "explicit-exception", notes: "Execution timeout guard remains direct for fail-closed validation." },
        { field: "expectOutput", role: "payload-bearing", notes: "Execution-output expectations remain direct structured payload input." },
        { field: "cwd", role: "safety-field", notes: "Execution root remains explicit, but the backend only checks that cwd exists as a directory; it does not repository-confine the path." },
        { field: "writeExecuted", role: "safety-field", notes: "Executed-notebook output directory remains explicit because the backend creates/uses the provided directory without repository confinement (and ignores it in script mode)." },
        { field: "noOverwrite", role: "safety-field", notes: "Overwrite behavior toggle remains explicit because it changes mutation semantics and backup policy." },
        { field: "noBackup", role: "safety-field", notes: "Backup suppression remains explicit because it weakens the default safety path." },
        { field: "skipValidation", role: "safety-field", notes: "Validation-bypass toggle remains explicit because it relaxes the normal pre-execution guardrail." },
      ],
      auditNotes: [
        "E27-M10 records the notebook execution baseline only; overwrite, backup, and validation-bypass controls stay explicit so later slimming does not hide execution-safety decisions.",
        "run_notebook path notes intentionally distinguish repository-confined notebook conversion outputs from cwd/writeExecuted inputs, which are validated for existence or directory shape but not repository confinement.",
      ],
    },
  ],
  [
    "run_cpp_lint_check",
    {
      fieldRoleAudit: [
        { field: "sourceDir", role: "safety-field", notes: "Repository-confined C++ source root for non-mutating lint checks." },
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for clang-tidy compile_commands access." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output and linter selection." },
      ],
      auditNotes: [
        "Split check wrapper is intentionally non-mutating and now routes output/linter selection through bounded options without exposing an auto-fix toggle.",
      ],
    },
  ],
  [
    "run_cpp_lint_fix",
    {
      fieldRoleAudit: [
        { field: "sourceDir", role: "safety-field", notes: "Repository-confined C++ source root for mutating lint-fix runs." },
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for clang-tidy compile_commands access." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output and linter selection." },
      ],
      auditNotes: [
        "Split fix wrapper preserves always-on auto-fix semantics without reintroducing a direct mode toggle in the schema surface.",
      ],
    },
  ],
  [
    "run_cpp_linters",
    {
      fieldRoleAudit: [
        { field: "sourceDir", role: "safety-field", notes: "Repository-confined C++ source root for compatibility lint runs." },
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for clang-tidy compile_commands access." },
        { field: "autoFix", role: "explicit-exception", notes: "Compatibility-only direct bridge between the split check and fix wrappers." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output and linter selection." },
      ],
      auditNotes: [
        "Compatibility C++ lint wrapper keeps autoFix explicit as a bridge surface while moving output/linter selection behind bounded options.",
      ],
    },
  ],
  [
    "run_cpp_coverage_summary",
    {
      fieldRoleAudit: [
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for routine coverage artifacts." },
        { field: "threshold", role: "explicit-exception", notes: "Coverage threshold guard remains direct for fail-closed validation." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
        { field: "tool", role: "explicit-exception", notes: "Advanced-only schema key intentionally retained for deterministic rejection in the summary wrapper." },
        { field: "filter", role: "explicit-exception", notes: "Advanced-only schema key intentionally retained for deterministic rejection in the summary wrapper." },
        { field: "html", role: "explicit-exception", notes: "Advanced-only schema key intentionally retained for deterministic rejection in the summary wrapper." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output-mode selection." },
      ],
      auditNotes: [
        "Routine coverage wrapper now uses bounded options for output-mode selection while intentionally keeping advanced-only keys in schema space only to fail closed and preserve the summary vs advanced split boundary.",
        "extraArgs is intentionally excluded from the shipped C++ coverage wrapper family until backend support exists.",
      ],
    },
  ],
  [
    "run_cpp_coverage_advanced",
    {
      fieldRoleAudit: [
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for advanced coverage artifacts." },
        { field: "threshold", role: "explicit-exception", notes: "Coverage threshold guard remains direct for fail-closed validation." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
        { field: "filter", role: "payload-bearing", notes: "Coverage include/filter selector remains direct freeform payload input." },
        { field: "html", role: "safety-field", notes: "Repository-confined HTML output directory remains explicit." },
        { field: "options", role: "explicit-exception", notes: "Bounded token carrier for output-mode and coverage-tool selection." },
      ],
      auditNotes: [
        "Advanced coverage wrapper routes output-mode and coverage-tool selection through bounded options while preserving direct filter/html safety boundaries.",
        "Validated buildDir/html paths are canonicalized before subprocess assembly, and nested new repo-confined html output directories remain allowed.",
        "extraArgs is intentionally excluded from the shipped C++ coverage wrapper family until backend support exists.",
      ],
    },
  ],
  [
    "run_sanitizers_basic",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for sanitizer-enabled binaries." },
        { field: "executable", role: "safety-field", notes: "Executable path remains explicit because the wrapper validates repo/buildDir confinement and file type." },
        { field: "sanitizer", role: "payload-bearing", notes: "Required sanitizer mode selector for routine runs." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
      ],
      auditNotes: [
        "Routine sanitizer wrapper keeps only shipped direct fields in counted inventory output.",
        "Advanced-only rejected keys remain part of runtime rejection behavior, but inventory output no longer represents them as accepted direct fields.",
      ],
    },
  ],
  [
    "run_sanitizers_advanced",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for sanitizer-enabled binaries." },
        { field: "executable", role: "safety-field", notes: "Executable path remains explicit because the wrapper validates repo/buildDir confinement and file type." },
        { field: "sanitizer", role: "payload-bearing", notes: "Required sanitizer mode selector for advanced runs." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
        { field: "suppressions", role: "safety-field", notes: "Optional suppressions file path remains explicit because it points to filesystem input that the backend reads directly without repository confinement." },
        { field: "options", role: "explicit-exception", notes: "Sanitizer environment-options string remains explicit because it is both powerful and wrapper-specific." },
        { field: "normalDuration", role: "explicit-exception", notes: "Positive-duration validation guard remains direct for fail-closed overhead calculations." },
        { field: "extraArgs", role: "payload-bearing", notes: "Executable passthrough arguments remain direct advanced payload input." },
      ],
      auditNotes: [
        "Advanced sanitizer wrapper records the live payload/safety split only; E27-M10 does not migrate the schema surface.",
      ],
    },
  ],
  [
    "run_sanitizers",
    {
      fieldRoleAudit: [
        { field: "outputMode", role: "bounded-option-candidate" },
        { field: "buildDir", role: "safety-field", notes: "Repository-confined build directory for sanitizer-enabled binaries." },
        { field: "executable", role: "safety-field", notes: "Executable path remains explicit because the wrapper validates repo/buildDir confinement and file type." },
        { field: "sanitizer", role: "payload-bearing", notes: "Required sanitizer mode selector on the compatibility bridge." },
        { field: "timeout", role: "explicit-exception", notes: "Timeout guard remains direct for fail-closed validation." },
        { field: "suppressions", role: "safety-field", notes: "Compatibility bridge forwards the suppressions file path as an explicit path-like safety input." },
        { field: "options", role: "explicit-exception", notes: "Compatibility bridge keeps the sanitizer options string explicit instead of hiding it behind a bounded carrier." },
        { field: "normalDuration", role: "explicit-exception", notes: "Positive-duration validation guard remains direct for fail-closed overhead calculations." },
        { field: "extraArgs", role: "payload-bearing", notes: "Compatibility bridge forwards executable passthrough arguments." },
      ],
      auditNotes: [
        "Compatibility sanitizer wrapper is a bridge surface, not the target end-state; E27-M10 records its direct-field rationale without changing runtime behavior.",
      ],
    },
  ],
]);
const ADW_COMMAND_WRAPPER_NAMES = new Set([
  "adw",
  "adw_status_health",
  "adw_setup",
  "adw_service",
  "create_workspace",
]);
const WRAPPER_FAMILY_SPECS: WrapperFamilySpec[] = [
  {
    ownerPlan: "E27-M3",
    matches: (basename) => basename === "platform_operations" || basename.startsWith("platform_"),
    oversizedClassification: "split-needed",
    testCandidateBasenames: ["platform_operations_compat_comment_pr_review"],
    testBasenamePrefixes: ["platform_"],
  },
  {
    ownerPlan: "E27-M4",
    matches: (basename) => basename.startsWith("git_"),
    oversizedClassification: "split-needed",
    testCandidateBasenames: ["git_diff", "git_merge", "git_stage", "git_commit", "git_branch", "git_worktree"],
    testBasenamePrefixes: ["git_"],
  },
  {
    ownerPlan: "E27-M5",
    matches: (basename) => ADW_COMMAND_WRAPPER_NAMES.has(basename),
    oversizedClassification: "split-needed",
    testCandidateBasenames: ["adw", "adw_status_health", "adw_setup", "adw_service"],
    testBasenamePrefixes: ["adw_"],
  },
  {
    ownerPlan: "E27-M6",
    matches: (basename) => basename.startsWith("adw_plans"),
    oversizedClassification: "bounded-options-needed",
    testCandidateBasenames: [
      "adw_plans_compat_required_args",
      "adw_plans_contract_shared",
      "adw_plans_read_required_args",
      "adw_plans_mutate_required_args",
    ],
    testBasenamePrefixes: ["adw_plans_"],
  },
  {
    ownerPlan: "E27-M6",
    matches: (basename) => basename.startsWith("adw_spec"),
    oversizedClassification: "split-needed",
    testCandidateBasenames: ["adw_spec", "adw_spec_read", "adw_spec_write", "adw_spec_messages", "adw_spec_shared"],
    testBasenamePrefixes: ["adw_spec_"],
  },
  {
    ownerPlan: "E27-M6",
    matches: (basename) => basename.startsWith("adw_notes"),
    oversizedClassification: "split-needed",
    testBasenamePrefixes: ["adw_notes_"],
  },
  {
    ownerPlan: "E27-M7",
    matches: (basename) => basename.startsWith("adw_issues_") || basename.startsWith("auto_mode_manifest"),
    oversizedClassification: "split-needed",
    testCandidateBasenames: [
      "adw_issues_spec",
      "adw_issues_batch_init",
      "adw_issues_batch_read",
      "adw_issues_batch_read_write",
      "adw_issues_batch_log_summary",
      "auto_mode_manifest",
    ],
    testBasenamePrefixes: ["adw_issues_", "auto_mode_manifest"],
  },
  {
    ownerPlan: "E27-M8",
    matches: (basename) =>
      basename === "find_files" ||
      basename === "search_content" ||
      basename.startsWith("ripgrep") ||
      basename.startsWith("refactor_astgrep") ||
      basename === "move" ||
      basename.startsWith("move_") ||
      basename === "workflow_builder" ||
      basename.startsWith("workflow_builder_"),
    oversizedClassification: "split-needed",
    testCandidateBasenames: [
      "ripgrep_advanced",
      "search_content",
      "find_files",
      "refactor_astgrep_apply",
      "move_safe",
    ],
    testBasenamePrefixes: ["ripgrep", "refactor_astgrep", "move_", "workflow_builder_"],
  },
  {
    ownerPlan: "E27-M9",
    matches: (basename) =>
      basename === "run_pytest_basic" ||
      basename === "run_pytest_advanced" ||
      basename === "run_bun_test" ||
      basename === "run_linters" ||
      basename === "run_validate_agent_references" ||
      basename.startsWith("build_mkdocs") ||
      basename.startsWith("clear_build") ||
      basename.startsWith("run_cmake") ||
      basename === "run_ctest",
    oversizedClassification: "split-needed",
    testCandidateBasenames: ["run_pytest_basic", "run_linters", "run_bun_test", "run_validate_agent_references"],
    testBasenamePrefixes: ["run_pytest_basic", "run_pytest_advanced", "build_mkdocs", "clear_build", "run_cmake"],
  },
  {
    ownerPlan: "E27-M10",
    matches: (basename) =>
      basename.startsWith("validate_notebook") ||
      (basename.startsWith("convert_") && basename.includes("notebook")) ||
      basename === "run_notebook" ||
      basename === "sync_notebook_pair" ||
      basename.startsWith("run_cpp_") ||
      basename.startsWith("run_sanitizers"),
    oversizedClassification: "split-needed",
    testCandidateBasenames: ["validate_notebook_readonly", "run_sanitizers_basic"],
    testBasenamePrefixes: ["validate_notebook", "run_cpp_", "run_sanitizers", "convert_", "sync_notebook_pair"],
  },
  {
    ownerPlan: "E27-M11",
    matches: (basename) =>
      basename === "get_datetime" || basename === "get_version" || basename === "feedback_log",
    oversizedClassification: "compatibility-defer-with-owner",
    testCandidateBasenames: ["feedback_log", "get_datetime", "get_version"],
  },
];

/**
 * Enumerates the committed `.opencode/tools/*.ts` audit surface and computes
 * direct-field counts using the E27-M1 rule: count top-level `args` keys,
 * always count `command`, exempt usage-only `help`, and exempt `options` only
 * when the wrapper clearly documents and deterministically parses bounded
 * command-scoped tokens.
 */
export async function generateWrapperSchemaInventory(
  repoRoot = getRepositoryRoot(),
): Promise<WrapperSchemaInventoryArtifact> {
  const context = await buildInventoryContext(repoRoot);
  const wrapperPaths = await listWrapperSourcePaths(context.toolsDir);
  const rows: WrapperSchemaInventoryRow[] = [];

  for (const wrapperPath of wrapperPaths) {
    try {
      const row = await inspectWrapperPath(wrapperPath, context);
      if (!shouldIncludeInventoryRow(row, context)) {
        continue;
      }
      rows.push(row);
    } catch (error) {
      rows.push(buildUnhandledFailureRow(wrapperPath, context, error));
    }
  }

  return {
    audit_scope: ".opencode/tools/*.ts",
    counting_rule: "direct_args_fields_with_help_and_bounded_options_exempt",
    rows,
  };
}

/** Writes the committed inventory artifact using temp-file staging. */
export async function writeWrapperSchemaInventoryArtifact(
  repoRoot = getRepositoryRoot(),
): Promise<WrapperSchemaInventoryArtifact> {
  const artifact = await generateWrapperSchemaInventory(repoRoot);
  const artifactPath = path.join(repoRoot, INVENTORY_ARTIFACT_PATH);
  const tempDir = path.join(repoRoot, TEMP_OUTPUT_DIR);
  const serialized = `${JSON.stringify(artifact, null, 2)}\n`;
  const tempPath = path.join(tempDir, `wrapper_schema_inventory.${process.pid}.tmp`);

  await assertSafeOutputPath(repoRoot, artifactPath);
  await assertSafeOutputPath(repoRoot, tempPath);
  await mkdir(tempDir, { recursive: true });
  await assertSafeOutputPath(repoRoot, tempPath);
  await writeFile(tempPath, serialized, "utf8");
  await rename(tempPath, artifactPath);
  return artifact;
}

export async function inspectWrapperSourceFile(
  absolutePath: string,
  repoRoot = getRepositoryRoot(),
): Promise<WrapperSchemaInventoryRow> {
  const context = await buildInventoryContext(repoRoot);
  return inspectWrapperPath(absolutePath, context);
}

export function inspectWrapperSourceText(
  absolutePath: string,
  sourceText: string,
  context: Pick<
    InventoryContext,
    | "repoRoot"
    | "docsByBasename"
    | "compatibilityWrappers"
    | "metadataDiagnostics"
    | "testPathsByBasename"
  >,
): WrapperSchemaInventoryRow {
  const wrapperPath = toRepoRelativePath(context.repoRoot, absolutePath);
  const basename = path.basename(absolutePath, ".ts");
  const companionDocPath = context.docsByBasename.get(basename) ?? null;
  const wrapperKind = classifyWrapperKind(basename, context.compatibilityWrappers);

  if (context.metadataDiagnostics.length > 0) {
    return buildInspectionFailedRow(
      wrapperPath,
      basename,
      companionDocPath,
      wrapperKind,
      `Supporting inventory metadata unavailable: ${context.metadataDiagnostics.join("; ")}`,
      context.testPathsByBasename,
    );
  }

  if (!TOOL_REGISTRATION_PATTERN.test(sourceText)) {
    return buildNoToolRow(
      wrapperPath,
      basename,
      companionDocPath,
      wrapperKind,
      context.testPathsByBasename,
    );
  }

  try {
    const argsObjectText = extractArgsObjectText(sourceText);
    if (!argsObjectText) {
      return buildNoToolRow(
        wrapperPath,
        basename,
        companionDocPath,
        wrapperKind,
        context.testPathsByBasename,
      );
    }

    const directFields = augmentInventoryDirectFields(basename, extractTopLevelArgsKeys(argsObjectText));
    const exemptFields = directFields.filter((field) => isExemptField(field, basename));
    const countedFields = directFields.filter((field) => !exemptFields.includes(field));

    return enrichInventoryRow(
      {
        wrapper_path: wrapperPath,
        companion_doc_path: companionDocPath,
        wrapper_kind: wrapperKind,
        status: "ok",
        current_shape: "direct_args_object",
        counted_fields: countedFields,
        exempt_fields: exemptFields,
        current_count: countedFields.length,
        diagnostic: null,
      },
      basename,
      directFields,
      context,
    );
  } catch (error) {
    return buildInspectionFailedRow(
      wrapperPath,
      basename,
      companionDocPath,
      wrapperKind,
      sanitizeDiagnostic(context.repoRoot, error),
      context.testPathsByBasename,
    );
  }
}

export function getRepositoryRoot(): string {
  return path.resolve(import.meta.dir, "../../..");
}

async function buildInventoryContext(repoRoot: string): Promise<InventoryContext> {
  const toolsDir = path.join(repoRoot, ".opencode/tools");
  const [docsResult, compatibilityResult, testPathsResult] = await Promise.allSettled([
    readCompanionDocsMap(toolsDir, repoRoot),
    readCompatibilityWrappers(repoRoot),
    readTestPathsMap(repoRoot),
  ] as const);
  const metadataDiagnostics: string[] = [];
  const docsByBasename =
    docsResult.status === "fulfilled"
      ? docsResult.value
      : (recordMetadataFailure(
          metadataDiagnostics,
          repoRoot,
          "companion docs",
          docsResult.reason,
        ),
        new Map<string, string>());
  const compatibilityWrappers =
    compatibilityResult.status === "fulfilled"
      ? compatibilityResult.value
      : (recordMetadataFailure(
          metadataDiagnostics,
          repoRoot,
          "compatibility wrapper metadata",
          compatibilityResult.reason,
        ),
        new Set<string>());
  const testPathsByBasename =
    testPathsResult.status === "fulfilled"
      ? testPathsResult.value
      : (recordMetadataFailure(metadataDiagnostics, repoRoot, "wrapper tests", testPathsResult.reason),
        new Map<string, string>());

  return {
    repoRoot,
    toolsDir,
    docsByBasename,
    compatibilityWrappers,
    testPathsByBasename,
    metadataDiagnostics,
  };
}

async function inspectWrapperPath(
  absolutePath: string,
  context: InventoryContext,
): Promise<WrapperSchemaInventoryRow> {
  const wrapperPath = toRepoRelativePath(context.repoRoot, absolutePath);
  const basename = path.basename(absolutePath, ".ts");
  const companionDocPath = context.docsByBasename.get(basename) ?? null;
  const wrapperKind = classifyWrapperKind(basename, context.compatibilityWrappers);

  try {
    const sourceText = await readBoundedUtf8File(absolutePath, context.repoRoot);
    return inspectWrapperSourceText(absolutePath, sourceText, context);
  } catch (error) {
    return buildInspectionFailedRow(
      wrapperPath,
      basename,
      companionDocPath,
      wrapperKind,
      `Unable to read wrapper source: ${sanitizeDiagnostic(context.repoRoot, error)}`,
      context.testPathsByBasename,
    );
  }
}

async function listWrapperSourcePaths(toolsDir: string): Promise<string[]> {
  let entries;

  try {
    entries = await readdir(toolsDir, { withFileTypes: true });
  } catch (error) {
    if (isMissingDirectoryError(error)) {
      return [];
    }
    throw error;
  }

  return entries
    .filter((entry) => (entry.isFile() || entry.isSymbolicLink()) && entry.name.endsWith(".ts"))
    .map((entry) => path.join(toolsDir, entry.name))
    .sort((left, right) => left.localeCompare(right));
}

function shouldIncludeInventoryRow(
  row: Pick<WrapperSchemaInventoryRow, "status" | "wrapper_path" | "companion_doc_path" | "wrapper_kind">,
  context: Pick<InventoryContext, "compatibilityWrappers">,
): boolean {
  if (row.status !== "no_tool_schema") {
    return true;
  }

  const basename = path.basename(row.wrapper_path, ".ts");
  return row.companion_doc_path !== null || isDeclaredWrapperBasename(basename, context.compatibilityWrappers);
}

function isDeclaredWrapperBasename(basename: string, compatibilityWrappers: ReadonlySet<string>): boolean {
  return (
    COMPATIBILITY_WRAPPER_NAMES.has(basename) ||
    SPLIT_WRAPPER_NAMES.has(basename) ||
    compatibilityWrappers.has(basename)
  );
}

async function readCompanionDocsMap(
  toolsDir: string,
  repoRoot: string,
): Promise<Map<string, string>> {
  let entries;

  try {
    entries = await readdir(toolsDir, { withFileTypes: true });
  } catch (error) {
    if (isMissingDirectoryError(error)) {
      return new Map<string, string>();
    }
    throw error;
  }

  const docs = new Map<string, string>();

  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".md")) {
      continue;
    }

    docs.set(
      path.basename(entry.name, ".md"),
      toRepoRelativePath(repoRoot, path.join(toolsDir, entry.name)),
    );
  }

  return docs;
}

async function readCompatibilityWrappers(repoRoot: string): Promise<Set<string>> {
  const raw = await readBoundedUtf8File(path.join(repoRoot, COMPATIBILITY_GUIDE_PATH), repoRoot);
  const parsed = JSON.parse(raw) as { wrappers?: Array<{ name?: unknown; status?: unknown }> };
  const names = new Set<string>();

  for (const wrapper of parsed.wrappers ?? []) {
    if (wrapper.status !== "exception_approved") {
      continue;
    }
    if (typeof wrapper.name === "string" && wrapper.name.trim()) {
      names.add(wrapper.name.trim());
    }
  }

  return names;
}

async function readTestPathsMap(repoRoot: string): Promise<Map<string, string>> {
  const testsDir = path.join(repoRoot, ".opencode/tools/__tests__");
  let entries;

  try {
    entries = await readdir(testsDir, { withFileTypes: true });
  } catch (error) {
    if (isMissingDirectoryError(error)) {
      return new Map<string, string>();
    }
    throw error;
  }

  const testsByBasename = new Map<string, string>();

  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".test.ts")) {
      continue;
    }

    testsByBasename.set(
      path.basename(entry.name, ".test.ts"),
      toRepoRelativePath(repoRoot, path.join(testsDir, entry.name)),
    );
  }

  return testsByBasename;
}

type RowAuditFields = Omit<
  WrapperSchemaInventoryRow,
  "classification" | "target_shape" | "target_count" | "owner_plan" | "nearest_test_path"
>;

function enrichInventoryRow(
  row: RowAuditFields,
  basename: string,
  directFields: string[] | null,
  context: Pick<InventoryContext, "testPathsByBasename">,
): WrapperSchemaInventoryRow {
  try {
    const classification = classifyInventoryRow(row, basename);
    const ownerPlan =
      classification === "already-compliant" ? null : resolveOwnerPlanForBasename(basename);

    return {
      ...row,
      classification,
      target_shape: resolveTargetShape(row, classification),
      target_count: resolveTargetCount(row, basename, classification),
      owner_plan: ownerPlan,
      nearest_test_path: resolveNearestTestPath(basename, context.testPathsByBasename),
      ...resolveOptionalAuditMetadata(basename, directFields),
    };
  } catch (error) {
    return buildDerivedInspectionFailureRow(row, basename, context, error);
  }
}

function buildNoToolRow(
  wrapperPath: string,
  basename: string,
  companionDocPath: string | null,
  wrapperKind: WrapperKind,
  testPathsByBasename: Map<string, string>,
): WrapperSchemaInventoryRow {
  return enrichInventoryRow(
    {
      wrapper_path: wrapperPath,
      companion_doc_path: companionDocPath,
      wrapper_kind: wrapperKind,
      status: "no_tool_schema",
      current_shape: "no_tool_schema",
      counted_fields: [],
      exempt_fields: [],
      current_count: 0,
      diagnostic: null,
    },
    basename,
    null,
    { testPathsByBasename },
  );
}

function buildInspectionFailedRow(
  wrapperPath: string,
  basename: string,
  companionDocPath: string | null,
  wrapperKind: WrapperKind,
  diagnostic: string,
  testPathsByBasename: Map<string, string>,
): WrapperSchemaInventoryRow {
  return enrichInventoryRow(
    {
      wrapper_path: wrapperPath,
      companion_doc_path: companionDocPath,
      wrapper_kind: wrapperKind,
      status: "inspection_failed",
      current_shape: "inspection_failed",
      counted_fields: [],
      exempt_fields: [],
      current_count: null,
      diagnostic,
    },
    basename,
    null,
    { testPathsByBasename },
  );
}

function classifyWrapperKind(
  basename: string,
  compatibilityWrappers: Set<string>,
): WrapperKind {
  if (COMPATIBILITY_WRAPPER_NAMES.has(basename) || compatibilityWrappers.has(basename)) {
    return "compatibility";
  }
  if (SPLIT_WRAPPER_NAMES.has(basename)) {
    return "split";
  }
  return "unsplit";
}

export function classifyInventoryRow(
  row: Pick<WrapperSchemaInventoryRow, "status" | "current_count">,
  basename: string,
): WrapperClassification {
  if (row.status === "no_tool_schema") {
    return "already-compliant";
  }
  if (row.status === "ok" && row.current_count !== null && row.current_count <= 4) {
    return "already-compliant";
  }
  return resolveOversizedClassificationForBasename(basename);
}

export function resolveOwnerPlanForBasename(basename: string): WrapperOwnerPlan {
  const matches = WRAPPER_FAMILY_SPECS.filter((rule) => rule.matches(basename)).map((rule) => rule.ownerPlan);

  if (matches.length !== 1) {
    throw new Error(
      `Expected exactly one owner plan for ${basename}; found ${matches.length}: ${matches.join(", ") || "none"}.`,
    );
  }

  return matches[0]!;
}

export function resolveNearestTestPath(
  basename: string,
  testPathsByBasename: ReadonlyMap<string, string>,
): string | null {
  const exactMatch = testPathsByBasename.get(basename);
  if (exactMatch) {
    return exactMatch;
  }

  for (const candidate of WRAPPER_TEST_FALLBACK_CANDIDATES.get(basename) ?? []) {
    const candidateMatch = testPathsByBasename.get(candidate);
    if (candidateMatch) {
      return candidateMatch;
    }
  }

  const orderedTestEntries = [...testPathsByBasename.entries()].sort(([left], [right]) =>
    left.localeCompare(right),
  );

  for (const prefix of WRAPPER_TEST_FALLBACK_PREFIXES.get(basename) ?? []) {
    const prefixMatch = orderedTestEntries.find(([candidate]) => candidate.startsWith(prefix))?.[1];
    if (prefixMatch) {
      return prefixMatch;
    }
  }

  if (AUDIT_ONLY_TEST_COVERED_WRAPPERS.has(basename)) {
    return testPathsByBasename.get("wrapper_schema_inventory") ?? null;
  }

  return null;
}

function resolveFieldRoleAudit(basename: string): WrapperFieldRoleAudit[] | undefined {
  const baseline = WRAPPER_AUDIT_BASELINES.get(basename);
  return baseline ? baseline.fieldRoleAudit.map((entry) => ({ ...entry })) : undefined;
}

function validateFieldRoleAuditParity(
  basename: string,
  directFields: string[] | null,
  fieldRoleAudit: WrapperFieldRoleAudit[] | undefined,
): void {
  if (!fieldRoleAudit) {
    return;
  }

  if (!directFields) {
    throw new Error(`field_role_audit parity requires live schema fields for ${basename}.`);
  }

  const duplicateAuditFields = fieldRoleAudit
    .map((entry) => entry.field)
    .filter((field, index, fields) => fields.indexOf(field) !== index);
  if (duplicateAuditFields.length > 0) {
    throw new Error(
      `field_role_audit contains duplicate entries for ${basename}: ${[...new Set(duplicateAuditFields)].join(", ")}.`,
    );
  }

  const auditFieldSet = new Set(fieldRoleAudit.map((entry) => entry.field));
  const directFieldSet = new Set(directFields);
  const missingAuditFields = directFields.filter((field) => !auditFieldSet.has(field));
  const unknownAuditEntries = fieldRoleAudit.filter((entry) => !directFieldSet.has(entry.field));
  const disallowedUnknownAuditFields = unknownAuditEntries
    .filter(
      (entry) =>
        entry.role !== "bounded-option-candidate" && entry.role !== "explicit-exception",
    )
    .map((entry) => entry.field);

  if (missingAuditFields.length > 0 || disallowedUnknownAuditFields.length > 0) {
    const details: string[] = [];
    if (missingAuditFields.length > 0) {
      details.push(`missing audited fields: ${missingAuditFields.join(", ")}`);
    }
    if (disallowedUnknownAuditFields.length > 0) {
      details.push(`unknown audited fields: ${disallowedUnknownAuditFields.join(", ")}`);
    }
    throw new Error(`field_role_audit parity mismatch for ${basename}: ${details.join("; ")}.`);
  }
}

function resolveAuditNotes(basename: string): string[] | undefined {
  const baseline = WRAPPER_AUDIT_BASELINES.get(basename);
  return baseline?.auditNotes ? [...baseline.auditNotes] : undefined;
}

function resolveOptionalAuditMetadata(
  basename: string,
  directFields: string[] | null,
): Pick<WrapperSchemaInventoryRow, "field_role_audit" | "audit_notes"> {
  const fieldRoleAudit = resolveFieldRoleAudit(basename);
  validateFieldRoleAuditParity(basename, directFields, fieldRoleAudit);
  const auditNotes = resolveAuditNotes(basename);
  return {
    ...(fieldRoleAudit ? { field_role_audit: fieldRoleAudit } : {}),
    ...(auditNotes ? { audit_notes: auditNotes } : {}),
  };
}

function resolveTargetShape(
  row: Pick<WrapperSchemaInventoryRow, "current_shape">,
  classification: WrapperClassification,
): WrapperTargetShape {
  if (classification === "already-compliant") {
    return row.current_shape === "no_tool_schema" ? "no_tool_schema" : "direct_args_object";
  }
  if (classification === "bounded-options-needed") {
    return "bounded-options";
  }
  if (classification === "split-needed") {
    return "split-wrapper-family";
  }
  return "compatibility-window";
}

function resolveTargetCount(
  row: Pick<WrapperSchemaInventoryRow, "current_count">,
  basename: string,
  classification: WrapperClassification,
): number | null {
  if (classification === "already-compliant") {
    return row.current_count;
  }
  if (classification === "bounded-options-needed") {
    return deriveBoundedOptionsTargetCount(basename, row.current_count);
  }
  return null;
}

function deriveBoundedOptionsTargetCount(basename: string, currentCount: number | null): number | null {
  if (!BOUNDED_OPTIONS_WRAPPERS.has(basename)) {
    return null;
  }
  return currentCount !== null && currentCount <= 4 ? currentCount : null;
}

function resolveOversizedClassificationForBasename(
  basename: string,
): Exclude<WrapperClassification, "already-compliant"> {
  const family = findSingleFamilySpec(basename);
  if (family) {
    return family.oversizedClassification;
  }
  if (BOUNDED_OPTIONS_WRAPPERS.has(basename)) {
    return "bounded-options-needed";
  }
  if (SPLIT_NEEDED_WRAPPERS.has(basename)) {
    return "split-needed";
  }
  return "compatibility-defer-with-owner";
}

function findSingleFamilySpec(basename: string): WrapperFamilySpec | null {
  const matches = WRAPPER_FAMILY_SPECS.filter((family) => family.matches(basename));
  return matches.length === 1 ? matches[0]! : null;
}

function buildUnhandledFailureRow(
  absolutePath: string,
  context: InventoryContext,
  error: unknown,
): WrapperSchemaInventoryRow {
  const wrapperPath = toRepoRelativePath(context.repoRoot, absolutePath);
  const basename = path.basename(absolutePath, ".ts");
  const companionDocPath = context.docsByBasename.get(basename) ?? null;
  const wrapperKind = classifyWrapperKind(basename, context.compatibilityWrappers);
  return buildInspectionFailedRow(
    wrapperPath,
    basename,
    companionDocPath,
    wrapperKind,
    `Unhandled wrapper inventory failure: ${sanitizeDiagnostic(context.repoRoot, error)}`,
    context.testPathsByBasename,
  );
}

function buildDerivedInspectionFailureRow(
  row: RowAuditFields,
  basename: string,
  context: Pick<InventoryContext, "testPathsByBasename">,
  error: unknown,
): WrapperSchemaInventoryRow {
  const ownerPlan = findSingleFamilySpec(basename)?.ownerPlan ?? null;
  const classification = ownerPlan
    ? resolveOversizedClassificationForBasename(basename)
    : "compatibility-defer-with-owner";
  const failedRow: RowAuditFields = {
    ...row,
    status: "inspection_failed",
    current_shape: "inspection_failed",
    counted_fields: [],
    exempt_fields: [],
    current_count: null,
    diagnostic: mergeDiagnostics(
      row.diagnostic,
      `Inventory metadata resolution failed: ${sanitizeDiagnostic("", error)}`,
    ),
  };
  return {
    ...failedRow,
    classification,
    target_shape: resolveTargetShape(failedRow, classification),
    target_count: resolveTargetCount(failedRow, basename, classification),
    owner_plan: ownerPlan,
    nearest_test_path: resolveNearestTestPath(basename, context.testPathsByBasename),
  };
}

function mergeDiagnostics(existing: string | null, next: string): string {
  return existing ? `${existing}; ${next}` : next;
}

function extractArgsObjectText(sourceText: string): string | null {
  const toolObjectText = extractToolRegistrationObjectText(sourceText);
  if (!toolObjectText) {
    return null;
  }

  ARGS_OBJECT_PATTERN.lastIndex = 0;
  const match = ARGS_OBJECT_PATTERN.exec(toolObjectText);
  if (!match) {
    return null;
  }
  const openBraceIndex = match.index + match[0].length - 1;
  const closeBraceIndex = findMatchingDelimiter(toolObjectText, openBraceIndex, "{", "}");
  if (closeBraceIndex < 0) {
    throw new Error("Unable to find the closing brace for the wrapper args object.");
  }
  return toolObjectText.slice(openBraceIndex, closeBraceIndex + 1);
}

function augmentInventoryDirectFields(_basename: string, directFields: string[]): string[] {
  return directFields;
}

function extractToolRegistrationObjectText(sourceText: string): string | null {
  const match = TOOL_REGISTRATION_PATTERN.exec(sourceText);
  if (!match) {
    return null;
  }
  const openBraceIndex = match.index + match[0].length - 1;
  const closeBraceIndex = findMatchingDelimiter(sourceText, openBraceIndex, "{", "}");
  if (closeBraceIndex < 0) {
    throw new Error("Unable to find the closing brace for the wrapper tool definition.");
  }
  return sourceText.slice(openBraceIndex, closeBraceIndex + 1);
}

function findMatchingDelimiter(
  sourceText: string,
  startIndex: number,
  openChar: string,
  closeChar: string,
): number {
  let depth = 0;
  let quote: "'" | '"' | "`" | null = null;
  let lineComment = false;
  let blockComment = false;

  for (let index = startIndex; index < sourceText.length; index += 1) {
    const char = sourceText[index];
    const next = sourceText[index + 1] ?? "";

    if (lineComment) {
      if (char === "\n") {
        lineComment = false;
      }
      continue;
    }

    if (blockComment) {
      if (char === "*" && next === "/") {
        blockComment = false;
        index += 1;
      }
      continue;
    }

    if (quote) {
      if (char === "\\") {
        index += 1;
        continue;
      }
      if (char === quote) {
        quote = null;
      }
      continue;
    }

    if (char === "/" && next === "/") {
      lineComment = true;
      index += 1;
      continue;
    }

    if (char === "/" && next === "*") {
      blockComment = true;
      index += 1;
      continue;
    }

    if (char === "'" || char === '"' || char === "`") {
      quote = char;
      continue;
    }

    if (char === openChar) {
      depth += 1;
      continue;
    }

    if (char === closeChar) {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
      continue;
    }
  }

  return -1;
}

function extractTopLevelArgsKeys(argsObjectText: string): string[] {
  const fields: string[] = [];
  let braceDepth = 0;
  let bracketDepth = 0;
  let parenDepth = 0;
  let quote: "'" | '"' | "`" | null = null;
  let lineComment = false;
  let blockComment = false;

  for (let index = 1; index < argsObjectText.length - 1; index += 1) {
    const char = argsObjectText[index];
    const next = argsObjectText[index + 1] ?? "";

    if (lineComment) {
      if (char === "\n") {
        lineComment = false;
      }
      continue;
    }

    if (blockComment) {
      if (char === "*" && next === "/") {
        blockComment = false;
        index += 1;
      }
      continue;
    }

    if (quote) {
      if (char === "\\") {
        index += 1;
        continue;
      }
      if (char === quote) {
        quote = null;
      }
      continue;
    }

    if (char === "/" && next === "/") {
      lineComment = true;
      index += 1;
      continue;
    }

    if (char === "/" && next === "*") {
      blockComment = true;
      index += 1;
      continue;
    }

    if ((char === "'" || char === '"') && braceDepth === 0 && bracketDepth === 0 && parenDepth === 0) {
      const keyEnd = findQuotedStringEnd(argsObjectText, index, char);
      const key = argsObjectText.slice(index + 1, keyEnd);
      const nextTokenIndex = skipWhitespace(argsObjectText, keyEnd + 1);
      if (argsObjectText[nextTokenIndex] === ":") {
        fields.push(key);
        index = keyEnd;
        continue;
      }
    }

    if (char === "'" || char === '"' || char === "`") {
      quote = char;
      continue;
    }

    if (char === "{") {
      braceDepth += 1;
      continue;
    }
    if (char === "}") {
      braceDepth -= 1;
      continue;
    }
    if (char === "[") {
      bracketDepth += 1;
      continue;
    }
    if (char === "]") {
      bracketDepth -= 1;
      continue;
    }
    if (char === "(") {
      parenDepth += 1;
      continue;
    }
    if (char === ")") {
      parenDepth -= 1;
      continue;
    }

    if (braceDepth !== 0 || bracketDepth !== 0 || parenDepth !== 0) {
      continue;
    }

    if (isIdentifierStart(char)) {
      const keyStart = index;
      index += 1;
      while (index < argsObjectText.length - 1 && isIdentifierPart(argsObjectText[index])) {
        index += 1;
      }
      const key = argsObjectText.slice(keyStart, index);
      const nextTokenIndex = skipWhitespace(argsObjectText, index);
      if (argsObjectText[nextTokenIndex] === ":") {
        fields.push(key);
      }
      index -= 1;
      continue;
    }
  }

  return fields;
}

function isExemptField(field: string, basename: string): boolean {
  if (field === "help") {
    return true;
  }
  if (field !== "options") {
    return false;
  }
  return BOUNDED_OPTIONS_WRAPPERS.has(basename);
}

function sanitizeDiagnostic(repoRoot: string, error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const normalizedMessage = repoRoot ? message.replaceAll(repoRoot, "<repo-root>") : message;
  const normalized = normalizedMessage.replace(/\s+/g, " ").trim();
  return normalized.length > 400 ? `${normalized.slice(0, 400)}... [truncated]` : normalized;
}

async function readBoundedUtf8File(absolutePath: string, repoRoot: string): Promise<string> {
  const confinedPath = await resolveRepoConfinedExistingPath(repoRoot, absolutePath);
  const fileStat = await stat(confinedPath);
  if (!fileStat.isFile()) {
    throw new Error(`Expected regular file: ${toRepoRelativePath(repoRoot, absolutePath)}`);
  }
  if (fileStat.size > MAX_REPO_CONTROLLED_READ_BYTES) {
    throw new Error(
      `File exceeds bounded read budget (${fileStat.size} bytes > ${MAX_REPO_CONTROLLED_READ_BYTES}) for ${toRepoRelativePath(repoRoot, absolutePath)}`,
    );
  }
  return readFile(confinedPath, "utf8");
}

async function assertSafeOutputPath(repoRoot: string, targetPath: string): Promise<void> {
  await assertPathWithinRepoRoot(repoRoot, targetPath);
  await assertSafeDirectoryChain(repoRoot, path.dirname(targetPath));
  try {
    const targetStats = await lstat(targetPath);
    if (targetStats.isSymbolicLink()) {
      throw new Error(`Refusing to write through symlinked target: ${toRepoRelativePath(repoRoot, targetPath)}`);
    }
  } catch (error) {
    if (!isMissingDirectoryError(error)) {
      throw error;
    }
  }
}

async function assertSafeDirectoryChain(repoRoot: string, targetDir: string): Promise<void> {
  const repoRealPath = await realpath(repoRoot);
  const resolvedTargetDir = path.resolve(targetDir);
  const relativePath = path.relative(repoRealPath, resolvedTargetDir);
  const segments = relativePath.split(path.sep).filter(Boolean);
  let currentPath = repoRealPath;

  for (const segment of segments) {
    currentPath = path.join(currentPath, segment);
    try {
      const stats = await lstat(currentPath);
      if (stats.isSymbolicLink()) {
        throw new Error(`Refusing to traverse symlinked directory: ${toRepoRelativePath(repoRoot, currentPath)}`);
      }
      if (!stats.isDirectory()) {
        throw new Error(`Expected directory while preparing output path: ${toRepoRelativePath(repoRoot, currentPath)}`);
      }
    } catch (error) {
      if (!isMissingDirectoryError(error)) {
        throw error;
      }
    }
  }
}

async function assertPathWithinRepoRoot(repoRoot: string, targetPath: string): Promise<void> {
  const repoRealPath = await realpath(repoRoot);
  const resolvedTargetPath = path.resolve(targetPath);
  const relativePath = path.relative(repoRealPath, resolvedTargetPath);
  if (
    relativePath === ".." ||
    relativePath.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativePath)
  ) {
    throw new Error("Path resolves outside repository root: <outside-repo>");
  }
}

async function resolveRepoConfinedExistingPath(repoRoot: string, targetPath: string): Promise<string> {
  await assertPathWithinRepoRoot(repoRoot, targetPath);
  const confinedRealPath = await realpath(targetPath);
  await assertPathWithinRepoRoot(repoRoot, confinedRealPath);
  return confinedRealPath;
}

function isMissingDirectoryError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    typeof (error as { code?: unknown }).code === "string" &&
    (error as { code: string }).code === "ENOENT"
  );
}

function recordMetadataFailure(
  diagnostics: string[],
  repoRoot: string,
  label: string,
  error: unknown,
): void {
  diagnostics.push(`${label}: ${sanitizeDiagnostic(repoRoot, error)}`);
}

function toRepoRelativePath(repoRoot: string, absolutePath: string): string {
  return path.relative(repoRoot, absolutePath).replaceAll(path.sep, "/");
}

function skipWhitespace(text: string, startIndex: number): number {
  let index = startIndex;
  while (index < text.length && /\s/.test(text[index] ?? "")) {
    index += 1;
  }
  return index;
}

function findQuotedStringEnd(text: string, startIndex: number, quote: "'" | '"'): number {
  for (let index = startIndex + 1; index < text.length; index += 1) {
    if (text[index] === "\\") {
      index += 1;
      continue;
    }
    if (text[index] === quote) {
      return index;
    }
  }
  throw new Error("Unterminated quoted property name in wrapper args object.");
}

function isIdentifierStart(char: string): boolean {
  return /[A-Za-z_$]/.test(char);
}

function isIdentifierPart(char: string): boolean {
  return /[A-Za-z0-9_$]/.test(char);
}
