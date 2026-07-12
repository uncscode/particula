import { describe, expect, it } from "bun:test";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  COMPACT_SCHEMA_FIELD_FIXTURES,
  COMPATIBILITY_METADATA_FIXTURES,
} from "./fixtures/wrapper_contract_fixtures";
import {
  assertCountedAndExemptFields,
  getCapturedToolDefinition,
  loadToolExecute,
  resetCapturedToolDefinition,
} from "./helpers/tool_harness";
import {
  classifyInventoryRow,
  generateWrapperSchemaInventory,
  getRepositoryRoot,
  inspectWrapperSourceFile,
  inspectWrapperSourceText,
  resolveNearestTestPath,
  resolveOwnerPlanForBasename,
  writeWrapperSchemaInventoryArtifact,
} from "../tooling/wrapper_schema_inventory";

const REAL_REPO_ROOT = path.resolve(import.meta.dir, "../../..");

function makeContext(
  repoRoot: string,
  companionDocPath: string | null = null,
  metadataDiagnostics: string[] = [],
  testPathsByBasename: Map<string, string> = new Map(),
) {
  const docsByBasename = new Map<string, string>();
  if (companionDocPath) {
    docsByBasename.set(path.basename(companionDocPath, ".md"), companionDocPath);
  }
  return {
    repoRoot,
    docsByBasename,
    compatibilityWrappers: new Set<string>(),
    metadataDiagnostics,
    testPathsByBasename,
  };
}

const ALLOWED_OWNER_PLANS = new Set([
  "E27-M3",
  "E27-M4",
  "E27-M5",
  "E27-M6",
  "E27-M7",
  "E27-M8",
  "E27-M9",
  "E27-M10",
  "E27-M11",
]);

describe("wrapper schema inventory", () => {
  it("catches counted/exempt bucket shifts instead of masking them with union-only equality", async () => {
    try {
      await loadToolExecute("../../find_files.ts");
      const definition = getCapturedToolDefinition();
      const wrapperPath = path.join(REAL_REPO_ROOT, ".opencode/tools/find_files.ts");
      const sourceText = await Bun.file(wrapperPath).text();
      const row = inspectWrapperSourceText(
        wrapperPath,
        sourceText,
        makeContext(REAL_REPO_ROOT, ".opencode/tools/find_files.md"),
      );

      expect(row.status).toBe("ok");
      expect(() =>
        assertCountedAndExemptFields(definition, {
          counted: row.exempt_fields,
          exempt: row.counted_fields,
          actualCounted: row.counted_fields,
          actualExempt: row.exempt_fields,
        }),
      ).toThrow("ASSERT: counted field classification did not match expectation");
    } finally {
      resetCapturedToolDefinition();
    }
  });

  it("exempts bounded options for issue-batch wrappers using options carriers", async () => {
    const wrapperPaths = [
      ".opencode/tools/adw_issues_batch_init.ts",
      ".opencode/tools/adw_issues_batch_read.ts",
      ".opencode/tools/adw_issues_batch_write.ts",
      ".opencode/tools/adw_issues_batch_log.ts",
      ".opencode/tools/adw_issues_batch_summary.ts",
    ];

    const rows = await Promise.all(
      wrapperPaths.map(async (relativePath) => {
        const wrapperPath = path.join(REAL_REPO_ROOT, relativePath);
        const sourceText = await Bun.file(wrapperPath).text();
        return inspectWrapperSourceText(wrapperPath, sourceText, makeContext(REAL_REPO_ROOT));
      }),
    );

    for (const row of rows) {
      expect(row.status).toBe("ok");
      expect(row.exempt_fields).toContain("options");
      expect(row.counted_fields).not.toContain("options");
    }
  });

  it("exempts bounded options for auto_mode_manifest.ts while keeping truthful counted fields", async () => {
    const wrapperPath = path.join(REAL_REPO_ROOT, ".opencode/tools/auto_mode_manifest.ts");
    const sourceText = await Bun.file(wrapperPath).text();
    const row = inspectWrapperSourceText(
      wrapperPath,
      sourceText,
      makeContext(REAL_REPO_ROOT, ".opencode/tools/auto_mode_manifest.md"),
    );

    expect(row.status).toBe("ok");
    expect(row.exempt_fields).toContain("options");
    expect(row.counted_fields).not.toContain("options");
    expect(row.current_count).toBe(14);
    expect(row.classification).toBe("split-needed");
  });

  it("keeps sanitizer options counted while excluding rejected advanced-only keys from basic inventory output", async () => {
    const wrapperPaths = [
      ".opencode/tools/run_sanitizers_basic.ts",
      ".opencode/tools/run_sanitizers_advanced.ts",
      ".opencode/tools/run_sanitizers.ts",
    ];

    const rows = await Promise.all(
      wrapperPaths.map(async (relativePath) => {
        const wrapperPath = path.join(REAL_REPO_ROOT, relativePath);
        const sourceText = await Bun.file(wrapperPath).text();
        return inspectWrapperSourceText(wrapperPath, sourceText, makeContext(REAL_REPO_ROOT));
      }),
    );

    for (const row of rows) {
      expect(row.status).toBe("ok");
      expect(row.exempt_fields).not.toContain("options");
    }

    const [basicRow, advancedRow, compatibilityRow] = rows;
    expect(basicRow.counted_fields).toEqual([
      "outputMode",
      "buildDir",
      "executable",
      "sanitizer",
      "timeout",
    ]);
    expect(basicRow.counted_fields).not.toContain("options");
    expect(basicRow.counted_fields).not.toContain("suppressions");
    expect(basicRow.counted_fields).not.toContain("normalDuration");
    expect(basicRow.counted_fields).not.toContain("extraArgs");

    expect(advancedRow.counted_fields).toContain("options");
    expect(compatibilityRow.counted_fields).toContain("options");
  });

  it("resolves dedicated sanitizer wrapper nearest test paths", async () => {
    const testPathsByBasename = new Map([
      ["run_sanitizers_basic", ".opencode/tools/__tests__/run_sanitizers_basic.test.ts"],
      ["run_sanitizers_advanced", ".opencode/tools/__tests__/run_sanitizers_advanced.test.ts"],
      ["run_sanitizers", ".opencode/tools/__tests__/run_sanitizers.test.ts"],
    ]);

    expect(resolveNearestTestPath("run_sanitizers_basic", testPathsByBasename)).toBe(
      ".opencode/tools/__tests__/run_sanitizers_basic.test.ts",
    );
    expect(resolveNearestTestPath("run_sanitizers_advanced", testPathsByBasename)).toBe(
      ".opencode/tools/__tests__/run_sanitizers_advanced.test.ts",
    );
    expect(resolveNearestTestPath("run_sanitizers", testPathsByBasename)).toBe(
      ".opencode/tools/__tests__/run_sanitizers.test.ts",
    );
  });

  it("exempts help for adw_setup.ts", async () => {
    const wrapperPath = path.join(REAL_REPO_ROOT, ".opencode/tools/adw_setup.ts");
    const sourceText = await Bun.file(wrapperPath).text();
    const row = inspectWrapperSourceText(
      wrapperPath,
      sourceText,
      makeContext(REAL_REPO_ROOT, ".opencode/tools/adw_setup.md"),
    );

    expect(row.status).toBe("ok");
    expect(row.exempt_fields).toContain("help");
    expect(row.counted_fields).not.toContain("help");
  });

  it("keeps focused ADW wrappers within the four-count budget", async () => {
    const wrapperPaths = [
      ".opencode/tools/adw_status_health.ts",
      ".opencode/tools/adw_setup.ts",
      ".opencode/tools/adw_service.ts",
    ];

    const rows = await Promise.all(
      wrapperPaths.map(async (relativePath) => {
        const wrapperPath = path.join(REAL_REPO_ROOT, relativePath);
        const sourceText = await Bun.file(wrapperPath).text();
        return inspectWrapperSourceText(
          wrapperPath,
          sourceText,
          makeContext(REAL_REPO_ROOT, relativePath.replace(/\.ts$/, ".md")),
        );
      }),
    );

    for (const row of rows) {
      expect(row.status).toBe("ok");
      expect(row.current_count).toBeLessThanOrEqual(4);
      expect(row.current_count).toBe(row.counted_fields.length);
    }

    const statusRow = rows[0];
    expect(statusRow.counted_fields).toEqual(["command"]);
    expect(statusRow.exempt_fields).toEqual(["help"]);

    const setupRow = rows[1];
    expect(setupRow.counted_fields).toEqual(["command", "wizard", "args"]);
    expect(setupRow.exempt_fields).toEqual(expect.arrayContaining(["options", "help"]));

    const serviceRow = rows[2];
    expect(serviceRow.counted_fields).toEqual(["command", "mode", "background", "force"]);
    expect(serviceRow.exempt_fields).toEqual(["help"]);
  });

  it("returns deterministic fallback output for no-tool modules", () => {
    const repoRoot = "/repo";
    const row = inspectWrapperSourceText(
      "/repo/.opencode/tools/helper_module.ts",
      "export function helper(): void {}\n",
      makeContext(repoRoot),
    );

    expect(row.status).toBe("no_tool_schema");
    expect(row.current_shape).toBe("no_tool_schema");
    expect(row.current_count).toBe(0);
    expect(row.wrapper_path).toBe(".opencode/tools/helper_module.ts");
    expect(row.companion_doc_path).toBeNull();
    expect(row.classification).toBe("already-compliant");
    expect(row.owner_plan).toBeNull();
    expect(row.target_shape).toBe("no_tool_schema");
    expect(row.target_count).toBe(0);
  });

  it("keeps sparse optional fields in the row without changing exemption behavior", () => {
    const repoRoot = "/repo";
    const row = inspectWrapperSourceText(
      "/repo/.opencode/tools/sample_wrapper.ts",
      COMPACT_SCHEMA_FIELD_FIXTURES.sparseOptionalInventory.sourceLines.join("\n"),
      makeContext(repoRoot),
    );

    expect(row.status).toBe("ok");
    expect(row.counted_fields).toEqual(COMPACT_SCHEMA_FIELD_FIXTURES.sparseOptionalInventory.counted);
    expect(row.exempt_fields).toEqual(COMPACT_SCHEMA_FIELD_FIXTURES.sparseOptionalInventory.exempt);
    expect(row.current_count).toBe(3);
  });

  it("anchors args extraction to the exported tool definition", () => {
    const repoRoot = "/repo";
    const row = inspectWrapperSourceText(
      "/repo/.opencode/tools/sample_wrapper.ts",
      [
        "import { tool } from '@opencode-ai/plugin';",
        "const helperConfig = {",
        "  args: {",
        "    wrong_field: tool.schema.string(),",
        "  },",
        "};",
        "export default tool({",
        "  args: {",
        "    command: tool.schema.enum(['show']),",
        "    path: tool.schema.string().optional(),",
        "    help: tool.schema.boolean().optional(),",
        "  },",
        "  async execute() {",
        "    return helperConfig.args.wrong_field;",
        "  },",
        "});",
        "",
      ].join("\n"),
      makeContext(repoRoot),
    );

    expect(row.status).toBe("ok");
    expect(row.counted_fields).toEqual(["command", "path"]);
    expect(row.exempt_fields).toEqual(["help"]);
    expect(row.counted_fields).not.toContain("wrong_field");
  });

  it("takes the cheap pre-scan path for non-wrapper files without importing them", () => {
    const repoRoot = "/repo";
    const row = inspectWrapperSourceText(
      "/repo/.opencode/tools/not_a_wrapper.ts",
      "export const broken = () => {\n  throw new Error('should never execute');\n// malformed on purpose\n",
      makeContext(repoRoot),
    );

    expect(row.status).toBe("no_tool_schema");
    expect(row.current_shape).toBe("no_tool_schema");
    expect(row.diagnostic).toBeNull();
  });

  it("downgrades malformed args inspection to a row-level failure with sanitized diagnostics", () => {
    const repoRoot = "/repo";
    const row = inspectWrapperSourceText(
      "/repo/.opencode/tools/feedback_log.ts",
      "import { tool } from '@opencode-ai/plugin';\nexport default tool({ args: { command: tool.schema.string().optional(), help: tool.schema.boolean().optional(), execute: true \n",
      makeContext(repoRoot),
    );

    expect(row.status).toBe("inspection_failed");
    expect(row.current_shape).toBe("inspection_failed");
    expect(row.current_count).toBeNull();
    expect(row.diagnostic).toContain("closing brace");
    expect(row.diagnostic).not.toContain("/repo");
    expect(row.classification).toBe("compatibility-defer-with-owner");
  });

  it("downgrades missing wrapper files to row-level failures instead of aborting", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        JSON.stringify(COMPATIBILITY_METADATA_FIXTURES.activeDisallowedConfig, null, 2),
        "utf8",
      );

      const row = await inspectWrapperSourceFile(
        path.join(tempRoot, ".opencode/tools/feedback_log.ts"),
        tempRoot,
      );

      expect(row.status).toBe("inspection_failed");
      expect(row.current_shape).toBe("inspection_failed");
      expect(row.wrapper_path).toBe(".opencode/tools/feedback_log.ts");
      expect(row.diagnostic).toContain("Unable to read wrapper source");
      expect(row.diagnostic).not.toContain(tempRoot);
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("inspects a wrapper source file from disk on the success path", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await writeFile(
        path.join(tempRoot, ".opencode/tools/sample_wrapper.ts"),
        [
          "import { tool } from '@opencode-ai/plugin';",
          "export default tool({",
          "  args: {",
          "    command: tool.schema.enum(['show']),",
          "    path: tool.schema.string().optional(),",
          "    help: tool.schema.boolean().optional(),",
          "  },",
          "  async execute() {",
          "    return 'ok';",
          "  },",
          "});",
          "",
        ].join("\n"),
        "utf8",
      );
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        JSON.stringify(COMPATIBILITY_METADATA_FIXTURES.activeDisallowedConfig, null, 2),
        "utf8",
      );

      const row = await inspectWrapperSourceFile(
        path.join(tempRoot, ".opencode/tools/sample_wrapper.ts"),
        tempRoot,
      );

      expect(row.status).toBe("ok");
      expect(row.wrapper_path).toBe(".opencode/tools/sample_wrapper.ts");
      expect(row.counted_fields).toEqual(["command", "path"]);
      expect(row.exempt_fields).toEqual(["help"]);
      expect(row.current_count).toBe(2);
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("downgrades symlink-escape wrapper reads to deterministic failure rows", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    const externalRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-external-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await writeFile(
        path.join(externalRoot, "escaped_wrapper.ts"),
        [
          "import { tool } from '@opencode-ai/plugin';",
          "export default tool({",
          "  args: { command: tool.schema.enum(['show']) },",
          "  async execute() {",
          "    return 'escaped';",
          "  },",
          "});",
          "",
        ].join("\n"),
        "utf8",
      );
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        JSON.stringify({ version: 1, wrappers: [] }, null, 2),
        "utf8",
      );
      await symlink(
        path.join(externalRoot, "escaped_wrapper.ts"),
        path.join(tempRoot, ".opencode/tools/escaped_wrapper.ts"),
      );

      const row = await inspectWrapperSourceFile(
        path.join(tempRoot, ".opencode/tools/escaped_wrapper.ts"),
        tempRoot,
      );

      expect(row.status).toBe("inspection_failed");
      expect(row.current_shape).toBe("inspection_failed");
      expect(row.wrapper_path).toBe(".opencode/tools/escaped_wrapper.ts");
      expect(row.diagnostic).toContain("Path resolves outside repository root");
      expect(row.diagnostic).not.toContain(tempRoot);
      expect(row.diagnostic).not.toContain(externalRoot);
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
      await rm(externalRoot, { recursive: true, force: true });
    }
  });

  it("downgrades malformed compatibility metadata to deterministic failure rows", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await writeFile(
        path.join(tempRoot, ".opencode/tools/feedback_log.ts"),
        [
          "import { tool } from '@opencode-ai/plugin';",
          "export default tool({",
          "  args: { command: tool.schema.enum(['show']) },",
          "  async execute() {",
          "    return 'ok';",
          "  },",
          "});",
          "",
        ].join("\n"),
        "utf8",
      );
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        COMPATIBILITY_METADATA_FIXTURES.malformedMetadataText,
        "utf8",
      );

      const artifact = await generateWrapperSchemaInventory(tempRoot);
      expect(artifact.rows).toHaveLength(1);
      expect(artifact.rows[0]?.status).toBe("inspection_failed");
      expect(artifact.rows[0]?.diagnostic).toContain("Supporting inventory metadata unavailable");
      expect(artifact.rows[0]?.diagnostic).not.toContain(tempRoot);
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("produces sorted repository-relative rows with deterministic classification fields", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const wrapperPaths = artifact.rows.map((row) => row.wrapper_path);
    const sortedPaths = [...wrapperPaths].sort((left, right) => left.localeCompare(right));

    expect(wrapperPaths).toEqual(sortedPaths);
    expect(wrapperPaths.every((wrapperPath) => !path.isAbsolute(wrapperPath))).toBe(true);
    expect(
      artifact.rows.every((row) =>
        ["already-compliant", "split-needed", "bounded-options-needed", "compatibility-defer-with-owner"].includes(
          row.classification,
        ),
      ),
    ).toBe(true);
  });

  it("is stable across repeat runs", async () => {
    const first = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const second = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    expect(second).toEqual(first);
  });

  it("classifies representative wrappers for each required classification path", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const rowsByBasename = new Map(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    expect(rowsByBasename.get("adw_issues_batch_init")?.classification).toBe("already-compliant");
    expect(rowsByBasename.get("platform_operations")?.classification).toBe("split-needed");
    expect(rowsByBasename.get("adw_issues_batch_log")?.classification).toBe("split-needed");
    expect(rowsByBasename.get("feedback_log")?.classification).toBe(
      "compatibility-defer-with-owner",
    );
  });

  it("routes representative wrapper families to the expected owner plans", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const rowsByBasename = new Map(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    expect(rowsByBasename.get("platform_operations")?.owner_plan).toBe("E27-M3");
    expect(rowsByBasename.get("adw")?.owner_plan).toBe("E27-M5");
    expect(rowsByBasename.get("auto_mode_manifest")?.owner_plan).toBe("E27-M7");
    expect(rowsByBasename.get("validate_notebook")?.owner_plan).toBe("E27-M10");
    expect(rowsByBasename.get("feedback_log")?.owner_plan).toBe("E27-M11");
    expect(rowsByBasename.get("get_datetime")?.owner_plan).toBeNull();
    expect(rowsByBasename.get("get_version")?.owner_plan).toBeNull();
  });

  it("derives nearest_test_path via exact match, family fallback, or null", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const rowsByBasename = new Map(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    expect(rowsByBasename.get("adw_issues_batch_init")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/adw_issues_batch_init.test.ts",
    );
    expect(rowsByBasename.get("adw_issues_batch_write")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/adw_issues_batch_read_write.test.ts",
    );
    expect(rowsByBasename.get("adw_issues_batch_log")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/adw_issues_batch_log_summary.test.ts",
    );
    expect(rowsByBasename.get("adw_issues_batch_summary")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/adw_issues_batch_log_summary.test.ts",
    );
    expect(rowsByBasename.get("auto_mode_manifest")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/auto_mode_manifest.test.ts",
    );
    expect(rowsByBasename.get("platform_operations")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/platform_operations_compat_comment_pr_review.test.ts",
    );
    expect(rowsByBasename.get("feedback_log")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/feedback_log.test.ts",
    );
    expect(rowsByBasename.get("workflow_builder")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/workflow_builder.test.ts",
    );
    expect(rowsByBasename.get("run_pytest_advanced")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_pytest_advanced.test.ts",
    );
    expect(rowsByBasename.get("build_mkdocs")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/build_mkdocs.test.ts",
    );
    expect(rowsByBasename.get("clear_build")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/clear_build.test.ts",
    );
    expect(rowsByBasename.get("clear_build_preview")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/clear_build.test.ts",
    );
    expect(rowsByBasename.get("clear_build_delete")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/clear_build.test.ts",
    );
  });

  it("fails closed when field_role_audit drifts from live wrapper schema fields", async () => {
    const wrapperPath = path.join(REAL_REPO_ROOT, ".opencode/tools/clear_build_preview.ts");
    const sourceText = await Bun.file(wrapperPath).text();
    const driftedSourceText = sourceText.replace(
      "buildDir: tool.schema",
      "buildDirRenamed: tool.schema",
    );

    const row = inspectWrapperSourceText(
      wrapperPath,
      driftedSourceText,
      makeContext(REAL_REPO_ROOT, ".opencode/tools/clear_build_preview.md"),
    );

    expect(row.status).toBe("inspection_failed");
    expect(row.diagnostic).toContain("field_role_audit parity mismatch");
    expect(row.diagnostic).toContain("missing audited fields: buildDirRenamed");
    expect(row.diagnostic).toContain("unknown audited fields: buildDir");
  });

  it("records E27-M8 field-role audit metadata and create_workspace ownership notes", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const rowsByBasename = new Map(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    const expectedAuditWrappers = [
      "create_workspace",
      "find_files",
      "move",
      "move_overwrite",
      "move_safe",
      "move_trash",
      "refactor_astgrep_apply",
      "refactor_astgrep_preview",
      "ripgrep_advanced",
      "search_content",
      "workflow_builder",
      "workflow_builder_mutate",
      "workflow_builder_read",
    ];

    for (const wrapperName of expectedAuditWrappers) {
      expect(rowsByBasename.get(wrapperName)?.field_role_audit).toBeDefined();
    }

    const findFilesRow = rowsByBasename.get("find_files");
    expect(findFilesRow?.counted_fields).toEqual(["pattern", "path"]);
    expect(findFilesRow?.exempt_fields).toEqual(["options"]);
    expect(findFilesRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "pattern", role: "payload-bearing" }),
        expect.objectContaining({ field: "path", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "fileType", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "contentPattern", role: "explicit-exception" }),
      ]),
    );

    const searchContentRow = rowsByBasename.get("search_content");
    expect(searchContentRow?.counted_fields).toEqual(["contentPattern", "path"]);
    expect(searchContentRow?.exempt_fields).toEqual(["options"]);
    expect(searchContentRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "contentPattern", role: "payload-bearing" }),
        expect.objectContaining({ field: "path", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "pattern", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "unrestricted", role: "explicit-exception" }),
      ]),
    );

    const ripgrepAdvancedRow = rowsByBasename.get("ripgrep_advanced");
    expect(ripgrepAdvancedRow?.counted_fields).toEqual(["contentPattern", "path"]);
    expect(ripgrepAdvancedRow?.exempt_fields).toEqual(["options"]);
    expect(ripgrepAdvancedRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "contentPattern", role: "payload-bearing" }),
        expect.objectContaining({ field: "path", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "contextLines", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "filesWithMatches", role: "bounded-option-candidate" }),
      ]),
    );

    for (const wrapperName of ["refactor_astgrep_apply", "refactor_astgrep_preview"]) {
      expect(rowsByBasename.get(wrapperName)?.field_role_audit).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ field: "pattern", role: "payload-bearing" }),
          expect.objectContaining({ field: "path", role: "safety-field" }),
        ]),
      );
    }

    const moveCompatibilityRow = rowsByBasename.get("move");
    expect(moveCompatibilityRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "source", role: "payload-bearing" }),
        expect.objectContaining({ field: "destination", role: "payload-bearing" }),
        expect.objectContaining({ field: "overwrite", role: "explicit-exception" }),
        expect.objectContaining({ field: "trash", role: "explicit-exception" }),
      ]),
    );

    for (const wrapperName of ["move_safe", "move_overwrite"]) {
      expect(rowsByBasename.get(wrapperName)?.field_role_audit).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ field: "source", role: "payload-bearing" }),
          expect.objectContaining({ field: "destination", role: "payload-bearing" }),
        ]),
      );
    }

    expect(rowsByBasename.get("move_trash")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "source", role: "payload-bearing" }),
      ]),
    );

    const workflowBuilderRow = rowsByBasename.get("workflow_builder");
    expect(workflowBuilderRow?.field_role_audit).toEqual([
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
      {
        field: "step_index",
        role: "required-identifier",
        commands: ["remove_step"],
        notes: "Alternative remove_step selector; paired with step_name.",
      },
      {
        field: "step_name",
        role: "required-identifier",
        commands: ["remove_step"],
        notes: "Alternative remove_step selector; paired with step_index.",
      },
      { field: "position", role: "bounded-option-candidate", commands: ["add_step"] },
      { field: "workflow_json", role: "payload-bearing", commands: ["update", "validate"] },
      { field: "output", role: "bounded-option-candidate" },
    ]);
    expect(workflowBuilderRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/workflow_builder.test.ts",
    );

    const workflowBuilderReadRow = rowsByBasename.get("workflow_builder_read");
    expect(workflowBuilderReadRow?.field_role_audit).toEqual([
      {
        field: "command",
        role: "command-selector",
        notes: "Read-only gate accepts list/get/validate only.",
      },
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
    ]);
    expect(workflowBuilderReadRow?.audit_notes).toEqual([
      "Split read wrapper intentionally keeps the broad schema but records mutate-only fields as explicit rejection semantics rather than supported read-surface inputs.",
    ]);
    expect(workflowBuilderReadRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/workflow_builder_read.test.ts",
    );

    const workflowBuilderMutateRow = rowsByBasename.get("workflow_builder_mutate");
    expect(workflowBuilderMutateRow?.field_role_audit).toEqual([
      {
        field: "command",
        role: "command-selector",
        notes: "Mutating gate accepts create/add_step/remove_step/update only.",
      },
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
    ]);
    expect(workflowBuilderMutateRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/workflow_builder_mutate.test.ts",
    );

    const createWorkspaceRow = rowsByBasename.get("create_workspace");
    expect(createWorkspaceRow?.owner_plan).toBe("E27-M5");
    expect(createWorkspaceRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "issueNumber", role: "required-identifier" }),
        expect.objectContaining({ field: "adwId", role: "safety-field" }),
      ]),
    );
    expect(createWorkspaceRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("create_workspace.py"),
        expect.stringContaining("Ownership remains with E27-M5"),
      ]),
    );
  });

  it("records E27-M9 field-role audit metadata for test/build/validation wrappers", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const rowsByBasename = new Map(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    const expectedAuditWrappers = [
      "run_pytest_basic",
      "run_pytest_advanced",
      "run_bun_test",
      "run_ctest",
      "run_cmake",
      "run_cmake_configure",
      "run_cmake_build",
      "run_linters",
      "run_validate_agent_references",
      "build_mkdocs",
      "build_mkdocs_validate",
      "build_mkdocs_build",
      "clear_build",
      "clear_build_preview",
      "clear_build_delete",
    ];

    for (const wrapperName of expectedAuditWrappers) {
      expect(rowsByBasename.get(wrapperName)?.field_role_audit).toBeDefined();
    }

    const runPytestBasicRow = rowsByBasename.get("run_pytest_basic");
    expect(runPytestBasicRow?.owner_plan).toBeNull();
    expect(runPytestBasicRow?.classification).toBe("already-compliant");
    expect(runPytestBasicRow?.counted_fields).toEqual(["minTests", "timeout", "cwd", "testPath"]);
    expect(runPytestBasicRow?.exempt_fields).toEqual(["options"]);
    expect(runPytestBasicRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "testPath", role: "payload-bearing" }),
        expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
        expect.objectContaining({ field: "cwd", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
      ]),
    );
    expect(runPytestBasicRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("compact routine fields"),
        expect.stringContaining("fail closed in execute()"),
      ]),
    );

    const runPytestAdvancedRow = rowsByBasename.get("run_pytest_advanced");
    expect(runPytestAdvancedRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_pytest_advanced.test.ts",
    );
    expect(runPytestAdvancedRow?.counted_fields).toEqual([
      "minTests",
      "timeout",
      "cwd",
      "testPath",
      "pytestArgs",
      "coverage",
      "coverageSource",
      "coverageThreshold",
      "overrideIni",
    ]);
    expect(runPytestAdvancedRow?.exempt_fields).toEqual(["options"]);
    expect(runPytestAdvancedRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "pytestArgs", role: "payload-bearing" }),
        expect.objectContaining({ field: "coverage", role: "explicit-exception" }),
        expect.objectContaining({ field: "coverageSource", role: "payload-bearing" }),
        expect.objectContaining({ field: "coverageThreshold", role: "explicit-exception" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "overrideIni", role: "payload-bearing" }),
      ]),
    );
    expect(runPytestAdvancedRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("keeps legacy routine/report toggles off the published schema"),
      ]),
    );

    const runBunTestRow = rowsByBasename.get("run_bun_test");
    expect(runBunTestRow?.owner_plan).toBeNull();
    expect(runBunTestRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_bun_test.test.ts",
    );
    expect(runBunTestRow?.counted_fields).toEqual([
      "testPath",
      "timeout",
      "minTests",
      "cwd",
    ]);
    expect(runBunTestRow?.exempt_fields).toEqual(["options"]);
    expect(runBunTestRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "cwd", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "failFast", role: "bounded-option-candidate" }),
      ]),
    );
    expect(runBunTestRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("keeps legacy output/filter/fail-fast toggles off the published schema"),
      ]),
    );

    const runCtestRow = rowsByBasename.get("run_ctest");
    expect(runCtestRow?.owner_plan).toBeNull();
    expect(runCtestRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_ctest.test.ts",
    );
    expect(runCtestRow?.counted_fields).toEqual(["buildDir", "timeout", "minTests"]);
    expect(runCtestRow?.exempt_fields).toEqual(["options"]);
    expect(runCtestRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "buildDir", role: "required-identifier" }),
        expect.objectContaining({ field: "parallel", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "minTests", role: "explicit-exception" }),
      ]),
    );

    const runCmakeCompatibilityRow = rowsByBasename.get("run_cmake");
    expect(runCmakeCompatibilityRow?.owner_plan).toBe("E27-M9");
    expect(runCmakeCompatibilityRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_cmake.test.ts",
    );
    expect(runCmakeCompatibilityRow?.exempt_fields).toEqual(["options"]);
    expect(runCmakeCompatibilityRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "build", role: "explicit-exception" }),
        expect.objectContaining({ field: "cmakeArgs", role: "payload-bearing" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
      ]),
    );
    expect(runCmakeCompatibilityRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("P3 follow-up"),
      ]),
    );

    const runCmakeBuildRow = rowsByBasename.get("run_cmake_build");
    expect(runCmakeBuildRow?.owner_plan).toBeNull();
    expect(runCmakeBuildRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_cmake.test.ts",
    );
    expect(runCmakeBuildRow?.counted_fields).toEqual(["preset", "buildDir", "buildTimeout", "timeout"]);
    expect(runCmakeBuildRow?.exempt_fields).toEqual(["options"]);
    expect(runCmakeBuildRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "preset", role: "required-identifier" }),
        expect.objectContaining({ field: "buildDir", role: "required-identifier" }),
        expect.objectContaining({ field: "jobs", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "buildTimeout", role: "explicit-exception" }),
      ]),
    );

    const runLintersRow = rowsByBasename.get("run_linters");
    expect(runLintersRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_linters.test.ts",
    );
    expect(runLintersRow?.owner_plan).toBeNull();
    expect(runLintersRow?.counted_fields).toEqual(["autoFix", "targetDir", "ruffTimeout", "mypyTimeout"]);
    expect(runLintersRow?.exempt_fields).toEqual(["options"]);
    expect(runLintersRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "autoFix", role: "safety-field" }),
        expect.objectContaining({ field: "linters", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "targetDir", role: "safety-field" }),
      ]),
    );

    const validateAgentReferencesRow = rowsByBasename.get("run_validate_agent_references");
    expect(validateAgentReferencesRow?.owner_plan).toBeNull();
    expect(validateAgentReferencesRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/run_validate_agent_references.test.ts",
    );
    expect(validateAgentReferencesRow?.counted_fields).toEqual(["cwd", "baselinePath"]);
    expect(validateAgentReferencesRow?.exempt_fields).toEqual([]);
    expect(validateAgentReferencesRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "cwd", role: "safety-field" }),
        expect.objectContaining({ field: "baselinePath", role: "safety-field" }),
      ]),
    );

    const buildMkdocsCompatibilityRow = rowsByBasename.get("build_mkdocs");
    expect(buildMkdocsCompatibilityRow?.owner_plan).toBeNull();
    expect(buildMkdocsCompatibilityRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/build_mkdocs.test.ts",
    );
    expect(buildMkdocsCompatibilityRow?.counted_fields).toEqual(["timeout", "cwd", "configFile", "validateOnly"]);
    expect(buildMkdocsCompatibilityRow?.exempt_fields).toEqual(["options"]);
    expect(buildMkdocsCompatibilityRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "cwd", role: "safety-field" }),
        expect.objectContaining({ field: "configFile", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
        expect.objectContaining({ field: "validateOnly", role: "explicit-exception" }),
      ]),
    );

    for (const wrapperName of ["build_mkdocs_validate", "build_mkdocs_build"]) {
      expect(rowsByBasename.get(wrapperName)?.nearest_test_path).toBe(
        ".opencode/tools/__tests__/build_mkdocs.test.ts",
      );
      expect(rowsByBasename.get(wrapperName)?.counted_fields).toEqual(["timeout", "cwd", "configFile"]);
      expect(rowsByBasename.get(wrapperName)?.exempt_fields).toEqual(["options"]);
      expect(rowsByBasename.get(wrapperName)?.field_role_audit).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ field: "cwd", role: "safety-field" }),
          expect.objectContaining({ field: "configFile", role: "safety-field" }),
          expect.objectContaining({ field: "options", role: "explicit-exception" }),
          expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
        ]),
      );
    }

    const clearBuildCompatibilityRow = rowsByBasename.get("clear_build");
    expect(clearBuildCompatibilityRow?.owner_plan).toBeNull();
    expect(clearBuildCompatibilityRow?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/clear_build.test.ts",
    );
    expect(clearBuildCompatibilityRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "buildDir", role: "safety-field" }),
        expect.objectContaining({ field: "dryRun", role: "explicit-exception" }),
        expect.objectContaining({ field: "force", role: "safety-field" }),
      ]),
    );

    expect(rowsByBasename.get("clear_build_preview")?.field_role_audit).toEqual([
      expect.objectContaining({ field: "buildDir", role: "safety-field" }),
    ]);
    expect(rowsByBasename.get("clear_build_preview")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/clear_build.test.ts",
    );
    expect(rowsByBasename.get("clear_build_delete")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "buildDir", role: "safety-field" }),
        expect.objectContaining({ field: "force", role: "safety-field" }),
      ]),
    );
    expect(rowsByBasename.get("clear_build_delete")?.nearest_test_path).toBe(
      ".opencode/tools/__tests__/clear_build.test.ts",
    );
  });

  it("records E27-M11 family membership, field-role audit metadata, and boundary notes", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const rowsByBasename = new Map(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    const e27M11Rows = artifact.rows.filter((row) => row.owner_plan === "E27-M11");

    expect(e27M11Rows.map((row) => path.basename(row.wrapper_path, ".ts")).sort()).toEqual([
      "feedback_log",
    ]);
    expect(
      artifact.rows
        .filter((row) => {
          const basename = path.basename(row.wrapper_path, ".ts");
          return basename === "feedback_log" || basename === "get_datetime" || basename === "get_version";
        })
        .map((row) => path.basename(row.wrapper_path, ".ts"))
        .sort(),
    ).toEqual([
      "feedback_log",
      "get_datetime",
      "get_version",
    ]);

    const feedbackLogRow = rowsByBasename.get("feedback_log");
    expect(feedbackLogRow?.classification).toBe("compatibility-defer-with-owner");
    expect(feedbackLogRow?.owner_plan).toBe("E27-M11");
    expect(feedbackLogRow?.companion_doc_path).toBe(".opencode/tools/feedback_log.md");
    expect(feedbackLogRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "category", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "severity", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "description", role: "payload-bearing" }),
      ]),
    );
    expect(feedbackLogRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("exactly feedback_log, get_datetime, and get_version"),
        expect.stringContaining("backend-only surface area"),
        expect.stringContaining("landed in E27-M11 P2"),
        expect.stringContaining("compact companion wrapper doc"),
      ]),
    );

    const getDatetimeRow = rowsByBasename.get("get_datetime");
    expect(getDatetimeRow?.classification).toBe("already-compliant");
    expect(getDatetimeRow?.owner_plan).toBeNull();
    expect(getDatetimeRow?.companion_doc_path).toBe(".opencode/tools/get_datetime.md");
    expect(getDatetimeRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "format", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "localtime", role: "safety-field" }),
      ]),
    );
    expect(getDatetimeRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("owner_plan null"),
        expect.stringContaining("no additional wrapper assignment"),
        expect.stringContaining("compact companion doc"),
      ]),
    );

    const getVersionRow = rowsByBasename.get("get_version");
    expect(getVersionRow?.classification).toBe("already-compliant");
    expect(getVersionRow?.owner_plan).toBeNull();
    expect(getVersionRow?.companion_doc_path).toBe(".opencode/tools/get_version.md");
    expect(getVersionRow?.field_role_audit).toEqual(
      expect.arrayContaining([expect.objectContaining({ field: "file", role: "payload-bearing" })]),
    );
    expect(getVersionRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("owner_plan null"),
        expect.stringContaining("no extra small utility wrapper"),
        expect.stringContaining("deferred to E27-M11 P3"),
        expect.stringContaining("blank-input omission"),
      ]),
    );

    const committed = JSON.parse(
      await readFile(
        path.join(REAL_REPO_ROOT, ".opencode/tools/wrapper_schema_inventory.json"),
        "utf8",
      ),
    );
    const committedRowsByBasename = new Map(
      committed.rows.map((row: { wrapper_path: string }) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    expect(committedRowsByBasename.get("feedback_log")?.companion_doc_path).toBe(
      ".opencode/tools/feedback_log.md",
    );
    expect(committedRowsByBasename.get("feedback_log")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "category", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "severity", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "description", role: "payload-bearing" }),
      ]),
    );
    expect(committedRowsByBasename.get("feedback_log")?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("exactly feedback_log, get_datetime, and get_version"),
        expect.stringContaining("backend-only surface area"),
        expect.stringContaining("landed in E27-M11 P2"),
        expect.stringContaining("compact companion wrapper doc"),
      ]),
    );
    expect(committedRowsByBasename.get("get_datetime")?.companion_doc_path).toBe(
      ".opencode/tools/get_datetime.md",
    );
    expect(committedRowsByBasename.get("get_datetime")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "format", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "localtime", role: "safety-field" }),
      ]),
    );
    expect(committedRowsByBasename.get("get_datetime")?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("owner_plan null"),
        expect.stringContaining("no additional wrapper assignment"),
        expect.stringContaining("compact companion doc"),
      ]),
    );
    expect(committedRowsByBasename.get("get_version")?.companion_doc_path).toBe(
      ".opencode/tools/get_version.md",
    );
    expect(committedRowsByBasename.get("get_version")?.field_role_audit).toEqual(
      expect.arrayContaining([expect.objectContaining({ field: "file", role: "payload-bearing" })]),
    );
    expect(committedRowsByBasename.get("get_version")?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("owner_plan null"),
        expect.stringContaining("no extra small utility wrapper"),
        expect.stringContaining("deferred to E27-M11 P3"),
        expect.stringContaining("blank-input omission"),
      ]),
    );
  });

  it("records E27-M10 field-role audit metadata for notebook, C++, and sanitizer wrappers", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const rowsByBasename = new Map(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    const expectedAuditWrappers = [
      "validate_notebook",
      "validate_notebook_readonly",
      "convert_notebook_to_py",
      "convert_py_to_notebook",
      "sync_notebook_pair",
      "run_notebook",
      "run_cpp_lint_check",
      "run_cpp_lint_fix",
      "run_cpp_linters",
      "run_cpp_coverage_summary",
      "run_cpp_coverage_advanced",
      "run_sanitizers_basic",
      "run_sanitizers_advanced",
      "run_sanitizers",
    ];

    for (const wrapperName of expectedAuditWrappers) {
      const row = rowsByBasename.get(wrapperName);
      expect(row?.field_role_audit).toBeDefined();
      if (row && row.classification !== "already-compliant") {
        expect(row.owner_plan).toBe("E27-M10");
      }
    }

    const validateNotebookRow = rowsByBasename.get("validate_notebook");
    expect(validateNotebookRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "notebookPath", role: "payload-bearing" }),
        expect.objectContaining({ field: "recursive", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "outputMode", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "convertToPy", role: "explicit-exception" }),
        expect.objectContaining({ field: "convertToIpynb", role: "explicit-exception" }),
        expect.objectContaining({ field: "sync", role: "explicit-exception" }),
        expect.objectContaining({ field: "outputDir", role: "explicit-exception" }),
      ]),
    );
    expect(validateNotebookRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("Compatibility notebook wrapper intentionally preserves convert/sync direct toggles"),
      ]),
    );

    const validateNotebookReadonlyRow = rowsByBasename.get("validate_notebook_readonly");
    expect(validateNotebookReadonlyRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "notebookPath", role: "payload-bearing" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "checkSync", role: "explicit-exception" }),
      ]),
    );
    expect(validateNotebookReadonlyRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("Read-only split wrapper omits mutating keys"),
      ]),
    );

    expect(rowsByBasename.get("convert_notebook_to_py")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "notebookPath", role: "payload-bearing" }),
        expect.objectContaining({ field: "outputDir", role: "safety-field" }),
      ]),
    );
    expect(rowsByBasename.get("convert_py_to_notebook")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "notebookPath", role: "payload-bearing" }),
        expect.objectContaining({ field: "outputDir", role: "safety-field" }),
      ]),
    );

    const syncNotebookPairRow = rowsByBasename.get("sync_notebook_pair");
    expect(syncNotebookPairRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "notebookPath", role: "payload-bearing" }),
        expect.objectContaining({ field: "outputDir", role: "explicit-exception" }),
      ]),
    );
    expect(syncNotebookPairRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("unsupported outputDir key only to fail closed"),
      ]),
    );

    const runNotebookRow = rowsByBasename.get("run_notebook");
    expect(runNotebookRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "notebookPath", role: "payload-bearing" }),
        expect.objectContaining({ field: "script", role: "safety-field" }),
        expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
        expect.objectContaining({ field: "cwd", role: "safety-field" }),
        expect.objectContaining({ field: "writeExecuted", role: "safety-field" }),
        expect.objectContaining({ field: "skipValidation", role: "safety-field" }),
      ]),
    );
    expect(runNotebookRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          field: "cwd",
          notes: expect.stringContaining("does not repository-confine"),
        }),
        expect.objectContaining({
          field: "writeExecuted",
          notes: expect.stringContaining("without repository confinement"),
        }),
      ]),
    );
    expect(runNotebookRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("not repository confinement"),
      ]),
    );

    const runCppLintersRow = rowsByBasename.get("run_cpp_linters");
    expect(runCppLintersRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "sourceDir", role: "safety-field" }),
        expect.objectContaining({ field: "buildDir", role: "safety-field" }),
        expect.objectContaining({ field: "autoFix", role: "explicit-exception" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
      ]),
    );
    expect(runCppLintersRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("autoFix explicit as a bridge surface"),
      ]),
    );

    for (const wrapperName of ["run_cpp_lint_check", "run_cpp_lint_fix"]) {
      expect(rowsByBasename.get(wrapperName)?.field_role_audit).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ field: "sourceDir", role: "safety-field" }),
          expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
          expect.objectContaining({ field: "options", role: "explicit-exception" }),
        ]),
      );
    }

    const runCppCoverageSummaryRow = rowsByBasename.get("run_cpp_coverage_summary");
    expect(runCppCoverageSummaryRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "buildDir", role: "safety-field" }),
        expect.objectContaining({ field: "threshold", role: "explicit-exception" }),
        expect.objectContaining({ field: "tool", role: "explicit-exception" }),
        expect.objectContaining({ field: "html", role: "explicit-exception" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
      ]),
    );
    expect(runCppCoverageSummaryRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("advanced-only keys in schema space only to fail closed"),
        expect.stringContaining("extraArgs is intentionally excluded"),
      ]),
    );

    expect(rowsByBasename.get("run_cpp_coverage_advanced")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "buildDir", role: "safety-field" }),
        expect.objectContaining({ field: "filter", role: "payload-bearing" }),
        expect.objectContaining({ field: "html", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
      ]),
    );
    expect(rowsByBasename.get("run_cpp_coverage_advanced")?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("canonicalized before subprocess assembly"),
        expect.stringContaining("extraArgs is intentionally excluded"),
      ]),
    );

    const runSanitizersCompatibilityRow = rowsByBasename.get("run_sanitizers");
    expect(runSanitizersCompatibilityRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "buildDir", role: "safety-field" }),
        expect.objectContaining({ field: "executable", role: "safety-field" }),
        expect.objectContaining({ field: "sanitizer", role: "payload-bearing" }),
        expect.objectContaining({ field: "suppressions", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "extraArgs", role: "payload-bearing" }),
      ]),
    );
    expect(runSanitizersCompatibilityRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("bridge surface, not the target end-state"),
      ]),
    );

    const runSanitizersBasicRow = rowsByBasename.get("run_sanitizers_basic");
    expect(runSanitizersBasicRow?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "outputMode", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "buildDir", role: "safety-field" }),
        expect.objectContaining({ field: "executable", role: "safety-field" }),
        expect.objectContaining({ field: "sanitizer", role: "payload-bearing" }),
        expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
      ]),
    );
    expect(runSanitizersBasicRow?.audit_notes).toEqual(
      expect.arrayContaining([
        expect.stringContaining("keeps only shipped direct fields in counted inventory output"),
        expect.stringContaining("inventory output no longer represents them as accepted direct fields"),
      ]),
    );

    expect(rowsByBasename.get("run_sanitizers_advanced")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "suppressions", role: "safety-field" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
        expect.objectContaining({ field: "normalDuration", role: "explicit-exception" }),
        expect.objectContaining({ field: "extraArgs", role: "payload-bearing" }),
      ]),
    );

    const committed = JSON.parse(
      await readFile(
        path.join(REAL_REPO_ROOT, ".opencode/tools/wrapper_schema_inventory.json"),
        "utf8",
      ),
    );
    const committedRowsByBasename = new Map(
      committed.rows.map((row: { wrapper_path: string }) => [path.basename(row.wrapper_path, ".ts"), row]),
    );

    expect(committedRowsByBasename.get("validate_notebook")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "convertToPy", role: "explicit-exception" }),
      ]),
    );
    expect(committedRowsByBasename.get("run_cpp_coverage_summary")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "tool", role: "explicit-exception" }),
        expect.objectContaining({ field: "options", role: "explicit-exception" }),
      ]),
    );
    expect(committedRowsByBasename.get("run_sanitizers_basic")?.field_role_audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "outputMode", role: "bounded-option-candidate" }),
        expect.objectContaining({ field: "timeout", role: "explicit-exception" }),
      ]),
    );
  });

  it("ensures non-compliant rows have one allowed owner and non-placeholder target metadata", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);

    for (const row of artifact.rows) {
      if (row.classification === "already-compliant") {
        expect(row.owner_plan).toBeNull();
        continue;
      }

      expect(row.owner_plan).not.toBeNull();
      expect(ALLOWED_OWNER_PLANS.has(row.owner_plan!)).toBe(true);
      expect(["split-wrapper-family", "bounded-options", "compatibility-window"]).toContain(
        row.target_shape,
      );
    }
  });

  it("matches the committed inventory artifact", async () => {
    const generated = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const committed = JSON.parse(
      await readFile(
        path.join(REAL_REPO_ROOT, ".opencode/tools/wrapper_schema_inventory.json"),
        "utf8",
      ),
    );

    expect(generated).toEqual(committed);
  });

  it("resolves the repository root relative to the tooling module", () => {
    expect(getRepositoryRoot()).toBe(REAL_REPO_ROOT);
  });

  it("records a missing companion doc as null", async () => {
    const wrapperPath = path.join(REAL_REPO_ROOT, ".opencode/tools/adw_setup.ts");
    const sourceText = await Bun.file(wrapperPath).text();
    const row = inspectWrapperSourceText(wrapperPath, sourceText, makeContext(REAL_REPO_ROOT));

    expect(row.companion_doc_path).toBeNull();
  });

  it("can inventory a minimal synthetic repo without absolute-path-only output", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await writeFile(
        path.join(tempRoot, ".opencode/tools/sample_wrapper.ts"),
        [
          "import { tool } from '@opencode-ai/plugin';",
          "export default tool({",
          "  args: {",
          "    command: tool.schema.enum(['show']),",
          "    path: tool.schema.string().optional(),",
          "    help: tool.schema.boolean().optional(),",
          "  },",
          "  async execute() {",
          "    return 'ok';",
          "  },",
          "});",
          "",
        ].join("\n"),
        "utf8",
      );
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        JSON.stringify({ version: 1, wrappers: [] }, null, 2),
        "utf8",
      );

      const artifact = await generateWrapperSchemaInventory(tempRoot);
      expect(artifact.rows).toHaveLength(1);
      expect(artifact.rows[0]?.wrapper_path).toBe(".opencode/tools/sample_wrapper.ts");
      expect(artifact.rows[0]?.counted_fields).toEqual(["command", "path"]);
      expect(artifact.rows[0]?.exempt_fields).toEqual(["help"]);
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("excludes helper modules without tool registrations from generated inventory rows", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await Promise.all([
        writeFile(
          path.join(tempRoot, ".opencode/tools/sample_wrapper.ts"),
          [
            "import { tool } from '@opencode-ai/plugin';",
            "export default tool({",
            "  args: { command: tool.schema.enum(['show']) },",
            "  async execute() {",
            "    return 'ok';",
            "  },",
            "});",
            "",
          ].join("\n"),
          "utf8",
        ),
        writeFile(
          path.join(tempRoot, ".opencode/tools/helper_module.ts"),
          "export function helper(): string {\n  return 'ok';\n}\n",
          "utf8",
        ),
        writeFile(
          path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
          JSON.stringify({ version: 1, wrappers: [] }, null, 2),
          "utf8",
        ),
      ]);

      const artifact = await generateWrapperSchemaInventory(tempRoot);

      expect(artifact.rows).toHaveLength(1);
      expect(artifact.rows[0]?.wrapper_path).toBe(".opencode/tools/sample_wrapper.ts");
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("classifies representative compatibility, split, and unsplit wrappers in a synthetic repo", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });

      const wrapperSource = [
        "import { tool } from '@opencode-ai/plugin';",
        "export default tool({",
        "  args: { command: tool.schema.enum(['show']) },",
        "  async execute() {",
        "    return 'ok';",
        "  },",
        "});",
        "",
      ].join("\n");

      await Promise.all([
        writeFile(path.join(tempRoot, ".opencode/tools/run_pytest_basic.ts"), wrapperSource, "utf8"),
        writeFile(path.join(tempRoot, ".opencode/tools/git_diff.ts"), wrapperSource, "utf8"),
        writeFile(path.join(tempRoot, ".opencode/tools/get_datetime.ts"), wrapperSource, "utf8"),
        writeFile(path.join(tempRoot, ".opencode/tools/get_datetime.md"), "# custom\n", "utf8"),
        writeFile(
          path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
          JSON.stringify(COMPATIBILITY_METADATA_FIXTURES.approvedHistoricalConfig, null, 2),
          "utf8",
        ),
      ]);

      const artifact = await generateWrapperSchemaInventory(tempRoot);
      const kinds = Object.fromEntries(
        artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row.wrapper_kind]),
      );
      const customRow = artifact.rows.find((row) => row.wrapper_path.endsWith("get_datetime.ts"));

      expect(kinds.run_pytest_basic).toBe("split");
      expect(kinds.git_diff).toBe("split");
      expect(kinds.get_datetime).toBe("unsplit");
      expect(customRow?.companion_doc_path).toBe(".opencode/tools/get_datetime.md");
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("classifies active legacy unified wrappers as compatibility surfaces", async () => {
    const artifact = await generateWrapperSchemaInventory(REAL_REPO_ROOT);
    const kinds = Object.fromEntries(
      artifact.rows.map((row) => [path.basename(row.wrapper_path, ".ts"), row.wrapper_kind]),
    );

    expect(kinds.adw_notes).toBe("compatibility");
    expect(kinds.clear_build).toBe("compatibility");
    expect(kinds.move).toBe("compatibility");
    expect(kinds.workflow_builder).toBe("compatibility");
  });

  it("resolves the dedicated compatibility adw test before family fallbacks", () => {
    const testPathsByBasename = new Map<string, string>([
      ["adw", ".opencode/tools/__tests__/adw.test.ts"],
    ]);

    expect(resolveNearestTestPath("adw", testPathsByBasename)).toBe(
      ".opencode/tools/__tests__/adw.test.ts",
    );
  });

  it("uses explicit bounded-options metadata instead of incidental source text", () => {
    const repoRoot = "/repo";
    const boundedRow = inspectWrapperSourceText(
      "/repo/.opencode/tools/adw_plans_read.ts",
      [
        "import { tool } from '@opencode-ai/plugin';",
        "export default tool({",
        "  args: {",
        "    command: tool.schema.enum(['list']),",
        "    options: tool.schema.string().optional(),",
        "    help: tool.schema.boolean().optional(),",
        "  },",
        "  async execute() {",
        "    return 'ok';",
        "  },",
        "});",
        "",
      ].join("\n"),
      makeContext(repoRoot),
    );
    const unboundedRow = inspectWrapperSourceText(
      "/repo/.opencode/tools/custom_wrapper.ts",
      [
        "import { tool } from '@opencode-ai/plugin';",
        "// Optional options: harmless wording should not change counting.",
        "export default tool({",
        "  args: {",
        "    command: tool.schema.enum(['list']),",
        "    options: tool.schema.string().optional(),",
        "    help: tool.schema.boolean().optional(),",
        "  },",
        "  async execute() {",
        "    return 'ok';",
        "  },",
        "});",
        "",
      ].join("\n"),
      makeContext(repoRoot),
    );

    expect(boundedRow.counted_fields).toContain("command");
    expect(boundedRow.exempt_fields).toContain("options");
    expect(unboundedRow.counted_fields).toContain("command");
    expect(unboundedRow.counted_fields).toContain("options");
    expect(unboundedRow.exempt_fields).toEqual(["help"]);
  });

  it("counts quoted top-level args keys", () => {
    const repoRoot = "/repo";
    const row = inspectWrapperSourceText(
      "/repo/.opencode/tools/quoted_keys.ts",
      [
        "import { tool } from '@opencode-ai/plugin';",
        "export default tool({",
        "  args: {",
        '    "command": tool.schema.enum(["show"]),',
        "    'path': tool.schema.string().optional(),",
        "    help: tool.schema.boolean().optional(),",
        "  },",
        "  async execute() {",
        "    return 'ok';",
        "  },",
        "});",
        "",
      ].join("\n"),
      makeContext(repoRoot),
    );

    expect(row.status).toBe("ok");
    expect(row.counted_fields).toEqual(["command", "path"]);
    expect(row.exempt_fields).toEqual(["help"]);
  });

  it("classifies synthetic no_tool_schema and inspection_failed rows deterministically", () => {
    expect(classifyInventoryRow({ status: "no_tool_schema", current_count: 0 }, "helper_shared")).toBe(
      "already-compliant",
    );
    expect(classifyInventoryRow({ status: "inspection_failed", current_count: null }, "feedback_log")).toBe(
      "compatibility-defer-with-owner",
    );
  });

  it("includes symlinked wrapper candidates so inventory generation fails closed per row", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    const externalRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-external-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await Promise.all([
        writeFile(
          path.join(externalRoot, "escaped_wrapper.ts"),
          [
            "import { tool } from '@opencode-ai/plugin';",
            "export default tool({",
            "  args: { command: tool.schema.enum(['show']) },",
            "  async execute() {",
            "    return 'escaped';",
            "  },",
            "});",
            "",
          ].join("\n"),
          "utf8",
        ),
        writeFile(
          path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
          JSON.stringify({ version: 1, wrappers: [] }, null, 2),
          "utf8",
        ),
      ]);
      await symlink(
        path.join(externalRoot, "escaped_wrapper.ts"),
        path.join(tempRoot, ".opencode/tools/escaped_wrapper.ts"),
      );

      const artifact = await generateWrapperSchemaInventory(tempRoot);

      expect(artifact.rows).toHaveLength(1);
      expect(artifact.rows[0]?.wrapper_path).toBe(".opencode/tools/escaped_wrapper.ts");
      expect(artifact.rows[0]?.status).toBe("inspection_failed");
      expect(artifact.rows[0]?.diagnostic).toContain("Path resolves outside repository root");
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
      await rm(externalRoot, { recursive: true, force: true });
    }
  });

  it("fails closed when owner routing is missing or ambiguous", () => {
    expect(() => resolveOwnerPlanForBasename("unknown_wrapper")).toThrow(
      "Expected exactly one owner plan",
    );
  });

  it("prefers exact tests before fallback and uses audit baseline coverage when needed", () => {
    const testsByBasename = new Map<string, string>([
      ["adw_issues_batch_read", ".opencode/tools/__tests__/adw_issues_batch_read.test.ts"],
      ["wrapper_schema_inventory", ".opencode/tools/__tests__/wrapper_schema_inventory.test.ts"],
      [
        "platform_operations_compat_comment_pr_review",
        ".opencode/tools/__tests__/platform_operations_compat_comment_pr_review.test.ts",
      ],
    ]);

    expect(resolveNearestTestPath("platform_operations", testsByBasename)).toBe(
      ".opencode/tools/__tests__/platform_operations_compat_comment_pr_review.test.ts",
    );
    expect(resolveNearestTestPath("build_mkdocs", testsByBasename)).toBe(
      ".opencode/tools/__tests__/wrapper_schema_inventory.test.ts",
    );
    expect(resolveNearestTestPath("clear_build", testsByBasename)).toBe(
      ".opencode/tools/__tests__/wrapper_schema_inventory.test.ts",
    );
    expect(resolveNearestTestPath("run_cmake", testsByBasename)).toBe(
      ".opencode/tools/__tests__/wrapper_schema_inventory.test.ts",
    );
    expect(resolveNearestTestPath("run_ctest", testsByBasename)).toBe(
      ".opencode/tools/__tests__/wrapper_schema_inventory.test.ts",
    );
    expect(resolveNearestTestPath("run_pytest_advanced", testsByBasename)).toBe(
      ".opencode/tools/__tests__/wrapper_schema_inventory.test.ts",
    );
    expect(resolveNearestTestPath("workflow_builder", testsByBasename)).toBeNull();
  });

  it("downgrades oversized supporting metadata reads without aborting the inventory", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await writeFile(
        path.join(tempRoot, ".opencode/tools/feedback_log.ts"),
        [
          "import { tool } from '@opencode-ai/plugin';",
          "export default tool({",
          "  args: { command: tool.schema.enum(['show']) },",
          "  async execute() {",
          "    return 'ok';",
          "  },",
          "});",
          "",
        ].join("\n"),
        "utf8",
      );
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        "{" + "x".repeat(1024 * 1024 + 32),
        "utf8",
      );

      const artifact = await generateWrapperSchemaInventory(tempRoot);
      expect(artifact.rows).toHaveLength(1);
      expect(artifact.rows[0]?.status).toBe("inspection_failed");
      expect(artifact.rows[0]?.diagnostic).toContain("bounded read budget");
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("rejects symlinked temp output paths when writing the committed artifact", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    const externalRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-external-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await mkdir(path.join(tempRoot, "adforge_local/opencode"), { recursive: true });
      await writeFile(
        path.join(tempRoot, ".opencode/tools/sample_wrapper.ts"),
        [
          "import { tool } from '@opencode-ai/plugin';",
          "export default tool({",
          "  args: { command: tool.schema.enum(['show']) },",
          "  async execute() {",
          "    return 'ok';",
          "  },",
          "});",
          "",
        ].join("\n"),
        "utf8",
      );
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        JSON.stringify({ version: 1, wrappers: [] }, null, 2),
        "utf8",
      );
      await symlink(externalRoot, path.join(tempRoot, "adforge_local/opencode/tmp"));

      await expect(writeWrapperSchemaInventoryArtifact(tempRoot)).rejects.toThrow(
        "Refusing to traverse symlinked directory",
      );
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
      await rm(externalRoot, { recursive: true, force: true });
    }
  });

  it("writes the inventory artifact through the committed path with staged output", async () => {
    const tempRoot = await mkdtemp(path.join(os.tmpdir(), "wrapper-schema-inventory-"));
    try {
      await mkdir(path.join(tempRoot, ".opencode/tools"), { recursive: true });
      await mkdir(path.join(tempRoot, ".opencode/guides"), { recursive: true });
      await writeFile(
        path.join(tempRoot, ".opencode/tools/sample_wrapper.ts"),
        [
          "import { tool } from '@opencode-ai/plugin';",
          "export default tool({",
          "  args: {",
          "    command: tool.schema.enum(['show']),",
          "    help: tool.schema.boolean().optional(),",
          "  },",
          "  async execute() {",
          "    return 'ok';",
          "  },",
          "});",
          "",
        ].join("\n"),
        "utf8",
      );
      await writeFile(
        path.join(tempRoot, ".opencode/guides/tool-wrapper-exceptions.json"),
        JSON.stringify({ version: 1, wrappers: [] }, null, 2),
        "utf8",
      );

      const artifact = await writeWrapperSchemaInventoryArtifact(tempRoot);
      const writtenText = await readFile(
        path.join(tempRoot, ".opencode/tools/wrapper_schema_inventory.json"),
        "utf8",
      );

      expect(JSON.parse(writtenText)).toEqual(artifact);
      expect(writtenText.endsWith("\n")).toBe(true);
    } finally {
      await rm(tempRoot, { recursive: true, force: true });
    }
  });
});
