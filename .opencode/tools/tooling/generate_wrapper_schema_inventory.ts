import { writeWrapperSchemaInventoryArtifact } from "./wrapper_schema_inventory";

/**
 * Repo-local generator entrypoint for the committed wrapper schema inventory.
 * Run with `bun .opencode/tools/tooling/generate_wrapper_schema_inventory.ts`.
 */
async function main(): Promise<void> {
  const artifact = await writeWrapperSchemaInventoryArtifact();
  process.stdout.write(`${JSON.stringify(artifact, null, 2)}\n`);
}

if (import.meta.main) {
  await main();
}
