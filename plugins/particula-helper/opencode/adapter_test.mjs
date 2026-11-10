import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";

// Pass a freshly extracted npm package to test the shipped artifact.
const entry = process.env.PARTICULA_PLUGIN_ENTRY || new URL("./index.js", import.meta.url).href;
const { default: plugin, loadSkills } = await import(
  entry.startsWith("file:") ? entry : pathToFileURL(resolve(entry)).href
);

test("installed package loads six skills and resolves supporting files", async () => {
  const skills = await loadSkills();
  assert.equal(skills.length, 6);
  assert.equal(new Set(skills.map((s) => s.id)).size, 6);
  for (const skill of skills) {
    assert.ok(skill.description);
    assert.ok(!skill.content.startsWith("---"));
    await access(skill.path);
    for (const match of skill.content.matchAll(/`((?:\.\.\/|\.\/)[^`\s]+)/g)) {
      await access(resolve(dirname(skill.path), match[1]));
    }
    assert.match(await readFile(skill.path, "utf8"), /Shared operating rules/);
  }
});

test("transform reload preserves duplicates and unrelated skills; cleanup disposes", async () => {
  let transform;
  let disposed = false;
  const cleanup = await plugin.setup({ skill: {
    transform: async (callback) => {
      transform = callback;
      return { dispose: async () => { disposed = true; } };
    },
  } });
  const custom = { id: "particula-setup", content: "project-owned" };
  for (let reload = 0; reload < 2; reload++) {
    const entries = new Map([[custom.id, custom], ["other", { id: "other" }]]);
    transform({ get: (id) => entries.get(id), add: (skill) => {
      assert.ok(!entries.has(skill.id));
      entries.set(skill.id, skill);
    } });
    assert.equal(entries.size, 7);
    assert.equal(entries.get(custom.id), custom);
    transform({ get: (id) => entries.get(id), add: () => assert.fail("duplicate") });
  }
  await cleanup();
  assert.ok(disposed);
});
