import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { Plugin, Skill } from "@opencode/plugin";

// Read package-local resources, never the active project's files.
export async function loadSkills() {
  const catalog = JSON.parse(
    await readFile(new URL("./payload/catalog.json", import.meta.url), "utf8"),
  );
  return Promise.all(catalog.map(async ({ name, description }) => {
    const url = new URL(`./payload/skills/${name}/SKILL.md`, import.meta.url);
    const markdown = await readFile(url, "utf8");
    return {
      id: Skill.ID.make(name),
      name: Skill.Name.make(name),
      description,
      path: Skill.Info.fields.path.make(fileURLToPath(url)),
      content: markdown.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, ""),
    };
  }));
}

export default Plugin.define({
  id: "particula-helper",
  async setup(ctx) {
    const skills = await loadSkills();
    const registration = await ctx.skill.transform((editor) => {
      for (const skill of skills) {
        // Preserve an existing project/user skill with the same ID.
        if (!editor.get(skill.id)) editor.add(skill);
      }
    });
    return () => registration.dispose();
  },
});
