import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
let count = 0;
async function walk(dir) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name.startsWith("__next.")) {
        for (const child of await readdir(p, { withFileTypes: true })) {
          if (child.isFile() && child.name.endsWith(".txt")) {
            await writeFile(
              path.join(dir, e.name + "." + child.name),
              await readFile(path.join(p, child.name)),
            );
            count++;
          }
        }
      }
      await walk(p);
    }
  }
}
await walk("out");

// The landing page uses static SVGs, native links and a details disclosure.
// Skip the framework bootstrap there. Interactive research routes keep it.
const home = path.join("out", "index.html");
const markup = await readFile(home, "utf8");
await writeFile(home, markup
  .replace(/<script\b(?![^>]*type="application\/ld\+json")[^>]*>[\s\S]*?<\/script>/gi, "")
  .replace(/<link\b(?=[^>]*\bas="script")[^>]*>/gi, ""));
console.log(
  `Wrote ${count} static segment aliases for portable App Router navigation.`,
);
