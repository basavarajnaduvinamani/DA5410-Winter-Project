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
console.log(
  `Wrote ${count} static segment aliases for portable App Router navigation.`,
);
