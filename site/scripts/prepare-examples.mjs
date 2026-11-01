import { readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
const data = JSON.parse(readFileSync("public/research.json", "utf8"));
if (data.examples.length > 24) throw new Error("Example limit exceeded");
for (const e of data.examples) {
  if (
    !/^COCO_val2014_\d{12}\.jpg$/.test(e.image) ||
    e.imageSrc !== `/examples/${e.image}` ||
    !/^[a-f0-9]{64}$/.test(e.imageSha256)
  )
    throw new Error("Invalid pinned COCO asset");
  const target = path.join("public/examples", e.image);
  const hash = (bytes) =>
    crypto.createHash("sha256").update(bytes).digest("hex");
  if (existsSync(target) && hash(readFileSync(target)) === e.imageSha256)
    continue;
  const response = await fetch(
    `https://s3.amazonaws.com/images.cocodataset.org/val2014/${e.image}`,
    { signal: AbortSignal.timeout(30000) },
  );
  if (!response.ok)
    throw new Error(`COCO asset download failed: ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length > 8 * 1024 * 1024 || hash(bytes) !== e.imageSha256)
    throw new Error("COCO asset hash mismatch");
  mkdirSync("public/examples", { recursive: true });
  writeFileSync(target, bytes);
}
console.log(
  `Verified ${data.examples.length} static photo assets; photographs remain outside Git.`,
);
