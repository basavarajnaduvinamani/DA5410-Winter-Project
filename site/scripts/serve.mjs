import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
const root = path.resolve("out");
const port = Number(process.env.PORT || 4173);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".woff2": "font/woff2",
  ".pdf": "application/pdf",
  ".txt": "text/plain; charset=utf-8",
};
http
  .createServer(async (req, res) => {
    try {
      let file = path.resolve(
        root,
        "." + decodeURIComponent(new URL(req.url, "http://localhost").pathname),
      );
      if (file !== root && !file.startsWith(root + path.sep)) {
        res.writeHead(403).end();
        return;
      }
      let info = await stat(file).catch(() => null);
      if (info?.isDirectory()) file = path.join(file, "index.html");
      let bytes = await readFile(file).catch(() => null);
      if (!bytes) {
        file = path.join(root, "404.html");
        bytes = await readFile(file);
        res.statusCode = 404;
      }
      res.setHeader(
        "Content-Type",
        path.basename(file).startsWith("__next.") && file.endsWith(".txt")
          ? "text/x-component"
          : types[path.extname(file)] || "application/octet-stream",
      );
      res.setHeader("X-Content-Type-Options", "nosniff");
      res.end(bytes);
    } catch {
      res.writeHead(500).end("Unable to serve the static export");
    }
  })
  .listen(port, "127.0.0.1", () =>
    console.log(`Static export: http://127.0.0.1:${port}`),
  );
