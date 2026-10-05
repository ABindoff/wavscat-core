// Run verify() in real browsers and require every golden bit to match.
//
//     node test-browser.mjs                  # chromium, firefox, webkit
//     BROWSERS=chrome,msedge node test-browser.mjs   # installed browsers
//
// Names "chromium", "firefox" and "webkit" use Playwright's own builds
// (npx playwright install). Any other name is launched as a Chromium channel,
// such as an installed "chrome" or "msedge".
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, firefox, webkit } from "playwright";

const root = join(fileURLToPath(import.meta.url), "..", "..");
const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm" };

// Serve the crate directory: browser.html and the pkg-web build next to it.
const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  const path = url.pathname === "/" ? "/js/browser.html" : url.pathname;
  const file = normalize(join(root, path));
  if (!file.startsWith(root)) {
    res.writeHead(403).end();
    return;
  }
  try {
    const body = await readFile(file);
    res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;

const engines = { chromium, firefox, webkit };
const names = (process.env.BROWSERS ?? "chromium,firefox,webkit").split(",").map((s) => s.trim());

let failed = false;
for (const name of names) {
  const browser = engines[name]
    ? await engines[name].launch()
    : await chromium.launch({ channel: name });
  try {
    const page = await browser.newPage();
    // The page imports ../pkg-web relative to js/, so serve it from js/.
    await page.goto(`${base}/js/browser.html`);
    await page.waitForFunction(() => window.__wavscat !== undefined, null, { timeout: 120_000 });
    const r = await page.evaluate(() => window.__wavscat);
    if (r.error || r.diffs.length > 0) {
      failed = true;
      console.error(`FAIL ${name}: ${r.error ?? r.diffs.join("\n")}`);
    } else {
      console.log(`ok   ${name} (${browser.version()}): numerics ${r.version}, verify ${r.ms.toFixed(0)} ms`);
    }
  } finally {
    await browser.close();
  }
}
server.close();
process.exit(failed ? 1 : 0);
