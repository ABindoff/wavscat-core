// Run a wasm32-wasip1 test binary under Node's built-in WASI.
//
// Usage: node tools/run-wasi.mjs <test.wasm> [args...]
// The repository's golden/ directory is mounted at /golden.
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import { argv, env, exit } from "node:process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const wasi = new WASI({
  version: "preview1",
  args: [argv[2], ...argv.slice(3)],
  env: { ...env, WAVSCAT_GOLDEN_DIR: "/golden" },
  preopens: { "/golden": join(root, "golden") },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(argv[2]));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
