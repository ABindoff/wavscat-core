// Run a wasm32-wasip1 test binary under Node's built-in WASI.
//
// Usage: node tools/run-wasi.mjs <test.wasm> [args...]
// The golden record is embedded in the binary, so no files are mounted.
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import { argv, exit } from "node:process";

const wasi = new WASI({
  version: "preview1",
  args: [argv[2], ...argv.slice(3)],
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(argv[2]));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
