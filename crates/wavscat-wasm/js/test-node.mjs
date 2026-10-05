// Tests of the wasm binding under Node. Build first with tools/build-wasm.sh.
//
//     node --test crates/wavscat-wasm/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const w = require("../pkg-node/wavscat_wasm.js");

test("every golden output bit matches the reference record", () => {
  const diffs = w.verify();
  assert.deepEqual(diffs, [], diffs.join("\n"));
});

test("numerics version is reported", () => {
  assert.match(w.numericsVersion(), /^\d+\.\d+\.\d+/);
});

// 30 s of a 3 Hz tap trace at 30 Hz, the video pipeline's case.
const sr = 30;
const n = 900;
const x = Float64Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * 3 * i) / sr));

test("time scattering: paths, shapes and labels agree", () => {
  const op = new w.Scattering1d({ n, J: 7, Q: [8, 1], T_sec: 4, sr });
  const paths = op.paths();
  const S = op.transform(x);
  assert.equal(S.length, paths.length);
  assert.equal(S.data.length, S.cols.reduce((a, c) => a + c, 0));
  assert.equal(paths[0].path, "S0");
  assert.equal(paths[1].path, "S1_1");
  assert.ok(paths.some((p) => p.order === 2 && p.path.startsWith("S2_")));
  // All paths share one time axis with local averaging.
  assert.ok(Array.from(S.cols).every((c) => c === S.cols[0]));
  // Renormalisation keeps the shape and leaves orders 0 and 1 alone.
  const R = op.renorm(S, 1e-12);
  assert.equal(R.length, S.length);
  assert.deepEqual(Array.from(R.path(1)), Array.from(S.path(1)));
});

test("joint scattering: both formats", () => {
  const time = new w.ScatteringJtfs({ n, J: 7, J_fr: 3, Q: 8, T_sec: 6, sr });
  const St = time.transform(x);
  assert.equal(St.length, time.paths().length);
  assert.ok(Array.from(St.rows).every((r) => r === 1));
  assert.ok(time.paths().some((p) => p.spin === 1) && time.paths().some((p) => p.spin === -1));

  const joint = new w.ScatteringJtfs({ n, J: 7, J_fr: 3, Q: 8, T_sec: 6, sr, format: "joint" });
  const Sj = joint.transform(x);
  assert.equal(Sj.length, joint.paths().length);
  assert.ok(Array.from(Sj.rows).some((r) => r > 1));
});

test("feature helpers", () => {
  const v = Float64Array.from([3, 1, 4, 1, 5, 9]);
  assert.equal(w.summarise(v, "max"), 9);
  assert.equal(w.summarise(v, "median"), 3.5);
  assert.equal(w.epsQuantile(Float64Array.from([10, 3, 1, 4, 2]), 0.3), 2.2);
  const lc = w.logCompress(Float64Array.from([-2, 0, 2]), 1);
  assert.equal(lc[1], 0);
  assert.equal(lc[0], -lc[2]);
});

test("bad parameters fail loudly", () => {
  assert.throws(() => new w.Scattering1d({ n, J: 7, Qq: 8 }), /unknown field/);
  assert.throws(() => new w.Scattering1d({ n, J: 12 }), /exceeds the signal length/);
  assert.throws(() => new w.Scattering1d({ n, J: 7, T: "local" }), /global/);
  const op = new w.Scattering1d({ n, J: 7 });
  assert.throws(() => op.transform(new Float64Array(10)), /length 900/);
  assert.throws(() => op.transform(Float64Array.from({ length: n }, () => NaN)), /non-finite/);
});
