// Tests of the video tapping session in the wasm binding, under Node.
// Build first with tools/build-wasm.sh.
//
//     node --test crates/wavscat-wasm/js/
import { test } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const w = require("../pkg-node/wavscat_wasm.js");

/// Render 30 s of a synthetic video and push it, frame by frame.
function capture(session, videoParams, staged) {
  const video = new w.SyntheticVideo(videoParams);
  const { width, height } = video;
  const frame = new Uint8Array(width * height);
  for (const t of w.frameTimes(30, 30, 0.004, 0.05, 41n)) {
    video.render(t, frame);
    const us = Math.round(t * 1e6);
    if (staged) {
      // Write straight into wasm memory, as VideoFrame.copyTo would.
      const ptr = session.stagingPointer(frame.length);
      new Uint8Array(w.wasmMemory().buffer, ptr, frame.length).set(frame);
      session.pushStaged(width, width, height, us, false);
    } else {
      session.pushFrame(frame, width, width, height, us);
    }
  }
  return video;
}

test("verify covers the video pipeline too", () => {
  assert.deepEqual(w.verify(), []);
});

test("a clean synthetic trial is accepted with features", () => {
  const session = new w.TappingSession();
  capture(session, { iti_sd: 0.01 }, false);
  assert.ok(session.frames > 800);
  const r = session.finish();
  assert.equal(r.accepted, true, r.reasons.join("; "));
  assert.ok(Math.abs(r.qc.f0_hz - 3) < 0.06, `f0 ${r.qc.f0_hz}`);
  assert.equal(r.features.names.length, r.features.values.length);
  assert.equal(r.features.names[0], "f0_hz");
  assert.ok(r.features.names.some((n) => n.startsWith("jtfs_J2")));
  assert.match(r.features.params_hash, /^[0-9a-f]{16}$/);
  const meanIti = r.itis.reduce((a, b) => a + b, 0) / r.itis.length;
  assert.ok(Math.abs(meanIti - 1 / 3) < 0.003, `mean ITI ${meanIti}`);
});

test("the zero-copy staging path gives the same report", () => {
  const a = new w.TappingSession();
  const b = new w.TappingSession();
  capture(a, { iti_sd: 0.01, seed: 3n }, false);
  capture(b, { iti_sd: 0.01, seed: 3n }, true);
  assert.deepEqual(b.finish(), a.finish());
});

test("a second rhythmic object is rejected with a reason", () => {
  const session = new w.TappingSession();
  capture(session, { iti_sd: 0.01, distractor_hz: 4.5 }, false);
  const r = session.finish();
  assert.equal(r.accepted, false);
  assert.equal(r.features, null);
  assert.ok(r.reasons.some((s) => s.includes("another rhythmic movement")), r.reasons.join("; "));
});

test("reset starts a new trial, and bad input is refused", () => {
  const session = new w.TappingSession({ capacity: 64 });
  capture(session, { width: 160, height: 120 }, false);
  assert.equal(session.frames, 64);
  session.reset();
  assert.equal(session.frames, 0);
  const frame = new Uint8Array(160 * 120);
  session.pushFrame(frame, 160, 160, 120, 1000);
  assert.throws(() => session.pushFrame(frame, 160, 160, 120, 1000), /does not follow/);
  assert.throws(() => new w.TappingSession({ capacty: 10 }), /unknown field/);
});

test("diagnose explains a rejected trial, and a landmark-like trace matches the video", () => {
  const session = new w.TappingSession();
  const video = capture(session, { iti_sd: 0.01, distractor_hz: 4.5 }, false);
  const d = session.diagnose();
  assert.equal(d.report.accepted, false);
  const a = d.analysis;
  assert.equal(a.loading.length, a.grid_width * a.grid_height);
  assert.equal(a.signal.length > 800, true);
  assert.equal(a.features.names[0], "f0_hz", a.features_error);
  assert.equal(d.timestamps.length, session.frames);

  // A trace that rises and falls once per true tap, as a fingertip
  // distance does, sampled at the frame times.
  const taps = video.taps();
  const phase = (t) => {
    let k = taps.findIndex((x) => x > t);
    if (k <= 0) return NaN;
    return k - 1 + (t - taps[k - 1]) / (taps[k] - taps[k - 1]);
  };
  const times = [], values = [];
  for (const t of d.timestamps) {
    const p = phase(t + d.timestamps[0]);
    if (Number.isFinite(p)) { times.push(t); values.push(2 + Math.cos(2 * Math.PI * p)); }
  }
  const r = w.analyseTrace(new Float64Array(times), new Float64Array(values));
  assert.ok(Math.abs(r.f0_hz - 3) < 0.06, `trace f0 ${r.f0_hz}`);
  assert.equal(r.features.names.length, a.features.names.length);
  assert.match(r.params_hash, /^[0-9a-f]{16}$/);
  assert.throws(() => w.analyseTrace(new Float64Array([0, 1]), new Float64Array([1])), /one timestamp per value/);
});
