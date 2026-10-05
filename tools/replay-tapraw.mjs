// Replay a .tapraw video, saved by apps/tap-compare, through the current
// tapping pipeline under Node, and print a summary: QC, the camera clock,
// the selected rhythm, where the tapping was, and the interval features.
// Build the wasm module first with tools/build-wasm.sh.
//
//     node tools/replay-tapraw.mjs trial.tapraw [more.tapraw ...] [--json out.json]
//
// --json writes each video's full diagnose() output, keyed by file name.
import { readFileSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const w = require("../crates/wavscat-wasm/pkg-node/wavscat_wasm.js");

const YUV = new Set(["I420", "I420A", "I422", "I444", "NV12"]);

export function readTapraw(path) {
  const bytes = readFileSync(path);
  if (bytes.subarray(0, 8).toString("latin1") !== "TAPRAW01") throw new Error(`${path} is not a .tapraw video`);
  const n = bytes.readUInt32LE(8);
  const header = JSON.parse(bytes.subarray(12, 12 + n).toString("utf8"));
  let offset = 12 + n;
  const frames = header.frames.map((f) => {
    const data = bytes.subarray(offset, offset + f.bytes);
    offset += f.bytes;
    return { ...f, data };
  });
  return { header, frames };
}

export function replay(path) {
  const { header, frames } = readTapraw(path);
  const session = new w.TappingSession({ capacity: Math.max(1024, frames.length) });
  for (const f of frames) {
    const plane = f.layout[0];
    const view = new Uint8Array(f.data.buffer, f.data.byteOffset + plane.offset, f.data.length - plane.offset);
    if (YUV.has(f.format)) {
      session.pushFrame(view, plane.stride, f.w, f.h, f.us);
    } else if (f.format === "RGBA" || f.format === "RGBX") {
      session.pushRgba(view, plane.stride, f.w, f.h, f.us);
    } else if (f.format === "BGRA" || f.format === "BGRX") {
      const rgba = new Uint8Array(view.length);
      for (let i = 0; i + 3 < view.length; i += 4) {
        rgba[i] = view[i + 2];
        rgba[i + 1] = view[i + 1];
        rgba[i + 2] = view[i];
        rgba[i + 3] = 255;
      }
      session.pushRgba(rgba, plane.stride, f.w, f.h, f.us);
    } else {
      throw new Error(`unsupported pixel format ${f.format}`);
    }
  }
  const d = session.diagnose();
  session.free();
  return { header, d };
}

function summary(name, header, d) {
  const q = d.report.qc;
  const a = d.analysis;
  const fmt = (x, k = 3) => (x === null || x === undefined || Number.isNaN(x) ? "-" : Number(x).toFixed(k));
  const lines = [`== ${name}: trial ${header.trial}, ${header.frames.length} frames, ${header.frames[0]?.format}, clock ${header.timestamp_source}`];
  lines.push(`QC: ${d.report.accepted ? "accepted" : `rejected: ${d.report.reasons.join("; ")}`}`);
  lines.push(`capture: ${fmt(q.effective_fps, 1)} fps effective, dropped ${fmt(100 * q.dropped_fraction, 1)}%, rate changes ${q.rate_changes ?? "-"}, timestamp error removed (RMS) ${fmt(1000 * (q.clock_rms ?? NaN), 1)} ms`);
  if (a?.clock) {
    for (const s of a.clock.segments) {
      lines.push(`  clock segment frames ${s.start}-${s.end - 1}: ${fmt(1 / s.period, 2)} fps, RMS residual ${fmt(1000 * s.rms_residual, 1)} ms, missing ${s.missing}, ${s.regular ? "regular" : "kept raw"}`);
    }
  }
  if (!a) return lines.join("\n");
  lines.push(`rhythm: f0 ${fmt(a.f0_hz)} Hz${a.from_harmonic ? " (from a harmonic)" : ""}, component ${a.component + 1}, competitor ratio ${fmt(a.competitor_ratio, 2)}`);
  lines.push(`  ranking: ${a.ranking.map((r) => `${r.component + 1}: score ${fmt(r.score, 2)} peak ${fmt(r.peak_hz, 2)} f0 ${fmt(r.f0_hz, 2)}${r.excluded ? " excluded" : ""}`).join("; ")}`);
  if (a.locations) lines.push(`motion box: ${a.locations.windows.length} windows, travel ${fmt(a.locations.travel, 1)} cells, largest jump ${fmt(a.locations.max_step, 1)} cells`);
  lines.push(`cycles: ${a.itis.length}, usable ${a.usable.filter((u) => u).length}`);
  if (a.features) {
    const keep = a.features.names.map((n, i) => [n, a.features.values[i]]).filter(([n]) => !n.startsWith("jtfs_"));
    lines.push(`features: ${keep.map(([n, v]) => `${n} ${fmt(v, 4)}`).join(", ")}`);
  } else {
    lines.push(`features: ${a.features_error}`);
  }
  return lines.join("\n");
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, "/").replace(/^\//, "")}` || process.argv[1].endsWith("replay-tapraw.mjs")) {
  const args = process.argv.slice(2);
  const j = args.indexOf("--json");
  const out = j >= 0 ? args.splice(j, 2)[1] : null;
  const all = {};
  for (const path of args) {
    const { header, d } = replay(path);
    console.log(summary(basename(path), header, d));
    all[basename(path)] = d;
  }
  if (out) writeFileSync(out, JSON.stringify(all));
}
