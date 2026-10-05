// Webcam tapping: the landmark-free video pipeline (tapvid, through wasm)
// against MediaPipe hand landmarks, on the same frames.
//
// Every captured frame goes to a TappingSession at once, and a snapshot of
// the same pixels goes to MediaPipe Hands in a worker, with one timestamp;
// hand detection never delays capture. After a trial, each hand's landmark
// trace is analysed by analyseTrace, which runs it through the same drift
// removal, resampling, fundamental, cycle timing and features as the video's
// selected component, so the two are compared like for like.
import init, { TappingSession, analyseTrace, verify } from "../../crates/wavscat-wasm/pkg-web/wavscat_wasm.js";

const RECORD_SEC = 10;
const COUNTDOWN_SEC = 3;
// Frames the landmark worker may fall behind by before frames are sent to it
// no more; those frames get no landmarks, and the video keeps every frame.
const MAX_PENDING = 15;
// Preview frames hand tracking must process before Record is enabled.
const WARMUP_FRAMES = 30;

const TRIALS = [
  {
    id: "left",
    title: "Left hand",
    text: "Tap your left thumb and index finger together, as quickly and as widely as you can, for 10 seconds. Keep your right hand still in your lap.",
    hands: ["left"],
  },
  {
    id: "right",
    title: "Right hand",
    text: "Tap your right thumb and index finger together, as quickly and as widely as you can, for 10 seconds. Keep your left hand still in your lap.",
    hands: ["right"],
  },
  {
    id: "both",
    title: "Both hands, anti-phase",
    text: "Tap with both hands at once, alternating: the left closes as the right opens. Keep a steady rhythm for 10 seconds, with both hands in view.",
    hands: ["left", "right", "left − right"],
  },
];

// MediaPipe hand landmark indices.
const WRIST = 0, THUMB_TIP = 4, INDEX_MCP = 5, INDEX_TIP = 8, MIDDLE_MCP = 9;
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLOURS = { video: () => css("--video"), left: () => css("--left"), right: () => css("--right"), "left − right": () => css("--diff") };

const state = {
  trialIndex: 0,
  phase: "idle", // idle | preview | countdown | recording | draining | analysing
  worker: null,
  delegate: null,
  session: null,
  canvas: null,
  ctx: null,
  // { us, id, hands: null until the worker answers, then [{ x, y, label, score }] }
  frames: [],
  byId: new Map(),
  nextId: 0,
  pending: 0,
  skipped: 0,
  previewSeen: 0,
  // Options, read from the page when a trial starts.
  tracking: true, // send frames to MediaPipe at all
  mpScale: 0.5, // MediaPipe input size, as a fraction of the camera frame
  mpCanvas: null,
  mpCtx: null,
  detectMs: [],
  firstUs: null,
  lastUs: null,
  lastMs: -1,
  phaseStart: 0,
  results: {},
  camera: null,
  verified: null,
};
// For testing from the console or a headless browser.
window.tapCompare = { state, TRIALS };

// ---------------------------------------------------------------- setup

async function setup() {
  status("Loading the wasm module…");
  await init();
  const bad = verify();
  state.verified = bad.length === 0;
  status(state.verified ? "wasm loaded; this device reproduces every reference bit. Loading MediaPipe…"
    : `wasm loaded, but this device failed ${bad.length} reference checks. Loading MediaPipe…`);
  state.worker = new Worker(new URL("landmarks-worker.js", import.meta.url), { type: "module" });
  await new Promise((resolve, reject) => {
    state.worker.onmessage = ({ data }) => {
      if (data.type === "ready") {
        state.delegate = data.delegate;
        resolve();
      } else if (data.type === "error") {
        reject(new Error(data.message));
      }
    };
    state.worker.onerror = (e) => reject(new Error(e.message || "the landmark worker failed to start"));
  });
  state.worker.onmessage = ({ data }) => onLandmarks(data);
  status(`Ready (MediaPipe on ${state.delegate}). Start the camera.`);
  $("start").disabled = false;
  renderChips();
  showTrial();
}

async function startCamera() {
  $("start").disabled = true;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: "user" },
    audio: false,
  });
  const video = $("video");
  video.srcObject = stream;
  await video.play();
  const s = stream.getVideoTracks()[0].getSettings();
  state.camera = { width: video.videoWidth, height: video.videoHeight, frameRate: s.frameRate ?? null, label: stream.getVideoTracks()[0].label };
  state.canvas = new OffscreenCanvas(video.videoWidth, video.videoHeight);
  state.ctx = state.canvas.getContext("2d", { willReadFrequently: true });
  sizeMediapipeCanvas();
  const overlay = $("overlay");
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  state.phase = "preview";
  readOptions();
  if (state.tracking) {
    status(`Camera ${video.videoWidth} x ${video.videoHeight}${state.camera.frameRate ? ` at ${state.camera.frameRate} fps` : ""}. Warming up hand tracking…`);
  } else {
    $("record").disabled = false;
    status(`Camera ${video.videoWidth} x ${video.videoHeight}${state.camera.frameRate ? ` at ${state.camera.frameRate} fps` : ""}. Hand tracking is off. Ready to record.`);
  }
  scheduleFrame();
}

// ---------------------------------------------------------------- capture

function scheduleFrame() {
  const video = $("video");
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback((now, meta) => {
      onFrame(meta.captureTime ?? meta.presentationTime ?? now);
      scheduleFrame();
    });
  } else {
    // Without frame callbacks, poll and take a new frame when the video's
    // time moves on.
    let last = -1;
    const poll = (now) => {
      if (video.currentTime !== last) {
        last = video.currentTime;
        onFrame(now);
      }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  }
}

/// The canvas MediaPipe's frames are drawn into, at `mpScale` of the camera
/// frame. Landmarks are in 0-1 image coordinates, so the scale changes their
/// precision but not their meaning.
function sizeMediapipeCanvas() {
  const w = Math.max(1, Math.round(state.camera.width * state.mpScale));
  const h = Math.max(1, Math.round(state.camera.height * state.mpScale));
  if (state.mpCanvas?.width === w && state.mpCanvas?.height === h) return;
  state.mpCanvas = new OffscreenCanvas(w, h);
  state.mpCtx = state.mpCanvas.getContext("2d");
}

/// Draw the current video frame, at MediaPipe's size, and send it to the
/// landmark worker. The bitmap is transferred, not copied.
function sendToWorker(id, tMs) {
  const video = $("video");
  state.mpCtx.drawImage(video, 0, 0, state.mpCanvas.width, state.mpCanvas.height);
  const bitmap = state.mpCanvas.transferToImageBitmap();
  state.pending += 1;
  state.worker.postMessage({ type: "frame", id, tMs, bitmap }, [bitmap]);
}

function onFrame(tMs) {
  const video = $("video");
  if (tMs <= state.lastMs) return;
  state.lastMs = tMs;
  const { width, height } = state.canvas;

  if (state.phase !== "recording") {
    // Preview and countdown: landmarks only, for the overlay, and only when
    // the worker is idle so that no backlog builds before recording.
    if (state.tracking && state.pending === 0 && (state.phase === "preview" || state.phase === "countdown")) {
      sendToWorker(-1, tMs);
    }
    if (state.phase === "countdown") {
      const elapsed = (performance.now() - state.phaseStart) / 1000;
      $("big").textContent = String(Math.max(1, Math.ceil(COUNTDOWN_SEC - elapsed)));
      if (elapsed >= COUNTDOWN_SEC) {
        state.phase = "recording";
        state.phaseStart = performance.now();
        $("big").textContent = "";
      }
    }
    return;
  }

  const us = Math.round(tMs * 1000);
  if (state.lastUs !== null && us <= state.lastUs) return;
  state.ctx.drawImage(video, 0, 0, width, height);
  const img = state.ctx.getImageData(0, 0, width, height);
  state.session.pushRgba(img.data, width * 4, width, height, us);
  img.data.fill(0);
  if (state.firstUs === null) state.firstUs = us;
  state.lastUs = us;
  const frame = { us, id: state.nextId++, hands: null };
  state.frames.push(frame);
  if (!state.tracking) {
    // Video only: no frame goes to MediaPipe.
  } else if (state.pending < MAX_PENDING) {
    state.byId.set(frame.id, frame);
    sendToWorker(frame.id, tMs);
  } else {
    state.skipped += 1;
  }

  const elapsed = (performance.now() - state.phaseStart) / 1000;
  $("progress").style.width = `${Math.min(100, (100 * elapsed) / RECORD_SEC)}%`;
  if (elapsed >= RECORD_SEC) {
    state.phase = "draining";
    $("progress").style.width = "100%";
    status("Waiting for the last landmarks…");
    drain();
  }
}

function onLandmarks(data) {
  if (data.type !== "landmarks") return;
  state.pending -= 1;
  drawHands(data);
  if (data.id < 0) {
    // Record becomes available once hand tracking has run steadily on the
    // preview for a while: its first frames can stall the camera.
    state.previewSeen += 1;
    if (state.previewSeen === WARMUP_FRAMES && state.phase === "preview" && $("record").disabled) {
      $("record").disabled = false;
      status(`${$("status").textContent.replace(" Warming up hand tracking…", "")} Ready to record.`);
    }
    return;
  }
  state.detectMs.push(data.ms);
  const frame = state.byId.get(data.id);
  if (!frame) return;
  state.byId.delete(data.id);
  frame.hands = data.landmarks.map((lm, i) => ({
    x: Float32Array.from(lm, (p) => p.x),
    y: Float32Array.from(lm, (p) => p.y),
    label: data.handedness[i]?.[0]?.categoryName ?? "",
    score: data.handedness[i]?.[0]?.score ?? 0,
  }));
}

/// Analyse once the worker has answered for every frame sent to it.
function drain() {
  const start = performance.now();
  const wait = () => {
    if (state.byId.size === 0 || performance.now() - start > 10000) {
      state.phase = "analysing";
      status("Analysing…");
      setTimeout(analyse, 0);
    } else {
      setTimeout(wait, 20);
    }
  };
  wait();
}

function drawHands(res) {
  const c = $("overlay");
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  g.lineWidth = 3;
  for (const lm of res.landmarks) {
    g.strokeStyle = "rgba(255,255,255,0.85)";
    g.beginPath();
    for (const [a, b] of HAND_EDGES) {
      g.moveTo(lm[a].x * c.width, lm[a].y * c.height);
      g.lineTo(lm[b].x * c.width, lm[b].y * c.height);
    }
    g.stroke();
    for (const k of [THUMB_TIP, INDEX_TIP]) {
      g.fillStyle = "#facc15";
      g.beginPath();
      g.arc(lm[k].x * c.width, lm[k].y * c.height, 6, 0, 2 * Math.PI);
      g.fill();
    }
  }
}

function readOptions() {
  state.tracking = $("tracking").checked;
  state.mpScale = Number($("mpScale").value);
  if (state.camera) sizeMediapipeCanvas();
  $("mpScale").disabled = !state.tracking;
  if (!state.tracking) $("overlay").getContext("2d").clearRect(0, 0, $("overlay").width, $("overlay").height);
}

function lockOptions(locked) {
  $("tracking").disabled = locked;
  $("mpScale").disabled = locked || !$("tracking").checked;
}

function startRecording() {
  readOptions();
  lockOptions(true);
  state.session?.free();
  state.session = new TappingSession();
  state.frames = [];
  state.byId.clear();
  state.skipped = 0;
  state.detectMs = [];
  state.firstUs = null;
  state.lastUs = null;
  state.phase = "countdown";
  state.phaseStart = performance.now();
  $("record").disabled = true;
  $("next").disabled = true;
  $("progress").style.width = "0";
  status("Get ready…");
}

// ---------------------------------------------------------------- landmark traces

/// Follow two hands through the frames, by nearest wrist, and name each by
/// the side of the image it is on: the camera is not mirrored, so the
/// person's left hand is on the image's right.
function trackHands(frames) {
  const slots = { left: [], right: [] };
  const last = { left: null, right: null };
  for (const f of frames) {
    if (!f.hands) continue;
    const hs = f.hands.map((h) => ({ ...h, wx: h.x[WRIST], wy: h.y[WRIST] }));
    let assign = {};
    if (hs.length >= 2) {
      const [a, b] = hs.slice(0, 2).sort((p, q) => p.wx - q.wx);
      assign = { right: a, left: b };
    } else if (hs.length === 1) {
      const h = hs[0];
      const d = (s) => (last[s] ? Math.hypot(h.wx - last[s].wx, h.wy - last[s].wy) : Infinity);
      const side = d("left") === Infinity && d("right") === Infinity ? (h.wx > 0.5 ? "left" : "right")
        : d("left") <= d("right") ? "left" : "right";
      assign = { [side]: h };
    }
    for (const side of ["left", "right"]) {
      if (assign[side]) {
        last[side] = assign[side];
        slots[side].push({ us: f.us, h: assign[side] });
      }
    }
  }
  return slots;
}

/// Thumb–index aperture over hand size (wrist to middle-finger knuckle), in
/// image pixels so that the aspect ratio is right. Dimensionless.
function aperture(h, w, ht) {
  const d = (a, b) => Math.hypot((h.x[a] - h.x[b]) * w, (h.y[a] - h.y[b]) * ht);
  return d(THUMB_TIP, INDEX_TIP) / d(WRIST, MIDDLE_MCP);
}

function traceOf(samples, firstUs) {
  const { width, height } = state.camera;
  return {
    times: Float64Array.from(samples, (s) => (s.us - firstUs) / 1e6),
    values: Float64Array.from(samples, (s) => aperture(s.h, width, height)),
  };
}

function zscore(v) {
  const m = mean(v);
  const sd = Math.sqrt(mean(v.map((x) => (x - m) ** 2))) || 1;
  return v.map((x) => (x - m) / sd);
}

/// The two hands' apertures, each standardised, left minus right, at frames
/// where both were found: one trace for an anti-phase trial.
function differenceTrace(slots, firstUs) {
  const r = new Map(slots.right.map((s) => [s.us, s]));
  const both = slots.left.filter((s) => r.has(s.us));
  if (both.length < 10) return null;
  const { width, height } = state.camera;
  const zl = zscore(both.map((s) => aperture(s.h, width, height)));
  const zr = zscore(both.map((s) => aperture(r.get(s.us).h, width, height)));
  return {
    times: Float64Array.from(both, (s) => (s.us - firstUs) / 1e6),
    values: Float64Array.from(zl, (v, i) => v - zr[i]),
  };
}

// ---------------------------------------------------------------- comparison

const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const median = (v) => {
  if (!v.length) return NaN;
  const s = [...v].sort((a, b) => a - b);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};

function corr(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb);
}

/// `y` sampled at `t0y + j / fs`, linearly interpolated at `t`; NaN outside.
function interp(y, t0y, fs, t) {
  const u = (t - t0y) * fs;
  const j = Math.floor(u);
  if (j < 0 || j + 1 >= y.length) return NaN;
  return y[j] + (u - j) * (y[j + 1] - y[j]);
}

/// Correlation of the video component with a landmark trace on the video's
/// grid, at the lag (within ±0.3 s) that maximises its magnitude. The
/// component's sign is arbitrary, so the sign of r only says which way round.
function waveformAgreement(v, tr) {
  const fs = v.fs;
  let best = { r: 0, lagMs: 0 };
  for (let lag = -Math.round(0.3 * fs); lag <= Math.round(0.3 * fs); lag++) {
    const a = [], b = [];
    v.signal.forEach((x, i) => {
      const y = interp(tr.signal, tr.t0, tr.fs, v.t0 + (i + lag) / fs);
      if (Number.isFinite(y)) { a.push(x); b.push(y); }
    });
    if (a.length < fs * 3) continue;
    const r = corr(a, b);
    if (Math.abs(r) > Math.abs(best.r)) best = { r, lagMs: (1000 * lag) / fs };
  }
  return best;
}

/// Match each video cycle boundary to the nearest landmark boundary within
/// half a cycle. The two phase origins differ, so a constant offset is
/// expected; its spread is what says whether the two time the same taps.
/// The landmark boundaries are first moved by the waveform lag and, when the
/// two are anti-correlated (the video's sign is arbitrary), by half a cycle,
/// so that matched boundaries sit close together rather than straddling the
/// half-cycle edge of the matching window.
function timingAgreement(v, tr, waveform) {
  const period = 1 / v.f0_hz;
  const half = 0.5 * period;
  const shift = -waveform.lagMs / 1000 + (waveform.r < 0 ? half : 0);
  const pairs = [];
  for (const b of v.boundaries) {
    let best = null;
    for (const c0 of tr.boundaries) {
      const c = c0 + shift;
      if (Math.abs(c - b) <= half && (best === null || Math.abs(c - b) < Math.abs(best - b))) best = c;
    }
    if (best !== null) pairs.push([b, best]);
  }
  if (pairs.length < 3) return { matched: pairs.length, of: v.boundaries.length };
  const off = pairs.map(([b, c]) => b - c);
  const mo = mean(off);
  const sdOff = Math.sqrt(off.reduce((s, x) => s + (x - mo) ** 2, 0) / (off.length - 1));
  // Intervals between consecutive matched boundaries, from each source.
  const iv = [], il = [];
  for (let i = 1; i < pairs.length; i++) {
    iv.push(pairs[i][0] - pairs[i - 1][0]);
    il.push(pairs[i][1] - pairs[i - 1][1]);
  }
  const keep = iv.map((x, i) => x < 1.5 / v.f0_hz && il[i] < 1.5 / v.f0_hz);
  const a = iv.filter((_, i) => keep[i]), b = il.filter((_, i) => keep[i]);
  return {
    matched: pairs.length,
    of: v.boundaries.length,
    offsetMs: 1000 * mo,
    offsetSdMs: 1000 * sdOff,
    itiMadMs: a.length ? 1000 * mean(a.map((x, i) => Math.abs(x - b[i]))) : NaN,
    itiR: a.length > 3 ? corr(a, b) : NaN,
  };
}

function featureAgreement(fv, ft) {
  const j = fv.names.map((n, i) => (n.startsWith("jtfs_") ? i : -1)).filter((i) => i >= 0);
  const a = j.map((i) => fv.values[i]), b = j.map((i) => ft.values[i]);
  return { n: j.length, r: corr(a, b), rmsDiff: Math.sqrt(mean(a.map((x, k) => (x - b[k]) ** 2))) };
}

// ---------------------------------------------------------------- analysis

function resultKey(trial, tracking) {
  return tracking ? trial.id : `${trial.id}-video`;
}

function analyse() {
  const trial = TRIALS[state.trialIndex];
  const key = resultKey(trial, state.tracking);
  const t = performance.now();
  const diag = state.session.diagnose();
  const ms = performance.now() - t;
  const slots = trackHands(state.frames);
  const n = state.frames.length;
  const traces = {};
  for (const side of ["left", "right"]) {
    const s = slots[side];
    const entry = { detected: s.length / n };
    if (!state.tracking) {
      entry.error = "hand tracking was off";
    } else if (s.length >= 0.5 * n) {
      const tr = traceOf(s, state.firstUs);
      try {
        entry.analysis = analyseTrace(tr.times, tr.values);
      } catch (e) {
        entry.error = String(e.message ?? e);
      }
      entry.positions = meanPositions(s);
    }
    traces[side] = entry;
  }
  if (trial.hands.includes("left − right")) {
    const d = differenceTrace(slots, state.firstUs);
    const entry = { detected: d ? d.times.length / n : 0 };
    if (!state.tracking) {
      entry.error = "hand tracking was off";
    } else if (d && d.times.length >= 0.5 * n) {
      try {
        entry.analysis = analyseTrace(d.times, d.values);
      } catch (e) {
        entry.error = String(e.message ?? e);
      }
    }
    traces["left − right"] = entry;
  }

  const comparisons = {};
  const a = diag.analysis;
  for (const [name, tr] of Object.entries(traces)) {
    if (!a || !tr.analysis) continue;
    const waveform = waveformAgreement(a, tr.analysis);
    comparisons[name] = {
      waveform,
      timing: timingAgreement(a, tr.analysis, waveform),
      features: a.features ? featureAgreement(a.features, tr.analysis.features) : null,
    };
  }

  state.results[key] = {
    trial: trial.id,
    hand_tracking: state.tracking,
    mediapipe_input: state.tracking ? { width: state.mpCanvas.width, height: state.mpCanvas.height } : null,
    recorded: new Date().toISOString(),
    camera: state.camera,
    frames: n,
    diagnose_ms: ms,
    video: diag,
    landmarks: traces,
    comparisons,
    landmarks_skipped: state.skipped,
    landmarks_missing: state.frames.filter((f) => !f.hands).length,
    landmark_ms_median: median(state.detectMs),
    mediapipe_delegate: state.delegate,
    landmark_frames: state.frames.map((f) => ({ us: f.us - state.firstUs, skipped: !f.hands, hands: (f.hands ?? []).map((h) => ({ x: Array.from(h.x), y: Array.from(h.y), label: h.label, score: h.score })) })),
  };
  state.session.reset();
  state.phase = "preview";
  renderResult(trial, key);
  renderSummary();
  renderChips();
  $("record").disabled = false;
  $("next").disabled = state.trialIndex >= TRIALS.length - 1;
  $("download").disabled = false;
  lockOptions(false);
  const missing = state.frames.filter((f) => !f.hands).length;
  const marks = state.tracking
    ? `landmarks for ${n - missing}, median ${fmt(median(state.detectMs), 0)} ms each on ${state.delegate} at ${state.mpCanvas.width} x ${state.mpCanvas.height}`
    : "hand tracking off";
  status(`Done: ${n} frames (${fmt(diag.report.qc.effective_fps, 1)} fps, ${fmt(100 * diag.report.qc.dropped_fraction, 0)}% dropped); ${marks}. Analysed in ${ms.toFixed(0)} ms. Record again, or go to the next trial.`);
}

/// Mean image position (0–1) of the wrist, thumb tip and index tip.
function meanPositions(samples) {
  const at = (k) => ({ x: mean(samples.map((s) => s.h.x[k])), y: mean(samples.map((s) => s.h.y[k])) });
  return { wrist: at(WRIST), thumb: at(THUMB_TIP), index: at(INDEX_TIP), knuckle: at(INDEX_MCP) };
}

// ---------------------------------------------------------------- rendering

const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : typeof x === "number" ? x.toFixed(d) : String(x));

function renderChips() {
  $("chips").innerHTML = TRIALS.map((t, i) =>
    `<span class="chip ${i === state.trialIndex ? "current" : ""} ${state.results[t.id] || state.results[`${t.id}-video`] ? "done" : ""}">${i + 1}. ${t.title}</span>`).join("");
}

function showTrial() {
  const t = TRIALS[state.trialIndex];
  $("trialTitle").textContent = `Trial ${state.trialIndex + 1} of ${TRIALS.length}: ${t.title}`;
  $("trialText").textContent = t.text;
  renderChips();
}

function status(s) {
  $("status").textContent = s;
}

function renderResult(trial, key) {
  const res = state.results[key];
  const diag = res.video;
  const a = diag.analysis;
  let el = document.getElementById(`result-${key}`);
  if (!el) {
    el = document.createElement("section");
    el.className = "panel";
    el.id = `result-${key}`;
    $("results").appendChild(el);
  }
  const rep = diag.report;
  const verdict = rep.accepted ? `<span class="verdict ok">accepted by QC</span>`
    : `<span class="verdict bad">rejected by QC</span>: ${rep.reasons.join("; ")}`;
  const names = Object.keys(res.landmarks);

  // Agreement table.
  const rows = names.map((name) => {
    const tr = res.landmarks[name];
    const c = res.comparisons[name];
    const t = tr.analysis;
    if (!t) return `<tr><td>${name}</td><td>${fmt(100 * tr.detected, 0)}%</td><td colspan="8" style="text-align:left">${tr.error ?? "hand not found often enough"}</td></tr>`;
    return `<tr><td><span style="color:${COLOURS[name]()}">■</span> ${name}</td><td>${fmt(100 * tr.detected, 0)}%</td>
      <td>${fmt(t.f0_hz, 2)}</td><td>${c ? fmt(c.waveform.r, 2) : "–"}</td><td>${c ? fmt(c.waveform.lagMs, 0) : "–"}</td>
      <td>${c ? `${c.timing.matched}/${c.timing.of}` : "–"}</td><td>${c ? fmt(c.timing.offsetSdMs, 1) : "–"}</td>
      <td>${c ? fmt(c.timing.itiMadMs, 1) : "–"}</td><td>${c ? fmt(c.timing.itiR, 2) : "–"}</td><td>${c?.features ? fmt(c.features.r, 3) : "–"}</td></tr>`;
  }).join("");

  el.innerHTML = `
    <h2>${trial.title}${res.hand_tracking ? "" : " (video only)"}</h2>
    <p>Video pipeline: ${verdict}. ${a ? `f<sub>0</sub> = ${fmt(a.f0_hz, 2)} Hz${a.from_harmonic ? " (timed from its second harmonic)" : ""}, component ${a.component + 1} of ${a.ranking.length}, ${a.itis.length} intervals, loading spread ${fmt(a.loading_spread, 3)}, competitor ratio ${fmt(a.competitor_ratio, 2)}.` : ""}
      ${res.frames} frames (${fmt(diag.report.qc.effective_fps, 1)} fps effective).</p>
    <h3>Agreement with each hand's landmark trace</h3>
    <div class="scroll"><table>
      <tr><th>landmark trace</th><th>hand found</th><th>f<sub>0</sub> (Hz)</th><th>waveform r</th><th>lag (ms)</th><th>taps matched</th>
        <th>timing SD (ms)</th><th>ITI |diff| (ms)</th><th>ITI r</th><th>JTFS feature r</th></tr>
      ${rows}
    </table></div>
    <p class="note">The trace is thumb–index aperture over hand size. Waveform r is the correlation of the video component with the trace, at the best lag within ±300 ms; the component's sign is arbitrary. Taps are cycle boundaries from each analytic signal, aligned by the waveform lag and sign, then matched within half a cycle; their phase origins differ, so a constant offset is expected and the timing SD (the spread of the offset) is what matters. ITI |diff| is the mean absolute difference between matched inter-tap intervals.</p>
    <div class="grid2">
      <div><h3>Video loading, and mean landmark positions</h3><canvas class="heat" id="heat-${key}"></canvas>
        <div class="legend"><span>red/blue: the selected component's loading, by sign</span><span>● index tip</span><span>▲ thumb tip</span><span>■ wrist</span></div></div>
      <div><h3>Signals (standardised)</h3><canvas class="plot" id="sig-${key}"></canvas>
        <div class="legend">${legend(["video", ...names.filter((n) => res.landmarks[n].analysis)])}<span>ticks: tap boundaries</span></div>
        <h3>Inter-tap intervals (s)</h3><canvas class="plot" id="iti-${key}"></canvas></div>
    </div>
    <h3>Features</h3>
    <div class="grid2">
      <div class="scroll">${itiTable(res, names)}</div>
      <div><canvas class="plot" id="feat-${key}" style="height:260px"></canvas>
        <p class="note">JTFS features, landmark (y) against video (x), one point per path; the line is equality.</p></div>
    </div>`;
  if (a) {
    drawHeat($(`heat-${key}`), a, res.landmarks);
    drawSignals($(`sig-${key}`), a, res);
    drawItis($(`iti-${key}`), a, res);
    drawFeatures($(`feat-${key}`), a, res);
  }
}

function legend(names) {
  return names.map((n) => `<span><i style="background:${COLOURS[n]()}"></i>${n}</span>`).join("");
}

function itiTable(res, names) {
  const a = res.video.analysis;
  if (!a?.features) return `<p class="note">No video features: ${a?.features_error ?? "the analysis did not run"}.</p>`;
  const cols = names.filter((n) => res.landmarks[n].analysis);
  const head = `<tr><th>feature</th><th>video</th>${cols.map((n) => `<th>${n}</th>`).join("")}</tr>`;
  const body = a.features.names.slice(0, 10).map((name, i) =>
    `<tr><td>${name}</td><td>${fmt(a.features.values[i], 4)}</td>${cols.map((n) => `<td>${fmt(res.landmarks[n].analysis.features.values[i], 4)}</td>`).join("")}</tr>`).join("");
  return `<table>${head}${body}</table>`;
}

function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}

/// The selected component's loading over the 64 x 48 grid, shown mirrored
/// like the preview, with each hand's mean landmark positions on top.
function drawHeat(c, a, landmarks) {
  const { g, w, h } = setupCanvas(c);
  const gw = a.grid_width, gh = a.grid_height;
  const peak = Math.max(...a.loading.map(Math.abs)) || 1;
  const cw = w / gw, ch = h / gh;
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const v = a.loading[y * gw + x] / peak;
      const s = Math.min(1, Math.abs(v));
      const [r, gg, b] = v >= 0 ? [255, 255 * (1 - s), 255 * (1 - s)] : [255 * (1 - s), 255 * (1 - s), 255];
      g.fillStyle = `rgb(${r | 0},${gg | 0},${b | 0})`;
      g.fillRect((gw - 1 - x) * cw, y * ch, cw + 0.5, ch + 0.5);
    }
  }
  // Landmarks: image coordinates to the centre-cropped grid, then mirrored.
  const { width, height } = state.camera;
  const target = gw / gh;
  let x0 = 0, y0 = 0, cwid = width, chei = height;
  if (width / height > target) { cwid = height * target; x0 = (width - cwid) / 2; } else { chei = width / target; y0 = (height - chei) / 2; }
  const px = (p) => ({ x: w - ((p.x * width - x0) / cwid) * w, y: ((p.y * height - y0) / chei) * h });
  for (const side of ["left", "right"]) {
    const pos = landmarks[side]?.positions;
    if (!pos) continue;
    g.fillStyle = COLOURS[side]();
    g.strokeStyle = "#000";
    g.lineWidth = 1;
    const mark = (p, shape) => {
      const q = px(p);
      g.beginPath();
      if (shape === "circle") g.arc(q.x, q.y, 6, 0, 2 * Math.PI);
      else if (shape === "tri") { g.moveTo(q.x, q.y - 7); g.lineTo(q.x + 6, q.y + 5); g.lineTo(q.x - 6, q.y + 5); g.closePath(); }
      else g.rect(q.x - 5, q.y - 5, 10, 10);
      g.fill();
      g.stroke();
    };
    mark(pos.index, "circle");
    mark(pos.thumb, "tri");
    mark(pos.wrist, "square");
  }
}

function axes(g, w, h, xr, yr, pad) {
  const X = (x) => pad.l + ((x - xr[0]) / (xr[1] - xr[0])) * (w - pad.l - pad.r);
  const Y = (y) => h - pad.b - ((y - yr[0]) / (yr[1] - yr[0])) * (h - pad.t - pad.b);
  g.strokeStyle = css("--line");
  g.fillStyle = css("--muted");
  g.font = "11px system-ui";
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(pad.l, pad.t);
  g.lineTo(pad.l, h - pad.b);
  g.lineTo(w - pad.r, h - pad.b);
  g.stroke();
  return { X, Y };
}

function line(g, X, Y, xs, ys, colour, width = 1.5) {
  g.strokeStyle = colour;
  g.lineWidth = width;
  g.beginPath();
  let on = false;
  xs.forEach((x, i) => {
    if (!Number.isFinite(ys[i])) { on = false; return; }
    on ? g.lineTo(X(x), Y(ys[i])) : g.moveTo(X(x), Y(ys[i]));
    on = true;
  });
  g.stroke();
}

function drawSignals(c, a, res) {
  const { g, w, h } = setupCanvas(c);
  const tEnd = a.t0 + a.signal.length / a.fs;
  const { X, Y } = axes(g, w, h, [a.t0, tEnd], [-3.5, 3.5], { l: 8, r: 8, t: 8, b: 20 });
  g.fillText("0 s", X(0) + 2, h - 6);
  g.fillText(`${tEnd.toFixed(0)} s`, X(tEnd) - 24, h - 6);
  const names = Object.keys(res.landmarks).filter((n) => res.landmarks[n].analysis);
  // Flip the video component to correlate positively with the first trace.
  const first = names.find((n) => res.comparisons[n]);
  const sign = first && res.comparisons[first].waveform.r < 0 ? -1 : 1;
  const tv = a.signal.map((_, i) => a.t0 + i / a.fs);
  line(g, X, Y, tv, zscore(a.signal).map((v) => sign * v), COLOURS.video(), 2);
  for (const n of names) {
    const t = res.landmarks[n].analysis;
    line(g, X, Y, t.signal.map((_, i) => t.t0 + i / t.fs), zscore(t.signal), COLOURS[n](), 1.3);
  }
  const ticks = (bs, colour, y) => {
    g.strokeStyle = colour;
    g.lineWidth = 1.5;
    for (const b of bs) { g.beginPath(); g.moveTo(X(b), y); g.lineTo(X(b), y + 7); g.stroke(); }
  };
  ticks(a.boundaries, COLOURS.video(), 2);
  names.forEach((n, i) => ticks(res.landmarks[n].analysis.boundaries, COLOURS[n](), 11 + 9 * i));
}

function drawItis(c, a, res) {
  const { g, w, h } = setupCanvas(c);
  const names = Object.keys(res.landmarks).filter((n) => res.landmarks[n].analysis);
  const all = [a.itis, ...names.map((n) => res.landmarks[n].analysis.itis)].flat();
  if (!all.length) return;
  const lo = Math.min(...all), hi = Math.max(...all), padY = 0.1 * (hi - lo || 0.1);
  const tEnd = a.t0 + a.signal.length / a.fs;
  const { X, Y } = axes(g, w, h, [a.t0, tEnd], [lo - padY, hi + padY], { l: 40, r: 8, t: 8, b: 16 });
  g.fillText((hi + padY).toFixed(3), 2, 14);
  g.fillText((lo - padY).toFixed(3), 2, h - 18);
  const mid = (bs) => bs.slice(1).map((b, i) => (b + bs[i]) / 2);
  line(g, X, Y, mid(a.boundaries), a.itis, COLOURS.video(), 2);
  for (const n of names) {
    const t = res.landmarks[n].analysis;
    line(g, X, Y, mid(t.boundaries), t.itis, COLOURS[n](), 1.3);
  }
}

function drawFeatures(c, a, res) {
  const { g, w, h } = setupCanvas(c);
  if (!a.features) return;
  const names = Object.keys(res.landmarks).filter((n) => res.landmarks[n].analysis);
  const idx = a.features.names.map((n, i) => (n.startsWith("jtfs_") ? i : -1)).filter((i) => i >= 0);
  const vals = [idx.map((i) => a.features.values[i]), ...names.map((n) => idx.map((i) => res.landmarks[n].analysis.features.values[i]))].flat();
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const { X, Y } = axes(g, w, h, [lo, hi], [lo, hi], { l: 36, r: 8, t: 8, b: 20 });
  g.fillText(lo.toFixed(1), X(lo), h - 6);
  g.fillText(hi.toFixed(1), X(hi) - 20, h - 6);
  g.fillText(hi.toFixed(1), 2, Y(hi) + 10);
  g.strokeStyle = css("--muted");
  g.setLineDash([4, 4]);
  g.beginPath();
  g.moveTo(X(lo), Y(lo));
  g.lineTo(X(hi), Y(hi));
  g.stroke();
  g.setLineDash([]);
  for (const n of names) {
    g.fillStyle = COLOURS[n]();
    const f = res.landmarks[n].analysis.features.values;
    for (const i of idx) {
      g.beginPath();
      g.arc(X(a.features.values[i]), Y(f[i]), 2.5, 0, 2 * Math.PI);
      g.fill();
    }
  }
}

function renderSummary() {
  const rows = [];
  for (const r of Object.values(state.results)) {
    const t = TRIALS.find((x) => x.id === r.trial);
    const title = `${t.title}${r.hand_tracking ? "" : " (video only)"}`;
    const q = r.video.report.qc;
    const a = r.video.analysis;
    const lead = `<td>${title}</td><td>${fmt(q.effective_fps, 1)}</td><td>${fmt(100 * q.dropped_fraction, 1)}</td>
      <td>${r.video.report.accepted ? "yes" : "no"}</td><td>${fmt(a?.f0_hz, 2)}</td><td>${fmt(a?.features?.values[3], 4)}</td>`;
    if (!r.hand_tracking) {
      rows.push(`<tr>${lead}<td>–</td><td colspan="6" style="text-align:left">hand tracking off</td></tr>`);
      continue;
    }
    for (const [name, tr] of Object.entries(r.landmarks)) {
      const c = r.comparisons[name];
      rows.push(`<tr>${lead}<td>${name}</td>
        <td>${fmt(tr.analysis?.f0_hz, 2)}</td><td>${c ? fmt(c.waveform.r, 2) : "–"}</td><td>${c ? fmt(c.timing.offsetSdMs, 1) : "–"}</td>
        <td>${c ? fmt(c.timing.itiMadMs, 1) : "–"}</td><td>${c?.features ? fmt(c.features.r, 3) : "–"}</td>
        <td>${fmt(tr.analysis?.features.values[3], 4)}</td></tr>`);
    }
  }
  $("summaryPanel").hidden = rows.length === 0;
  $("summary").innerHTML = `<tr><th>trial</th><th>fps</th><th>dropped (%)</th><th>QC</th><th>video f<sub>0</sub> (Hz)</th><th>video ITI CV</th>
    <th>landmark trace</th><th>trace f<sub>0</sub> (Hz)</th><th>waveform r</th><th>timing SD (ms)</th><th>ITI |diff| (ms)</th>
    <th>JTFS feature r</th><th>trace ITI CV</th></tr>${rows.join("")}`;
}

function download() {
  const blob = new Blob([JSON.stringify({
    app: "tap-compare",
    user_agent: navigator.userAgent,
    device_verified: state.verified,
    trials: state.results,
  })], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `tap-compare-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

$("start").onclick = () => startCamera().catch((e) => { status(`Camera failed: ${e.message}`); $("start").disabled = false; });
$("record").onclick = startRecording;
$("next").onclick = () => {
  state.trialIndex = Math.min(TRIALS.length - 1, state.trialIndex + 1);
  $("next").disabled = true;
  showTrial();
};
$("download").onclick = download;
$("tracking").onchange = () => {
  readOptions();
  if (state.phase === "preview" && $("record").disabled && !state.tracking) $("record").disabled = false;
};
$("mpScale").onchange = readOptions;
setup().catch((e) => status(`Setup failed: ${e.message}`));
