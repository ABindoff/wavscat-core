// MediaPipe Hands in a worker. The page sends frames, as VideoFrames or
// ImageBitmaps, with their times: preview frames for the live overlay, and
// after a trial every recorded frame. The worker answers with the
// landmarks, in the same order.

// MediaPipe loads its wasm glue with importScripts, which a module worker
// does not have. Workers may fetch synchronously, so do that and evaluate
// the script in the global scope, as importScripts would.
self.importScripts = (...urls) => {
  for (const url of urls) {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url, false);
    xhr.send();
    if (xhr.status !== 200) throw new Error(`Could not load ${url}: ${xhr.status}`);
    (0, eval)(xhr.responseText);
  }
};

const { FilesetResolver, HandLandmarker } = await import(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs"
);
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let landmarker = null;
let delegate = null;

async function setup() {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const opts = (d) => ({
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: d },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(vision, opts("GPU"));
    delegate = "GPU";
  } catch {
    landmarker = await HandLandmarker.createFromOptions(vision, opts("CPU"));
    delegate = "CPU";
  }
  // The first inferences compile GPU programs, which can stall the page's
  // video for a second; do that now, on blank frames, not during a trial.
  // Camera timestamps are far later than these, so time still runs forward.
  const blank = new OffscreenCanvas(640, 480);
  blank.getContext("2d").fillRect(0, 0, 640, 480);
  for (let t = 1; t <= 5; t++) {
    const bitmap = blank.transferToImageBitmap();
    landmarker.detectForVideo(bitmap, t);
    bitmap.close();
    blank.getContext("2d").fillRect(0, 0, 640, 480);
  }
}

const ready = setup().then(
  () => self.postMessage({ type: "ready", delegate }),
  (e) => self.postMessage({ type: "error", message: String(e?.message ?? e) }),
);

// Frames are handled strictly in order, one at a time.
let queue = ready;
let lastT = -Infinity;

self.onmessage = ({ data }) => {
  if (data.type !== "frame") return;
  // A frame that fails still gets an answer, so the page's count of frames
  // in flight stays right and later frames are not held up.
  queue = queue.then(() => detect(data)).catch((e) => {
    self.postMessage({ type: "landmarks", id: data.id, ms: 0, landmarks: [], handedness: [], error: String(e?.message ?? e) });
  });
};

async function detect({ id, tMs, source, width, height }) {
  const start = performance.now();
  let res = { landmarks: [], handedness: [] };
  let bitmap = null;
  try {
    // A VideoFrame or ImageBitmap from the page, scaled to MediaPipe's size.
    bitmap = await createImageBitmap(source, { resizeWidth: width, resizeHeight: height, resizeQuality: "high" });
    // Video mode needs strictly increasing times.
    const t = Math.max(tMs, lastT + 1);
    lastT = t;
    res = landmarker.detectForVideo(bitmap, t);
  } finally {
    source.close();
    bitmap?.close();
  }
  self.postMessage({
    type: "landmarks",
    id,
    ms: performance.now() - start,
    // Plain arrays of { x, y } in 0-1 image coordinates; z is not used.
    landmarks: res.landmarks.map((lm) => lm.map((p) => ({ x: p.x, y: p.y }))),
    handedness: res.handedness.map((h) => [{ categoryName: h[0]?.categoryName ?? "", score: h[0]?.score ?? 0 }]),
  });
}
