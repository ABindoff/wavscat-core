//! The video tapping pipeline for the browser: a streaming session that takes
//! camera frames as they arrive and, at the end of a trial, returns QC and
//! features.
//!
//! ```js
//! const session = new TappingSession();
//! // For each frame, either pass its luma plane...
//! session.pushFrame(luma, stride, width, height, timestampUs);
//! // ...or, to avoid any copy, write it straight into wasm memory:
//! const ptr = session.stagingPointer(byteLength);
//! await videoFrame.copyTo(new Uint8Array(wasmMemory().buffer, ptr, byteLength), { rect, layout });
//! session.pushStaged(stride, width, height, timestampUs, false);
//! // At the end of the trial:
//! const report = session.finish(); // { accepted, reasons, qc, features, itis }
//! ```
//!
//! Only a 64 x 48 grid of each frame is kept. The staging buffer is zeroed
//! after every push, so no raw frame stays even in wasm memory.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use tapvid::features::FeatureParams;
use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::PipelineParams;
use tapvid::qc::QcParams;
use tapvid::report::trial_report;
use tapvid::synth::{frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};

use crate::js_err;

fn timestamp(us: f64) -> Result<i64, JsError> {
    if !us.is_finite() {
        return Err(JsError::new("The timestamp must be a finite number of microseconds."));
    }
    Ok(us.round() as i64)
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SessionArgs {
    #[serde(default)]
    capacity: Option<usize>,
    #[serde(default)]
    grid_width: Option<usize>,
    #[serde(default)]
    grid_height: Option<usize>,
}

const SESSION_KEYS: &[&str] = &["capacity", "grid_width", "grid_height"];

/// One trial of webcam finger tapping, captured frame by frame.
#[wasm_bindgen]
pub struct TappingSession {
    ingest: Ingest,
    staging: Vec<u8>,
    pipeline: PipelineParams,
    qc: QcParams,
    features: FeatureParams,
}

#[wasm_bindgen]
impl TappingSession {
    /// Start a session: `{ capacity, grid_width, grid_height }`, all optional.
    #[wasm_bindgen(constructor)]
    pub fn new(params: JsValue) -> Result<TappingSession, JsError> {
        let args: SessionArgs = if params.is_undefined() || params.is_null() {
            SessionArgs::default()
        } else {
            crate::parse(params, SESSION_KEYS)?
        };
        let d = IngestParams::default();
        let ip = IngestParams {
            capacity: args.capacity.unwrap_or(d.capacity),
            grid_width: args.grid_width.unwrap_or(d.grid_width),
            grid_height: args.grid_height.unwrap_or(d.grid_height),
            ..d
        };
        Ok(TappingSession {
            ingest: Ingest::new(ip).map_err(js_err)?,
            staging: Vec::new(),
            pipeline: PipelineParams::default(),
            qc: QcParams::default(),
            features: FeatureParams::default(),
        })
    }

    /// Add a luma (Y) plane captured at `timestamp_us` microseconds.
    #[wasm_bindgen(js_name = pushFrame)]
    pub fn push_frame(&mut self, luma: &[u8], stride: usize, width: usize, height: usize, timestamp_us: f64) -> Result<(), JsError> {
        self.ingest.push_frame(luma, stride, width, height, timestamp(timestamp_us)?).map_err(js_err)
    }

    /// Add an RGBA image, as from a canvas.
    #[wasm_bindgen(js_name = pushRgba)]
    pub fn push_rgba(&mut self, rgba: &[u8], stride: usize, width: usize, height: usize, timestamp_us: f64) -> Result<(), JsError> {
        self.ingest.push_rgba(rgba, stride, width, height, timestamp(timestamp_us)?).map_err(js_err)
    }

    /// Byte offset in wasm memory of a staging buffer of at least `len`
    /// bytes, for writing a frame without an intermediate copy. Ask again
    /// before every frame: growing wasm memory moves the buffer.
    #[wasm_bindgen(js_name = stagingPointer)]
    pub fn staging_pointer(&mut self, len: usize) -> usize {
        if self.staging.len() < len {
            self.staging.resize(len, 0);
        }
        self.staging.as_ptr() as usize
    }

    /// Add the frame written into the staging buffer: a luma plane, or RGBA
    /// if `rgba` is true. The buffer is zeroed afterwards.
    #[wasm_bindgen(js_name = pushStaged)]
    pub fn push_staged(&mut self, stride: usize, width: usize, height: usize, timestamp_us: f64, rgba: bool) -> Result<(), JsError> {
        let ts = timestamp(timestamp_us)?;
        let result = if rgba {
            self.ingest.push_rgba(&self.staging, stride, width, height, ts)
        } else {
            self.ingest.push_frame(&self.staging, stride, width, height, ts)
        };
        self.staging.fill(0);
        result.map_err(js_err)
    }

    /// Frames held.
    #[wasm_bindgen(getter)]
    pub fn frames(&self) -> usize {
        self.ingest.len()
    }

    /// Analyse and gate the trial: `{ accepted, reasons, qc, features, itis }`.
    /// Features and intervals are present only when the trial is accepted.
    pub fn finish(&self) -> Result<JsValue, JsError> {
        let report = trial_report(&self.ingest, &self.pipeline, &self.qc, &self.features).map_err(js_err)?;
        let ser = serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true);
        report.serialize(&ser).map_err(js_err)
    }

    /// Discard every frame, ready for the next trial.
    pub fn reset(&mut self) {
        self.ingest.clear();
        self.staging.fill(0);
    }
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SynthArgs {
    #[serde(default)]
    width: Option<usize>,
    #[serde(default)]
    height: Option<usize>,
    #[serde(default)]
    iti_sd: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    distractor_hz: Option<f64>,
}

const SYNTH_KEYS: &[&str] = &["width", "height", "iti_sd", "seed", "distractor_hz"];

/// A synthetic tapping video with known taps, for demonstrations and tests:
/// a bright blob tapping at 3 Hz on a textured background, with sensor noise,
/// gain drift and an exposure step. `{ width, height, iti_sd, seed,
/// distractor_hz }`, all optional.
#[wasm_bindgen]
pub struct SyntheticVideo {
    synth: VideoSynth,
    work: Vec<f32>,
}

#[wasm_bindgen]
impl SyntheticVideo {
    #[wasm_bindgen(constructor)]
    pub fn new(params: JsValue) -> Result<SyntheticVideo, JsError> {
        let a: SynthArgs = if params.is_undefined() || params.is_null() {
            SynthArgs::default()
        } else {
            crate::parse(params, SYNTH_KEYS)?
        };
        let d = VideoSpec::default();
        let seed = a.seed.unwrap_or(d.seed);
        let spec = VideoSpec {
            width: a.width.unwrap_or(d.width),
            height: a.height.unwrap_or(d.height),
            tap: TapSpec { iti_sd: a.iti_sd.unwrap_or(0.0), seed, ..TapSpec::default() },
            distractor: a.distractor_hz.map(|rate_hz| Distractor {
                rate_hz,
                displacement: 15.0,
                radius: 10.0,
                contrast: 90.0,
                centre: (0.8, 0.3),
            }),
            seed,
            ..d
        };
        let work = vec![0f32; spec.width * spec.height];
        Ok(SyntheticVideo { synth: VideoSynth::new(spec), work })
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.synth.spec().width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.synth.spec().height
    }

    /// Render the frame captured at `t` seconds into `out`, a luma plane of
    /// `width * height` bytes.
    pub fn render(&mut self, t: f64, out: &mut [u8]) -> Result<(), JsError> {
        if out.len() != self.work.len() {
            return Err(JsError::new(&format!("out must hold {} bytes.", self.work.len())));
        }
        self.synth.render(t, out, &mut self.work);
        Ok(())
    }

    /// True tap times, in seconds.
    pub fn taps(&self) -> Vec<f64> {
        self.synth.taps().to_vec()
    }
}

/// The module's `WebAssembly.Memory`, for writing frames into the buffer at
/// `TappingSession.stagingPointer` without an intermediate copy.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// Capture times, in seconds, of a webcam clock at `fps` over `duration`
/// seconds, with timing jitter and dropped frames.
#[wasm_bindgen(js_name = frameTimes)]
pub fn frame_times_js(duration: f64, fps: f64, jitter_sd: f64, drop_prob: f64, seed: u64) -> Vec<f64> {
    frame_times(&FrameClock { fps, jitter_sd, drop_prob, seed }, duration)
}
