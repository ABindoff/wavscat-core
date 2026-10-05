//! Synthetic tapping videos with known ground truth, for end-to-end tests.
//!
//! A bright blob, the hand, moves periodically on a textured background. Its
//! displacement follows the same tap phase as [`crate::synth`], so the true
//! tap times are known exactly, and it can carry interval jitter, amplitude
//! decrement and a second harmonic. On top of that come the nuisances of home
//! recordings: sensor noise, a slow drift in global gain, an abrupt exposure
//! step, and optionally a second moving blob, standing in for another person
//! or a pet.
//!
//! Frames are rendered one at a time, at whatever capture times the clock
//! produces, into a buffer the caller reuses, so a whole video is never held.
//! Every pixel value comes from integer-seeded randomness and `libm`, so the
//! video is identical on every platform.

use crate::rng::SplitMix64;
use crate::synth::{phase_at, tap_times, TapSpec};
use wavscat_core::math;

/// A second blob moving sinusoidally at its own rate.
#[derive(Debug, Clone)]
pub struct Distractor {
    /// Rate of its motion, in Hz.
    pub rate_hz: f64,
    /// Amplitude of its motion, in pixels, along the horizontal.
    pub displacement: f64,
    /// Gaussian radius (standard deviation), in pixels.
    pub radius: f64,
    /// Peak brightness added, in luma units.
    pub contrast: f64,
    /// Centre, as fractions of the width and height.
    pub centre: (f64, f64),
}

/// The scene and its nuisances.
#[derive(Debug, Clone)]
pub struct VideoSpec {
    pub width: usize,
    pub height: usize,
    /// The tapping: intervals, waveform and amplitude over time. `noise_sd`
    /// is ignored; see `sensor_noise`.
    pub tap: TapSpec,
    /// Blob displacement, in pixels, for a waveform amplitude of one. The
    /// motion is vertical, as a finger opening and closing.
    pub displacement: f64,
    /// Gaussian radius of the blob, in pixels.
    pub radius: f64,
    /// Peak brightness of the blob above the background, in luma units.
    pub contrast: f64,
    /// Rest position of the blob, as fractions of the width and height.
    pub centre: (f64, f64),
    /// Peak-to-mean amplitude of the background texture, in luma units.
    pub texture: f64,
    /// Sensor noise, uniform with this standard deviation, in luma units.
    pub sensor_noise: f64,
    /// Relative amplitude and rate of slow gain drift.
    pub gain_drift: f64,
    pub gain_drift_hz: f64,
    /// An exposure step: from this time, in seconds, gain is multiplied by
    /// this factor.
    pub exposure_step: Option<(f64, f64)>,
    pub distractor: Option<Distractor>,
    pub seed: u64,
}

impl Default for VideoSpec {
    /// 320 x 240, regular 3 Hz tapping, mild noise and drift, an exposure
    /// step half way through a 30 s trial, no distractor.
    fn default() -> Self {
        VideoSpec {
            width: 320,
            height: 240,
            tap: TapSpec::default(),
            displacement: 20.0,
            radius: 10.0,
            contrast: 90.0,
            centre: (0.5, 0.5),
            texture: 40.0,
            sensor_noise: 2.0,
            gain_drift: 0.1,
            gain_drift_hz: 0.05,
            exposure_step: Some((15.0, 1.3)),
            distractor: None,
            seed: 1,
        }
    }
}

/// A renderer for one synthetic video.
pub struct VideoSynth {
    spec: VideoSpec,
    taps: Vec<f64>,
    background: Vec<f32>,
}

/// Add a Gaussian blob of peak `contrast` at `(cx, cy)` to `img`, within
/// three radii of its centre.
fn add_blob(img: &mut [f32], width: usize, height: usize, cx: f64, cy: f64, radius: f64, contrast: f64) {
    let reach = 3.0 * radius;
    let (x0, x1) = (((cx - reach).floor().max(0.0)) as usize, ((cx + reach).ceil() as usize).min(width));
    let (y0, y1) = (((cy - reach).floor().max(0.0)) as usize, ((cy + reach).ceil() as usize).min(height));
    let two_r2 = 2.0 * radius * radius;
    for y in y0..y1 {
        let dy = y as f64 + 0.5 - cy;
        for x in x0..x1 {
            let dx = x as f64 + 0.5 - cx;
            img[y * width + x] += (contrast * math::exp(-(dx * dx + dy * dy) / two_r2)) as f32;
        }
    }
}

impl VideoSynth {
    pub fn new(spec: VideoSpec) -> VideoSynth {
        let taps = tap_times(&spec.tap);
        // Background: a few random gratings plus fixed per-pixel grain,
        // around mid grey.
        let mut g = SplitMix64::new(spec.seed ^ 0xbac6_6400);
        let gratings: Vec<(f64, f64, f64, f64)> = (0..6)
            .map(|_| (0.02 + 0.1 * g.uniform(), 0.02 + 0.1 * g.uniform(), 2.0 * std::f64::consts::PI * g.uniform(), g.uniform()))
            .collect();
        let mut background = vec![0f32; spec.width * spec.height];
        for y in 0..spec.height {
            for x in 0..spec.width {
                let mut v = 0.0;
                for &(fx, fy, ph, w) in &gratings {
                    v += w * math::sin(fx * x as f64 + fy * y as f64 + ph);
                }
                let grain = g.uniform() - 0.5;
                background[y * spec.width + x] = (110.0 + spec.texture * (v / 3.0 + 0.3 * grain)) as f32;
            }
        }
        VideoSynth { spec, taps, background }
    }

    /// True tap times, in seconds.
    pub fn taps(&self) -> &[f64] {
        &self.taps
    }

    pub fn spec(&self) -> &VideoSpec {
        &self.spec
    }

    /// The global gain at time `t`.
    fn gain(&self, t: f64) -> f64 {
        let s = &self.spec;
        let drift = 1.0 + s.gain_drift * math::sin(2.0 * std::f64::consts::PI * s.gain_drift_hz * t);
        match s.exposure_step {
            Some((at, factor)) if t >= at => drift * factor,
            _ => drift,
        }
    }

    /// Render the frame captured at `t` seconds into `out`, a `width x
    /// height` luma plane. `work` is scratch of the same size, reused
    /// between calls.
    pub fn render(&self, t: f64, out: &mut [u8], work: &mut [f32]) {
        let s = &self.spec;
        let (w, h) = (s.width, s.height);
        work.copy_from_slice(&self.background);

        let ph = phase_at(&self.taps, t);
        let wave = s.tap.fundamental * math::cos(ph) + s.tap.second_harmonic * math::cos(2.0 * ph + 1.0);
        let amp = s.tap.amplitude + s.tap.amplitude_slope * t;
        let cx = s.centre.0 * w as f64;
        let cy = s.centre.1 * h as f64 + s.displacement * amp * wave;
        add_blob(work, w, h, cx, cy, s.radius, s.contrast);

        if let Some(d) = &s.distractor {
            let dx = d.displacement * math::sin(2.0 * std::f64::consts::PI * d.rate_hz * t);
            add_blob(work, w, h, d.centre.0 * w as f64 + dx, d.centre.1 * h as f64, d.radius, d.contrast);
        }

        // Gain, sensor noise, and 8-bit quantisation. Noise is seeded by the
        // capture time, so a frame does not depend on render order.
        let gain = self.gain(t);
        let mut noise = SplitMix64::new(s.seed ^ t.to_bits());
        let spread = s.sensor_noise * 12f64.sqrt();
        for (o, &v) in out.iter_mut().zip(work.iter()) {
            let n = spread * (noise.uniform() - 0.5);
            *o = (gain * v as f64 + n).round().clamp(0.0, 255.0) as u8;
        }
    }
}
