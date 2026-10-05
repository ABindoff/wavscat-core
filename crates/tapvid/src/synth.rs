//! Synthetic tapping signals with known ground truth.
//!
//! The estimators are judged against these. Taps are generated first, as
//! inter-tap intervals drawn around a mean, and the oscillation is built
//! through them: its phase is a monotone cubic through `(t_k, 2 pi k)`, so the
//! true tap times are known exactly and the instantaneous frequency is
//! continuous. The oscillation is then sampled at the timestamps of a webcam
//! clock, which jitters and drops frames, as real capture does.

use std::f64::consts::PI;

use crate::rng::SplitMix64;
use wavscat_core::math;

/// The tapping behaviour to simulate.
#[derive(Debug, Clone)]
pub struct TapSpec {
    /// Length of the recording, in seconds.
    pub duration: f64,
    /// Mean inter-tap interval, in seconds. 1/3 s is 3 Hz tapping.
    pub mean_iti: f64,
    /// Standard deviation of the inter-tap intervals, in seconds.
    pub iti_sd: f64,
    /// Oscillation amplitude at time zero, in signal units.
    pub amplitude: f64,
    /// Change in amplitude per second: negative for a decrement.
    pub amplitude_slope: f64,
    /// Amplitude of a second harmonic, relative to the fundamental. Real
    /// tapping is not sinusoidal: opening and closing differ in speed.
    pub second_harmonic: f64,
    /// Standard deviation of additive white noise on each frame.
    pub noise_sd: f64,
    pub seed: u64,
}

impl Default for TapSpec {
    /// 30 s of perfectly regular 3 Hz tapping.
    fn default() -> Self {
        TapSpec {
            duration: 30.0,
            mean_iti: 1.0 / 3.0,
            iti_sd: 0.0,
            amplitude: 1.0,
            amplitude_slope: 0.0,
            second_harmonic: 0.0,
            noise_sd: 0.0,
            seed: 1,
        }
    }
}

/// How frames are captured.
#[derive(Debug, Clone)]
pub struct FrameClock {
    /// Nominal frame rate, in frames per second.
    pub fps: f64,
    /// Standard deviation of each frame's timing jitter, in seconds.
    pub jitter_sd: f64,
    /// Probability that any given frame is dropped.
    pub drop_prob: f64,
    pub seed: u64,
}

impl Default for FrameClock {
    /// A perfect 30 fps clock.
    fn default() -> Self {
        FrameClock { fps: 30.0, jitter_sd: 0.0, drop_prob: 0.0, seed: 2 }
    }
}

/// A synthetic recording and the truth behind it.
#[derive(Debug, Clone)]
pub struct SynthRecording {
    /// Capture timestamps, in seconds, strictly increasing.
    pub timestamps: Vec<f64>,
    /// The signal at each timestamp.
    pub values: Vec<f64>,
    /// True tap times, in seconds, covering the whole recording and a margin
    /// on each side.
    pub taps: Vec<f64>,
}

impl SynthRecording {
    /// True inter-tap intervals of the taps that lie within `[t0, t1]`.
    pub fn itis_between(&self, t0: f64, t1: f64) -> Vec<f64> {
        let inside: Vec<f64> = self.taps.iter().copied().filter(|&t| t >= t0 && t <= t1).collect();
        inside.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

/// Tap times from `-2 mean_iti` to `duration + 2 mean_iti`, so that the phase
/// is defined over the whole recording. Intervals are normal around the mean,
/// floored at 30% of it.
pub fn tap_times(spec: &TapSpec) -> Vec<f64> {
    let mut rng = SplitMix64::new(spec.seed);
    let mut t = -2.0 * spec.mean_iti;
    let mut taps = vec![t];
    while t < spec.duration + 2.0 * spec.mean_iti {
        let iti = (spec.mean_iti + spec.iti_sd * rng.normal()).max(0.3 * spec.mean_iti);
        t += iti;
        taps.push(t);
    }
    taps
}

/// The oscillation phase at time `t`: a monotone cubic Hermite interpolant
/// through `(taps[k], 2 pi k)`, with Fritsch-Butland tangents (the harmonic
/// mean of adjacent slopes), so that frequency stays positive and continuous.
pub fn phase_at(taps: &[f64], t: f64) -> f64 {
    let n = taps.len();
    assert!(n >= 2 && t >= taps[0] && t <= taps[n - 1], "t outside the tap range");
    let k = match taps.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
        Ok(i) => return 2.0 * PI * i as f64,
        Err(i) => i - 1,
    };
    let slope = |i: usize| 2.0 * PI / (taps[i + 1] - taps[i]);
    let tangent = |i: usize| -> f64 {
        if i == 0 {
            slope(0)
        } else if i == n - 1 {
            slope(n - 2)
        } else {
            2.0 / (1.0 / slope(i - 1) + 1.0 / slope(i))
        }
    };
    let h = taps[k + 1] - taps[k];
    let s = (t - taps[k]) / h;
    let (s2, s3) = (s * s, s * s * s);
    let p0 = 2.0 * PI * k as f64;
    let p1 = p0 + 2.0 * PI;
    (2.0 * s3 - 3.0 * s2 + 1.0) * p0
        + (s3 - 2.0 * s2 + s) * h * tangent(k)
        + (-2.0 * s3 + 3.0 * s2) * p1
        + (s3 - s2) * h * tangent(k + 1)
}

/// Capture timestamps over `[0, duration)`: nominal frame times plus jitter,
/// with dropped frames removed. Jitter is limited to a third of a frame so
/// that timestamps stay strictly increasing.
pub fn frame_times(clock: &FrameClock, duration: f64) -> Vec<f64> {
    let mut rng = SplitMix64::new(clock.seed);
    let dt = 1.0 / clock.fps;
    let limit = dt / 3.0;
    let mut out = Vec::new();
    let mut j = 0usize;
    loop {
        let nominal = j as f64 * dt;
        if nominal >= duration {
            break;
        }
        let jitter = (clock.jitter_sd * rng.normal()).clamp(-limit, limit);
        let dropped = rng.uniform() < clock.drop_prob;
        if !dropped {
            out.push((nominal + jitter).max(0.0));
        }
        j += 1;
    }
    out
}

/// Render a recording, `A(t) (cos(phase) + h cos(2 phase + 1)) + noise` at
/// each capture time, with `h` the second harmonic and
/// `A(t) = amplitude + amplitude_slope * t`.
pub fn render(spec: &TapSpec, clock: &FrameClock) -> SynthRecording {
    let taps = tap_times(spec);
    let timestamps = frame_times(clock, spec.duration);
    let mut noise = SplitMix64::new(spec.seed ^ 0x5eed_5eed_5eed_5eed);
    let values = timestamps
        .iter()
        .map(|&t| {
            let a = spec.amplitude + spec.amplitude_slope * t;
            let ph = phase_at(&taps, t);
            let wave = math::cos(ph) + spec.second_harmonic * math::cos(2.0 * ph + 1.0);
            a * wave + spec.noise_sd * noise.normal()
        })
        .collect();
    SynthRecording { timestamps, values, taps }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_passes_through_every_tap_and_increases() {
        let spec = TapSpec { iti_sd: 0.03, ..TapSpec::default() };
        let taps = tap_times(&spec);
        for (k, &t) in taps.iter().enumerate() {
            assert!((phase_at(&taps, t) - 2.0 * PI * k as f64).abs() < 1e-9);
        }
        let mut last = phase_at(&taps, taps[0]);
        let mut t = taps[0];
        while t < taps[taps.len() - 1] {
            let p = phase_at(&taps, t);
            assert!(p >= last, "phase decreased at t = {t}");
            last = p;
            t += 0.001;
        }
    }

    #[test]
    fn clock_drops_and_jitters_but_stays_ordered() {
        let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, ..FrameClock::default() };
        let ts = frame_times(&clock, 30.0);
        assert!(ts.windows(2).all(|w| w[1] > w[0]));
        let kept = ts.len() as f64 / 900.0;
        assert!((kept - 0.95).abs() < 0.03, "kept {kept}");
    }
}
