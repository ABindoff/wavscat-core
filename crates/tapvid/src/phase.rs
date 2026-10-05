//! Stage 6: the analytic-signal phase, inter-tap intervals and amplitude.
//!
//! This is the direct estimator of tap-to-tap variability. Picking peaks frame
//! by frame quantises every tap time to the frame clock, which at 30 fps adds
//! about 14 ms of standard deviation to each inter-tap interval (one frame
//! over the square root of six), comparable to healthy variability itself. The
//! phase of the analytic signal is continuous, so cycle boundaries can be
//! placed between frames:
//!
//! 1. band-pass around the fundamental `f0` with a Gaussian window in
//!    frequency, zero-phase, on a reflection-padded copy;
//! 2. form the analytic signal by zeroing negative frequencies (done in the
//!    same multiplication), and unwrap its phase;
//! 3. place a cycle boundary wherever the unwrapped phase crosses a multiple
//!    of `2 pi`, interpolating linearly between samples;
//! 4. discard the first and last cycles, which the padding distorts.

use std::f64::consts::PI;

use wavscat_core::complex::C64;
use wavscat_core::{fft, math, pad, Error};

/// Settings for [`analyse`].
#[derive(Debug, Clone)]
pub struct PhaseParams {
    /// Standard deviation of the Gaussian band-pass, as a fraction of `f0`.
    /// Wider keeps more cycle-to-cycle variation but admits more noise and,
    /// on real tapping, more of the second harmonic.
    pub bandwidth: f64,
    /// Reflection padding at each end, in cycles of `f0`.
    pub pad_cycles: f64,
    /// Number of lags, in cycles, over which phase diffusion is fitted.
    pub diffusion_lags: usize,
}

impl Default for PhaseParams {
    fn default() -> Self {
        PhaseParams { bandwidth: 0.5, pad_cycles: 3.0, diffusion_lags: 8 }
    }
}

/// Summaries of the inter-tap intervals and cycle amplitudes. Times are in
/// seconds; amplitudes are in the units of the input signal.
#[derive(Debug, Clone, PartialEq)]
pub struct ItiSummary {
    /// Number of cycles kept, after discarding the first and last.
    pub n_cycles: usize,
    pub mean: f64,
    /// Sample standard deviation (denominator `n - 1`).
    pub sd: f64,
    /// `sd / mean`.
    pub cv: f64,
    pub median: f64,
    /// Median absolute deviation, scaled by 1.4826 to estimate the SD for
    /// normal data.
    pub mad: f64,
    /// `mad / median`.
    pub robust_cv: f64,
    /// Lag-1 autocorrelation of successive intervals.
    pub lag1_autocorrelation: f64,
    /// Mean of the per-cycle amplitudes.
    pub mean_amplitude: f64,
    /// Least-squares slope of per-cycle amplitude against time, per second:
    /// negative for a decrement.
    pub amplitude_slope: f64,
    /// `amplitude_slope / mean_amplitude`, per second, which does not depend
    /// on the arbitrary scale of the signal.
    pub relative_amplitude_slope: f64,
    /// Growth rate of the variance of the detrended phase, in rad^2 per
    /// second: `D` in `E[(r(t + tau) - r(t))^2] = 2 D tau`.
    pub phase_diffusion: f64,
}

/// The cycle-level result of [`analyse`].
#[derive(Debug, Clone)]
pub struct CycleAnalysis {
    /// Every cycle boundary found, in seconds on the input's time axis.
    pub boundaries: Vec<f64>,
    /// Inter-tap intervals of the kept cycles, in seconds.
    pub itis: Vec<f64>,
    /// Mean envelope over each kept cycle.
    pub amplitudes: Vec<f64>,
    pub summary: ItiSummary,
}

/// Analyse a uniformly sampled oscillation, `x[i]` at time `t0 + i / fs`, with
/// fundamental frequency `f0` in Hz.
pub fn analyse(x: &[f64], fs: f64, t0: f64, f0: f64, p: &PhaseParams) -> Result<CycleAnalysis, Error> {
    analyse_at(x, fs, t0, f0, 1, p)
}

/// [`analyse`], timing the cycles from harmonic `harmonic` of `f0` instead
/// of from the fundamental.
///
/// When the second harmonic dominates, as stage 5 reports, it is the cleaner
/// signal: a phase-locked harmonic has phase `2 phi + c`, so one tapping
/// cycle is one `4 pi` advance of its phase. The band-pass keeps the same
/// width in Hz, `bandwidth * f0`, because the harmonic's modulation
/// sidebands sit at the same offsets as the fundamental's. Phase diffusion
/// is reported on the fundamental's scale.
pub fn analyse_at(
    x: &[f64],
    fs: f64,
    t0: f64,
    f0: f64,
    harmonic: u32,
    p: &PhaseParams,
) -> Result<CycleAnalysis, Error> {
    let n = x.len();
    let h = harmonic as f64;
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error("The signal contains gaps or non-finite values; split or reject the trial.".into()));
    }
    if !(fs > 0.0) || !(f0 > 0.0) || harmonic == 0 || h * f0 >= fs / 2.0 {
        return Err(Error("Need fs > 0, harmonic >= 1 and 0 < harmonic * f0 < fs / 2.".into()));
    }
    if !(p.bandwidth > 0.0) {
        return Err(Error("bandwidth must be positive.".into()));
    }

    let z = analytic_band(x, fs, h * f0, p.bandwidth * f0, p.pad_cycles * h);

    // Unwrapped phase and envelope.
    // Unwrap: each step is the change in wrapped angle, reduced to the
    // nearest equivalent in [-pi, pi].
    let wrapped: Vec<f64> = z.iter().map(|v| math::atan2(v.im, v.re)).collect();
    let mut phase = Vec::with_capacity(n);
    phase.push(wrapped[0]);
    for i in 1..n {
        let mut d = wrapped[i] - wrapped[i - 1];
        d -= 2.0 * PI * (d / (2.0 * PI)).round();
        phase.push(phase[i - 1] + d);
    }
    let envelope: Vec<f64> = z.iter().map(|v| v.norm()).collect();
    let time = |i: f64| t0 + i / fs;

    // Cycle boundaries: the first crossing of each multiple of 2 pi h.
    let period = 2.0 * PI * h;
    let mut boundaries = Vec::new();
    let mut k = (phase[0] / period).floor() + 1.0;
    let mut i = 1;
    while i < n {
        let target = period * k;
        if phase[i] >= target && phase[i - 1] < target {
            let frac = (target - phase[i - 1]) / (phase[i] - phase[i - 1]);
            boundaries.push(time((i - 1) as f64 + frac));
            k += 1.0;
            continue;
        }
        i += 1;
    }

    if boundaries.len() < 5 {
        return Err(Error(format!(
            "Only {} cycle boundaries found; at least five are needed.",
            boundaries.len()
        )));
    }
    // Cycles are the intervals between boundaries; drop the first and last.
    let kept = &boundaries[1..boundaries.len() - 1];
    let itis: Vec<f64> = kept.windows(2).map(|w| w[1] - w[0]).collect();

    // Per-cycle amplitude: mean envelope over the samples within the cycle.
    let sample_of = |t: f64| (t - t0) * fs;
    let amplitudes: Vec<f64> = kept
        .windows(2)
        .map(|w| {
            let a = sample_of(w[0]).ceil().max(0.0) as usize;
            let b = (sample_of(w[1]).ceil().max(0.0) as usize).min(n);
            let span = &envelope[a..b.max(a + 1).min(n)];
            span.iter().sum::<f64>() / span.len() as f64
        })
        .collect();
    let mids: Vec<f64> = kept.windows(2).map(|w| 0.5 * (w[0] + w[1])).collect();

    // Phase diffusion over the span of the kept cycles.
    let a = sample_of(kept[0]).ceil() as usize;
    let b = (sample_of(kept[kept.len() - 1]).floor() as usize).min(n - 1);
    let diffusion = if harmonic == 1 {
        phase_diffusion(&phase[a..=b], fs, f0, p.diffusion_lags)
    } else {
        let scaled: Vec<f64> = phase[a..=b].iter().map(|v| v / h).collect();
        phase_diffusion(&scaled, fs, f0, p.diffusion_lags)
    };

    let summary = summarise(&itis, &amplitudes, &mids, diffusion);
    Ok(CycleAnalysis { boundaries, itis, amplitudes, summary })
}


/// The analytic signal of `x` (sampled at `fs` Hz) band-passed around `fc`
/// Hz with a Gaussian of standard deviation `sigma` Hz.
///
/// The mean is removed and the series padded by reflection, `pad_cycles`
/// cycles of `fc` at each end, so the band-pass sees no wrap-around step. The
/// band-pass and the analytic signal are one multiplication in frequency:
/// twice the gain at positive frequencies, once at zero and Nyquist, zero at
/// negative frequencies.
pub(crate) fn analytic_band(x: &[f64], fs: f64, fc: f64, sigma: f64, pad_cycles: f64) -> Vec<C64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let centred: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let padding = ((pad_cycles * fs / fc).ceil() as usize).min(n.saturating_sub(1));
    let padded = pad::pad_reflect(&centred, padding, padding);
    let len = padded.len();

    let gain = |f: f64| math::exp(-((f - fc) * (f - fc)) / (2.0 * sigma * sigma));
    let mut spec = fft::fft_real(&padded);
    for (k, v) in spec.iter_mut().enumerate() {
        let weight = if k == 0 {
            gain(0.0)
        } else if 2 * k < len {
            2.0 * gain(k as f64 * fs / len as f64)
        } else if 2 * k == len {
            gain(fs / 2.0)
        } else {
            0.0
        };
        *v = v.scale(weight);
    }
    fft::ifft(&mut spec);
    spec[padding..padding + n].to_vec()
}

fn mean(v: &[f64]) -> f64 {
    let mut acc = 0.0;
    for x in v {
        acc += x;
    }
    acc / v.len() as f64
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let h = s.len() / 2;
    if s.len() % 2 == 1 { s[h] } else { (s[h - 1] + s[h]) / 2.0 }
}

/// Least-squares slope and intercept of `y` on `x`.
fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let (mx, my) = (mean(x), mean(y));
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
    }
    let slope = sxy / sxx;
    (slope, my - slope * mx)
}

/// `D` such that the mean squared increment of the detrended phase over a lag
/// of `tau` seconds is `2 D tau`, fitted through the origin over lags of
/// `1, ..., lags` cycles.
fn phase_diffusion(phase: &[f64], fs: f64, f0: f64, lags: usize) -> f64 {
    let n = phase.len();
    let t: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let (slope, intercept) = ols(&t, phase);
    let r: Vec<f64> = phase.iter().zip(&t).map(|(p, ti)| p - (intercept + slope * ti)).collect();
    let mut num = 0.0;
    let mut den = 0.0;
    for m in 1..=lags {
        let lag = (m as f64 * fs / f0).round() as usize;
        if lag == 0 || lag >= n / 2 {
            break;
        }
        let mut msd = 0.0;
        for i in 0..n - lag {
            let d = r[i + lag] - r[i];
            msd += d * d;
        }
        msd /= (n - lag) as f64;
        let tau = lag as f64 / fs;
        num += tau * msd;
        den += tau * tau;
    }
    if den > 0.0 { num / (2.0 * den) } else { f64::NAN }
}

fn summarise(itis: &[f64], amplitudes: &[f64], mids: &[f64], phase_diffusion: f64) -> ItiSummary {
    let n = itis.len();
    let m = mean(itis);
    let mut ss = 0.0;
    for x in itis {
        ss += (x - m) * (x - m);
    }
    let sd = if n > 1 { (ss / (n - 1) as f64).sqrt() } else { f64::NAN };
    let med = median(itis);
    let deviations: Vec<f64> = itis.iter().map(|x| (x - med).abs()).collect();
    let mad = 1.4826 * median(&deviations);
    let lag1 = if n > 2 && ss > 0.0 {
        let mut c = 0.0;
        for w in itis.windows(2) {
            c += (w[0] - m) * (w[1] - m);
        }
        c / ss
    } else {
        f64::NAN
    };
    let mean_amplitude = mean(amplitudes);
    let (amplitude_slope, _) = ols(mids, amplitudes);
    ItiSummary {
        n_cycles: n,
        mean: m,
        sd,
        cv: sd / m,
        median: med,
        mad,
        robust_cv: mad / med,
        lag1_autocorrelation: lag1,
        mean_amplitude,
        amplitude_slope,
        relative_amplitude_slope: amplitude_slope / mean_amplitude,
        phase_diffusion,
    }
}
