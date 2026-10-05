//! Welch power spectral density.
//!
//! Averaging the periodograms of overlapping, tapered segments trades
//! frequency resolution for variance, which is what component selection needs:
//! a stable picture of where the power sits, not the finest line. Segments are
//! zero-padded so that the peak can be located between the natural bins.

use std::f64::consts::PI;

use wavscat_core::{fft, math, Error};

/// Settings for [`welch`].
#[derive(Debug, Clone)]
pub struct WelchParams {
    /// Segment length, in seconds. 8 s resolves 0.125 Hz, and fits several
    /// overlapping segments into a 30 s trial.
    pub segment_sec: f64,
    /// Fraction by which successive segments overlap, in `[0, 1)`.
    pub overlap: f64,
    /// Minimum transform length; segments are zero-padded up to it.
    pub min_nfft: usize,
}

impl Default for WelchParams {
    fn default() -> Self {
        WelchParams { segment_sec: 8.0, overlap: 0.5, min_nfft: 1024 }
    }
}

/// A one-sided power spectral density.
#[derive(Debug, Clone)]
pub struct Psd {
    /// Bin frequencies, in Hz, from 0 to the Nyquist frequency.
    pub freqs: Vec<f64>,
    /// Power density at each bin, in signal units squared per Hz.
    pub power: Vec<f64>,
}

impl Psd {
    /// Spacing of the bins, in Hz.
    pub fn df(&self) -> f64 {
        self.freqs[1] - self.freqs[0]
    }
}

/// Welch's estimate for a uniformly sampled series at `fs` Hz.
///
/// Each segment has its mean removed and a periodic Hann taper applied. The
/// density is scaled so that summing it over frequency, times the bin width,
/// recovers the variance. A series shorter than one segment is analysed as a
/// single segment.
pub fn welch(x: &[f64], fs: f64, p: &WelchParams) -> Result<Psd, Error> {
    let n = x.len();
    if n < 8 {
        return Err(Error("At least eight samples are needed for a spectrum.".into()));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error("The series contains gaps or non-finite values.".into()));
    }
    if !(fs > 0.0) || !(0.0..1.0).contains(&p.overlap) || !(p.segment_sec > 0.0) {
        return Err(Error("Need fs > 0, segment_sec > 0 and overlap in [0, 1).".into()));
    }

    let seg = ((p.segment_sec * fs).round() as usize).clamp(8, n);
    let step = (((1.0 - p.overlap) * seg as f64).round() as usize).max(1);
    let nfft = seg.next_power_of_two().max(p.min_nfft.next_power_of_two());
    let window: Vec<f64> = (0..seg).map(|i| 0.5 - 0.5 * math::cos(2.0 * PI * i as f64 / seg as f64)).collect();
    let mut wsum = 0.0;
    for w in &window {
        wsum += w * w;
    }

    let bins = nfft / 2 + 1;
    let mut power = vec![0.0; bins];
    let mut segments = 0usize;
    let mut start = 0;
    while start + seg <= n {
        let chunk = &x[start..start + seg];
        let mut m = 0.0;
        for v in chunk {
            m += v;
        }
        m /= seg as f64;
        let mut buf = vec![0.0; nfft];
        for (i, (v, w)) in chunk.iter().zip(&window).enumerate() {
            buf[i] = (v - m) * w;
        }
        let spec = fft::fft_real(&buf);
        for (k, acc) in power.iter_mut().enumerate() {
            let v = spec[k];
            *acc += v.re * v.re + v.im * v.im;
        }
        segments += 1;
        start += step;
    }

    // Density scaling, doubled except at zero and Nyquist for one side.
    let scale = 1.0 / (fs * wsum * segments as f64);
    for (k, v) in power.iter_mut().enumerate() {
        *v *= scale;
        if k != 0 && !(nfft % 2 == 0 && k == nfft / 2) {
            *v *= 2.0;
        }
    }
    let freqs = (0..bins).map(|k| k as f64 * fs / nfft as f64).collect();
    Ok(Psd { freqs, power })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sinusoid_peaks_at_its_frequency_with_its_variance() {
        let fs = 30.0;
        let x: Vec<f64> = (0..900).map(|i| 2.0 * math::sin(2.0 * PI * 3.0 * i as f64 / fs)).collect();
        let psd = welch(&x, fs, &WelchParams::default()).unwrap();
        let k = (0..psd.power.len()).max_by(|&a, &b| psd.power[a].total_cmp(&psd.power[b])).unwrap();
        assert!((psd.freqs[k] - 3.0).abs() < psd.df());
        // Integrated power equals the variance, 2^2 / 2.
        let total: f64 = psd.power.iter().sum::<f64>() * psd.df();
        assert!((total - 2.0).abs() < 0.02, "integrated power {total}");
    }
}
