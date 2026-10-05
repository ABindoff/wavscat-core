//! Stage 5: choose the tapping component and find its fundamental.
//!
//! Each SVD component is scored by how much of its power is tapping-like: in
//! the tapping band, and concentrated in one peak there. The best component
//! is selected, and the runner-up's score is kept for QC, since a close second
//! often means another moving object in the frame.
//!
//! The fundamental is not simply the highest peak. From some camera angles
//! opening and closing look alike, and the dominant energy sits at twice the
//! tapping rate. So after finding the peak at `f`, two things are checked at
//! `f / 2`:
//!
//! - its power, integrated over a window, must be a non-trivial share of the
//!   power at `f`;
//! - it must be phase-locked to `f`. A true second harmonic has exactly twice
//!   the phase of its fundamental, cycle by cycle, however irregular the
//!   tapping, so `z1^2 conj(z2)` keeps a constant angle, where `z1` and `z2`
//!   are the analytic signals around `f / 2` and `f`. Noise near `f / 2` has
//!   no such relation.
//!
//! Power alone cannot make this call: cycle-to-cycle jitter broadens the
//! harmonic's line twice as much as the fundamental's, and its skirt fills the
//! spectrum below. Locking is unaffected by that broadening.

use wavscat_core::{math, Error};

use crate::phase::analytic_band;
use crate::spectrum::{welch, Psd, WelchParams};

/// Settings for scoring and for the fundamental.
#[derive(Debug, Clone)]
pub struct BandParams {
    /// Tapping band, in Hz.
    pub lo: f64,
    pub hi: f64,
    /// Half-width of the peak, in Hz, for the sharpness measure.
    pub peak_halfwidth: f64,
    /// Half-width of the power windows around `f` and `f / 2`, as a fraction
    /// of their centre frequency.
    pub half_tolerance: f64,
    /// The power near `f / 2` must be at least this fraction of the power
    /// near `f`. 0.01 is an amplitude ratio of 0.1.
    pub half_min_ratio: f64,
    /// ...and at least this phase-locked to `f`, on a scale from 0 (none) to
    /// 1 (perfect).
    pub half_min_locking: f64,
    /// Width of the band-passes used to measure locking, as a fraction of
    /// their centre frequency.
    pub locking_bandwidth: f64,
}

impl Default for BandParams {
    fn default() -> Self {
        BandParams {
            lo: 1.0,
            hi: 7.0,
            peak_halfwidth: 0.25,
            half_tolerance: 0.1,
            half_min_ratio: 0.01,
            half_min_locking: 0.5,
            locking_bandwidth: 0.3,
        }
    }
}

/// Whether the fundamental is the strongest peak or half of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F0Case {
    /// The strongest in-band peak is the tapping rate.
    Fundamental,
    /// The strongest peak is the second harmonic; the rate is half of it.
    Harmonic,
}

/// How tapping-like one component is.
#[derive(Debug, Clone, PartialEq)]
pub struct Periodicity {
    /// Share of the total power that lies in the tapping band.
    pub band_fraction: f64,
    /// Share of the in-band power within `peak_halfwidth` of the peak.
    pub sharpness: f64,
    /// `band_fraction * sharpness`: high only for one clean in-band peak.
    pub score: f64,
    /// Frequency of the dominant peak, in Hz, sought up to twice the top of
    /// the band so that a second harmonic above it is seen.
    pub peak_hz: f64,
    /// The tapping fundamental, in Hz.
    pub f0_hz: f64,
    pub case: F0Case,
    /// Power near `peak_hz / 2` relative to the power near `peak_hz`.
    pub half_ratio: f64,
    /// Phase locking of `peak_hz / 2` to `peak_hz`, from 0 to 1.
    pub half_locking: f64,
}

impl Periodicity {
    /// Which harmonic of `f0` stage 6 should time cycles from: the second
    /// when it dominates, otherwise the fundamental. Timing from a weak
    /// fundamental fails once it is a fifth of the harmonic (see the
    /// `harmonic_timing` example), while timing from the harmonic does not.
    pub fn timing_harmonic(&self) -> u32 {
        match self.case {
            F0Case::Fundamental => 1,
            F0Case::Harmonic => 2,
        }
    }
}

/// Index of the bin nearest `f`.
fn bin(psd: &Psd, f: f64) -> usize {
    ((f / psd.df()).round() as usize).min(psd.freqs.len() - 1)
}

/// Peak frequency refined by a parabola through the log power of the peak bin
/// and its neighbours.
fn refine(psd: &Psd, k: usize) -> f64 {
    let p = &psd.power;
    if k == 0 || k + 1 >= p.len() || p[k - 1] <= 0.0 || p[k + 1] <= 0.0 || p[k] <= 0.0 {
        return psd.freqs[k];
    }
    let (a, b, c) = (math::ln(p[k - 1]), math::ln(p[k]), math::ln(p[k + 1]));
    let denom = a - 2.0 * b + c;
    let delta = if denom < 0.0 { 0.5 * (a - c) / denom } else { 0.0 };
    psd.freqs[k] + delta.clamp(-0.5, 0.5) * psd.df()
}

fn argmax(p: &[f64], range: std::ops::RangeInclusive<usize>) -> usize {
    let mut best = *range.start();
    for k in range {
        if p[k] > p[best] {
            best = k;
        }
    }
    best
}

fn sum(v: &[f64]) -> f64 {
    let mut acc = 0.0;
    for x in v {
        acc += x;
    }
    acc
}

/// Phase locking of the band around `f / 2` to the band around `f`:
/// `|sum z1^2 conj(z2)| / sum |z1|^2 |z2|`, which is 1 when the angle of
/// `z1^2 conj(z2)` never changes and near 0 when it wanders.
fn locking(x: &[f64], fs: f64, f: f64, bandwidth: f64) -> f64 {
    let z1 = analytic_band(x, fs, f / 2.0, bandwidth * f / 2.0, 3.0);
    let z2 = analytic_band(x, fs, f, bandwidth * f, 3.0);
    let (mut re, mut im, mut norm) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in z1.iter().zip(&z2) {
        let c = *a * *a * b.conj();
        re += c.re;
        im += c.im;
        norm += c.norm();
    }
    if norm > 0.0 { (re * re + im * im).sqrt() / norm } else { 0.0 }
}

/// Score one uniformly sampled component, at `fs` Hz, and locate its
/// fundamental.
pub fn periodicity(x: &[f64], fs: f64, w: &WelchParams, b: &BandParams) -> Result<Periodicity, Error> {
    let psd = welch(x, fs, w)?;
    let nyquist = psd.freqs[psd.freqs.len() - 1];
    if !(b.lo > 0.0 && b.lo < b.hi && b.hi <= nyquist) {
        return Err(Error(format!("The band must satisfy 0 < lo < hi <= {nyquist} Hz.")));
    }
    let p = &psd.power;
    let (lo, hi) = (bin(&psd, b.lo).max(1), bin(&psd, b.hi));

    let total = sum(&p[1..]);
    let in_band = sum(&p[lo..=hi]);
    let band_fraction = if total > 0.0 { in_band / total } else { 0.0 };

    // Score the strongest peak within the tapping band.
    let k_band = argmax(p, lo..=hi);
    let w = (b.peak_halfwidth / psd.df()).round() as usize;
    let around = sum(&p[k_band.saturating_sub(w).max(lo)..=(k_band + w).min(hi)]);
    let sharpness = if in_band > 0.0 { around / in_band } else { 0.0 };

    // The band bounds the fundamental, not its harmonic: a 4 Hz tapper's
    // second harmonic is at 8 Hz. So the dominant peak is sought up to twice
    // the band's top, short of Nyquist.
    let search_hi = bin(&psd, (2.0 * b.hi).min(0.9 * nyquist)).max(hi);
    let k = argmax(p, lo..=search_hi);
    let peak_hz = refine(&psd, k);

    // Is the dominant peak the second harmonic? Its half must lie in the band,
    // carry real power, and be phase-locked to it.
    let window = |fc: f64| (bin(&psd, fc * (1.0 - b.half_tolerance)).max(1), bin(&psd, fc * (1.0 + b.half_tolerance)));
    let power_in = |(a, z): (usize, usize)| sum(&p[a..=z.max(a)]);
    let at_peak = power_in(window(peak_hz));
    let half_window = window(peak_hz / 2.0);
    let half_in_band = peak_hz / 2.0 >= b.lo && peak_hz / 2.0 <= b.hi;
    let half_ratio = if at_peak > 0.0 { power_in(half_window) / at_peak } else { 0.0 };
    let half_locking = if half_in_band && half_ratio >= b.half_min_ratio {
        locking(x, fs, peak_hz, b.locking_bandwidth)
    } else {
        0.0
    };

    let (case, f0_hz) = if half_in_band && half_ratio >= b.half_min_ratio && half_locking >= b.half_min_locking {
        (F0Case::Harmonic, refine(&psd, argmax(p, half_window.0..=half_window.1)))
    } else {
        // Not a harmonic: the fundamental is the strongest in-band peak.
        (F0Case::Fundamental, refine(&psd, k_band))
    };

    Ok(Periodicity {
        band_fraction,
        sharpness,
        score: band_fraction * sharpness,
        peak_hz,
        f0_hz,
        case,
        half_ratio,
        half_locking,
    })
}

/// Components ranked by score, best first.
#[derive(Debug, Clone)]
pub struct Selection {
    /// `(component index, periodicity)`, in descending order of score.
    pub ranking: Vec<(usize, Periodicity)>,
}

impl Selection {
    pub fn best(&self) -> &(usize, Periodicity) {
        &self.ranking[0]
    }

    pub fn runner_up(&self) -> Option<&(usize, Periodicity)> {
        self.ranking.get(1)
    }
}

/// Score every component, each a uniformly sampled series at `fs` Hz, and
/// rank them. Ties keep the original order.
pub fn select(components: &[Vec<f64>], fs: f64, w: &WelchParams, b: &BandParams) -> Result<Selection, Error> {
    if components.is_empty() {
        return Err(Error("No components to select from.".into()));
    }
    let mut ranking = components
        .iter()
        .enumerate()
        .map(|(i, x)| Ok((i, periodicity(x, fs, w, b)?)))
        .collect::<Result<Vec<_>, Error>>()?;
    ranking.sort_by(|x, y| y.1.score.total_cmp(&x.1.score));
    Ok(Selection { ranking })
}
