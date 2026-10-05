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
    /// Which harmonic of `f0_hz` the dominant peak of this component is.
    pub harmonic: u32,
    /// Set by [`select`] when the component is a harmonic of motion below the
    /// tapping band, such as a sway; it cannot be selected.
    pub excluded: bool,
}

impl Periodicity {
    /// Which harmonic of `f0` stage 6 should time cycles from: the one that
    /// dominates this component. Timing from a weak fundamental fails once it
    /// is a fifth of the harmonic (see the `harmonic_timing` example), while
    /// timing from the harmonic does not.
    pub fn timing_harmonic(&self) -> u32 {
        self.harmonic
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
        harmonic: if case == F0Case::Harmonic { 2 } else { 1 },
        excluded: false,
    })
}

/// Components ranked by score, best first.
#[derive(Debug, Clone)]
pub struct Selection {
    /// `(component index, periodicity)`: selectable components in descending
    /// order of score, then excluded ones.
    pub ranking: Vec<(usize, Periodicity)>,
    /// Frequency tolerance, as a fraction, for telling oscillators apart.
    tolerance: f64,
}

impl Selection {
    /// The selected component.
    pub fn best(&self) -> &(usize, Periodicity) {
        &self.ranking[0]
    }

    /// The next selectable component in the ranking, whatever it is.
    pub fn runner_up(&self) -> Option<&(usize, Periodicity)> {
        self.ranking.get(1).filter(|(_, p)| !p.excluded)
    }

    /// The best-scoring selectable component that belongs to a different
    /// oscillator from the selected one. A moving hand spreads over several
    /// components that share one fundamental, so the plain runner-up is
    /// usually the same oscillator; this is the one that signals a second
    /// moving object.
    ///
    /// A component whose frequency is within the tolerance of 1, 2, 3 or 4
    /// times the selected fundamental, or of a half, third or quarter of it,
    /// counts as the same oscillator: a harmonic whose link to the
    /// fundamental was too weak to confirm in its own component.
    pub fn competitor(&self) -> Option<&(usize, Periodicity)> {
        let f0 = self.best().1.f0_hz;
        let related = |f: f64| {
            let r = if f >= f0 { f / f0 } else { f0 / f };
            let k = r.round();
            (1.0..=4.0).contains(&k) && (r - k).abs() <= self.tolerance * k
        };
        self.ranking[1..].iter().filter(|(_, p)| !p.excluded).find(|(_, p)| !related(p.f0_hz))
    }
}

/// Phase locking of the band of `low` around `f / m` to the band of `high`
/// around `f`: `|sum z_low^m conj(z_high)| / sum |z_low|^m |z_high|`, near 1
/// when `f` is the `m`-th harmonic of the oscillation at `f / m`.
fn cross_locking(low: &[f64], high: &[f64], fs: f64, f: f64, m: u32, bandwidth: f64) -> f64 {
    let fl = f / m as f64;
    let z1 = analytic_band(low, fs, fl, bandwidth * fl, 3.0);
    let z2 = analytic_band(high, fs, f, bandwidth * f, 3.0);
    let (mut re, mut im, mut norm) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in z1.iter().zip(&z2) {
        let mut zp = *a;
        for _ in 1..m {
            zp = zp * *a;
        }
        let c = zp * b.conj();
        re += c.re;
        im += c.im;
        norm += c.norm();
    }
    if norm > 0.0 { (re * re + im * im).sqrt() / norm } else { 0.0 }
}

/// Score every component, each a uniformly sampled series at `fs` Hz, find
/// the oscillator each belongs to, and rank them.
///
/// The SVD spreads one moving object over several components, and one
/// component can carry a single harmonic of the motion: a hand tapping at
/// 3 Hz can leave a pure 6 Hz component, and a body swaying at 0.4 Hz a pure
/// 1.2 Hz one, sharper than any tapping peak. So the dominant peak `f` of
/// each component is tested against subharmonics `f / m`, `m = 4, 3, 2`, in
/// every component: one with real power that is phase-locked to `f` makes
/// `f` its `m`-th harmonic. If that fundamental lies in the band, the
/// component is timed at harmonic `m`; if it lies below, the component is a
/// harmonic of slow motion and is excluded from selection.
pub fn select(components: &[Vec<f64>], fs: f64, w: &WelchParams, b: &BandParams) -> Result<Selection, Error> {
    if components.is_empty() {
        return Err(Error("No components to select from.".into()));
    }
    let psds = components.iter().map(|x| welch(x, fs, w)).collect::<Result<Vec<_>, Error>>()?;
    let mut ranking = components
        .iter()
        .enumerate()
        .map(|(i, x)| Ok((i, periodicity(x, fs, w, b)?)))
        .collect::<Result<Vec<_>, Error>>()?;

    let window = |psd: &Psd, fc: f64| {
        (bin(psd, fc * (1.0 - b.half_tolerance)).max(1), bin(psd, fc * (1.0 + b.half_tolerance)))
    };
    let window_power = |psd: &Psd, fc: f64| {
        let (a, z) = window(psd, fc);
        sum(&psd.power[a..=z.max(a)])
    };
    // A subharmonic must be a genuine peak in its component: at least twice
    // the mean power of its flanks. The phase-noise skirt of a strong line
    // slopes rather than peaks, and can otherwise pass for one.
    let prominent = |psd: &Psd, fc: f64| {
        let flanks = 0.5 * (window_power(psd, 0.75 * fc) + window_power(psd, 1.25 * fc));
        window_power(psd, fc) >= 2.0 * flanks
    };
    for (i, per) in ranking.iter_mut() {
        // Walk down the harmonic chain from the dominant peak: at each step
        // take the smallest m whose subharmonic is a prominent peak in some
        // component and phase-locked to the current frequency, until none is.
        // 12 Hz goes to 6 and then 3; a sway's 1.2 Hz goes to 0.4.
        let mut f = per.peak_hz;
        let mut source = *i;
        let mut total = 1u32;
        let mut moved = false;
        loop {
            let at_f = window_power(&psds[source], f);
            if at_f <= 0.0 {
                break;
            }
            let mut step: Option<(u32, usize, f64)> = None;
            for m in 2..=4u32 {
                let target = f / m as f64;
                if target < 2.0 * psds[source].df() {
                    break;
                }
                let mut found: Option<(usize, f64)> = None;
                for (j, x) in components.iter().enumerate() {
                    if window_power(&psds[j], target) / at_f < b.half_min_ratio || !prominent(&psds[j], target) {
                        continue;
                    }
                    let lock = cross_locking(x, &components[source], fs, f, m, b.locking_bandwidth);
                    let better = match found {
                        Some((_, l)) => lock > l,
                        None => true,
                    };
                    if lock >= b.half_min_locking && better {
                        found = Some((j, lock));
                    }
                }
                if let Some((j, _)) = found {
                    let psd = &psds[j];
                    let (a, z) = window(psd, target);
                    step = Some((m, j, refine(psd, argmax(&psd.power, a..=z.max(a)))));
                    break;
                }
            }
            match step {
                Some((m, j, below)) => {
                    f = below;
                    source = j;
                    total *= m;
                    moved = true;
                }
                None => break,
            }
        }
        if moved {
            if f < b.lo {
                per.excluded = true;
            } else if f <= b.hi {
                per.f0_hz = f;
                per.harmonic = total;
                per.case = F0Case::Harmonic;
            }
        }
    }

    // Selectable components by score, then excluded ones.
    ranking.sort_by(|x, y| x.1.excluded.cmp(&y.1.excluded).then(y.1.score.total_cmp(&x.1.score)));
    if ranking[0].1.excluded {
        return Err(Error(
            "Every component is a harmonic of motion below the tapping band; no tapping found.".into(),
        ));
    }
    Ok(Selection { ranking, tolerance: b.half_tolerance })
}
