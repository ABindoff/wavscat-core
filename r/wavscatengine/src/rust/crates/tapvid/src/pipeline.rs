//! The post-capture pipeline: from captured frames to inter-tap intervals.
//!
//! [`analyse_trial`] runs stages 2 to 6 over the frames an [`Ingest`] holds:
//! preprocess, randomized SVD, resample every component to a uniform grid,
//! select the tapping component and its fundamental, and time its cycles.
//! QC gating and JTFS features build on its result. [`analyse_trace`] runs
//! the same stages from drift removal on, over one trace that did not come
//! from frames, such as a landmark distance, for comparison with the video.

use wavscat_core::Error;

use crate::clock::{regularise, ClockParams, ClockQc};
use crate::extract::{extract, ExtractParams};
use crate::ingest::{Ingest, IngestQc};
use crate::phase::{analyse_at, CycleAnalysis, PhaseParams};
use crate::preprocess::{detrend, preprocess, GainQc, PreprocessParams};
use crate::resample::resample;
use crate::select::{periodicity, select, BandParams, F0Case, Periodicity, Selection};
use crate::spectrum::WelchParams;
use crate::svd::{randomized_svd, SvdParams};

/// Settings for every stage of [`analyse_trial`].
#[derive(Debug, Clone)]
pub struct PipelineParams {
    /// Re-estimate frame times from the camera's regular clock; `None`
    /// uses the timestamps as they came.
    pub clock: Option<ClockParams>,
    pub preprocess: PreprocessParams,
    pub svd: SvdParams,
    /// Combine every component by its power in the tapping band (see
    /// [`crate::extract`]); `None` keeps the single selected component.
    pub extract: Option<ExtractParams>,
    /// Uniform rate the components are resampled to, in Hz.
    pub fs: f64,
    /// Longest capture gap, in seconds, that resampling may bridge.
    pub max_gap: f64,
    pub welch: WelchParams,
    pub band: BandParams,
    pub phase: PhaseParams,
}

impl Default for PipelineParams {
    fn default() -> Self {
        PipelineParams {
            clock: Some(ClockParams::default()),
            preprocess: PreprocessParams::default(),
            svd: SvdParams::default(),
            extract: Some(ExtractParams::default()),
            fs: 30.0,
            max_gap: 0.15,
            welch: WelchParams::default(),
            band: BandParams::default(),
            phase: PhaseParams::default(),
        }
    }
}

/// Everything one trial produces before gating.
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub ingest: IngestQc,
    /// Frame times used, in seconds from the oldest frame: re-estimated from
    /// the camera's clock unless that was turned off.
    pub timestamps: Vec<f64>,
    /// What re-estimating the frame times found, if it was done.
    pub clock: Option<ClockQc>,
    pub gain: GainQc,
    /// Singular values of the preprocessed frames.
    pub singular_values: Vec<f64>,
    /// Participation ratio of the selected component's spatial loading: the
    /// effective fraction of grid cells it involves, from near 0 (one cell)
    /// to 1 (all equally). A hand is localised; a lighting change or motion
    /// of the whole frame is not.
    pub loading_spread: f64,
    /// The selected component's spatial loading, only if
    /// `svd.return_loadings` was set. It shows where the hand is.
    pub loading: Option<Vec<f64>>,
    /// Components ranked by periodicity score.
    pub selection: Selection,
    /// Index of the selected component, which sets the rate.
    pub component: usize,
    /// The tapping signal, resampled: `t0 + i / fs`. The combination of
    /// components extracted by their power in the tapping band, or the
    /// selected component alone if extraction is off.
    pub t0: f64,
    pub signal: Vec<f64>,
    /// Weight of each component in `signal`, and the share of its power in
    /// the tapping bands; `None` without extraction.
    pub weights: Option<Vec<f64>>,
    pub band_fraction: Option<f64>,
    /// Periodicity score of `signal`, for QC.
    pub score: f64,
    /// The harmonic of `f0_hz` the cycles were timed from.
    pub timing_harmonic: u32,
    /// Tapping fundamental, in Hz, and whether it was read from the second
    /// harmonic.
    pub f0_hz: f64,
    pub case: F0Case,
    /// Score of the best component from a different oscillator, over the
    /// selected score: near one means a second moving object in the frame
    /// was almost as tapping-like. Zero if there is none.
    pub competitor_ratio: f64,
    pub cycles: CycleAnalysis,
}

/// Analyse the frames held by `ing`. Times in the result are seconds from
/// the oldest held frame.
/// Overlap of two spatial maps: the cosine similarity of their squares, from
/// 0 when they move different pixels to 1 when they move the same ones in
/// the same proportions.
pub fn map_overlap(a: &[f64], b: &[f64]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (x2, y2) = (x * x, y * y);
        ab += x2 * y2;
        aa += x2 * x2;
        bb += y2 * y2;
    }
    if aa > 0.0 && bb > 0.0 { ab / (aa * bb).sqrt() } else { 0.0 }
}

/// The frame times to analyse with: `raw`, re-estimated from the camera's
/// clock if `p.clock` asks for it.
pub fn frame_times(raw: Vec<f64>, p: &PipelineParams) -> Result<(Vec<f64>, Option<ClockQc>), Error> {
    match &p.clock {
        Some(c) => {
            let (t, qc) = regularise(&raw, c)?;
            Ok((t, Some(qc)))
        }
        None => Ok((raw, None)),
    }
}

pub fn analyse_trial(ing: &Ingest, p: &PipelineParams) -> Result<TrialResult, Error> {
    let (ts, clock) = frame_times(ing.timestamps(), p)?;
    let (pre, gain) = preprocess(ing, &ts, &p.preprocess)?;
    // Loadings are always computed for QC; they leave only if requested.
    let mut svd_params = p.svd.clone();
    svd_params.return_loadings = true;
    let mut svd = randomized_svd(&pre, &svd_params)?;

    let mut uniform = Vec::with_capacity(svd.scores.len());
    let mut t0 = 0.0;
    for score in &svd.scores {
        let u = resample(&ts, score, p.fs, p.max_gap)?;
        if let Some(g) = u.gaps.first() {
            return Err(Error(format!(
                "A capture gap from {:.3} s to {:.3} s is longer than {} s; split or reject the trial.",
                g.start, g.end, p.max_gap
            )));
        }
        t0 = u.t0;
        uniform.push(u.values);
    }

    let selection = select(&uniform, p.fs, &p.welch, &p.band)?;
    let (component, best) = selection.best().clone();
    let loadings = svd.loadings.take().expect("requested");

    // The tapping signal: every component weighted by its power in the
    // tapping band, or the selected one alone.
    let (signal, weights, band_fraction, harmonic, map) = match &p.extract {
        Some(ep) => {
            let ex = extract(&uniform, p.fs, best.f0_hz, component, ep)?;
            // The pixel map of the signal's pattern, not of its weights.
            let mut map = vec![0.0; loadings[0].len()];
            for (w, l) in ex.pattern.iter().zip(&loadings) {
                for (m, v) in map.iter_mut().zip(l) {
                    *m += w * v;
                }
            }
            let norm = map.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for m in map.iter_mut() {
                    *m /= norm;
                }
            }
            let h = ex.timing_harmonic();
            (ex.signal, Some(ex.weights), Some(ex.band_fraction), h, map)
        }
        None => (uniform[component].clone(), None, None, best.timing_harmonic(), loadings[component].clone()),
    };
    let score = if p.extract.is_some() { periodicity(&signal, p.fs, &p.welch, &p.band)?.score } else { best.score };
    // The strongest other rhythm that is fast enough to be someone else
    // tapping rather than the participant swaying, and somewhere else in the
    // frame, against the tapping's strength.
    let competitor = selection
        .competitors_above(p.band.competitor_min_hz)
        .find(|(i, _)| map_overlap(&loadings[*i], &map) < p.band.competitor_max_overlap);
    let tapping_strength = selection.oscillator_strength();
    let competitor_ratio = match competitor {
        Some((_, r)) if tapping_strength > 0.0 => r.strength / tapping_strength,
        _ => 0.0,
    };
    let cycles = analyse_at(&signal, p.fs, t0, best.f0_hz, harmonic, &p.phase)?;

    // Loading spread: 1 / (cells * sum v^4) for a unit-norm loading.
    let mut fourth = 0.0;
    for v in &map {
        fourth += v * v * v * v;
    }
    let loading_spread = if fourth > 0.0 { 1.0 / (map.len() as f64 * fourth) } else { 0.0 };
    let loading = if p.svd.return_loadings { Some(map) } else { None };

    Ok(TrialResult {
        ingest: ing.qc(),
        timestamps: ts,
        clock,
        gain,
        singular_values: svd.singular_values,
        loading_spread,
        loading,
        component,
        t0,
        signal,
        weights,
        band_fraction,
        score,
        timing_harmonic: harmonic,
        f0_hz: best.f0_hz,
        case: best.case,
        competitor_ratio,
        cycles,
        selection,
    })
}

/// One trace through the stages after the SVD.
#[derive(Debug, Clone)]
pub struct TraceResult {
    /// The trace, detrended and resampled: `t0 + i / fs`.
    pub t0: f64,
    pub signal: Vec<f64>,
    pub periodicity: Periodicity,
    pub f0_hz: f64,
    pub case: F0Case,
    pub cycles: CycleAnalysis,
}

/// Analyse a single trace sampled at `timestamps` seconds, such as the
/// distance between two hand landmarks, as [`analyse_trial`] analyses the
/// selected video component: drift removed as each pixel's is, resampled to
/// the same grid, its fundamental found the same way, and its cycles timed.
///
/// The frame times are re-estimated from the camera's clock as they are for
/// video, so pass the trace's frames' own capture times, gaps and all.
pub fn analyse_trace(timestamps: &[f64], values: &[f64], p: &PipelineParams) -> Result<TraceResult, Error> {
    if values.len() != timestamps.len() {
        return Err(Error("There must be one timestamp per value.".into()));
    }
    let (ts, _) = frame_times(timestamps.to_vec(), p)?;
    let clean = detrend(&ts, values, p.preprocess.detrend_cutoff)?;
    let u = resample(&ts, &clean, p.fs, p.max_gap)?;
    if let Some(g) = u.gaps.first() {
        return Err(Error(format!(
            "A gap in the trace from {:.3} s to {:.3} s is longer than {} s.",
            g.start, g.end, p.max_gap
        )));
    }
    let selection = select(std::slice::from_ref(&u.values), p.fs, &p.welch, &p.band)?;
    let best = selection.best().1.clone();
    if best.excluded {
        return Err(Error("The trace's rhythm is a harmonic of motion below the tapping band.".into()));
    }
    let cycles = analyse_at(&u.values, p.fs, u.t0, best.f0_hz, best.timing_harmonic(), &p.phase)?;
    Ok(TraceResult { t0: u.t0, f0_hz: best.f0_hz, case: best.case, periodicity: best, cycles, signal: u.values })
}
