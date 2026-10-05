//! The post-capture pipeline: from captured frames to inter-tap intervals.
//!
//! [`analyse_trial`] runs stages 2 to 6 over the frames an [`Ingest`] holds:
//! preprocess, randomized SVD, resample every component to a uniform grid,
//! select the tapping component and its fundamental, and time its cycles.
//! QC gating and JTFS features build on its result. [`analyse_trace`] runs
//! the same stages from drift removal on, over one trace that did not come
//! from frames, such as a landmark distance, for comparison with the video.

use wavscat_core::Error;

use crate::ingest::{Ingest, IngestQc};
use crate::phase::{analyse_at, CycleAnalysis, PhaseParams};
use crate::preprocess::{detrend, preprocess, GainQc, PreprocessParams};
use crate::resample::resample;
use crate::select::{select, BandParams, F0Case, Periodicity, Selection};
use crate::spectrum::WelchParams;
use crate::svd::{randomized_svd, SvdParams};

/// Settings for every stage of [`analyse_trial`].
#[derive(Debug, Clone)]
pub struct PipelineParams {
    pub preprocess: PreprocessParams,
    pub svd: SvdParams,
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
            preprocess: PreprocessParams::default(),
            svd: SvdParams::default(),
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
    /// Index of the selected component.
    pub component: usize,
    /// The selected component, resampled: `t0 + i / fs`.
    pub t0: f64,
    pub signal: Vec<f64>,
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
pub fn analyse_trial(ing: &Ingest, p: &PipelineParams) -> Result<TrialResult, Error> {
    let ts = ing.timestamps();
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
    let competitor_ratio = match selection.competitor() {
        Some((_, r)) if best.score > 0.0 => r.score / best.score,
        _ => 0.0,
    };
    let cycles = analyse_at(&uniform[component], p.fs, t0, best.f0_hz, best.timing_harmonic(), &p.phase)?;

    // Loading spread: 1 / (cells * sum v^4) for a unit-norm loading.
    let mut loadings = svd.loadings.take().expect("requested");
    let map = loadings.swap_remove(component);
    let mut fourth = 0.0;
    for v in &map {
        fourth += v * v * v * v;
    }
    let loading_spread = if fourth > 0.0 { 1.0 / (map.len() as f64 * fourth) } else { 0.0 };
    let loading = if p.svd.return_loadings { Some(map) } else { None };

    Ok(TrialResult {
        ingest: ing.qc(),
        gain,
        singular_values: svd.singular_values,
        loading_spread,
        loading,
        component,
        t0,
        signal: uniform.swap_remove(component),
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
pub fn analyse_trace(timestamps: &[f64], values: &[f64], p: &PipelineParams) -> Result<TraceResult, Error> {
    let clean = detrend(timestamps, values, p.preprocess.detrend_cutoff)?;
    let u = resample(timestamps, &clean, p.fs, p.max_gap)?;
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
