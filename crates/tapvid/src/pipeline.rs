//! The post-capture pipeline: from captured frames to inter-tap intervals.
//!
//! [`analyse_trial`] runs stages 2 to 6 over the frames an [`Ingest`] holds:
//! preprocess, randomized SVD, resample every component to a uniform grid,
//! select the tapping component and its fundamental, and time its cycles.
//! QC gating and JTFS features build on its result.

use wavscat_core::Error;

use crate::ingest::{Ingest, IngestQc};
use crate::phase::{analyse_at, CycleAnalysis, PhaseParams};
use crate::preprocess::{preprocess, GainQc, PreprocessParams};
use crate::resample::resample;
use crate::select::{select, BandParams, F0Case, Selection};
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
    let svd = randomized_svd(&pre, &p.svd)?;

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

    Ok(TrialResult {
        ingest: ing.qc(),
        gain,
        singular_values: svd.singular_values,
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
