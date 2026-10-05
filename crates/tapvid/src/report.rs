//! The trial report every binding returns: QC always, features and intervals
//! when the trial is accepted. One definition, so that a report has the same
//! keys in JavaScript, Python and R. With the `serde` feature it serialises.

use wavscat_core::Error;

use crate::features::{trial_features, FeatureParams};
use crate::ingest::Ingest;
use crate::pipeline::PipelineParams;
use crate::qc::{run_trial, QcParams};
use crate::select::F0Case;

/// QC measurements; `None` where the analysis did not get that far.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct QcSummary {
    pub frames_accepted: u64,
    pub frames_rejected: u64,
    pub frames_dropped: u64,
    pub effective_fps: f64,
    pub longest_gap: f64,
    pub dropped_fraction: f64,
    pub gain_min: Option<f64>,
    pub gain_max: Option<f64>,
    pub abrupt_changes: Option<usize>,
    pub dark_frames: Option<usize>,
    pub score: Option<f64>,
    pub competitor_ratio: Option<f64>,
    /// Whether the cycles were timed from a harmonic of the fundamental.
    pub from_harmonic: Option<bool>,
    pub f0_hz: Option<f64>,
    pub usable_cycles: Option<usize>,
    pub loading_spread: Option<f64>,
}

/// The feature vector and its version stamp.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FeatureSummary {
    pub schema_version: u32,
    pub crate_version: String,
    pub numerics_version: String,
    pub params_hash: String,
    pub names: Vec<String>,
    pub values: Vec<f64>,
}

/// Everything a trial produces for the caller.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TrialReport {
    pub accepted: bool,
    /// Rejection reasons, in words.
    pub reasons: Vec<String>,
    pub qc: QcSummary,
    /// Present only for an accepted trial.
    pub features: Option<FeatureSummary>,
    /// Inter-tap intervals in seconds, for an accepted trial.
    pub itis: Option<Vec<f64>>,
}

/// Analyse, gate and featurise the trial held by `ing`.
pub fn trial_report(ing: &Ingest, p: &PipelineParams, q: &QcParams, f: &FeatureParams) -> Result<TrialReport, Error> {
    let out = run_trial(ing, p, q);
    let qc = &out.qc;
    let mut report = TrialReport {
        accepted: qc.accepted(),
        reasons: qc.reasons.iter().map(|r| r.to_string()).collect(),
        qc: QcSummary {
            frames_accepted: qc.ingest.frames_accepted,
            frames_rejected: qc.ingest.frames_rejected,
            frames_dropped: qc.ingest.frames_dropped,
            effective_fps: qc.ingest.effective_fps,
            longest_gap: qc.ingest.longest_gap,
            dropped_fraction: qc.dropped_fraction,
            gain_min: qc.gain_min,
            gain_max: qc.gain_max,
            abrupt_changes: qc.abrupt_changes,
            dark_frames: qc.dark_frames,
            score: qc.score,
            competitor_ratio: qc.competitor_ratio,
            from_harmonic: qc.case.map(|c| c == F0Case::Harmonic),
            f0_hz: qc.f0_hz,
            usable_cycles: qc.usable_cycles,
            loading_spread: qc.loading_spread,
        },
        features: None,
        itis: None,
    };
    if let Some(r) = out.accepted() {
        let feats = trial_features(r, p, q, f)?;
        report.features = Some(FeatureSummary {
            schema_version: feats.schema_version,
            crate_version: feats.crate_version.to_string(),
            numerics_version: feats.numerics_version.to_string(),
            params_hash: feats.params_hash,
            names: feats.names,
            values: feats.values,
        });
        report.itis = Some(r.cycles.itis.clone());
    }
    Ok(report)
}
