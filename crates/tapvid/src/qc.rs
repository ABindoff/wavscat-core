//! Stage 8: quality control and gating.
//!
//! Every trial gets a QC report, whether or not it yields features. The
//! report holds the measurements and a list of reasons to reject; a trial is
//! accepted when the list is empty. Every threshold is a parameter, not a
//! constant: the defaults are starting points, to be calibrated on the
//! existing video corpus.
//!
//! The usual failures in uncontrolled home recordings, and what catches them:
//!
//! - the camera stalling or the app losing frames: frame rate, dropped
//!   frames, capture gaps;
//! - the lighting changing: the exposure trace (abrupt changes, range, dark
//!   frames);
//! - the participant pausing or stopping: too few usable cycles;
//! - the hand leaving the frame, or nothing rhythmic in it: a weak
//!   periodicity score;
//! - a second moving object, a person or a pet: a competing oscillator;
//! - motion of the whole frame rather than of a hand: a spread-out spatial
//!   loading.

use std::fmt;

use crate::ingest::{Ingest, IngestQc};
use crate::pipeline::{analyse_trial, PipelineParams, TrialResult};
use crate::select::F0Case;

/// Gating thresholds. The defaults are starting points for calibration.
#[derive(Debug, Clone)]
pub struct QcParams {
    pub min_frames: usize,
    pub min_fps: f64,
    /// Largest acceptable fraction of frames dropped.
    pub max_dropped_fraction: f64,
    pub min_score: f64,
    pub max_competitor_ratio: f64,
    pub min_usable_cycles: usize,
    pub max_abrupt_changes: usize,
    /// Largest acceptable ratio of brightest to darkest frame mean.
    pub max_gain_ratio: f64,
    pub max_dark_frames: usize,
    /// Largest acceptable participation ratio of the selected loading.
    pub max_loading_spread: f64,
    /// A cycle is usable when its interval is within this factor of the
    /// median interval, either way...
    pub usable_iti_factor: f64,
    /// ...and its amplitude at least this fraction of the 90th percentile of
    /// cycle amplitudes, which is what full tapping looks like in this trial.
    pub usable_min_amplitude: f64,
    /// Smallest acceptable fraction of cycles that are usable. A stop or a
    /// long pause leaves many low-amplitude cycles.
    pub min_usable_fraction: f64,
}

impl Default for QcParams {
    fn default() -> Self {
        QcParams {
            min_frames: 150,
            min_fps: 15.0,
            max_dropped_fraction: 0.10,
            min_score: 0.5,
            max_competitor_ratio: 0.5,
            min_usable_cycles: 20,
            max_abrupt_changes: 3,
            max_gain_ratio: 2.0,
            max_dark_frames: 0,
            max_loading_spread: 0.1,
            usable_iti_factor: 2.0,
            usable_min_amplitude: 0.25,
            min_usable_fraction: 0.8,
        }
    }
}

/// Why a trial was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum Rejection {
    TooFewFrames { frames: usize },
    LowFrameRate { fps: f64 },
    DroppedFrames { fraction: f64 },
    /// A capture gap, in seconds from the first frame, too long to bridge.
    /// The caller may split the trial here instead of rejecting it.
    CaptureGap { start: f64, end: f64 },
    AbruptExposure { changes: usize },
    LightingRange { ratio: f64 },
    DarkFrames { count: usize },
    /// The analysis could not run, for example because nothing in the frame
    /// was periodic in the tapping band.
    AnalysisFailed { reason: String },
    WeakPeriodicity { score: f64 },
    CompetingOscillator { ratio: f64 },
    FewCycles { usable: usize },
    /// Too many cycles were unusable: the tapping paused or stopped.
    Interrupted { usable_fraction: f64 },
    DiffuseMotion { spread: f64 },
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rejection::TooFewFrames { frames } => write!(f, "only {frames} frames captured"),
            Rejection::LowFrameRate { fps } => write!(f, "frame rate {fps:.1} fps is too low"),
            Rejection::DroppedFrames { fraction } => write!(f, "{:.0}% of frames were dropped", 100.0 * fraction),
            Rejection::CaptureGap { start, end } => {
                write!(f, "capture stopped from {start:.2} s to {end:.2} s; split or reject the trial")
            }
            Rejection::AbruptExposure { changes } => write!(f, "exposure changed abruptly {changes} times"),
            Rejection::LightingRange { ratio } => write!(f, "brightness varied {ratio:.1}-fold"),
            Rejection::DarkFrames { count } => write!(f, "{count} frames were too dark"),
            Rejection::AnalysisFailed { reason } => write!(f, "analysis failed: {reason}"),
            Rejection::WeakPeriodicity { score } => write!(f, "no clear rhythmic movement (score {score:.2})"),
            Rejection::CompetingOscillator { ratio } => {
                write!(f, "another rhythmic movement was almost as strong (ratio {ratio:.2})")
            }
            Rejection::FewCycles { usable } => write!(f, "only {usable} usable tapping cycles"),
            Rejection::Interrupted { usable_fraction } => {
                write!(f, "tapping paused or stopped: only {:.0}% of cycles usable", 100.0 * usable_fraction)
            }
            Rejection::DiffuseMotion { spread } => {
                write!(f, "the movement is spread over the frame (spread {spread:.2}), not localised like a hand")
            }
        }
    }
}

/// The measurements behind the gating, and the verdict.
#[derive(Debug, Clone)]
pub struct QcReport {
    pub ingest: IngestQc,
    /// Fraction of frames dropped: from the camera's clock when frame times
    /// were re-estimated (empty slots), otherwise from long intervals.
    pub dropped_fraction: f64,
    /// Changes of frame rate during the trial, and the timestamp error the
    /// clock fit removed (RMS, seconds); `None` without re-estimation.
    pub rate_changes: Option<usize>,
    pub clock_rms: Option<f64>,
    /// Exposure: darkest and brightest frame means, abrupt changes, dark
    /// frames. `None` when the analysis did not get that far.
    pub gain_min: Option<f64>,
    pub gain_max: Option<f64>,
    pub abrupt_changes: Option<usize>,
    pub dark_frames: Option<usize>,
    /// Periodicity score of the selected component.
    pub score: Option<f64>,
    /// Score of the best competing oscillator over the selected score.
    pub competitor_ratio: Option<f64>,
    pub case: Option<F0Case>,
    pub f0_hz: Option<f64>,
    pub usable_cycles: Option<usize>,
    pub loading_spread: Option<f64>,
    pub reasons: Vec<Rejection>,
}

impl QcReport {
    pub fn accepted(&self) -> bool {
        self.reasons.is_empty()
    }
}

/// Number of cycles whose interval and amplitude are both plausible; see
/// [`usable_mask`].
pub fn usable_cycles(itis: &[f64], amplitudes: &[f64], factor: f64, min_amp: f64) -> usize {
    usable_mask(itis, amplitudes, factor, min_amp).iter().filter(|u| **u).count()
}

/// Which cycles have an interval and amplitude that are both plausible: the
/// interval within `factor` of the median either way, the amplitude at least
/// `min_amp` of the 90th percentile of amplitudes. The reference is a high
/// quantile, not the median, because after a stop most cycles are noise and
/// the median would be noise too. Pauses and stops fail one or both, and so
/// does a stretch where the signal lost the hand.
pub fn usable_mask(itis: &[f64], amplitudes: &[f64], factor: f64, min_amp: f64) -> Vec<bool> {
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        let h = s.len() / 2;
        if s.is_empty() {
            0.0
        } else if s.len() % 2 == 1 {
            s[h]
        } else {
            (s[h - 1] + s[h]) / 2.0
        }
    };
    let mi = median(itis);
    let ma = {
        let mut s = amplitudes.to_vec();
        s.sort_by(|a, b| a.total_cmp(b));
        if s.is_empty() { 0.0 } else { s[((s.len() - 1) as f64 * 0.9).round() as usize] }
    };
    itis.iter()
        .zip(amplitudes)
        .map(|(i, a)| *i >= mi / factor && *i <= mi * factor && *a >= min_amp * ma)
        .collect()
}

/// A trial: its QC report always, its analysis when it got that far.
#[derive(Debug, Clone)]
pub struct TrialOutput {
    pub qc: QcReport,
    pub result: Option<TrialResult>,
}

impl TrialOutput {
    /// The analysis, if the trial passed every gate.
    pub fn accepted(&self) -> Option<&TrialResult> {
        if self.qc.accepted() { self.result.as_ref() } else { None }
    }
}

/// Analyse the trial held by `ing` and gate it.
pub fn run_trial(ing: &Ingest, p: &PipelineParams, q: &QcParams) -> TrialOutput {
    let iq = ing.qc();
    let total = iq.frames_accepted + iq.frames_dropped;
    let mut dropped_fraction = if total > 0 { iq.frames_dropped as f64 / total as f64 } else { 0.0 };
    // The camera's clock, if it fits, counts dropped frames properly: an
    // interval of 1.5 frames on a coarse timestamp grid is not a drop.
    let clock = match &p.clock {
        Some(c) if ing.len() >= 3 => crate::clock::regularise(&ing.timestamps(), c).ok().map(|(_, q)| q),
        _ => None,
    };
    if let Some(c) = &clock {
        if c.segments.iter().any(|s| s.regular) {
            dropped_fraction = c.missing_fraction;
        }
    }
    let mut qc = QcReport {
        ingest: iq.clone(),
        dropped_fraction,
        rate_changes: clock.as_ref().map(|c| c.rate_changes),
        clock_rms: clock.as_ref().map(|c| c.rms_residual),
        gain_min: None,
        gain_max: None,
        abrupt_changes: None,
        dark_frames: None,
        score: None,
        competitor_ratio: None,
        case: None,
        f0_hz: None,
        usable_cycles: None,
        loading_spread: None,
        reasons: Vec::new(),
    };

    // Capture: enough frames, fast enough, few enough lost.
    if ing.len() < q.min_frames {
        qc.reasons.push(Rejection::TooFewFrames { frames: ing.len() });
        return TrialOutput { qc, result: None };
    }
    if iq.effective_fps < q.min_fps {
        qc.reasons.push(Rejection::LowFrameRate { fps: iq.effective_fps });
    }
    if dropped_fraction > q.max_dropped_fraction {
        qc.reasons.push(Rejection::DroppedFrames { fraction: dropped_fraction });
    }
    let ts = ing.timestamps();
    if let Some(w) = ts.windows(2).find(|w| w[1] - w[0] > p.max_gap) {
        qc.reasons.push(Rejection::CaptureGap { start: w[0], end: w[1] });
        return TrialOutput { qc, result: None };
    }

    let r = match analyse_trial(ing, p) {
        Ok(r) => r,
        Err(e) => {
            qc.reasons.push(Rejection::AnalysisFailed { reason: e.0 });
            return TrialOutput { qc, result: None };
        }
    };

    // Lighting.
    qc.gain_min = Some(r.gain.min);
    qc.gain_max = Some(r.gain.max);
    qc.abrupt_changes = Some(r.gain.abrupt_changes);
    qc.dark_frames = Some(r.gain.dark_frames);
    if r.gain.abrupt_changes > q.max_abrupt_changes {
        qc.reasons.push(Rejection::AbruptExposure { changes: r.gain.abrupt_changes });
    }
    if r.gain.min > 0.0 && r.gain.max / r.gain.min > q.max_gain_ratio {
        qc.reasons.push(Rejection::LightingRange { ratio: r.gain.max / r.gain.min });
    }
    if r.gain.dark_frames > q.max_dark_frames {
        qc.reasons.push(Rejection::DarkFrames { count: r.gain.dark_frames });
    }

    // The tapping itself.
    let score = r.score;
    let usable = usable_cycles(&r.cycles.itis, &r.cycles.amplitudes, q.usable_iti_factor, q.usable_min_amplitude);
    qc.score = Some(score);
    qc.competitor_ratio = Some(r.competitor_ratio);
    qc.case = Some(r.case);
    qc.f0_hz = Some(r.f0_hz);
    qc.usable_cycles = Some(usable);
    qc.loading_spread = Some(r.loading_spread);
    if score < q.min_score {
        qc.reasons.push(Rejection::WeakPeriodicity { score });
    }
    if r.competitor_ratio > q.max_competitor_ratio {
        qc.reasons.push(Rejection::CompetingOscillator { ratio: r.competitor_ratio });
    }
    if usable < q.min_usable_cycles {
        qc.reasons.push(Rejection::FewCycles { usable });
    }
    let usable_fraction = usable as f64 / r.cycles.itis.len().max(1) as f64;
    if usable_fraction < q.min_usable_fraction {
        qc.reasons.push(Rejection::Interrupted { usable_fraction });
    }
    if r.loading_spread > q.max_loading_spread {
        qc.reasons.push(Rejection::DiffuseMotion { spread: r.loading_spread });
    }
    TrialOutput { qc, result: Some(r) }
}
