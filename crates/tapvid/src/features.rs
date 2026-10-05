//! Stage 7: the feature vector of a trial, and the version stamp that makes
//! feature vectors comparable.
//!
//! Two sets of features describe the selected tapping component:
//!
//! - from stage 6, the inter-tap intervals and amplitude: mean, SD, CV,
//!   robust equivalents, lag-1 autocorrelation, relative amplitude slope and
//!   phase diffusion;
//! - from joint time-frequency scattering (wavscat-core), modulation
//!   features: period jitter moves energy between adjacent first-order bands,
//!   which lands in the second-order and frequential-modulation paths.
//!
//! The component is scaled to unit RMS before scattering. Scattering is
//! homogeneous, `S(c x) = |c| S(x)`, so this removes overall amplitude, which
//! depends on camera distance and hand size and carries no clinical signal,
//! from every path. Second-order paths are then optionally divided by the
//! first-order energy they ride on, and all are log-compressed with a fixed
//! `eps` and averaged over time, giving one value per path.
//!
//! The log uses a fixed `eps`, not one taken from each recording's own
//! quantiles as the R recipe does, so that trials share one scale.
//!
//! Features from different settings are not comparable, so every vector
//! carries the feature-schema version, the crate and numerics versions, and a
//! hash of every parameter that produced it, the SVD seed included.

use std::fmt::Write as _;

use wavscat_core::features::log_compress;
use wavscat_core::jtfs::{ParamsJtfs, ScatteringJtfs};
use wavscat_core::Error;

use crate::pipeline::{PipelineParams, TrialResult};
use crate::qc::QcParams;

/// Version of the feature definitions: names, order and meaning. Bumped
/// whenever a feature is added, removed or redefined.
pub const FEATURE_SCHEMA_VERSION: u32 = 1;

/// Settings for the scattering features.
#[derive(Debug, Clone)]
pub struct FeatureParams {
    /// Log-scale of the temporal transform: the widest wavelet spans `2^J`
    /// samples, 4.3 s at 30 Hz for `J = 7`.
    pub j: u32,
    /// Wavelets per octave at orders one and two.
    pub q: [u32; 2],
    /// Log-scale of the frequential transform, in first-order bands.
    pub j_fr: u32,
    /// Averaging window, in seconds.
    pub t_sec: f64,
    /// Divide second-order paths by first-order energy.
    pub renorm: bool,
    /// Floor added to the first-order energy when renormalising.
    pub renorm_eps: f64,
    /// Log compression `log1p(v / eps)`, relative to the unit-RMS signal.
    pub log_eps: f64,
}

impl Default for FeatureParams {
    fn default() -> Self {
        FeatureParams { j: 7, q: [8, 1], j_fr: 3, t_sec: 6.0, renorm: true, renorm_eps: 1e-6, log_eps: 1e-3 }
    }
}

/// A named feature vector and the stamp of what produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct Features {
    pub schema_version: u32,
    pub crate_version: &'static str,
    pub numerics_version: &'static str,
    /// Hash of every parameter, as 16 hexadecimal digits.
    pub params_hash: String,
    pub names: Vec<String>,
    pub values: Vec<f64>,
}

/// Scattering features of a uniformly sampled signal at `fs` Hz: one value
/// per joint path, named `jtfs_<path>`.
pub fn jtfs_features(signal: &[f64], fs: f64, p: &FeatureParams) -> Result<(Vec<String>, Vec<f64>), Error> {
    let n = signal.len();
    let mut ss = 0.0;
    for v in signal {
        ss += v * v;
    }
    let rms = (ss / n as f64).sqrt();
    if !(rms > 0.0) {
        return Err(Error("The signal is flat; there is nothing to scatter.".into()));
    }
    let x: Vec<f64> = signal.iter().map(|v| v / rms).collect();

    let mut params = ParamsJtfs::new(n, p.j, p.q.to_vec());
    params.j_fr = p.j_fr;
    params.time.t_sec = Some(p.t_sec);
    params.time.sr = Some(fs);
    let op = ScatteringJtfs::new(&params)?;
    let (mut coefs, s1) = op.transform_with_s1(&x)?;
    if p.renorm {
        op.renorm(&mut coefs, &s1, p.renorm_eps)?;
    }
    let mut values = Vec::with_capacity(coefs.len());
    for c in coefs.iter_mut() {
        log_compress(&mut c.data, p.log_eps)?;
        let mut acc = 0.0;
        for v in &c.data {
            acc += v;
        }
        values.push(acc / c.data.len() as f64);
    }
    let names = op.paths.iter().map(|path| format!("jtfs_{}", path.label)).collect();
    Ok((names, values))
}

/// Names of the interval and amplitude features, in vector order.
pub const ITI_FEATURES: [&str; 10] = [
    "f0_hz",
    "iti_mean",
    "iti_sd",
    "iti_cv",
    "iti_median",
    "iti_mad",
    "iti_robust_cv",
    "iti_lag1",
    "amplitude_relative_slope",
    "phase_diffusion",
];

/// The full feature vector of an analysed trial.
pub fn trial_features(
    r: &TrialResult,
    pipeline: &PipelineParams,
    qc: &QcParams,
    p: &FeatureParams,
) -> Result<Features, Error> {
    let s = &r.cycles.summary;
    let mut names: Vec<String> = ITI_FEATURES.iter().map(|n| n.to_string()).collect();
    let mut values = vec![
        r.f0_hz,
        s.mean,
        s.sd,
        s.cv,
        s.median,
        s.mad,
        s.robust_cv,
        s.lag1_autocorrelation,
        s.relative_amplitude_slope,
        s.phase_diffusion,
    ];
    let (jn, jv) = jtfs_features(&r.signal, pipeline.fs, p)?;
    names.extend(jn);
    values.extend(jv);
    Ok(Features {
        schema_version: FEATURE_SCHEMA_VERSION,
        crate_version: env!("CARGO_PKG_VERSION"),
        numerics_version: wavscat_core::NUMERICS_VERSION,
        params_hash: params_hash(pipeline, qc, p),
        names,
        values,
    })
}

/// Every parameter as one canonical string: `name=value` lines in a fixed
/// order, numbers in Rust's shortest round-trip form. Written out by hand
/// rather than taken from `Debug`, whose format is not promised to be
/// stable.
pub fn canonical_params(pipeline: &PipelineParams, qc: &QcParams, p: &FeatureParams) -> String {
    let mut s = String::new();
    let mut put = |k: &str, v: String| {
        writeln!(s, "{k}={v}").unwrap();
    };
    let opt = |v: Option<f64>| v.map_or("none".to_string(), |x| format!("{x:?}"));
    put("schema", FEATURE_SCHEMA_VERSION.to_string());
    put("numerics", wavscat_core::NUMERICS_VERSION.to_string());
    let pp = &pipeline.preprocess;
    put("preprocess.detrend_cutoff", opt(pp.detrend_cutoff));
    put("preprocess.abrupt_change", format!("{:?}", pp.abrupt_change));
    put("preprocess.dark_level", format!("{:?}", pp.dark_level));
    let sv = &pipeline.svd;
    put("svd.rank", sv.rank.to_string());
    put("svd.oversample", sv.oversample.to_string());
    put("svd.power_iters", sv.power_iters.to_string());
    put("svd.seed", sv.seed.to_string());
    put("fs", format!("{:?}", pipeline.fs));
    put("max_gap", format!("{:?}", pipeline.max_gap));
    let w = &pipeline.welch;
    put("welch.segment_sec", format!("{:?}", w.segment_sec));
    put("welch.overlap", format!("{:?}", w.overlap));
    put("welch.min_nfft", w.min_nfft.to_string());
    let b = &pipeline.band;
    put("band.lo", format!("{:?}", b.lo));
    put("band.hi", format!("{:?}", b.hi));
    put("band.peak_halfwidth", format!("{:?}", b.peak_halfwidth));
    put("band.half_tolerance", format!("{:?}", b.half_tolerance));
    put("band.half_min_ratio", format!("{:?}", b.half_min_ratio));
    put("band.half_min_locking", format!("{:?}", b.half_min_locking));
    put("band.locking_bandwidth", format!("{:?}", b.locking_bandwidth));
    let ph = &pipeline.phase;
    put("phase.bandwidth", format!("{:?}", ph.bandwidth));
    put("phase.pad_cycles", format!("{:?}", ph.pad_cycles));
    put("phase.diffusion_lags", ph.diffusion_lags.to_string());
    put("qc.min_frames", qc.min_frames.to_string());
    put("qc.min_fps", format!("{:?}", qc.min_fps));
    put("qc.max_dropped_fraction", format!("{:?}", qc.max_dropped_fraction));
    put("qc.min_score", format!("{:?}", qc.min_score));
    put("qc.max_competitor_ratio", format!("{:?}", qc.max_competitor_ratio));
    put("qc.min_usable_cycles", qc.min_usable_cycles.to_string());
    put("qc.max_abrupt_changes", qc.max_abrupt_changes.to_string());
    put("qc.max_gain_ratio", format!("{:?}", qc.max_gain_ratio));
    put("qc.max_dark_frames", qc.max_dark_frames.to_string());
    put("qc.max_loading_spread", format!("{:?}", qc.max_loading_spread));
    put("qc.usable_iti_factor", format!("{:?}", qc.usable_iti_factor));
    put("qc.usable_min_amplitude", format!("{:?}", qc.usable_min_amplitude));
    put("qc.min_usable_fraction", format!("{:?}", qc.min_usable_fraction));
    put("features.j", p.j.to_string());
    put("features.q", format!("{},{}", p.q[0], p.q[1]));
    put("features.j_fr", p.j_fr.to_string());
    put("features.t_sec", format!("{:?}", p.t_sec));
    put("features.renorm", p.renorm.to_string());
    put("features.renorm_eps", format!("{:?}", p.renorm_eps));
    put("features.log_eps", format!("{:?}", p.log_eps));
    s
}

/// 64-bit FNV-1a hash of [`canonical_params`], as 16 hexadecimal digits.
pub fn params_hash(pipeline: &PipelineParams, qc: &QcParams, p: &FeatureParams) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in canonical_params(pipeline, qc, p).bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
