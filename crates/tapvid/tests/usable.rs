//! Interval features over usable cycles only, and hesitations: a pause, or a
//! stretch where the signal lost the hand, must not distort the interval
//! statistics, and must show up as a hesitation instead.

use tapvid::features::{hesitations, trace_features, FeatureParams, ITI_FEATURES};
use tapvid::phase::{summarise_kept, CycleAnalysis, ItiSummary};
use tapvid::pipeline::{analyse_trace, PipelineParams};
use tapvid::qc::{usable_mask, QcParams};
use tapvid::synth::{render, FrameClock, TapSpec};

/// Cycles at 250 ms with a little alternation, except two merged cycles
/// (500 ms, low amplitude) as a slip leaves them.
fn cycles_with_a_slip() -> CycleAnalysis {
    let mut itis = Vec::new();
    let mut amplitudes = Vec::new();
    for i in 0..30 {
        if i == 10 || i == 11 {
            itis.push(0.5);
            amplitudes.push(0.1);
        } else {
            itis.push(if i % 2 == 0 { 0.24 } else { 0.26 });
            amplitudes.push(1.0);
        }
    }
    let mut boundaries = vec![0.0, 0.25];
    for d in &itis {
        boundaries.push(boundaries.last().unwrap() + d);
    }
    boundaries.push(boundaries.last().unwrap() + 0.25);
    let summary = ItiSummary {
        n_cycles: itis.len(),
        mean: 0.0,
        sd: 0.0,
        cv: 0.0,
        median: 0.0,
        mad: 0.0,
        robust_cv: 0.0,
        lag1_autocorrelation: 0.0,
        mean_amplitude: 0.0,
        amplitude_slope: 0.0,
        relative_amplitude_slope: 0.0,
        phase_diffusion: 0.5,
    };
    CycleAnalysis { boundaries, itis, amplitudes, summary }
}

#[test]
fn merged_cycles_are_left_out_and_counted_as_one_hesitation() {
    let c = cycles_with_a_slip();
    let q = QcParams::default();
    let usable = usable_mask(&c.itis, &c.amplitudes, q.usable_iti_factor, q.usable_min_amplitude);
    assert_eq!(usable.iter().filter(|u| !**u).count(), 2);
    let s = summarise_kept(&c, &usable);
    assert_eq!(s.n_cycles, 28);
    assert!((s.mean - 0.25).abs() < 1e-12, "mean {}", s.mean);
    assert!(s.cv < 0.05, "cv {}", s.cv);
    // Strict alternation, and the gap does not join intervals 9 and 12.
    assert!(s.lag1_autocorrelation < -0.9, "lag 1 {}", s.lag1_autocorrelation);
    assert_eq!(s.phase_diffusion, 0.5);
    let (n, longest) = hesitations(&c, &usable);
    assert_eq!(n, 1);
    assert!((longest - 1.0).abs() < 1e-12);
}

#[test]
fn with_too_few_cycles_kept_the_statistics_are_nan_not_an_error() {
    let c = cycles_with_a_slip();
    let mut keep = vec![false; c.itis.len()];
    keep[0] = true;
    let s = summarise_kept(&c, &keep);
    assert_eq!(s.n_cycles, 1);
    assert!(s.sd.is_nan() && s.lag1_autocorrelation.is_nan() && s.amplitude_slope.is_nan());
    let none = summarise_kept(&c, &vec![false; c.itis.len()]);
    assert!(none.mean.is_nan());
}

/// A 10 s tapping trace at 4 Hz, optionally with the movement stopped from
/// 4.5 s to 6 s.
fn trace(pause: bool) -> (Vec<f64>, Vec<f64>) {
    let spec = TapSpec { duration: 10.0, mean_iti: 0.25, iti_sd: 0.008, noise_sd: 0.05, seed: 11, ..TapSpec::default() };
    let clock = FrameClock { jitter_sd: 0.002, drop_prob: 0.0, seed: 12, ..FrameClock::default() };
    let rec = render(&spec, &clock);
    let values = rec
        .timestamps
        .iter()
        .zip(&rec.values)
        .map(|(t, v)| if pause && (4.5..6.0).contains(t) { 0.02 * v } else { *v })
        .collect();
    (rec.timestamps, values)
}

fn feature(names: &[String], values: &[f64], name: &str) -> f64 {
    values[names.iter().position(|n| n == name).unwrap()]
}

#[test]
fn a_pause_becomes_a_hesitation_and_leaves_the_intervals_alone() {
    let p = PipelineParams::default();
    let (q, f) = (QcParams::default(), FeatureParams::default());
    let (t, clean) = trace(false);
    let (_, paused) = trace(true);
    let a = trace_features(&analyse_trace(&t, &clean, &p).unwrap(), &p, &q, &f).unwrap();
    let r = analyse_trace(&t, &paused, &p).unwrap();
    let b = trace_features(&r, &p, &q, &f).unwrap();
    assert_eq!(&b.names[..ITI_FEATURES.len()], &ITI_FEATURES.map(String::from)[..]);

    let (cv_a, cv_b) = (feature(&a.names, &a.values, "iti_cv"), feature(&b.names, &b.values, "iti_cv"));
    assert!(cv_b < 1.5 * cv_a, "CV with the pause {cv_b}, without {cv_a}");
    let (mean_a, mean_b) = (feature(&a.names, &a.values, "iti_mean"), feature(&b.names, &b.values, "iti_mean"));
    assert!((mean_b - mean_a).abs() < 0.005, "mean {mean_b} vs {mean_a}");

    assert_eq!(feature(&a.names, &a.values, "hesitations"), 0.0);
    assert!(feature(&b.names, &b.values, "hesitations") >= 1.0);
    let longest = feature(&b.names, &b.values, "longest_hesitation_s");
    assert!((0.75..=2.5).contains(&longest), "longest hesitation {longest} s for a 1.5 s pause");
    assert!(feature(&b.names, &b.values, "usable_fraction") < 1.0);
}
