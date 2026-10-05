//! Tests of `analyse_trace`: one trace, such as a landmark distance, through
//! the stages after the SVD, for comparison with the video.

use tapvid::features::{trace_features, FeatureParams, ITI_FEATURES};
use tapvid::pipeline::{analyse_trace, PipelineParams};
use tapvid::preprocess::detrend;
use tapvid::qc::QcParams;
use tapvid::synth::{render, FrameClock, TapSpec};

#[test]
fn a_ten_second_trace_gives_its_intervals_and_every_feature() {
    let spec = TapSpec { duration: 10.0, iti_sd: 0.01, noise_sd: 0.05, seed: 3, ..TapSpec::default() };
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 9, ..FrameClock::default() };
    let rec = render(&spec, &clock);
    // A slow drift and an offset, as a landmark distance has.
    let values: Vec<f64> = rec.timestamps.iter().zip(&rec.values).map(|(t, v)| 4.0 + 0.3 * t + v).collect();

    let p = PipelineParams::default();
    let r = analyse_trace(&rec.timestamps, &values, &p).unwrap();
    assert!((r.f0_hz - 1.0 / spec.mean_iti).abs() < 0.1, "f0 {}", r.f0_hz);
    let truth = rec.itis_between(r.cycles.boundaries[0], *r.cycles.boundaries.last().unwrap());
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    assert!((mean(&r.cycles.itis) - mean(&truth)).abs() < 0.003);

    let f = trace_features(&r, &p, &QcParams::default(), &FeatureParams::default()).unwrap();
    assert_eq!(&f.names[..ITI_FEATURES.len()], &ITI_FEATURES.map(String::from)[..]);
    assert!(f.names.len() > ITI_FEATURES.len() + 10);
    assert!(f.values.iter().all(|v| v.is_finite()));
}

#[test]
fn detrending_a_trace_removes_its_offset_and_slope() {
    let t: Vec<f64> = (0..300).map(|i| i as f64 / 30.0 + 0.001 * (i % 3) as f64).collect();
    let v: Vec<f64> = t.iter().map(|t| 5.0 - 0.2 * t).collect();
    let d = detrend(&t, &v, Some(0.3)).unwrap();
    assert!(d.iter().all(|x| x.abs() < 1e-9));
    let c = detrend(&t, &v, None).unwrap();
    assert!(c.iter().sum::<f64>().abs() < 1e-9);
}
