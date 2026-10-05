//! Tests of stage 7: the feature vector and its version stamp.

use tapvid::features::{canonical_params, jtfs_features, params_hash, trial_features, FeatureParams, ITI_FEATURES};
use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::PipelineParams;
use tapvid::qc::{run_trial, QcParams};
use tapvid::resample::resample;
use tapvid::synth::{frame_times, render, FrameClock, TapSpec};
use tapvid::synth_video::{VideoSpec, VideoSynth};

/// A 30 s tapping trace resampled to 30 Hz.
fn trace(iti_sd: f64, seed: u64) -> Vec<f64> {
    let spec = TapSpec { iti_sd, noise_sd: 0.05, second_harmonic: 0.4, seed, ..TapSpec::default() };
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 50, ..FrameClock::default() };
    let rec = render(&spec, &clock);
    resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap().values
}

/// Mean of the second-order features, over a few recordings.
fn second_order(iti_sd: f64) -> f64 {
    let mut acc = 0.0;
    for seed in 1..=4 {
        let (names, values) = jtfs_features(&trace(iti_sd, seed), 30.0, &FeatureParams::default()).unwrap();
        let o2: Vec<f64> = names.iter().zip(&values).filter(|(n, _)| n.starts_with("jtfs_J2")).map(|(_, v)| *v).collect();
        acc += o2.iter().sum::<f64>() / o2.len() as f64 / 4.0;
    }
    acc
}

#[test]
fn period_jitter_raises_second_order_energy() {
    // The premise of the scattering features (jitter_features example).
    let (a, b, c) = (second_order(0.0), second_order(0.010), second_order(0.030));
    assert!(a + 0.2 < b && b + 0.2 < c, "second order at 0, 10, 30 ms jitter: {a}, {b}, {c}");
}

#[test]
fn features_do_not_depend_on_amplitude() {
    let x = trace(0.015, 7);
    let p = FeatureParams::default();
    let (_, base) = jtfs_features(&x, 30.0, &p).unwrap();
    for scale in [0.01, 3.0, 250.0] {
        let scaled: Vec<f64> = x.iter().map(|v| v * scale).collect();
        let (_, v) = jtfs_features(&scaled, 30.0, &p).unwrap();
        for (a, b) in base.iter().zip(&v) {
            assert!((a - b).abs() < 1e-9, "scale {scale}: {a} vs {b}");
        }
    }
}

#[test]
fn an_accepted_trial_gives_a_complete_named_vector() {
    let spec = VideoSpec { tap: TapSpec { iti_sd: 0.010, ..TapSpec::default() }, ..VideoSpec::default() };
    let video = VideoSynth::new(spec.clone());
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 41, ..FrameClock::default() };
    let mut ing = Ingest::new(IngestParams::default()).unwrap();
    let (w, h) = (spec.width, spec.height);
    let (mut frame, mut work) = (vec![0u8; w * h], vec![0f32; w * h]);
    for &t in &frame_times(&clock, 30.0) {
        video.render(t, &mut frame, &mut work);
        ing.push_frame(&frame, w, w, h, (t * 1e6).round() as i64).unwrap();
    }
    let (pp, qp, fp) = (PipelineParams::default(), QcParams::default(), FeatureParams::default());
    let out = run_trial(&ing, &pp, &qp);
    let r = out.accepted().expect("a clean trial is accepted");
    let f = trial_features(r, &pp, &qp, &fp).unwrap();
    assert_eq!(f.names.len(), f.values.len());
    assert_eq!(&f.names[..ITI_FEATURES.len()], &ITI_FEATURES.map(String::from)[..]);
    assert!(f.names[ITI_FEATURES.len()..].iter().all(|n| n.starts_with("jtfs_")));
    let mut unique = f.names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), f.names.len(), "feature names must be unique");
    assert!(f.values.iter().all(|v| v.is_finite()));
    assert_eq!(f.params_hash, params_hash(&pp, &qp, &fp));
    assert!((f.values[0] - 3.0).abs() < 0.06, "f0 {}", f.values[0]);
}

#[test]
fn the_params_hash_covers_every_setting() {
    let (pp, qp, fp) = (PipelineParams::default(), QcParams::default(), FeatureParams::default());
    let base = params_hash(&pp, &qp, &fp);
    assert_eq!(base, params_hash(&pp, &qp, &fp), "the hash is a pure function");
    assert_eq!(base.len(), 16);

    let mut variants = Vec::new();
    let mut p = pp.clone();
    p.svd.seed += 1;
    variants.push(params_hash(&p, &qp, &fp));
    let mut p = pp.clone();
    p.preprocess.detrend_cutoff = None;
    variants.push(params_hash(&p, &qp, &fp));
    let mut p = pp.clone();
    p.band.hi = 6.5;
    variants.push(params_hash(&p, &qp, &fp));
    let mut q = qp.clone();
    q.min_usable_cycles += 1;
    variants.push(params_hash(&pp, &q, &fp));
    let mut f = fp.clone();
    f.t_sec = 4.0;
    variants.push(params_hash(&pp, &qp, &f));
    let mut f = fp.clone();
    f.renorm = false;
    variants.push(params_hash(&pp, &qp, &f));
    for (i, v) in variants.iter().enumerate() {
        assert_ne!(v, &base, "variant {i} did not change the hash");
    }
    // Floats are written in shortest round-trip form.
    assert!(canonical_params(&pp, &qp, &fp).contains("band.lo=1.0\n"));
}
