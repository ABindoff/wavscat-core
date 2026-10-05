//! Acceptance tests for the inter-tap interval estimator, against synthetic
//! recordings with known ground truth.
//!
//! The headline requirement: tap-to-tap variability must be recovered more
//! accurately than frame quantisation allows. At 30 fps, snapping each tap to
//! a frame adds about 14 ms of SD to the intervals (1/30 s over the square
//! root of six).
//!
//! Thresholds sit above what the estimator achieves (see the `iti_sweep`
//! example) with a margin, so that they catch regressions without being
//! fragile.

use tapvid::phase::{analyse, CycleAnalysis, PhaseParams};
use tapvid::resample::resample;
use tapvid::synth::{render, FrameClock, SynthRecording, TapSpec};

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn rough_clock(seed: u64) -> FrameClock {
    FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed, ..FrameClock::default() }
}

/// Resample to 30 Hz and analyse, assuming the fundamental is known.
fn run(rec: &SynthRecording, f0: f64) -> CycleAnalysis {
    let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
    analyse(&u.values, u.fs, u.t0, f0, &PhaseParams::default()).unwrap()
}

/// True intervals over the span of the cycles the estimator kept.
fn truth(rec: &SynthRecording, a: &CycleAnalysis) -> Vec<f64> {
    let kept = &a.boundaries[1..a.boundaries.len() - 1];
    rec.itis_between(kept[0] - 0.05, kept[kept.len() - 1] + 0.05)
}

#[test]
fn regular_tapping_has_almost_no_variability() {
    // Noise-free, and on a perfect clock.
    let rec = render(&TapSpec::default(), &FrameClock::default());
    let a = run(&rec, 3.0);
    assert!(a.summary.sd < 0.0005, "SD {} s", a.summary.sd);
    assert!((a.summary.mean - 1.0 / 3.0).abs() < 0.0005, "mean {} s", a.summary.mean);
    assert!(a.summary.n_cycles >= 80);

    // With noise, jitter and dropped frames, still far below the floor.
    for seed in 1..=10 {
        let spec = TapSpec { noise_sd: 0.05, seed, ..TapSpec::default() };
        let a = run(&render(&spec, &rough_clock(seed + 100)), 3.0);
        assert!(a.summary.sd < 0.003, "seed {seed}: SD {} s", a.summary.sd);
    }
}

#[test]
fn variability_below_the_frame_floor_is_recovered() {
    // True SD of 10 ms, below the ~14 ms that frame quantisation adds.
    let mut errors = Vec::new();
    let mut inflation = Vec::new();
    for seed in 1..=20 {
        let spec = TapSpec { iti_sd: 0.010, noise_sd: 0.05, seed, ..TapSpec::default() };
        let rec = render(&spec, &rough_clock(seed + 100));
        let a = run(&rec, 3.0);
        let true_itis = truth(&rec, &a);
        errors.push(a.summary.sd - sd(&true_itis));

        // The baseline: each tap snapped to its nearest captured frame.
        let kept = &a.boundaries[1..a.boundaries.len() - 1];
        let snapped: Vec<f64> = rec
            .taps
            .iter()
            .filter(|&&t| t >= kept[0] - 0.05 && t <= kept[kept.len() - 1] + 0.05)
            .map(|&t| *rec.timestamps.iter().min_by(|x, y| (*x - t).abs().total_cmp(&(*y - t).abs())).unwrap())
            .collect();
        let frame_itis: Vec<f64> = snapped.windows(2).map(|w| w[1] - w[0]).collect();
        inflation.push(sd(&frame_itis) - sd(&true_itis));
    }
    let rms = |v: &[f64]| (v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64).sqrt();
    let (ours, frames) = (rms(&errors), rms(&inflation));
    assert!(ours < 0.003, "estimator SD error {:.2} ms", ours * 1e3);
    assert!(frames > 0.005, "frame quantisation error only {:.2} ms", frames * 1e3);
    assert!(ours < frames / 2.5, "{:.2} ms vs frame floor {:.2} ms", ours * 1e3, frames * 1e3);
}

#[test]
fn each_tap_is_placed_between_frames() {
    let spec = TapSpec { iti_sd: 0.010, noise_sd: 0.05, seed: 7, ..TapSpec::default() };
    let rec = render(&spec, &rough_clock(107));
    let a = run(&rec, 3.0);
    let kept = &a.boundaries[1..a.boundaries.len() - 1];
    let errs: Vec<f64> = kept
        .iter()
        .map(|&b| rec.taps.iter().map(|t| b - t).min_by(|x, y| x.abs().total_cmp(&y.abs())).unwrap())
        .collect();
    // A constant offset is a phase convention, not an error.
    let bias = errs.iter().sum::<f64>() / errs.len() as f64;
    let rms = (errs.iter().map(|e| (e - bias) * (e - bias)).sum::<f64>() / errs.len() as f64).sqrt();
    assert!(rms < 0.004, "per-tap timing error {:.2} ms against a 33 ms frame", rms * 1e3);
}

#[test]
fn a_waveform_with_a_strong_second_harmonic_is_handled() {
    let mut errors = Vec::new();
    for seed in 1..=10 {
        let spec = TapSpec { iti_sd: 0.010, noise_sd: 0.05, second_harmonic: 1.0, seed, ..TapSpec::default() };
        let rec = render(&spec, &rough_clock(seed + 100));
        let a = run(&rec, 3.0);
        errors.push(a.summary.sd - sd(&truth(&rec, &a)));
    }
    let worst = errors.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    assert!(worst < 0.004, "worst SD error {:.2} ms", worst * 1e3);
}

#[test]
fn a_misjudged_fundamental_still_works() {
    // Stage 5 estimates f0; allow it to be 10% out either way.
    for f0 in [2.7, 3.3] {
        let spec = TapSpec { iti_sd: 0.010, noise_sd: 0.05, seed: 3, ..TapSpec::default() };
        let rec = render(&spec, &rough_clock(103));
        let a = run(&rec, f0);
        let err = a.summary.sd - sd(&truth(&rec, &a));
        assert!(err.abs() < 0.004, "f0 = {f0}: SD error {:.2} ms", err * 1e3);
        assert!((a.summary.mean - 1.0 / 3.0).abs() < 0.002, "f0 = {f0}: mean {}", a.summary.mean);
    }
}

#[test]
fn amplitude_decrement_is_recovered() {
    // From 1.0 to 0.4 over 30 s: a slope of -0.02 per second.
    for seed in 1..=5 {
        let spec = TapSpec { amplitude_slope: -0.02, noise_sd: 0.05, iti_sd: 0.010, seed, ..TapSpec::default() };
        let a = run(&render(&spec, &rough_clock(seed + 100)), 3.0);
        let s = a.summary.amplitude_slope;
        assert!((s + 0.02).abs() < 0.001, "seed {seed}: slope {s} per s");
    }
    // Scale-free: the same tapping recorded in units 50 times larger, as a
    // nearer camera might, gives the same relative slope.
    let relative = |scale: f64| {
        let spec = TapSpec { amplitude: scale, amplitude_slope: -0.02 * scale, ..TapSpec::default() };
        run(&render(&spec, &FrameClock::default()), 3.0).summary.relative_amplitude_slope
    };
    let (small, large) = (relative(1.0), relative(50.0));
    assert!(small < 0.0, "relative slope {small}");
    assert!((small - large).abs() < 1e-9 * small.abs(), "{small} vs {large}");
}

#[test]
fn irregular_tapping_diffuses_in_phase_and_regular_tapping_does_not() {
    let diffusion = |iti_sd: f64| {
        let spec = TapSpec { iti_sd, noise_sd: 0.02, seed: 5, ..TapSpec::default() };
        run(&render(&spec, &rough_clock(105)), 3.0).summary.phase_diffusion
    };
    let (still, wandering) = (diffusion(0.0), diffusion(0.020));
    assert!(wandering > 10.0 * still.max(1e-6), "regular {still}, irregular {wandering}");
}

#[test]
fn gaps_must_be_resolved_before_analysis() {
    let mut rec = render(&TapSpec::default(), &FrameClock::default());
    // Lose a third of a second of frames.
    rec.timestamps.drain(300..310);
    rec.values.drain(300..310);
    let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
    assert_eq!(u.gaps.len(), 1);
    let err = analyse(&u.values, u.fs, u.t0, 3.0, &PhaseParams::default()).unwrap_err();
    assert!(err.0.contains("split or reject"), "{err}");
}
