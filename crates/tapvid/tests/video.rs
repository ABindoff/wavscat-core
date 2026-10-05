//! End-to-end tests on synthetic videos: frames rendered at a jittery,
//! dropping clock, pushed through `push_frame`, and analysed into inter-tap
//! intervals, against the known taps. Thresholds sit outside what the
//! `video_e2e` example measures.

use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::{analyse_trial, PipelineParams, TrialResult};
use tapvid::select::F0Case;
use tapvid::synth::{frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Render 30 s of `spec` at a rough 30 fps clock, analyse it, and return the
/// result with the true intervals over the cycles the analysis kept.
fn run(spec: &VideoSpec, seed: u64) -> (TrialResult, Vec<f64>) {
    let spec = VideoSpec { seed, tap: TapSpec { seed, ..spec.tap.clone() }, ..spec.clone() };
    let video = VideoSynth::new(spec.clone());
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 40, ..FrameClock::default() };
    let times = frame_times(&clock, 30.0);
    let mut ing = Ingest::new(IngestParams::default()).unwrap();
    let (w, h) = (spec.width, spec.height);
    let (mut frame, mut work) = (vec![0u8; w * h], vec![0f32; w * h]);
    for &t in &times {
        video.render(t, &mut frame, &mut work);
        ing.push_frame(&frame, w, w, h, (t * 1e6).round() as i64).unwrap();
    }
    let r = analyse_trial(&ing, &PipelineParams::default()).unwrap();
    let kept = &r.cycles.boundaries[1..r.cycles.boundaries.len() - 1];
    let inside: Vec<f64> = video
        .taps()
        .iter()
        .map(|t| t - times[0])
        .filter(|&t| t >= kept[0] - 0.05 && t <= kept[kept.len() - 1] + 0.05)
        .collect();
    (r, inside.windows(2).map(|w| w[1] - w[0]).collect())
}

fn tapping() -> TapSpec {
    TapSpec { iti_sd: 0.010, ..TapSpec::default() }
}

/// The tapping was found and its intervals recovered.
fn assert_recovered(r: &TrialResult, truth: &[f64], label: &str) {
    assert!((r.f0_hz - 3.0).abs() < 0.06, "{label}: f0 {}", r.f0_hz);
    let mean_truth = truth.iter().sum::<f64>() / truth.len() as f64;
    assert!((r.cycles.summary.mean - mean_truth).abs() < 0.002, "{label}: mean ITI {} vs {mean_truth}", r.cycles.summary.mean);
    let err = r.cycles.summary.sd - sd(truth);
    assert!(err.abs() < 0.004, "{label}: SD error {:.2} ms", err * 1e3);
}

#[test]
fn a_clean_trial_with_exposure_changes_is_recovered() {
    for seed in 1..=2 {
        let (r, truth) = run(&VideoSpec { tap: tapping(), ..VideoSpec::default() }, seed);
        assert_recovered(&r, &truth, "clean");
        assert!(r.competitor_ratio < 0.3, "competitor {}", r.competitor_ratio);
        // The 30% exposure step at 15 s, and the dropped frames, are seen.
        assert!(r.gain.abrupt_changes >= 1);
        let dropped = r.ingest.frames_dropped as f64 / 900.0;
        assert!((0.02..0.08).contains(&dropped), "dropped {dropped}");
    }
}

#[test]
fn a_dominant_second_harmonic_is_handled_end_to_end() {
    let tap = TapSpec { fundamental: 0.3, second_harmonic: 1.0, ..tapping() };
    for seed in 1..=2 {
        let (r, truth) = run(&VideoSpec { tap: tap.clone(), ..VideoSpec::default() }, seed);
        assert_eq!(r.case, F0Case::Harmonic);
        assert_recovered(&r, &truth, "harmonic");
    }
}

#[test]
fn an_amplitude_decrement_shows_in_the_slope() {
    let tap = TapSpec { amplitude_slope: -0.015, ..tapping() };
    for seed in 1..=2 {
        let (decrement, _) = run(&VideoSpec { tap: tap.clone(), ..VideoSpec::default() }, seed);
        let (steady, _) = run(&VideoSpec { tap: tapping(), ..VideoSpec::default() }, seed);
        let (d, s) = (decrement.cycles.summary.relative_amplitude_slope, steady.cycles.summary.relative_amplitude_slope);
        assert!(d < -0.007 && d < 2.0 * s, "decrement {d} vs steady {s}");
    }
}

#[test]
fn a_swaying_body_does_not_capture_the_selection() {
    // A large, slow, perfectly regular sway leaves sharp in-band harmonics
    // (1.2 Hz, 1.6 Hz) that would outscore jittery tapping on their own.
    let spec = VideoSpec {
        tap: tapping(),
        distractor: Some(Distractor { rate_hz: 0.4, displacement: 40.0, radius: 25.0, contrast: 80.0, centre: (0.25, 0.3) }),
        ..VideoSpec::default()
    };
    for seed in 1..=2 {
        let (r, truth) = run(&spec, seed);
        assert_recovered(&r, &truth, "sway");
        assert!(r.competitor_ratio < 0.3, "competitor {}", r.competitor_ratio);
    }
}

#[test]
fn a_second_rhythmic_object_is_flagged() {
    // Another object moving at 4.5 Hz, in the tapping band and as strong as
    // the hand: which one is the tapping is ambiguous, so QC must say so.
    let spec = VideoSpec {
        tap: tapping(),
        distractor: Some(Distractor { rate_hz: 4.5, displacement: 15.0, radius: 10.0, contrast: 90.0, centre: (0.8, 0.3) }),
        ..VideoSpec::default()
    };
    for seed in 1..=2 {
        let (r, _) = run(&spec, seed);
        assert!(r.competitor_ratio > 0.7, "competitor {}", r.competitor_ratio);
    }
}
