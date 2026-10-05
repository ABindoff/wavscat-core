//! QC gating on synthetic videos of the failures typical of home recordings.
//! Each scenario must be accepted or rejected for the right reason; values
//! from the `qc_scenarios` example.

use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::PipelineParams;
use tapvid::qc::{run_trial, usable_cycles, QcParams, Rejection, TrialOutput};
use tapvid::synth::{frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};

fn base() -> VideoSpec {
    VideoSpec { tap: TapSpec { iti_sd: 0.010, ..TapSpec::default() }, ..VideoSpec::default() }
}

/// Render 30 s of `spec`, skipping frames in `gap` (by index), and gate it.
fn gate(spec: &VideoSpec, gap: Option<(usize, usize)>) -> TrialOutput {
    let video = VideoSynth::new(spec.clone());
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 41, ..FrameClock::default() };
    let mut ing = Ingest::new(IngestParams::default()).unwrap();
    let (w, h) = (spec.width, spec.height);
    let (mut frame, mut work) = (vec![0u8; w * h], vec![0f32; w * h]);
    for (k, &t) in frame_times(&clock, 30.0).iter().enumerate() {
        if matches!(gap, Some((a, b)) if (a..b).contains(&k)) {
            continue;
        }
        video.render(t, &mut frame, &mut work);
        ing.push_frame(&frame, w, w, h, (t * 1e6).round() as i64).unwrap();
    }
    run_trial(&ing, &PipelineParams::default(), &QcParams::default())
}

fn has(out: &TrialOutput, pred: impl Fn(&Rejection) -> bool, label: &str) {
    assert!(out.qc.reasons.iter().any(pred), "{label}: reasons {:?}", out.qc.reasons);
}

#[test]
fn a_good_trial_is_accepted() {
    let out = gate(&base(), None);
    assert!(out.qc.accepted(), "{:?}", out.qc.reasons);
    assert!(out.accepted().is_some());
    assert!(out.qc.loading_spread.unwrap() < 0.05, "a hand is localised");
}

#[test]
fn a_second_rhythmic_object_is_rejected() {
    let spec = VideoSpec {
        distractor: Some(Distractor { rate_hz: 4.5, displacement: 15.0, radius: 10.0, contrast: 90.0, centre: (0.8, 0.3) }),
        ..base()
    };
    has(&gate(&spec, None), |r| matches!(r, Rejection::CompetingOscillator { .. }), "second object");
}

#[test]
fn stopping_part_way_is_rejected() {
    let out = gate(&VideoSpec { stop_at: Some(8.0), ..base() }, None);
    has(&out, |r| matches!(r, Rejection::Interrupted { .. }), "stop");
    assert!(out.accepted().is_none());
}

#[test]
fn a_still_hand_is_rejected() {
    has(&gate(&VideoSpec { displacement: 0.0, ..base() }, None), |r| matches!(r, Rejection::WeakPeriodicity { .. }), "still");
}

#[test]
fn a_shaking_camera_is_not_mistaken_for_a_hand() {
    let spec = VideoSpec { displacement: 0.0, camera_shake: Some((3.0, 3.0)), ..base() };
    has(&gate(&spec, None), |r| matches!(r, Rejection::DiffuseMotion { .. }), "shake");
}

#[test]
fn unstable_lighting_is_rejected() {
    let out = gate(&VideoSpec { gain_drift: 0.6, gain_drift_hz: 0.2, ..base() }, None);
    has(&out, |r| matches!(r, Rejection::LightingRange { .. }), "lighting range");
    has(&out, |r| matches!(r, Rejection::AbruptExposure { .. }), "abrupt exposure");
}

#[test]
fn going_dark_is_rejected() {
    has(&gate(&VideoSpec { exposure_step: Some((20.0, 0.002)), ..base() }, None), |r| matches!(r, Rejection::DarkFrames { .. }), "dark");
}

#[test]
fn a_capture_gap_is_reported_with_its_place() {
    let out = gate(&base(), Some((400, 412)));
    assert!(out.result.is_none());
    match out.qc.reasons.as_slice() {
        [Rejection::CaptureGap { start, end }] => assert!(end - start > 0.15 && *start > 10.0 && *end < 20.0),
        other => panic!("reasons {other:?}"),
    }
}

#[test]
fn usable_cycles_judge_amplitude_against_full_tapping() {
    // 20 tapping cycles, then 60 noise-level ones after a stop: against the
    // median, the noise would pass; against the 90th percentile it does not.
    let itis = vec![0.33; 80];
    let amps: Vec<f64> = (0..80).map(|i| if i < 20 { 1.0 } else { 0.05 }).collect();
    assert_eq!(usable_cycles(&itis, &amps, 2.0, 0.25), 20);
    // A cycle twice the median interval is a hesitation, not usable.
    let mut itis = vec![0.33; 30];
    itis[10] = 0.8;
    assert_eq!(usable_cycles(&itis, &[1.0; 30], 2.0, 0.25), 29);
}
