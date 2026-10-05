//! A participant is not a robot: the hand wanders and sways, and a tremor
//! may shake the other hand. None of that is a reason to reject a trial,
//! and none of it may be mistaken for the tapping. Synthetic 10 s videos of
//! 3 Hz tapping with 10 ms of true interval jitter, plus movement that is not
//! tapping, must each be accepted, give the tapping rate, and give intervals
//! close to the true ones.

use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::PipelineParams;
use tapvid::qc::{run_trial, QcParams};
use tapvid::synth::{frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};

fn base() -> VideoSpec {
    VideoSpec {
        tap: TapSpec { iti_sd: 0.010, duration: 12.0, ..TapSpec::default() },
        exposure_step: Some((5.0, 1.2)),
        centre: (0.4, 0.5),
        ..VideoSpec::default()
    }
}

/// A rest tremor in the other hand: 5.5 Hz, 4 px.
fn tremor() -> Distractor {
    Distractor { rate_hz: 5.5, displacement: 4.0, radius: 10.0, contrast: 90.0, centre: (0.8, 0.5) }
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (m, (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt())
}

fn check(label: &str, spec: VideoSpec) {
    for seed in 1..=3u64 {
        let spec = VideoSpec { seed, tap: TapSpec { seed, ..spec.tap.clone() }, ..spec.clone() };
        let video = VideoSynth::new(spec.clone());
        let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.03, seed: seed + 40, ..FrameClock::default() };
        let times = frame_times(&clock, 10.0);
        let mut ing = Ingest::new(IngestParams::default()).unwrap();
        let (w, h) = (spec.width, spec.height);
        let (mut frame, mut work) = (vec![0u8; w * h], vec![0f32; w * h]);
        for &t in &times {
            video.render(t, &mut frame, &mut work);
            ing.push_frame(&frame, w, w, h, (t * 1e6).round() as i64).unwrap();
        }
        let out = run_trial(&ing, &PipelineParams::default(), &QcParams::default());
        assert!(out.qc.accepted(), "{label}, seed {seed}: rejected: {:?}", out.qc.reasons);
        let r = out.result.unwrap();
        assert!((r.f0_hz - 3.0).abs() < 0.1, "{label}, seed {seed}: f0 {}", r.f0_hz);
        // True intervals over the span of the kept cycles.
        let b = &r.cycles.boundaries;
        let taps: Vec<f64> = video.taps().iter().map(|t| t - times[0]).filter(|&t| t >= b[1] - 0.1 && t <= b[b.len() - 2] + 0.1).collect();
        let truth: Vec<f64> = taps.windows(2).map(|w| w[1] - w[0]).collect();
        let (tm, tsd) = mean_sd(&truth);
        let (m, sd) = mean_sd(&r.cycles.itis);
        assert!((m - tm).abs() < 0.004, "{label}, seed {seed}: mean interval {m}, truth {tm}");
        assert!(sd < tsd + 0.006, "{label}, seed {seed}: interval SD {sd}, truth {tsd}");
        assert!(r.cycles.itis.len() + 1 >= truth.len(), "{label}, seed {seed}: {} cycles for {} true", r.cycles.itis.len(), truth.len());
    }
}

#[test]
fn a_still_hand() {
    check("still", base());
}

#[test]
fn a_wandering_hand() {
    // Poor postural control: a smooth irregular path, 4% and 8% of the
    // frame width (13 and 26 px).
    check("wander 4%", VideoSpec { wander: Some(0.04), ..base() });
    check("wander 8%", VideoSpec { wander: Some(0.08), ..base() });
}

#[test]
fn a_swaying_hand() {
    // A regular 0.8 Hz sway: its sidebands around the tapping sit on the
    // hand itself and must not count as another rhythm.
    check("sway", VideoSpec { sway: Some((0.8, 0.03, 0.02)), ..base() });
}

#[test]
fn a_tremor_in_the_other_hand() {
    // A pure 5.5 Hz rhythm is cleaner than tapping, but smaller; it must
    // neither be taken for the tapping nor reject the trial.
    check("tremor", VideoSpec { distractor: Some(tremor()), ..base() });
}

#[test]
fn wandering_with_a_tremor() {
    check("wander and tremor", VideoSpec { wander: Some(0.06), distractor: Some(tremor()), ..base() });
}
