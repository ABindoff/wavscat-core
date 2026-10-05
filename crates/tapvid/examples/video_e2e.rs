//! End to end on synthetic videos: frames through `push_frame` to inter-tap
//! intervals, against the ground truth, in several scenarios.
//!
//!     cargo run --release -p tapvid --example video_e2e
use std::time::Instant;

use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::{analyse_trial, PipelineParams};
use tapvid::synth::{frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() {
    let tap = TapSpec { iti_sd: 0.010, ..TapSpec::default() };
    let scenarios: Vec<(&str, VideoSpec)> = vec![
        ("clean", VideoSpec { tap: tap.clone(), ..VideoSpec::default() }),
        ("harmonic", VideoSpec { tap: TapSpec { fundamental: 0.3, second_harmonic: 1.0, ..tap.clone() }, ..VideoSpec::default() }),
        ("decrement", VideoSpec { tap: TapSpec { amplitude_slope: -0.015, ..tap.clone() }, ..VideoSpec::default() }),
        ("slow sway", VideoSpec {
            tap: tap.clone(),
            distractor: Some(Distractor { rate_hz: 0.4, displacement: 40.0, radius: 25.0, contrast: 80.0, centre: (0.25, 0.3) }),
            ..VideoSpec::default()
        }),
        ("in-band other", VideoSpec {
            tap: tap.clone(),
            distractor: Some(Distractor { rate_hz: 4.5, displacement: 15.0, radius: 10.0, contrast: 90.0, centre: (0.8, 0.3) }),
            ..VideoSpec::default()
        }),
    ];
    println!("{:<14} {:>5} {:>6} {:>8} {:>8} {:>8} {:>8} {:>7} {:>8}", "scenario", "comp", "f0", "case", "true sd", "est sd", "mean ms", "compet", "rel slope");
    for (name, spec) in scenarios {
        for seed in 1..=3u64 {
            let spec = VideoSpec { seed, tap: TapSpec { seed, ..spec.tap.clone() }, ..spec.clone() };
            let video = VideoSynth::new(spec.clone());
            let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 40, ..FrameClock::default() };
            let mut ing = Ingest::new(IngestParams::default()).unwrap();
            let (w, h) = (spec.width, spec.height);
            let (mut frame, mut work) = (vec![0u8; w * h], vec![0f32; w * h]);
            let times = frame_times(&clock, 30.0);
            for &t in &times {
                video.render(t, &mut frame, &mut work);
                ing.push_frame(&frame, w, w, h, (t * 1e6).round() as i64).unwrap();
            }
            let start = Instant::now();
            let r = analyse_trial(&ing, &PipelineParams::default()).unwrap();
            let ms = start.elapsed().as_secs_f64() * 1e3;
            // Truth over the kept cycles; trial time is relative to the first frame.
            let offset = times[0];
            let kept = &r.cycles.boundaries[1..r.cycles.boundaries.len() - 1];
            let inside: Vec<f64> = video.taps().iter().map(|t| t - offset).filter(|&t| t >= kept[0] - 0.05 && t <= kept[kept.len() - 1] + 0.05).collect();
            let truth: Vec<f64> = inside.windows(2).map(|w| w[1] - w[0]).collect();
            println!(
                "{name:<14} {:>5} {:>6.3} {:>8?} {:>8.2} {:>8.2} {:>8.1} {:>7.2} {:>9.4}   ({ms:.0} ms)",
                r.component, r.f0_hz, r.case, sd(&truth) * 1e3, r.cycles.summary.sd * 1e3,
                r.cycles.summary.mean * 1e3, r.competitor_ratio, r.cycles.summary.relative_amplitude_slope
            );
        }
    }
}
