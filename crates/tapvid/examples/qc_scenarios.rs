//! QC measurements and verdicts across the failures of home recordings.
//!
//!     cargo run --release -p tapvid --example qc_scenarios
use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::PipelineParams;
use tapvid::qc::{run_trial, QcParams};
use tapvid::synth::{frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};

fn main() {
    let tap = TapSpec { iti_sd: 0.010, ..TapSpec::default() };
    let base = VideoSpec { tap: tap.clone(), ..VideoSpec::default() };
    let scenarios: Vec<(&str, VideoSpec, Option<(usize, usize)>)> = vec![
        ("clean", base.clone(), None),
        ("second object 4.5 Hz", VideoSpec {
            distractor: Some(Distractor { rate_hz: 4.5, displacement: 15.0, radius: 10.0, contrast: 90.0, centre: (0.8, 0.3) }),
            ..base.clone()
        }, None),
        ("stops at 8 s", VideoSpec { stop_at: Some(8.0), ..base.clone() }, None),
        ("no tapping", VideoSpec { displacement: 0.0, ..base.clone() }, None),
        ("camera shake 3 Hz", VideoSpec { displacement: 0.0, camera_shake: Some((3.0, 3.0)), ..base.clone() }, None),
        ("lighting swings", VideoSpec { gain_drift: 0.6, gain_drift_hz: 0.2, ..base.clone() }, None),
        ("goes dark", VideoSpec { exposure_step: Some((20.0, 0.002)), ..base.clone() }, None),
        ("capture gap", base.clone(), Some((400, 412))),
    ];
    for (name, spec, gap) in scenarios {
        let video = VideoSynth::new(spec.clone());
        let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 41, ..FrameClock::default() };
        let mut ing = Ingest::new(IngestParams::default()).unwrap();
        let (w, h) = (spec.width, spec.height);
        let (mut frame, mut work) = (vec![0u8; w * h], vec![0f32; w * h]);
        for (k, &t) in frame_times(&clock, 30.0).iter().enumerate() {
            if let Some((a, b)) = gap {
                if (a..b).contains(&k) {
                    continue;
                }
            }
            video.render(t, &mut frame, &mut work);
            ing.push_frame(&frame, w, w, h, (t * 1e6).round() as i64).unwrap();
        }
        let out = run_trial(&ing, &PipelineParams::default(), &QcParams::default());
        let q = &out.qc;
        let f = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.3}"));
        println!(
            "{name:<22} {} score {} compet {} usable {} spread {} gain {}-{} abrupt {:?} dark {:?}",
            if q.accepted() { "ACCEPT" } else { "reject" },
            f(q.score), f(q.competitor_ratio), q.usable_cycles.map_or("-".into(), |u| u.to_string()),
            f(q.loading_spread), f(q.gain_min), f(q.gain_max), q.abrupt_changes, q.dark_frames
        );
        for r in &q.reasons {
            println!("{:<24}- {r}", "");
        }
    }
}
