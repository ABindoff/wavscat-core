//! Re-estimating frame times from the camera's regular clock: coarse
//! timestamps, dropped frames, a frame-rate change part way through, and a
//! clock that is not regular at all.

use tapvid::clock::{regularise, ClockParams};
use tapvid::rng::SplitMix64;

/// Windows-style timestamps: true times floored onto a 15.625 ms grid, with
/// an arbitrary phase.
fn quantise(t: &[f64], phase: f64) -> Vec<f64> {
    let q = 0.015625;
    t.iter().map(|x| ((x + phase) / q).floor() * q).collect()
}

/// True capture times of a camera at `fps` from `start` to `end`, dropping
/// each frame with probability `drop`. Returns (times, frames dropped).
fn camera(fps: f64, start: f64, end: f64, drop: f64, rng: &mut SplitMix64) -> (Vec<f64>, usize) {
    let mut out = Vec::new();
    let mut dropped = 0;
    let mut k = 0;
    loop {
        let t = start + k as f64 / fps;
        if t >= end {
            break;
        }
        if rng.uniform() < drop {
            dropped += 1;
        } else {
            out.push(t);
        }
        k += 1;
    }
    (out, dropped)
}

fn errors(a: &[f64], b: &[f64]) -> (f64, f64) {
    let e: Vec<f64> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
    (rms, e.iter().copied().fold(0.0, f64::max))
}

#[test]
fn coarse_timestamps_are_brought_back_onto_the_camera_clock() {
    for seed in 1..=5 {
        let mut rng = SplitMix64::new(seed);
        let (truth, dropped) = camera(30.0, 0.0, 10.0, 0.02, &mut rng);
        // Quantised timestamps are offset by the grid phase; compare
        // intervals' worth of error by removing each one's mean offset.
        let raw = quantise(&truth, 0.003 * seed as f64);
        let (fixed, qc) = regularise(&raw, &ClockParams::default()).unwrap();
        let shift = |v: &[f64]| {
            let m = v.iter().zip(&truth).map(|(x, y)| x - y).sum::<f64>() / v.len() as f64;
            v.iter().map(|x| x - m).collect::<Vec<f64>>()
        };
        let (raw_rms, _) = errors(&shift(&raw), &truth);
        let (rms, worst) = errors(&shift(&fixed), &truth);
        assert!(rms < 0.001 && worst < 0.002, "seed {seed}: corrected rms {rms}, worst {worst}; raw rms {raw_rms}");
        assert!(raw_rms > 0.003, "seed {seed}: raw rms {raw_rms}");
        assert_eq!(qc.segments.len(), 1, "seed {seed}: {:?}", qc.segments);
        assert_eq!(qc.rate_changes, 0);
        assert_eq!(qc.missing, dropped, "seed {seed}");
        assert!((qc.segments[0].period - 1.0 / 30.0).abs() < 1e-4);
    }
}

#[test]
fn a_frame_rate_change_part_way_through_starts_a_new_segment() {
    // Auto-exposure: 30 fps, then 20 fps from 5 s, as the light dims.
    for seed in 1..=5 {
        let mut rng = SplitMix64::new(10 + seed);
        let (mut truth, _) = camera(30.0, 0.0, 5.0, 0.01, &mut rng);
        let first_slow = truth.len();
        let (slow, _) = camera(20.0, 5.0, 10.0, 0.01, &mut rng);
        truth.extend(slow);
        let raw = quantise(&truth, 0.007);
        let (fixed, qc) = regularise(&raw, &ClockParams::default()).unwrap();
        assert_eq!(qc.rate_changes, 1, "seed {seed}: {:?}", qc.segments);
        let boundary = qc.segments[1].start;
        assert!(boundary.abs_diff(first_slow) <= 3, "seed {seed}: split at {boundary}, change at {first_slow}");
        assert!((qc.segments[0].period - 1.0 / 30.0).abs() < 3e-4, "{:?}", qc.segments[0]);
        assert!((qc.segments[1].period - 1.0 / 20.0).abs() < 3e-4, "{:?}", qc.segments[1]);
        // Away from the change, every frame is back on its clock.
        let m = fixed.iter().zip(&truth).map(|(x, y)| x - y).sum::<f64>() / fixed.len() as f64;
        for i in (0..truth.len()).filter(|&i| i.abs_diff(first_slow) > 4) {
            let e = (fixed[i] - m - truth[i]).abs();
            assert!(e < 0.004, "seed {seed}: frame {i} off by {e}");
        }
        assert!(fixed.windows(2).all(|w| w[1] > w[0]));
    }
}

#[test]
fn an_irregular_clock_keeps_its_own_timestamps() {
    let mut rng = SplitMix64::new(99);
    let mut t = 0.0;
    let raw: Vec<f64> = (0..300)
        .map(|_| {
            t += 0.010 + 0.060 * rng.uniform();
            t
        })
        .collect();
    let (fixed, qc) = regularise(&raw, &ClockParams::default()).unwrap();
    assert!(qc.segments.iter().all(|s| !s.regular), "{:?}", qc.segments);
    assert_eq!(fixed, raw);
}

#[test]
fn bad_input_is_refused_and_tiny_input_passes_through() {
    assert!(regularise(&[0.0, 0.0, 1.0], &ClockParams::default()).is_err());
    let (t, qc) = regularise(&[0.0, 0.033], &ClockParams::default()).unwrap();
    assert_eq!(t, vec![0.0, 0.033]);
    assert!(qc.segments.is_empty());
}

#[test]
fn coarse_timestamps_no_longer_blur_each_interval() {
    use tapvid::pipeline::{analyse_trace, PipelineParams, TraceResult};
    use tapvid::synth::{render, FrameClock, TapSpec};
    // 5 Hz tapping with 5 ms of true interval jitter, sampled at a regular
    // 30 fps with a few drops, then stamped on a 15.625 ms grid. Per-cycle
    // interval error against the true taps: re-estimated frame times should
    // do as well as the true capture times, and raw stamps clearly worse.
    let (mut fixed, mut raw, mut exact) = (0.0, 0.0, 0.0);
    for seed in 1..=8 {
        let spec = TapSpec { duration: 10.0, mean_iti: 0.2, iti_sd: 0.005, noise_sd: 0.03, seed, ..TapSpec::default() };
        let clock = FrameClock { jitter_sd: 0.0, drop_prob: 0.02, seed: seed + 20, ..FrameClock::default() };
        let rec = render(&spec, &clock);
        let stamped = quantise(&rec.timestamps, 0.004 * seed as f64);
        let off = PipelineParams { clock: None, ..PipelineParams::default() };
        let error = |r: &TraceResult| {
            let b = &r.cycles.boundaries;
            let truth = rec.itis_between(b[1] - 0.05, b[b.len() - 2] + 0.05);
            assert_eq!(truth.len(), r.cycles.itis.len(), "seed {seed}");
            r.cycles.itis.iter().zip(&truth).map(|(p, q)| (p - q).abs()).sum::<f64>() / truth.len() as f64
        };
        fixed += error(&analyse_trace(&stamped, &rec.values, &PipelineParams::default()).unwrap());
        raw += error(&analyse_trace(&stamped, &rec.values, &off).unwrap());
        exact += error(&analyse_trace(&rec.timestamps, &rec.values, &off).unwrap());
    }
    assert!(fixed < 1.1 * exact, "mean |interval error|: {fixed} re-estimated, {exact} true times");
    assert!(fixed < 0.75 * raw, "mean |interval error|: {fixed} re-estimated, {raw} raw stamps");
}
