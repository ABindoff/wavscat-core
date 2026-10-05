//! How well the phase estimator recovers tap-to-tap variability, against the
//! frame-quantisation floor, across band-pass widths and capture conditions.
//!
//!     cargo run --release -p tapvid --example iti_sweep
use tapvid::phase::{analyse, PhaseParams};
use tapvid::resample::resample;
use tapvid::synth::{render, FrameClock, TapSpec};

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() {
    let clocks = [
        ("perfect 30 fps", FrameClock::default()),
        ("jitter 4 ms, 5% dropped", FrameClock { jitter_sd: 0.004, drop_prob: 0.05, ..FrameClock::default() }),
    ];
    println!("{:<26} {:<4} {:>4} {:>5} | {:>8} {:>8} {:>8} | {:>9} {:>9}", "clock", "harm", "sd", "bw", "true", "est", "frames", "err ms", "event ms");
    for (name, clock) in &clocks {
        for (harmonic, iti_sd) in [(0.0, 0.0), (0.0, 0.010), (0.5, 0.0), (0.5, 0.010), (1.0, 0.010)] {
            for bw in [0.3, 0.4, 0.5, 0.7] {
                let mut ests = Vec::new();
                let mut truths = Vec::new();
                let mut frames = Vec::new();
                let mut event_rms = Vec::new();
                for seed in 1..=20u64 {
                    let spec = TapSpec { iti_sd, noise_sd: 0.05, second_harmonic: harmonic, seed, ..TapSpec::default() };
                    let clock = FrameClock { seed: seed + 100, ..clock.clone() };
                    let rec = render(&spec, &clock);
                    let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
                    let p = PhaseParams { bandwidth: bw, ..PhaseParams::default() };
                    let a = analyse(&u.values, u.fs, u.t0, 3.0, &p).unwrap();
                    let kept = &a.boundaries[1..a.boundaries.len() - 1];
                    let (t0, t1) = (kept[0] - 0.05, kept[kept.len() - 1] + 0.05);
                    let truth = rec.itis_between(t0, t1);
                    // Peak picking at frame resolution: each tap moved to the
                    // nearest captured frame.
                    let snapped: Vec<f64> = rec.taps.iter().filter(|&&t| t >= t0 && t <= t1)
                        .map(|&t| *rec.timestamps.iter().min_by(|a, b| (*a - t).abs().total_cmp(&(*b - t).abs())).unwrap())
                        .collect();
                    let fq: Vec<f64> = snapped.windows(2).map(|w| w[1] - w[0]).collect();
                    // Per-event timing error: each estimated boundary to the
                    // nearest true tap.
                    let errs: Vec<f64> = kept.iter().map(|&b| {
                        rec.taps.iter().map(|t| b - t).min_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap()
                    }).collect();
                    let bias = errs.iter().sum::<f64>() / errs.len() as f64;
                    let rms = (errs.iter().map(|e| (e - bias) * (e - bias)).sum::<f64>() / errs.len() as f64).sqrt();
                    ests.push(a.summary.sd);
                    truths.push(sd(&truth));
                    frames.push(sd(&fq));
                    event_rms.push(rms);
                }
                let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
                let err: Vec<f64> = ests.iter().zip(&truths).map(|(e, t)| (e - t) * 1e3).collect();
                let err_rms = (err.iter().map(|e| e * e).sum::<f64>() / err.len() as f64).sqrt();
                println!(
                    "{:<26} h{:<3.1} {:>4.0} {:>5.1} | {:>8.2} {:>8.2} {:>8.2} | {:>9.2} {:>9.2}",
                    name, harmonic, iti_sd * 1e3, bw, mean(&truths) * 1e3, mean(&ests) * 1e3, mean(&frames) * 1e3,
                    err_rms, mean(&event_rms) * 1e3
                );
            }
        }
    }
}
