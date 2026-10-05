//! When the second harmonic dominates, timing cycles from it rather than from
//! the weak fundamental.
//!
//!     cargo run --release -p tapvid --example harmonic_timing
use tapvid::phase::{analyse_at, PhaseParams};
use tapvid::resample::resample;
use tapvid::synth::{render, FrameClock, TapSpec};

fn sd(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() {
    println!("{:>11} {:>8} | {:>14} {:>14}", "fundamental", "true sd", "from f0 (ms)", "from 2 f0 (ms)");
    for fundamental in [1.0, 0.5, 0.3, 0.2] {
        let mut err = [Vec::new(), Vec::new()];
        let mut truth_sd = 0.0;
        for seed in 1..=20u64 {
            let spec = TapSpec { iti_sd: 0.010, fundamental, second_harmonic: 1.0, noise_sd: 0.05, seed, ..TapSpec::default() };
            let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 100, ..FrameClock::default() };
            let rec = render(&spec, &clock);
            let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
            for (slot, h) in [1u32, 2].iter().enumerate() {
                let a = analyse_at(&u.values, u.fs, u.t0, 3.0, *h, &PhaseParams::default()).unwrap();
                let kept = &a.boundaries[1..a.boundaries.len() - 1];
                let t = sd(&rec.itis_between(kept[0] - 0.05, kept[kept.len() - 1] + 0.05));
                err[slot].push(a.summary.sd - t);
                truth_sd += t / 40.0;
            }
        }
        let rms = |v: &[f64]| (v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64).sqrt() * 1e3;
        println!("{fundamental:>11.1} {:>8.2} | {:>14.2} {:>14.2}", truth_sd * 1e3, rms(&err[0]), rms(&err[1]));
    }
}
