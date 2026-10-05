//! Does period jitter show up as second-order scattering energy, as the
//! design brief expects? Features of tapping at increasing ITI jitter.
//!
//!     cargo run --release -p tapvid --example jitter_features
use tapvid::features::{jtfs_features, FeatureParams};
use tapvid::resample::resample;
use tapvid::synth::{render, FrameClock, TapSpec};

fn main() {
    let p = FeatureParams::default();
    println!("{:>7} | {:>12} {:>12} {:>10}", "iti sd", "mean order 1", "mean order 2", "n paths");
    let mut base: Option<Vec<f64>> = None;
    for iti_sd in [0.0, 0.010, 0.020, 0.030, 0.045] {
        let (mut o1, mut o2) = (0.0, 0.0);
        let mut moved = 0usize;
        let reps = 6;
        let mut first = Vec::new();
        for seed in 1..=reps {
            let spec = TapSpec { iti_sd, noise_sd: 0.05, second_harmonic: 0.4, seed, ..TapSpec::default() };
            let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 50, ..FrameClock::default() };
            let rec = render(&spec, &clock);
            let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
            let (names, values) = jtfs_features(&u.values, 30.0, &p).unwrap();
            let (mut s1, mut n1, mut s2, mut n2) = (0.0, 0, 0.0, 0);
            for (n, v) in names.iter().zip(&values) {
                if n.starts_with("jtfs_J1") {
                    s1 += v;
                    n1 += 1;
                } else if n.starts_with("jtfs_J2") {
                    s2 += v;
                    n2 += 1;
                }
            }
            o1 += s1 / n1 as f64 / reps as f64;
            o2 += s2 / n2 as f64 / reps as f64;
            if seed == 1 {
                first = values.clone();
            }
        }
        if let Some(b) = &base {
            moved = b.iter().zip(&first).filter(|(a, c)| (*a - *c).abs() > 0.1).count();
        } else {
            base = Some(first);
        }
        println!("{:>5.0} ms | {o1:>12.3} {o2:>12.3} {moved:>10}", iti_sd * 1e3);
    }
}
