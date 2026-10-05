//! How the fundamental decision behaves as the fundamental weakens relative
//! to the second harmonic, across tapping rates and noise levels.
//!
//!     cargo run --release -p tapvid --example f0_sweep
use tapvid::resample::resample;
use tapvid::select::{select, BandParams, F0Case};
use tapvid::spectrum::WelchParams;
use tapvid::synth::{render, FrameClock, TapSpec};

fn main() {
    println!("{:>5} {:>5} {:>5} | {:>9} {:>14} {:>10} {:>9}", "rate", "fund", "noise", "harmonic", "f0 within 5%", "half ratio", "locking");
    for rate in [1.5, 3.0, 4.0, 6.0] {
        for noise in [0.05, 0.3] {
            for fundamental in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0] {
                let (mut harmonic, mut close, mut ratio, mut contrast) = (0, 0, 0.0, 0.0);
                for seed in 1..=10u64 {
                    let spec = TapSpec {
                        mean_iti: 1.0 / rate,
                        iti_sd: 0.1 / rate,
                        fundamental,
                        second_harmonic: 1.0,
                        noise_sd: noise,
                        seed,
                        ..TapSpec::default()
                    };
                    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 50, ..FrameClock::default() };
                    let rec = render(&spec, &clock);
                    let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
                    let s = select(&[u.values], 30.0, &WelchParams::default(), &BandParams::default()).unwrap();
                    let p = &s.best().1;
                    harmonic += (p.case == F0Case::Harmonic) as u32;
                    close += ((p.f0_hz - rate).abs() < 0.05 * rate) as u32;
                    ratio += p.half_ratio / 10.0;
                    contrast += p.half_locking / 10.0;
                }
                println!("{rate:>5.1} {fundamental:>5.2} {noise:>5.2} | {harmonic:>7}/10 {close:>12}/10 {ratio:>10.4} {contrast:>9.2}");
            }
        }
    }
}
