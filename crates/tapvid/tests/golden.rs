//! Bit-for-bit reproducibility of the video pipeline across targets.
//!
//! Regenerate after a deliberate change to the numerics:
//!
//! ```text
//! WAVSCAT_UPDATE_GOLDEN=1 cargo test -p tapvid --test golden
//! ```

use std::fmt::Write as _;

use tapvid::phase::{analyse, PhaseParams};
use tapvid::resample::resample;
use tapvid::rng::SplitMix64;
use tapvid::synth::{render, FrameClock, TapSpec};
use wavscat_core::verify::{differences, hash};

const RECORD: &str = include_str!("../golden.tsv");

fn report() -> String {
    let mut out = String::new();
    writeln!(out, "# tapvid golden hashes, numerics {}", wavscat_core::NUMERICS_VERSION).unwrap();

    // The random stream behind the SVD test matrix.
    let mut g = SplitMix64::new(0x7a9);
    let draws: Vec<f64> = (0..10_000).map(|_| g.normal()).collect();
    writeln!(out, "rng\tnormal\t{:016x}", hash(draws)).unwrap();

    let spec = TapSpec {
        iti_sd: 0.012,
        amplitude_slope: -0.01,
        second_harmonic: 0.4,
        noise_sd: 0.05,
        seed: 21,
        ..TapSpec::default()
    };
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 22, ..FrameClock::default() };
    let rec = render(&spec, &clock);
    writeln!(out, "synth\ttimestamps\t{:016x}", hash(rec.timestamps.iter().copied())).unwrap();
    writeln!(out, "synth\tvalues\t{:016x}", hash(rec.values.iter().copied())).unwrap();

    let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
    writeln!(out, "resample\tvalues\t{:016x}", hash(u.values.iter().copied())).unwrap();

    let a = analyse(&u.values, u.fs, u.t0, 3.0, &PhaseParams::default()).unwrap();
    writeln!(out, "phase\tboundaries\t{:016x}", hash(a.boundaries.iter().copied())).unwrap();
    writeln!(out, "phase\tamplitudes\t{:016x}", hash(a.amplitudes.iter().copied())).unwrap();
    let s = &a.summary;
    let summary = [
        s.n_cycles as f64, s.mean, s.sd, s.cv, s.median, s.mad, s.robust_cv, s.lag1_autocorrelation,
        s.mean_amplitude, s.amplitude_slope, s.relative_amplitude_slope, s.phase_diffusion,
    ];
    writeln!(out, "phase\tsummary\t{:016x}", hash(summary)).unwrap();
    out
}

#[test]
fn outputs_are_bit_identical_to_the_golden_record() {
    let got = report();
    if std::env::var_os("WAVSCAT_UPDATE_GOLDEN").is_some() {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/golden.tsv"), &got).unwrap();
        return;
    }
    let diffs = differences(&got, RECORD);
    assert!(diffs.is_empty(), "outputs differ on this target:\n{}", diffs.join("\n"));
}
