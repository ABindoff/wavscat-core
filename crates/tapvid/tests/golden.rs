//! Bit-for-bit reproducibility of the video pipeline across targets.
//!
//! Regenerate after a deliberate change to the numerics:
//!
//! ```text
//! WAVSCAT_UPDATE_GOLDEN=1 cargo test -p tapvid --test golden
//! ```

use std::fmt::Write as _;

use tapvid::ingest::{Ingest, IngestParams};
use tapvid::phase::{analyse, analyse_at, PhaseParams};
use tapvid::select::{periodicity, BandParams};
use tapvid::spectrum::{welch, WelchParams};
use tapvid::svd::{randomized_svd, DenseF32, SvdParams};
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

    // Ingest: luma and RGBA frames of several sizes, padded strides and an
    // irregular clock with a gap, through the ring and its QC.
    let mut ing = Ingest::new(IngestParams { capacity: 16, ..IngestParams::default() }).unwrap();
    let mut t = 0i64;
    for i in 0..24usize {
        let (w, h, pad) = [(640, 480, 0), (1280, 720, 16), (100, 75, 3)][i % 3];
        let stride = w + pad;
        t += if i == 10 { 120_000 } else { 33_000 + (i as i64 * 977) % 2_000 };
        if i % 4 == 3 {
            let rgba: Vec<u8> = (0..4 * stride * h).map(|k| ((k * 31 + i * 7) % 256) as u8).collect();
            ing.push_rgba(&rgba, 4 * stride, w, h, t).unwrap();
        } else {
            let luma: Vec<u8> = (0..stride * h).map(|k| ((k * 13 + i * 11) % 256) as u8).collect();
            ing.push_frame(&luma, stride, w, h, t).unwrap();
        }
    }
    let ring: Vec<f64> = (0..ing.len()).flat_map(|f| ing.frame(f).iter().map(|&v| v as f64).collect::<Vec<_>>()).collect();
    writeln!(out, "ingest\tframes\t{:016x}", hash(ring)).unwrap();
    let qc = ing.qc();
    let qc_fields = [
        qc.frames_accepted as f64, qc.frames_rejected as f64, qc.frames_dropped as f64,
        qc.effective_fps, qc.longest_gap,
    ];
    writeln!(out, "ingest\tqc\t{:016x}", hash(qc_fields)).unwrap();

    // The randomized SVD of a 300 x 500 f32 matrix: four structured
    // components plus noise, with the default seed and sign convention.
    let (m, n) = (300usize, 500usize);
    let mut g = SplitMix64::new(0x5bd);
    let factors: Vec<f64> = (0..(m + n) * 4).map(|_| g.normal()).collect();
    let data: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let mut acc = 0.1 * g.normal();
            for (k, s) in [8.0, 4.0, 2.0, 1.0].iter().enumerate() {
                acc += s * factors[i * 4 + k] * factors[(m + j) * 4 + k] / 20.0;
            }
            acc as f32
        })
        .collect();
    let p = SvdParams { return_loadings: true, ..SvdParams::default() };
    let svd = randomized_svd(&DenseF32 { data: &data, rows: m, cols: n }, &p).unwrap();
    writeln!(out, "svd\tsingular_values\t{:016x}", hash(svd.singular_values.iter().copied())).unwrap();
    writeln!(out, "svd\tscores\t{:016x}", hash(svd.scores.iter().flatten().copied())).unwrap();
    let loadings = svd.loadings.unwrap();
    writeln!(out, "svd\tloadings\t{:016x}", hash(loadings.iter().flatten().copied())).unwrap();

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

    // Stage 5 on a fundamental-dominant and a harmonic-dominant waveform, and
    // stage 6 timed from whichever stage 5 chooses.
    for (name, fundamental) in [("fundamental", 1.0), ("harmonic", 0.3)] {
        let spec = TapSpec { iti_sd: 0.012, fundamental, second_harmonic: 1.0, noise_sd: 0.05, seed: 23, ..TapSpec::default() };
        let rec = render(&spec, &clock);
        let u = resample(&rec.timestamps, &rec.values, 30.0, 0.15).unwrap();
        let psd = welch(&u.values, 30.0, &WelchParams::default()).unwrap();
        writeln!(out, "welch-{name}\tpower\t{:016x}", hash(psd.power.iter().copied())).unwrap();
        let p = periodicity(&u.values, 30.0, &WelchParams::default(), &BandParams::default()).unwrap();
        let fields = [
            p.band_fraction, p.sharpness, p.score, p.peak_hz, p.f0_hz,
            p.timing_harmonic() as f64, p.half_ratio, p.half_locking,
        ];
        writeln!(out, "select-{name}\tperiodicity\t{:016x}", hash(fields)).unwrap();
        let a = analyse_at(&u.values, u.fs, u.t0, p.f0_hz, p.timing_harmonic(), &PhaseParams::default()).unwrap();
        writeln!(out, "phase-{name}\tboundaries\t{:016x}", hash(a.boundaries.iter().copied())).unwrap();
    }
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
