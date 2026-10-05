//! Bit-for-bit reproducibility of the video pipeline across targets.
//!
//! Regenerate after a deliberate change to the numerics:
//!
//! ```text
//! WAVSCAT_UPDATE_GOLDEN=1 cargo test -p tapvid --test golden
//! ```

use std::fmt::Write as _;

use tapvid::features::{trial_features, FeatureParams};
use tapvid::ingest::{Ingest, IngestParams};
use tapvid::phase::{analyse, analyse_at, PhaseParams};
use tapvid::pipeline::{analyse_trial, PipelineParams};
use tapvid::qc::{run_trial, QcParams};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};
use tapvid::preprocess::{preprocess, PreprocessParams};
use tapvid::select::{periodicity, BandParams};
use tapvid::spectrum::{welch, WelchParams};
use tapvid::svd::{randomized_svd, DenseF32, SvdParams};
use tapvid::resample::resample;
use tapvid::rng::SplitMix64;
use tapvid::synth::{frame_times, render, FrameClock, TapSpec};
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

    // Preprocessing on irregular timestamps, and the SVD of the result: a
    // textured scene with an oscillating patch under a gain ramp and step.
    let ts = frame_times(&FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 31, ..FrameClock::default() }, 20.0);
    let (rows, cols) = (ts.len(), 120);
    let mut g = SplitMix64::new(32);
    let scene: Vec<f64> = (0..cols).map(|_| 60.0 + 80.0 * g.uniform()).collect();
    let frames: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            let t = ts[r];
            let patch = if (30..50).contains(&c) { 15.0 * wavscat_core::math::sin(6.0 * std::f64::consts::PI * t) } else { 0.0 };
            let gain = (1.0 + 0.01 * t) * if t > 10.0 { 1.4 } else { 1.0 };
            (gain * (scene[c] + patch) + 0.5 * g.normal()) as f32
        })
        .collect();
    let mat = DenseF32 { data: &frames, rows, cols };
    let (pre, gain_qc) = preprocess(&mat, &ts, &PreprocessParams::default()).unwrap();
    writeln!(out, "preprocess\tgain_trace\t{:016x}", hash(gain_qc.trace.iter().copied())).unwrap();
    let gq = [gain_qc.min, gain_qc.max, gain_qc.abrupt_changes as f64, gain_qc.dark_frames as f64];
    writeln!(out, "preprocess\tgain_qc\t{:016x}", hash(gq)).unwrap();
    let svd = randomized_svd(&pre, &SvdParams::default()).unwrap();
    writeln!(out, "preprocess\tsvd_scores\t{:016x}", hash(svd.scores.iter().flatten().copied())).unwrap();

    // End to end: a small synthetic video with a sway distractor, rendered
    // at a rough clock, through ingest and the whole post-capture pipeline.
    let vspec = VideoSpec {
        width: 160,
        height: 120,
        radius: 6.0,
        displacement: 10.0,
        tap: TapSpec { duration: 12.0, iti_sd: 0.010, seed: 41, ..TapSpec::default() },
        exposure_step: Some((6.0, 1.3)),
        distractor: Some(Distractor { rate_hz: 0.4, displacement: 20.0, radius: 12.0, contrast: 80.0, centre: (0.25, 0.3) }),
        seed: 42,
        ..VideoSpec::default()
    };
    let video = VideoSynth::new(vspec.clone());
    let vclock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 43, ..FrameClock::default() };
    let mut ving = Ingest::new(IngestParams::default()).unwrap();
    let (mut vframe, mut vwork) = (vec![0u8; 160 * 120], vec![0f32; 160 * 120]);
    for &t in &frame_times(&vclock, 12.0) {
        video.render(t, &mut vframe, &mut vwork);
        ving.push_frame(&vframe, 160, 160, 120, (t * 1e6).round() as i64).unwrap();
    }
    let trial = analyse_trial(&ving, &PipelineParams::default()).unwrap();
    writeln!(out, "video\tsignal\t{:016x}", hash(trial.signal.iter().copied())).unwrap();
    let tf = [trial.component as f64, trial.f0_hz, trial.selection.best().1.timing_harmonic() as f64, trial.competitor_ratio];
    writeln!(out, "video\tselection\t{:016x}", hash(tf)).unwrap();
    writeln!(out, "video\tboundaries\t{:016x}", hash(trial.cycles.boundaries.iter().copied())).unwrap();
    let gated = run_trial(&ving, &PipelineParams::default(), &QcParams::default());
    let q = &gated.qc;
    let qf = [
        q.accepted() as u8 as f64,
        q.reasons.len() as f64,
        q.dropped_fraction,
        q.score.unwrap_or(-1.0),
        q.competitor_ratio.unwrap_or(-1.0),
        q.usable_cycles.map_or(-1.0, |u| u as f64),
        q.loading_spread.unwrap_or(-1.0),
    ];
    writeln!(out, "video\tqc\t{:016x}", hash(qf)).unwrap();
    let fp = FeatureParams::default();
    let feats = trial_features(&trial, &PipelineParams::default(), &QcParams::default(), &fp).unwrap();
    writeln!(out, "video\tfeatures\t{:016x}", hash(feats.values.iter().copied())).unwrap();
    writeln!(out, "video\tparams_hash\t{}", feats.params_hash).unwrap();

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
