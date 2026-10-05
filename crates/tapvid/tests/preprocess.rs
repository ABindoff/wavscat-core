//! Tests of stage 2, preprocessing.

use std::f64::consts::PI;

use tapvid::preprocess::{preprocess, PreprocessParams};
use tapvid::rng::SplitMix64;
use tapvid::svd::{randomized_svd, DenseF32, Operator, SvdParams};
use tapvid::synth::{frame_times, FrameClock};
use wavscat_core::math;

fn rough_times(seed: u64) -> Vec<f64> {
    frame_times(&FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed, ..FrameClock::default() }, 30.0)
}

/// The preprocessed matrix, materialised by applying the operator to the
/// identity, column by column.
fn dense(a: &impl Operator) -> Vec<f64> {
    let (rows, cols) = (a.rows(), a.cols());
    let mut eye = vec![0.0; cols * cols];
    for c in 0..cols {
        eye[c * cols + c] = 1.0;
    }
    let mut out = vec![0.0; rows * cols];
    a.mul(&eye, cols, &mut out);
    out
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

#[test]
fn a_static_scene_under_changing_exposure_vanishes() {
    let ts = rough_times(1);
    let (rows, cols) = (ts.len(), 40);
    let mut g = SplitMix64::new(2);
    let scene: Vec<f64> = (0..cols).map(|_| 80.0 + 40.0 * g.uniform()).collect();
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            let gain = 1.0 + 0.3 * math::sin(0.4 * ts[r]) + if ts[r] > 12.0 { 0.25 } else { 0.0 };
            (gain * scene[c]) as f32
        })
        .collect();
    let m = DenseF32 { data: &data, rows, cols };
    let (a, qc) = preprocess(&m, &ts, &PreprocessParams::default()).unwrap();
    let biggest = dense(&a).iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    // f32 storage of the frames bounds how exactly the gain cancels.
    assert!(biggest < 1e-6, "residual {biggest}");
    assert_eq!(qc.abrupt_changes, 1);
}

#[test]
fn centring_alone_zeroes_every_pixel_mean() {
    let ts = rough_times(3);
    let (rows, cols) = (ts.len(), 30);
    let mut g = SplitMix64::new(4);
    let data: Vec<f32> = (0..rows * cols).map(|_| (100.0 + 10.0 * g.normal()) as f32).collect();
    let m = DenseF32 { data: &data, rows, cols };
    let p = PreprocessParams { detrend_cutoff: None, ..PreprocessParams::default() };
    let (a, _) = preprocess(&m, &ts, &p).unwrap();
    let d = dense(&a);
    for c in 0..cols {
        let mean: f64 = (0..rows).map(|r| d[r * cols + c]).sum::<f64>() / rows as f64;
        assert!(mean.abs() < 1e-14, "pixel {c}: mean {mean}");
    }
}

#[test]
fn drift_is_removed_and_the_tapping_band_kept() {
    // Bounds from the detrend_response example, with a margin.
    let ts = rough_times(5);
    let (rows, cols) = (ts.len(), 1000);
    for (f, lo, hi) in [(0.05, 0.0, 0.001), (0.2, 0.0, 0.01), (1.0, 0.97, 1.0), (3.0, 0.99, 1.0)] {
        let s: Vec<f64> = ts.iter().map(|&t| math::sin(2.0 * PI * f * t + 0.3)).collect();
        let mut data = vec![100f32; rows * cols];
        for r in 0..rows {
            data[r * cols] = (100.0 + s[r]) as f32;
        }
        let m = DenseF32 { data: &data, rows, cols };
        let (a, _) = preprocess(&m, &ts, &PreprocessParams::default()).unwrap();
        let mut e0 = vec![0.0; cols];
        e0[0] = 1.0;
        let mut col = vec![0.0; rows];
        a.mul(&e0, 1, &mut col);
        let mean_s = s.iter().sum::<f64>() / rows as f64;
        let var_in: f64 = s.iter().map(|v| (v - mean_s) * (v - mean_s)).sum();
        let kept = col.iter().map(|v| 1e4 * v * v).sum::<f64>() / var_in;
        assert!(kept >= lo && kept <= hi, "{f} Hz: kept {kept}");
    }
}

#[test]
fn the_products_are_exact_adjoints() {
    let ts = rough_times(6);
    let (rows, cols) = (ts.len(), 25);
    let mut g = SplitMix64::new(7);
    let data: Vec<f32> = (0..rows * cols).map(|_| (100.0 + 20.0 * g.normal()) as f32).collect();
    let m = DenseF32 { data: &data, rows, cols };
    let (a, _) = preprocess(&m, &ts, &PreprocessParams::default()).unwrap();
    let k = 3;
    let x: Vec<f64> = (0..cols * k).map(|_| g.normal()).collect();
    let y: Vec<f64> = (0..rows * k).map(|_| g.normal()).collect();
    let (mut ax, mut aty) = (vec![0.0; rows * k], vec![0.0; cols * k]);
    a.mul(&x, k, &mut ax);
    a.mul_t(&y, k, &mut aty);
    let lhs: f64 = ax.iter().zip(&y).map(|(p, q)| p * q).sum();
    let rhs: f64 = x.iter().zip(&aty).map(|(p, q)| p * q).sum();
    assert!((lhs - rhs).abs() < 1e-12 * lhs.abs().max(1.0), "{lhs} vs {rhs}");
}

#[test]
fn the_tapping_oscillator_leads_despite_an_exposure_step() {
    // A textured scene, a patch oscillating at 3 Hz, and auto-exposure that
    // ramps slowly and then steps up by 40% half way through.
    let ts = rough_times(8);
    let (rows, cols) = (ts.len(), 200);
    let mut g = SplitMix64::new(9);
    let scene: Vec<f64> = (0..cols).map(|_| 60.0 + 80.0 * g.uniform()).collect();
    let osc: Vec<f64> = ts.iter().map(|&t| math::sin(2.0 * PI * 3.0 * t)).collect();
    let gain = |t: f64| (1.0 + 0.01 * t) * if t > 15.0 { 1.4 } else { 1.0 };
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            let patch = if (40..60).contains(&c) { 15.0 * osc[r] } else { 0.0 };
            (gain(ts[r]) * (scene[c] + patch) + 0.5 * g.normal()) as f32
        })
        .collect();
    let m = DenseF32 { data: &data, rows, cols };

    let (a, qc) = preprocess(&m, &ts, &PreprocessParams::default()).unwrap();
    let svd = randomized_svd(&a, &SvdParams::default()).unwrap();
    let c = corr(&svd.scores[0], &osc).abs();
    assert!(c > 0.99, "leading component vs the oscillation: {c}");
    assert_eq!(qc.abrupt_changes, 1);

    // Without gain removal, the step dominates instead.
    let mut centred = vec![0f32; rows * cols];
    for col in 0..cols {
        let mean = (0..rows).map(|r| data[r * cols + col] as f64).sum::<f64>() / rows as f64;
        for r in 0..rows {
            centred[r * cols + col] = (data[r * cols + col] as f64 - mean) as f32;
        }
    }
    let raw = randomized_svd(&DenseF32 { data: &centred, rows, cols }, &SvdParams::default()).unwrap();
    let step: Vec<f64> = ts.iter().map(|&t| if t > 15.0 { 1.0 } else { 0.0 }).collect();
    assert!(corr(&raw.scores[0], &step).abs() > 0.9, "the step should lead without normalisation");
}

#[test]
fn dark_frames_are_counted_and_kept_finite() {
    let ts = rough_times(10);
    let (rows, cols) = (ts.len(), 10);
    let data: Vec<f32> = (0..rows * cols).map(|i| if (100..110).contains(&(i / cols)) { 0.0 } else { 50.0 }).collect();
    let m = DenseF32 { data: &data, rows, cols };
    let (a, qc) = preprocess(&m, &ts, &PreprocessParams::default()).unwrap();
    assert_eq!(qc.dark_frames, 10);
    assert_eq!(qc.min, 0.0);
    assert!(dense(&a).iter().all(|v| v.is_finite()));
}
