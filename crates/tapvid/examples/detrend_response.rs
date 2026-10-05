//! Frequency response of the drift removal on irregular timestamps: the
//! fraction of a sinusoid's variance that survives, by frequency and cutoff.
//!
//!     cargo run --release -p tapvid --example detrend_response
use std::f64::consts::PI;

use tapvid::preprocess::{preprocess, PreprocessParams};
use tapvid::svd::{DenseF32, Operator};
use tapvid::synth::{frame_times, FrameClock};
use wavscat_core::math;

fn main() {
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, ..FrameClock::default() };
    let ts = frame_times(&clock, 30.0);
    let (rows, cols) = (ts.len(), 1000);
    let freqs = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 3.0];
    print!("{:>7} |", "cutoff");
    for f in freqs {
        print!(" {f:>6}");
    }
    println!("   (fraction of variance kept, by frequency in Hz)");
    for cutoff in [0.15, 0.3, 0.5] {
        print!("{cutoff:>7} |");
        for f in freqs {
            let s: Vec<f64> = ts.iter().map(|&t| math::sin(2.0 * PI * f * t + 0.3)).collect();
            let mut data = vec![100f32; rows * cols];
            for r in 0..rows {
                data[r * cols] = (100.0 + s[r]) as f32;
            }
            let m = DenseF32 { data: &data, rows, cols };
            let p = PreprocessParams { detrend_cutoff: Some(cutoff), ..PreprocessParams::default() };
            let (a, _) = preprocess(&m, &ts, &p).unwrap();
            let mut e0 = vec![0.0; cols];
            e0[0] = 1.0;
            let mut col = vec![0.0; rows];
            a.mul(&e0, 1, &mut col);
            // Undo the gain of ~1/100 and compare with the centred input.
            let mean_s = s.iter().sum::<f64>() / rows as f64;
            let var_in: f64 = s.iter().map(|v| (v - mean_s) * (v - mean_s)).sum();
            let var_out: f64 = col.iter().map(|v| (v * 100.0) * (v * 100.0)).sum();
            print!(" {:>6.3}", var_out / var_in);
        }
        println!();
    }
}
