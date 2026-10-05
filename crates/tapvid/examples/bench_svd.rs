//! Timing of the randomized SVD at the video size: 30 s at 30 fps of a
//! 64 x 48 grid, 900 frames by 3072 pixels.
//!
//!     cargo run --release -p tapvid --example bench_svd
use std::time::Instant;
use tapvid::rng::SplitMix64;
use tapvid::svd::{randomized_svd, DenseF32, SvdParams};

fn main() {
    let (m, n) = (900, 3072);
    let mut g = SplitMix64::new(1);
    let data: Vec<f32> = (0..m * n).map(|_| g.normal() as f32).collect();
    let a = DenseF32 { data: &data, rows: m, cols: n };
    for power_iters in [0, 1, 2] {
        let p = SvdParams { power_iters, ..SvdParams::default() };
        randomized_svd(&a, &p).unwrap();
        let reps = 5;
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(randomized_svd(&a, &p).unwrap());
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let products = 2 + 2 * power_iters;
        let gflop = 2.0 * (products * m * n * 16) as f64 / 1e9;
        println!("power_iters = {power_iters}: {ms:.1} ms ({gflop:.2} GFLOP in products, {:.2} GFLOP/s)", gflop / (ms / 1e3));
    }
}
