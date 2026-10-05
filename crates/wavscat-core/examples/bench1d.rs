//! Rough timing of operator construction and one transform.
//!
//!     cargo run --release --example bench1d -- <n> <J> <Q1>
use std::time::Instant;
use wavscat_core::scattering1d::{Params1d, Scattering1d};

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (n, j, q) = (a[0], a[1] as u32, a[2] as u32);
    let x: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1013) as f64 / 1013.0 - 0.5).collect();
    let t0 = Instant::now();
    let op = Scattering1d::new(&Params1d::new(n, j, vec![q])).unwrap();
    let t1 = Instant::now();
    let reps = 10;
    for _ in 0..reps {
        std::hint::black_box(op.transform(&x).unwrap());
    }
    let t2 = Instant::now();
    println!(
        "n = {n}, J = {j}, Q = {q}: {} paths, build {:.1} ms, transform {:.1} ms",
        op.paths.len(),
        (t1 - t0).as_secs_f64() * 1e3,
        (t2 - t1).as_secs_f64() * 1e3 / reps as f64
    );
}
