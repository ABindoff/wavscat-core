//! Rough timing of joint scattering.
//!
//!     cargo run --release --example bench_jtfs -- <n> <J> <J_fr> <Q1>
use std::time::Instant;
use wavscat_core::jtfs::{ParamsJtfs, ScatteringJtfs};

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (n, j, j_fr, q) = (a[0], a[1] as u32, a[2] as u32, a[3] as u32);
    let x: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1013) as f64 / 1013.0 - 0.5).collect();
    let mut p = ParamsJtfs::new(n, j, vec![q]);
    p.j_fr = j_fr;
    let t0 = Instant::now();
    let op = ScatteringJtfs::new(&p).unwrap();
    let t1 = Instant::now();
    let reps = 5;
    for _ in 0..reps {
        std::hint::black_box(op.transform(&x).unwrap());
    }
    let t2 = Instant::now();
    println!(
        "n = {n}, J = {j}, J_fr = {j_fr}, Q = {q}: {} paths, build {:.0} ms, transform {:.0} ms",
        op.paths.len(),
        (t1 - t0).as_secs_f64() * 1e3,
        (t2 - t1).as_secs_f64() * 1e3 / reps as f64
    );
}
