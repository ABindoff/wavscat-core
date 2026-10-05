//! Time the FFT at the lengths the cascade uses.
//!
//!     cargo run --release --example bench_fft
use std::time::Instant;
use wavscat_core::complex::C64;
use wavscat_core::fft;

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / reps as f64
}

fn main() {
    println!("{:>8} {:>12} {:>12} {:>12}", "n", "complex us", "real fwd us", "real inv us");
    for n in [1024usize, 4096, 32000, 65536, 262144] {
        let x: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1013) as f64 / 1013.0 - 0.5).collect();
        let xc: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.25 * v)).collect();
        let reps = (4_000_000 / n).max(5);
        let c = time(reps, || {
            let mut y = xc.clone();
            fft::fft(&mut y);
            std::hint::black_box(y);
        });
        let rf = time(reps, || {
            std::hint::black_box(fft::fft_real(&x));
        });
        let spec = fft::fft_real(&x);
        let ri = time(reps, || {
            std::hint::black_box(fft::ifft_real_part(spec.clone()));
        });
        println!("{n:>8} {c:>12.1} {rf:>12.1} {ri:>12.1}");
    }
}
