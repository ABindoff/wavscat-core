//! Padding and timing for the video tapping configuration: 30 s at 30 Hz.
use std::time::Instant;
use wavscat_core::jtfs::{ParamsJtfs, ScatteringJtfs};
use wavscat_core::scattering1d::TSpec;

fn main() {
    let n = 900;
    let x: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1013) as f64 / 1013.0 - 0.5).collect();
    for (q, t) in [(8, TSpec::Samples(180.0)), (12, TSpec::Samples(180.0)), (8, TSpec::Global)] {
        let mut p = ParamsJtfs::new(n, 7, vec![q]);
        p.time.t = t;
        p.j_fr = 3;
        let t0 = Instant::now();
        let op = ScatteringJtfs::new(&p).unwrap();
        let t1 = Instant::now();
        let out = op.transform(&x).unwrap();
        let t2 = Instant::now();
        println!(
            "J = 7, Q = {q}, T = {t:?}: n_padded {}, pad {}+{}, n_padded_fr {}, {} paths x {} samples, build {:.1} ms, transform {:.1} ms",
            op.time.n_padded, op.time.pad_left, op.time.pad_right, op.n_padded_fr, out.len(), out[1].cols,
            (t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3
        );
    }
}
