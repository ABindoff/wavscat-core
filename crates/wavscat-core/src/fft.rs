//! A deterministic discrete Fourier transform for any length.
//!
//! Library FFTs pick SIMD kernels at run time, so the same call performs its
//! additions in a different order on an AVX laptop, an ARM phone and in wasm.
//! The results agree to rounding but not to the bit. This one is scalar with a
//! fixed order of operations, which is the price of bit-identity.
//!
//! Powers of two use an iterative radix-2 Cooley-Tukey transform. Any other
//! length uses Bluestein's algorithm, which rewrites the transform as a
//! convolution and evaluates that with a power-of-two FFT. Other lengths do
//! occur: the padding of the averaging filter is sized at the raw signal
//! length, and the frequential axis of joint scattering is padded to a multiple
//! of `2^J_fr` rather than to a power of two.
//!
//! Conventions match `numpy.fft`: the forward transform is unnormalised and
//! uses `exp(-2 pi i k n / N)`; the inverse divides by `N`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use crate::complex::C64;
use crate::math;

/// Forward transform in place, unnormalised.
pub fn fft(x: &mut [C64]) {
    if x.len() > 1 {
        plan(x.len()).forward(x);
    }
}

/// Inverse transform in place, divided by the length.
pub fn ifft(x: &mut [C64]) {
    let n = x.len();
    if n > 1 {
        plan(n).inverse_unnormalised(x);
    }
    let nf = n as f64;
    for v in x.iter_mut() {
        *v = v.unscale(nf);
    }
}

/// Forward transform of a real vector.
pub fn fft_real(x: &[f64]) -> Vec<C64> {
    let mut out: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft(&mut out);
    out
}

/// Inverse transform of a real vector.
pub fn ifft_real(x: &[f64]) -> Vec<C64> {
    let mut out: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    ifft(&mut out);
    out
}

/// `exp(-i pi num / den)`, with `num` already reduced into `[0, 2 den)`.
///
/// Reducing the angle in integers first keeps the argument to `sin` and `cos`
/// small, so large transforms lose no precision to an inexact `2 pi k`.
fn unit_root(num: u64, den: u64) -> C64 {
    let theta = PI * (num as f64) / (den as f64);
    C64::new(math::cos(theta), -math::sin(theta))
}

enum Kind {
    Radix2 {
        /// `exp(-2 pi i k / n)` for `k < n / 2`.
        twiddles: Vec<C64>,
        /// Bit-reversal permutation.
        rev: Vec<u32>,
    },
    Bluestein {
        /// Power-of-two length of the inner convolution, at least `2 n - 1`.
        m: usize,
        /// `exp(-i pi k^2 / n)`.
        chirp: Vec<C64>,
        /// Forward transform of the conjugate chirp, wrapped to length `m`.
        kernel_hat: Vec<C64>,
        inner: Arc<Plan>,
    },
}

struct Plan {
    n: usize,
    kind: Kind,
}

fn cache() -> &'static Mutex<HashMap<usize, Arc<Plan>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Plan>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Fetch or build the plan for length `n`.
///
/// Plans are deterministic functions of `n`, so caching them is invisible in
/// the output. The lock is released while building, because a Bluestein plan
/// builds its inner plan through this same function.
fn plan(n: usize) -> Arc<Plan> {
    if let Some(p) = cache().lock().unwrap_or_else(|e| e.into_inner()).get(&n) {
        return Arc::clone(p);
    }
    let built = Arc::new(Plan::new(n));
    let mut guard = cache().lock().unwrap_or_else(|e| e.into_inner());
    Arc::clone(guard.entry(n).or_insert(built))
}

impl Plan {
    fn new(n: usize) -> Plan {
        assert!(n >= 2);
        if n.is_power_of_two() {
            let n64 = n as u64;
            let twiddles = (0..n / 2).map(|k| unit_root(2 * k as u64, n64)).collect();
            let bits = n.trailing_zeros();
            let rev = (0..n as u32)
                .map(|i| i.reverse_bits() >> (32 - bits))
                .collect();
            Plan { n, kind: Kind::Radix2 { twiddles, rev } }
        } else {
            let m = (2 * n - 1).next_power_of_two();
            let n64 = n as u64;
            let chirp: Vec<C64> = (0..n64)
                .map(|k| unit_root((k * k) % (2 * n64), n64))
                .collect();
            let mut kernel = vec![C64::ZERO; m];
            kernel[0] = chirp[0].conj();
            for k in 1..n {
                kernel[k] = chirp[k].conj();
                kernel[m - k] = chirp[k].conj();
            }
            let inner = plan(m);
            inner.forward(&mut kernel);
            Plan { n, kind: Kind::Bluestein { m, chirp, kernel_hat: kernel, inner } }
        }
    }

    fn forward(&self, x: &mut [C64]) {
        debug_assert_eq!(x.len(), self.n);
        match &self.kind {
            Kind::Radix2 { twiddles, rev } => radix2(x, twiddles, rev),
            Kind::Bluestein { m, chirp, kernel_hat, inner } => {
                let n = self.n;
                let mut a = vec![C64::ZERO; *m];
                for k in 0..n {
                    a[k] = x[k] * chirp[k];
                }
                inner.forward(&mut a);
                for (v, h) in a.iter_mut().zip(kernel_hat) {
                    *v = *v * *h;
                }
                inner.inverse_unnormalised(&mut a);
                let mf = *m as f64;
                for k in 0..n {
                    x[k] = chirp[k] * a[k].unscale(mf);
                }
            }
        }
    }

    /// The inverse as the conjugate of the forward transform of the conjugate.
    /// Conjugation is exact, so this adds no rounding.
    fn inverse_unnormalised(&self, x: &mut [C64]) {
        for v in x.iter_mut() {
            *v = v.conj();
        }
        self.forward(x);
        for v in x.iter_mut() {
            *v = v.conj();
        }
    }
}

fn radix2(x: &mut [C64], twiddles: &[C64], rev: &[u32]) {
    let n = x.len();
    for i in 0..n {
        let j = rev[i] as usize;
        if i < j {
            x.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let a = x[start + k];
                let b = x[start + k + half] * twiddles[k * step];
                x[start + k] = a + b;
                x[start + k + half] = a - b;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft(x: &[C64]) -> Vec<C64> {
        let n = x.len() as u64;
        (0..n)
            .map(|k| {
                let mut acc = C64::ZERO;
                for (t, v) in x.iter().enumerate() {
                    acc = acc + *v * unit_root((2 * k * t as u64) % (2 * n), n);
                }
                acc
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<C64> {
        // A deterministic, irregular test signal.
        (0..n)
            .map(|i| {
                let t = i as f64;
                C64::new(math::sin(0.37 * t) + 0.1 * t / n as f64, math::cos(1.3 * t * t / n as f64))
            })
            .collect()
    }

    fn max_rel_err(a: &[C64], b: &[C64]) -> f64 {
        let scale = b.iter().map(|v| v.norm()).fold(0.0, f64::max);
        a.iter().zip(b).map(|(p, q)| (*p - *q).norm()).fold(0.0, f64::max) / scale
    }

    #[test]
    fn matches_naive_dft_for_assorted_lengths() {
        for n in [2, 3, 5, 7, 8, 12, 64, 100, 120, 127, 256, 1000] {
            let x = signal(n);
            let mut got = x.clone();
            fft(&mut got);
            let want = naive_dft(&x);
            let err = max_rel_err(&got, &want);
            assert!(err < 1e-13, "n = {n}: relative error {err:e}");
        }
    }

    #[test]
    fn inverse_round_trips() {
        for n in [1, 2, 6, 64, 120, 1500, 4096] {
            let x = signal(n);
            let mut y = x.clone();
            fft(&mut y);
            ifft(&mut y);
            let err = max_rel_err(&y, &x);
            assert!(err < 1e-13, "n = {n}: relative error {err:e}");
        }
    }
}
