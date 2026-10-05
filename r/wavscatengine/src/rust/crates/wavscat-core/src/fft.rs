//! A deterministic discrete Fourier transform for any length.
//!
//! Library FFTs pick SIMD kernels at run time, so the same call performs its
//! additions in a different order on an AVX laptop, an ARM phone and in wasm.
//! The results agree to rounding but not to the bit. This one is scalar with a
//! fixed order of operations, which is the price of bit-identity.
//!
//! Powers of two use the textbook radix-2 decimation-in-time transform,
//! executed in a cache-blocked order that provably does not change a bit (see
//! [`radix2`]). Other lengths do occur: the averaging filter's padding is sized
//! at the raw signal length, and the frequential axis of joint scattering is
//! padded to a multiple of `2^J_fr`, such as 152 = 19 x 8. When every odd prime
//! factor is at most 64, a mixed-radix step peels off the largest one with a
//! direct DFT (see [`mixed`]). Otherwise Bluestein's algorithm rewrites the
//! transform as a convolution evaluated with power-of-two FFTs.
//!
//! Many short transforms of the same length, such as every time column of a
//! joint scattering path along frequency, run as one batch: see
//! [`fft_columns`]. A batch is bit-identical to separate transforms.
//!
//! Real signals of even length are transformed through a complex FFT of half
//! the length, packing even samples into the real part and odd samples into
//! the imaginary part. The scattering cascade transforms real envelopes and
//! takes real parts of inverses constantly, so this halves most of its work.
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
        plan(x.len()).forward(x, 1);
    }
}

/// Inverse transform in place, divided by the length.
pub fn ifft(x: &mut [C64]) {
    let n = x.len();
    if n > 1 {
        plan(n).inverse_unnormalised(x, 1);
    }
    let nf = n as f64;
    for v in x.iter_mut() {
        *v = v.unscale(nf);
    }
}

/// Columns transformed together. A block of this width stays in cache for
/// any transform length the cascade uses (152 rows of 64 complex values is
/// 152 KiB), where a full-width batch of tens of thousands of columns would
/// stream every row from memory on every pass.
const COLUMN_BLOCK: usize = 64;

/// Apply `f` to each block of up to `COLUMN_BLOCK` columns of a row-major
/// matrix, copied out to a contiguous buffer and back. Columns never mix, so
/// the blocking is invisible in the result.
fn by_column_blocks(x: &mut [C64], cols: usize, f: impl Fn(&mut [C64], usize)) {
    let n = x.len() / cols;
    assert_eq!(n * cols, x.len(), "the data are not a whole number of rows");
    if cols <= COLUMN_BLOCK {
        f(x, cols);
        return;
    }
    let mut buf = vec![C64::ZERO; n * COLUMN_BLOCK];
    let mut c0 = 0;
    while c0 < cols {
        let w = COLUMN_BLOCK.min(cols - c0);
        let block = &mut buf[..n * w];
        for r in 0..n {
            block[r * w..(r + 1) * w].copy_from_slice(&x[r * cols + c0..r * cols + c0 + w]);
        }
        f(block, w);
        for r in 0..n {
            x[r * cols + c0..r * cols + c0 + w].copy_from_slice(&block[r * w..(r + 1) * w]);
        }
        c0 += w;
    }
}

/// Forward transform of each column of a row-major matrix with `cols`
/// columns, in place. Bit-identical to transforming each column alone.
pub fn fft_columns(x: &mut [C64], cols: usize) {
    let n = x.len() / cols;
    if n > 1 {
        let p = plan(n);
        by_column_blocks(x, cols, |blk, w| p.forward(blk, w));
    }
}

/// Inverse transform of each column, divided by the column length.
/// Bit-identical to transforming each column alone.
pub fn ifft_columns(x: &mut [C64], cols: usize) {
    let n = x.len() / cols;
    let p = if n > 1 { Some(plan(n)) } else { None };
    let nf = n as f64;
    by_column_blocks(x, cols, |blk, w| {
        if let Some(p) = &p {
            p.inverse_unnormalised(blk, w);
        }
        for v in blk.iter_mut() {
            *v = v.unscale(nf);
        }
    });
}

/// Forward transform of a real vector: the full, Hermitian-symmetric spectrum.
pub fn fft_real(x: &[f64]) -> Vec<C64> {
    let n = x.len();
    if n >= 4 && n % 2 == 0 {
        return real_plan(n).forward(x);
    }
    let mut out: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft(&mut out);
    out
}

/// Inverse transform of a real vector, which is `conj(fft(x)) / n`.
pub fn ifft_real(x: &[f64]) -> Vec<C64> {
    let nf = x.len() as f64;
    let mut out = fft_real(x);
    for v in out.iter_mut() {
        *v = v.conj().unscale(nf);
    }
    out
}

/// Real part of the inverse transform.
///
/// Computed as the inverse of the Hermitian part of `x`, which has the same
/// real part and lets a half-length complex FFT do the work.
pub fn ifft_real_part(mut x: Vec<C64>) -> Vec<f64> {
    let n = x.len();
    if n >= 4 && n % 2 == 0 {
        return real_plan(n).inverse_real_part(&x);
    }
    ifft(&mut x);
    x.iter().map(|v| v.re).collect()
}

/// `exp(-i pi num / den)`, with `num` already reduced into `[0, 2 den)`.
///
/// Reducing the angle in integers first keeps the argument to `sin` and `cos`
/// small, so large transforms lose no precision to an inexact `2 pi k`.
fn unit_root(num: u64, den: u64) -> C64 {
    let theta = PI * (num as f64) / (den as f64);
    C64::new(math::cos(theta), -math::sin(theta))
}

/// `exp(-2 pi i k / n)` for `k < len`.
fn roots(n: usize, len: usize) -> Vec<C64> {
    (0..len as u64).map(|k| unit_root(2 * k, n as u64)).collect()
}

/// Largest prime handled by a direct DFT step. Lengths with a larger prime
/// factor go through Bluestein instead.
const MAX_DIRECT_PRIME: usize = 64;

enum Kind {
    Pow2 {
        /// Bit-reversal permutation.
        rev: Vec<u32>,
        /// Twiddles for every stage, concatenated: the stage of length `len`
        /// uses `exp(-2 pi i k / len)` for `k < len / 2`, stored from offset
        /// `len / 2 - 1`.
        stage_twiddles: Vec<C64>,
    },
    /// One decimation-in-time step `n = p m`, with `p` the largest odd prime
    /// factor of `n`: `p` transforms of length `m`, combined by direct
    /// `p`-point DFTs.
    Mixed {
        p: usize,
        m: usize,
        /// The plan for length `m`, absent when `m = 1`.
        sub: Option<Arc<Plan>>,
        /// `exp(-2 pi i e / n)` for `e < n`.
        twiddles: Vec<C64>,
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

/// Plans are deterministic functions of the length, so caching them is
/// invisible in the output. The lock is released while building, because a
/// plan may build another through the same cache.
fn cached<T>(
    cache: &'static OnceLock<Mutex<HashMap<usize, Arc<T>>>>,
    n: usize,
    build: impl FnOnce() -> T,
) -> Arc<T> {
    let map = cache.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = map.lock().unwrap_or_else(|e| e.into_inner()).get(&n) {
        return Arc::clone(p);
    }
    let built = Arc::new(build());
    let mut guard = map.lock().unwrap_or_else(|e| e.into_inner());
    Arc::clone(guard.entry(n).or_insert(built))
}

fn plan(n: usize) -> Arc<Plan> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<Plan>>>> = OnceLock::new();
    cached(&CACHE, n, || Plan::new(n))
}

fn real_plan(n: usize) -> Arc<RealPlan> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<RealPlan>>>> = OnceLock::new();
    cached(&CACHE, n, || RealPlan::new(n))
}

/// The largest odd prime factor of `n`, or 1 if `n` is a power of two.
fn largest_odd_prime_factor(mut n: usize) -> usize {
    while n % 2 == 0 {
        n /= 2;
    }
    let mut largest = 1;
    let mut f = 3;
    while f * f <= n {
        while n % f == 0 {
            largest = f;
            n /= f;
        }
        f += 2;
    }
    largest.max(n)
}

impl Plan {
    fn new(n: usize) -> Plan {
        assert!(n >= 2);
        if n.is_power_of_two() {
            let bits = n.trailing_zeros();
            let rev = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
            let mut stage_twiddles = Vec::with_capacity(n - 1);
            let mut len = 2;
            while len <= n {
                stage_twiddles.extend(roots(len, len / 2));
                len *= 2;
            }
            return Plan { n, kind: Kind::Pow2 { rev, stage_twiddles } };
        }
        let p = largest_odd_prime_factor(n);
        if p <= MAX_DIRECT_PRIME {
            let m = n / p;
            let sub = if m > 1 { Some(plan(m)) } else { None };
            return Plan { n, kind: Kind::Mixed { p, m, sub, twiddles: roots(n, n) } };
        }
        let m = (2 * n - 1).next_power_of_two();
        let n64 = n as u64;
        let chirp: Vec<C64> = (0..n64).map(|k| unit_root((k * k) % (2 * n64), n64)).collect();
        let mut kernel = vec![C64::ZERO; m];
        kernel[0] = chirp[0].conj();
        for k in 1..n {
            kernel[k] = chirp[k].conj();
            kernel[m - k] = chirp[k].conj();
        }
        let inner = plan(m);
        inner.forward(&mut kernel, 1);
        Plan { n, kind: Kind::Bluestein { m, chirp, kernel_hat: kernel, inner } }
    }

    /// Transform `b` interleaved vectors at once: `x` holds `n` rows of `b`
    /// values, row-major, and each column is transformed independently.
    ///
    /// Every column sees exactly the arithmetic it would see alone, so a batch
    /// is bit-identical to `b` separate transforms. Batching only shares the
    /// twiddle loads and lets the compiler vectorise across columns, which is
    /// exact because each lane does the same IEEE operations as scalar code.
    fn forward(&self, x: &mut [C64], b: usize) {
        debug_assert_eq!(x.len(), self.n * b);
        match &self.kind {
            Kind::Pow2 { rev, stage_twiddles } => {
                if b == 1 {
                    radix2(x, rev, stage_twiddles)
                } else {
                    radix2_batch(x, b, rev, stage_twiddles)
                }
            }
            Kind::Mixed { p, m, sub, twiddles } => mixed(x, b, *p, *m, sub.as_deref(), twiddles),
            Kind::Bluestein { m, chirp, kernel_hat, inner } => {
                let n = self.n;
                let mut a = vec![C64::ZERO; m * b];
                for k in 0..n {
                    for c in 0..b {
                        a[k * b + c] = x[k * b + c] * chirp[k];
                    }
                }
                inner.forward(&mut a, b);
                for (i, h) in kernel_hat.iter().enumerate() {
                    for v in &mut a[i * b..(i + 1) * b] {
                        *v = *v * *h;
                    }
                }
                inner.inverse_unnormalised(&mut a, b);
                let mf = *m as f64;
                for k in 0..n {
                    for c in 0..b {
                        x[k * b + c] = chirp[k] * a[k * b + c].unscale(mf);
                    }
                }
            }
        }
    }

    /// The inverse as the conjugate of the forward transform of the conjugate.
    /// Conjugation is exact, so this adds no rounding.
    fn inverse_unnormalised(&self, x: &mut [C64], b: usize) {
        for v in x.iter_mut() {
            *v = v.conj();
        }
        self.forward(x, b);
        for v in x.iter_mut() {
            *v = v.conj();
        }
    }
}

/// One mixed-radix step for `n = p m`.
///
/// With `Y_r` the length-`m` transform of the decimated sequence
/// `x[j p + r]`, the output is
/// `X[k + m q] = sum_r W_n^(r k) W_p^(r q) Y_r[k]`, summed over `r` in order.
/// Twiddles that are exactly one (exponent zero) are not multiplied.
fn mixed(x: &mut [C64], b: usize, p: usize, m: usize, sub: Option<&Plan>, tw: &[C64]) {
    let n = p * m;
    let mut y = vec![C64::ZERO; n * b];
    for r in 0..p {
        for j in 0..m {
            let src = (j * p + r) * b;
            let dst = (r * m + j) * b;
            y[dst..dst + b].copy_from_slice(&x[src..src + b]);
        }
    }
    if let Some(sub) = sub {
        for r in 0..p {
            sub.forward(&mut y[r * m * b..(r + 1) * m * b], b);
        }
    }

    let mut t = vec![C64::ZERO; p * b];
    let mut sd = vec![C64::ZERO; (p - 1) * b];
    for k in 0..m {
        for r in 0..p {
            let row = &y[(r * m + k) * b..(r * m + k + 1) * b];
            let dst = &mut t[r * b..(r + 1) * b];
            let e = r * k;
            if e == 0 {
                dst.copy_from_slice(row);
            } else {
                let w = tw[e];
                for (d, v) in dst.iter_mut().zip(row) {
                    *d = *v * w;
                }
            }
        }
        prime_dft(x, b, p, m, k, &mut t, &mut sd, tw);
    }
}

/// The direct `p`-point DFT of the rows `t_0, ..., t_(p-1)`, for odd prime
/// `p`, written to rows `k + m q` of `x`.
///
/// Pairs `r` with `p - r`, whose twiddles are complex conjugates. With
/// `s_r = t_r + t_(p-r)`, `d_r = t_r - t_(p-r)`, `c = cos(2 pi r q / p)` and
/// `s = sin(2 pi r q / p)`:
///
/// - `X_0 = t_0 + s_1 + ... + s_h`
/// - `A_q = t_0 + sum_r c s_r` and `B_q = sum_r s d_r`, over `r = 1, ..., h`
///   in order, where `h = (p - 1) / 2`
/// - `X_q = A_q - i B_q` and `X_(p-q) = A_q + i B_q`
///
/// Every product is real times complex, a quarter of the floating-point work
/// of the unpaired sum.
#[allow(clippy::too_many_arguments)]
fn prime_dft(x: &mut [C64], b: usize, p: usize, m: usize, k: usize, t: &mut [C64], sd: &mut [C64], tw: &[C64]) {
    let h = (p - 1) / 2;
    let (s, d) = sd.split_at_mut(h * b);
    for r in 1..=h {
        let (tr, tpr) = (&t[r * b..(r + 1) * b], &t[(p - r) * b..(p - r + 1) * b]);
        let (sr, dr) = (&mut s[(r - 1) * b..r * b], &mut d[(r - 1) * b..r * b]);
        for c in 0..b {
            sr[c] = tr[c] + tpr[c];
            dr[c] = tr[c] - tpr[c];
        }
    }
    let t0 = &t[..b];

    let out0 = &mut x[k * b..(k + 1) * b];
    out0.copy_from_slice(t0);
    for r in 1..=h {
        for (o, v) in out0.iter_mut().zip(&s[(r - 1) * b..r * b]) {
            *o = *o + *v;
        }
    }

    let mut a = vec![C64::ZERO; b];
    let mut bb = vec![C64::ZERO; b];
    for q in 1..=h {
        a.copy_from_slice(t0);
        for r in 1..=h {
            // exp(-2 pi i e / p) = cos - i sin, read from the length-n table.
            let w = tw[((r * q) % p) * m];
            let (cs, sn) = (w.re, -w.im);
            let sr = &s[(r - 1) * b..r * b];
            let dr = &d[(r - 1) * b..r * b];
            if r == 1 {
                for c in 0..b {
                    a[c] = a[c] + sr[c].scale(cs);
                    bb[c] = dr[c].scale(sn);
                }
            } else {
                for c in 0..b {
                    a[c] = a[c] + sr[c].scale(cs);
                    bb[c] = bb[c] + dr[c].scale(sn);
                }
            }
        }
        let (lo, hi) = (k + m * q, k + m * (p - q));
        for c in 0..b {
            // -i B and +i B, exactly.
            x[lo * b + c] = a[c] + C64::new(bb[c].im, -bb[c].re);
            x[hi * b + c] = a[c] + C64::new(-bb[c].im, bb[c].re);
        }
    }
}

/// The textbook radix-2 transform applied to `b` interleaved columns.
fn radix2_batch(x: &mut [C64], b: usize, rev: &[u32], tw: &[C64]) {
    let n = x.len() / b;
    for i in 0..n {
        let j = rev[i] as usize;
        if i < j {
            for c in 0..b {
                x.swap(i * b + c, j * b + c);
            }
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let w = &tw[half - 1..len - 1];
        for blk in x.chunks_exact_mut(len * b) {
            let (lo, hi) = blk.split_at_mut(half * b);
            for k in 0..half {
                let wk = w[k];
                let lo_k = &mut lo[k * b..(k + 1) * b];
                let hi_k = &mut hi[k * b..(k + 1) * b];
                for (a, h) in lo_k.iter_mut().zip(hi_k.iter_mut()) {
                    let av = *a;
                    let bv = *h * wk;
                    *a = av + bv;
                    *h = av - bv;
                }
            }
        }
        len *= 2;
    }
}

/// Multiply by `i`, exactly.
#[inline]
fn times_i(v: C64) -> C64 {
    C64::new(-v.im, v.re)
}

/// Size of the blocks within which the early stages run to completion: 2048
/// complex values are 32 KiB, inside the level-one data cache of any current
/// phone or laptop core.
const BLOCK: usize = 2048;

/// Iterative radix-2 decimation-in-time FFT of a power-of-two length.
///
/// The numerical definition is the textbook one: bit-reverse, then for each
/// stage length `len = 2, 4, ..., n`, replace every pair `(a, b)` at distance
/// `len / 2` by `(a + w b, a - w b)` with `w = exp(-2 pi i k / len)`.
///
/// The schedule below does the same butterflies in a cache-friendlier order.
/// Stages up to `BLOCK` only mix values within aligned blocks of that size, so
/// each block is finished before the next is touched, and stages are fused in
/// pairs so that each pass over memory does two. Every butterfly still sees
/// exactly the same inputs, so the output is bit-identical to the textbook
/// order; `blocked_schedule_is_bit_identical` checks that.
fn radix2(x: &mut [C64], rev: &[u32], tw: &[C64]) {
    let n = x.len();
    for i in 0..n {
        let j = rev[i] as usize;
        if i < j {
            x.swap(i, j);
        }
    }
    let block = BLOCK.min(n);
    for chunk in x.chunks_exact_mut(block) {
        stages(chunk, 2, block, tw);
    }
    stages(x, 2 * block, n, tw);
}

/// Apply the stages of length `lo, 2 lo, ..., hi` to `x`, fusing pairs.
fn stages(x: &mut [C64], lo: usize, hi: usize, tw: &[C64]) {
    let mut len = lo;
    while len <= hi {
        if 2 * len <= hi {
            stage_pair(x, len, tw);
            len *= 4;
        } else {
            stage(x, len, tw);
            len *= 2;
        }
    }
}

fn stage(x: &mut [C64], len: usize, tw: &[C64]) {
    let half = len / 2;
    let w = &tw[half - 1..len - 1];
    for blk in x.chunks_exact_mut(len) {
        let (lo, hi) = blk.split_at_mut(half);
        for k in 0..half {
            let a = lo[k];
            let b = hi[k] * w[k];
            lo[k] = a + b;
            hi[k] = a - b;
        }
    }
}

/// The stages of length `len` and `2 len` in one pass, with the same
/// arithmetic as two calls to [`stage`].
fn stage_pair(x: &mut [C64], len: usize, tw: &[C64]) {
    let h = len / 2;
    let w1 = &tw[h - 1..len - 1];
    let w2 = &tw[len - 1..2 * len - 1];
    for blk in x.chunks_exact_mut(2 * len) {
        let (first, second) = blk.split_at_mut(len);
        let (x0s, x1s) = first.split_at_mut(h);
        let (x2s, x3s) = second.split_at_mut(h);
        for k in 0..h {
            // Stage `len`: pairs (0, 1) and (2, 3), both with twiddle w1[k].
            let b1 = x1s[k] * w1[k];
            let a0 = x0s[k] + b1;
            let a1 = x0s[k] - b1;
            let b3 = x3s[k] * w1[k];
            let a2 = x2s[k] + b3;
            let a3 = x2s[k] - b3;
            // Stage `2 len`: pairs (0, 2) with w2[k] and (1, 3) with w2[k + h].
            let c2 = a2 * w2[k];
            x0s[k] = a0 + c2;
            x2s[k] = a0 - c2;
            let c3 = a3 * w2[k + h];
            x1s[k] = a1 + c3;
            x3s[k] = a1 - c3;
        }
    }
}

/// Transforms of real signals of even length `n = 2 h` through a complex FFT
/// of length `h`.
struct RealPlan {
    n: usize,
    half: Arc<Plan>,
    /// `exp(-2 pi i k / n)` for `k < h`.
    twiddles: Vec<C64>,
}

impl RealPlan {
    fn new(n: usize) -> RealPlan {
        RealPlan { n, half: plan(n / 2), twiddles: roots(n, n / 2) }
    }

    /// With `z[m] = x[2m] + i x[2m+1]` and `Z = FFT_h(z)`, the spectra of the
    /// even and odd samples are `E_k = (Z_k + conj Z_{h-k}) / 2` and
    /// `O_k = -i (Z_k - conj Z_{h-k}) / 2`, and `X_k = E_k + W^k O_k`. The
    /// upper half is filled by conjugate symmetry, which is exact.
    fn forward(&self, x: &[f64]) -> Vec<C64> {
        let n = self.n;
        let h = n / 2;
        let mut z: Vec<C64> = (0..h).map(|m| C64::new(x[2 * m], x[2 * m + 1])).collect();
        self.half.forward(&mut z, 1);

        let mut out = vec![C64::ZERO; n];
        out[0] = C64::new(z[0].re + z[0].im, 0.0);
        out[h] = C64::new(z[0].re - z[0].im, 0.0);
        for k in 1..h {
            let zc = z[h - k].conj();
            let e = (z[k] + zc).scale(0.5);
            let d = z[k] - zc;
            let o = C64::new(d.im, -d.re).scale(0.5);
            let v = e + self.twiddles[k] * o;
            out[k] = v;
            out[n - k] = v.conj();
        }
        out
    }

    /// `Re(ifft(x))`. The Hermitian part `H_k = (X_k + conj X_{n-k}) / 2` has
    /// the same real inverse and a real inverse exactly. Its even and odd
    /// sample spectra are `E_k = (H_k + H_{k+h}) / 2` and
    /// `O_k = (H_k - H_{k+h}) conj(W^k) / 2`, and `IFFT_h(E + i O)` interleaves
    /// the even and odd samples as its real and imaginary parts.
    fn inverse_real_part(&self, x: &[C64]) -> Vec<f64> {
        let n = self.n;
        let h = n / 2;
        let herm = |k: usize| (x[k] + x[(n - k) % n].conj()).scale(0.5);
        let mut z: Vec<C64> = (0..h)
            .map(|k| {
                let a = herm(k);
                let b = herm(k + h);
                let e = (a + b).scale(0.5);
                let o = ((a - b) * self.twiddles[k].conj()).scale(0.5);
                e + times_i(o)
            })
            .collect();
        self.half.inverse_unnormalised(&mut z, 1);
        let hf = h as f64;
        let mut out = vec![0.0; n];
        for (m, v) in z.iter().enumerate() {
            out[2 * m] = v.re / hf;
            out[2 * m + 1] = v.im / hf;
        }
        out
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
                C64::new(
                    math::sin(0.37 * t) + 0.1 * t / n as f64,
                    math::cos(1.3 * t * t / n as f64),
                )
            })
            .collect()
    }

    fn max_rel_err(a: &[C64], b: &[C64]) -> f64 {
        let scale = b.iter().map(|v| v.norm()).fold(0.0, f64::max);
        a.iter().zip(b).map(|(p, q)| (*p - *q).norm()).fold(0.0, f64::max) / scale
    }

    const LENGTHS: [usize; 16] = [2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64, 100, 120, 127, 256, 1000];

    #[test]
    fn matches_naive_dft_for_assorted_lengths() {
        for n in LENGTHS.into_iter().chain([512, 2048]) {
            let x = signal(n);
            let mut got = x.clone();
            fft(&mut got);
            let err = max_rel_err(&got, &naive_dft(&x));
            assert!(err < 1e-13, "n = {n}: relative error {err:e}");
        }
    }

    #[test]
    fn real_transforms_match_complex_ones() {
        for n in LENGTHS {
            let x: Vec<f64> = signal(n).iter().map(|v| v.re).collect();
            let as_complex: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
            let want = naive_dft(&as_complex);
            let err = max_rel_err(&fft_real(&x), &want);
            assert!(err < 1e-13, "fft_real n = {n}: relative error {err:e}");

            // A spectrum that is not Hermitian: the real part of its inverse.
            let spec = signal(n);
            let mut full = spec.clone();
            ifft(&mut full);
            let want_re: Vec<C64> = full.iter().map(|v| C64::new(v.re, 0.0)).collect();
            let got_re: Vec<C64> = ifft_real_part(spec).iter().map(|&v| C64::new(v, 0.0)).collect();
            let err = max_rel_err(&got_re, &want_re);
            assert!(err < 1e-13, "ifft_real_part n = {n}: relative error {err:e}");
        }
    }

    /// The textbook loop order: bit-reverse, then every stage over the whole
    /// array, smallest first.
    fn textbook(x: &mut [C64]) {
        let n = x.len();
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = (i as u32).reverse_bits() as usize >> (32 - bits);
            if i < j {
                x.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let w = unit_root(2 * k as u64, len as u64);
                    let a = x[start + k];
                    let b = x[start + k + len / 2] * w;
                    x[start + k] = a + b;
                    x[start + k + len / 2] = a - b;
                }
            }
            len *= 2;
        }
    }

    #[test]
    fn blocked_schedule_is_bit_identical() {
        // Odd and even stage counts, below, at and above the block size.
        for bits in 1..=15 {
            let n = 1usize << bits;
            let x = signal(n);
            let mut want = x.clone();
            textbook(&mut want);
            let mut got = x;
            fft(&mut got);
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g.re.to_bits() == w.re.to_bits() && g.im.to_bits() == w.im.to_bits(),
                    "n = {n}, bin {k}: {g:?} != {w:?}"
                );
            }
        }
    }

    #[test]
    fn mixed_radix_lengths_match_naive_dft() {
        // 152 = 19 x 8 and its subsamplings, the frequential axis of a joint
        // operator; products of several odd primes; a 61-point direct DFT.
        for n in [152, 76, 38, 19, 60, 105, 225, 3 * 5 * 7 * 4, 61, 2 * 61] {
            assert!(largest_odd_prime_factor(n) <= MAX_DIRECT_PRIME);
            let x = signal(n);
            let mut got = x.clone();
            fft(&mut got);
            let err = max_rel_err(&got, &naive_dft(&x));
            assert!(err < 1e-13, "n = {n}: relative error {err:e}");
        }
        assert_eq!(largest_odd_prime_factor(152), 19);
        assert_eq!(largest_odd_prime_factor(32000), 5);
        assert_eq!(largest_odd_prime_factor(1024), 1);
        assert_eq!(largest_odd_prime_factor(127), 127);
    }

    #[test]
    fn batches_are_bit_identical_to_single_transforms() {
        // Power of two, mixed radix, direct prime, and Bluestein.
        for n in [8, 64, 4096, 152, 19, 120, 127, 254] {
            for b in [1, 3, 7, 150] {
                let cols: Vec<Vec<C64>> = (0..b)
                    .map(|c| signal(n + c).into_iter().take(n).collect())
                    .collect();
                let mut batch: Vec<C64> = (0..n).flat_map(|r| cols.iter().map(move |c| c[r])).collect();
                fft_columns(&mut batch, b);
                let mut inv = batch.clone();
                ifft_columns(&mut inv, b);
                for (c, col) in cols.iter().enumerate() {
                    let mut single = col.clone();
                    fft(&mut single);
                    let mut back = single.clone();
                    ifft(&mut back);
                    for r in 0..n {
                        let (g, w) = (batch[r * b + c], single[r]);
                        assert!(g.re.to_bits() == w.re.to_bits() && g.im.to_bits() == w.im.to_bits(),
                                "forward n = {n}, b = {b}, column {c}, row {r}");
                        let (g, w) = (inv[r * b + c], back[r]);
                        assert!(g.re.to_bits() == w.re.to_bits() && g.im.to_bits() == w.im.to_bits(),
                                "inverse n = {n}, b = {b}, column {c}, row {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_round_trips() {
        for n in [1, 2, 6, 64, 120, 1500, 4096, 8192] {
            let x = signal(n);
            let mut y = x.clone();
            fft(&mut y);
            ifft(&mut y);
            let err = max_rel_err(&y, &x);
            assert!(err < 1e-13, "n = {n}: relative error {err:e}");
        }
    }
}
