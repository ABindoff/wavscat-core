//! Array primities shared by the cascades.
//!
//! The Rust counterpart of `R/backend-r.R`. The order of every reduction here
//! is part of the numerical definition: blocks are summed first to last, and a
//! mean is that sum divided by the count.

use crate::complex::C64;

/// Subsample a signal by `k` through its spectrum. Kept as the reference
/// definition that [`filter_periodize`] is tested against.
///
/// Element `i` of the result is the mean of `x[b * m + i]` over blocks
/// `b = 0, ..., k - 1`, where `m = len / k`. The `1 / k` keeps the inverse
/// transform correctly scaled.
#[cfg(test)]
pub(crate) fn periodize_mean(x: &[C64], k: usize) -> Vec<C64> {
    if k <= 1 {
        return x.to_vec();
    }
    let kf = k as f64;
    periodize(x, k, |acc| acc.unscale(kf))
}

/// Rescale a real filter to a coarser resolution by summing blocks.
pub(crate) fn periodize_sum(x: &[f64], k: usize) -> Vec<f64> {
    if k <= 1 {
        return x.to_vec();
    }
    assert_eq!(x.len() % k, 0, "cannot periodise {} by {}", x.len(), k);
    let m = x.len() / k;
    (0..m)
        .map(|i| {
            let mut acc = x[i];
            for b in 1..k {
                acc += x[b * m + i];
            }
            acc
        })
        .collect()
}

#[cfg(test)]
fn periodize(x: &[C64], k: usize, finish: impl Fn(C64) -> C64) -> Vec<C64> {
    assert_eq!(x.len() % k, 0, "cannot periodise {} by {}", x.len(), k);
    let m = x.len() / k;
    (0..m)
        .map(|i| {
            let mut acc = x[i];
            for b in 1..k {
                acc = acc + x[b * m + i];
            }
            finish(acc)
        })
        .collect()
}

/// Filter a spectrum and subsample the result by `k`, without materialising
/// the full-length product.
///
/// Exactly `periodize_mean(&cdgmm(a, f), k)`: the same products, summed over
/// blocks in the same order, divided by `k` once. Fusing them only avoids
/// writing and rereading a full-length temporary, which on the first order is
/// a megabyte per wavelet.
pub(crate) fn filter_periodize(a: &[C64], f: &[f64], k: usize) -> Vec<C64> {
    assert_eq!(a.len(), f.len());
    if k <= 1 {
        return cdgmm(a, f);
    }
    assert_eq!(a.len() % k, 0, "cannot periodise {} by {}", a.len(), k);
    let m = a.len() / k;
    let kf = k as f64;
    (0..m)
        .map(|i| {
            let mut acc = a[i].scale(f[i]);
            for b in 1..k {
                let t = b * m + i;
                acc = acc + a[t].scale(f[t]);
            }
            acc.unscale(kf)
        })
        .collect()
}

/// [`filter_periodize`] applied to every column of a row-major matrix with
/// `cols` columns, as the frequential axis of joint scattering needs.
///
/// Row `i` of the result accumulates rows `i, m + i, 2m + i, ...` in that
/// order, so each column sees exactly the arithmetic of [`filter_periodize`].
pub(crate) fn filter_periodize_rows(a: &[C64], cols: usize, f: &[f64], k: usize) -> Vec<C64> {
    let n = f.len();
    assert_eq!(a.len(), n * cols);
    assert_eq!(n % k.max(1), 0, "cannot periodise {} rows by {}", n, k);
    let k = k.max(1);
    let m = n / k;
    let kf = k as f64;
    let mut out = vec![C64::ZERO; m * cols];
    for i in 0..m {
        let acc = &mut out[i * cols..(i + 1) * cols];
        let h = f[i];
        for (o, v) in acc.iter_mut().zip(&a[i * cols..(i + 1) * cols]) {
            *o = v.scale(h);
        }
        for b in 1..k {
            let t = b * m + i;
            let h = f[t];
            for (o, v) in acc.iter_mut().zip(&a[t * cols..(t + 1) * cols]) {
                *o = *o + v.scale(h);
            }
        }
        if k > 1 {
            for o in acc.iter_mut() {
                *o = o.unscale(kf);
            }
        }
    }
    out
}

/// Pointwise product of a spectrum with a real filter.
pub(crate) fn cdgmm(a: &[C64], f: &[f64]) -> Vec<C64> {
    debug_assert_eq!(a.len(), f.len());
    a.iter().zip(f).map(|(v, &h)| v.scale(h)).collect()
}

/// Complex moduli.
pub(crate) fn modulus(x: &[C64]) -> Vec<f64> {
    x.iter().map(|v| v.norm()).collect()
}

/// Sum in index order.
pub(crate) fn sum(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_filter_periodize_is_bit_identical() {
        let n = 1024;
        let a: Vec<C64> = (0..n)
            .map(|i| C64::new(((i * 37) % 101) as f64 / 7.0 - 3.0, ((i * 53) % 97) as f64 / 11.0))
            .collect();
        let f: Vec<f64> = (0..n).map(|i| ((i * 29) % 89) as f64 / 13.0 - 1.5).collect();
        for k in [1, 2, 4, 8, 64, 1024] {
            let want = periodize_mean(&cdgmm(&a, &f), k);
            let got = filter_periodize(&a, &f, k);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert!(g.re.to_bits() == w.re.to_bits() && g.im.to_bits() == w.im.to_bits(), "k = {k}");
            }
        }
    }

    #[test]
    fn row_wise_filter_periodize_matches_each_column() {
        let (n, cols) = (152, 5);
        let a: Vec<C64> = (0..n * cols)
            .map(|i| C64::new(((i * 37) % 101) as f64 / 7.0 - 3.0, ((i * 53) % 97) as f64 / 11.0))
            .collect();
        let f: Vec<f64> = (0..n).map(|i| ((i * 29) % 89) as f64 / 13.0 - 1.5).collect();
        for k in [1, 2, 4, 8] {
            let got = filter_periodize_rows(&a, cols, &f, k);
            for c in 0..cols {
                let col: Vec<C64> = (0..n).map(|r| a[r * cols + c]).collect();
                let want = filter_periodize(&col, &f, k);
                for (r, w) in want.iter().enumerate() {
                    let g = got[r * cols + c];
                    assert!(g.re.to_bits() == w.re.to_bits() && g.im.to_bits() == w.im.to_bits(),
                            "k = {k}, column {c}, row {r}");
                }
            }
        }
    }
}
