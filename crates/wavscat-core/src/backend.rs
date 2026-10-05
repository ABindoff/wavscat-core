//! Array primitives shared by the cascades.
//!
//! The Rust counterpart of `R/backend-r.R`. The order of every reduction here
//! is part of the numerical definition: blocks are summed first to last, and a
//! mean is that sum divided by the count.

use crate::complex::C64;

/// Subsample a signal by `k` through its spectrum.
///
/// Element `i` of the result is the mean of `x[b * m + i]` over blocks
/// `b = 0, ..., k - 1`, where `m = len / k`. The `1 / k` keeps the inverse
/// transform correctly scaled.
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

/// Pointwise product of a spectrum with a real filter.
pub(crate) fn cdgmm(a: &[C64], f: &[f64]) -> Vec<C64> {
    debug_assert_eq!(a.len(), f.len());
    a.iter().zip(f).map(|(v, &h)| v.scale(h)).collect()
}

/// Complex moduli.
pub(crate) fn modulus(x: &[C64]) -> Vec<f64> {
    x.iter().map(|v| v.norm()).collect()
}

/// Real parts.
pub(crate) fn real(x: &[C64]) -> Vec<f64> {
    x.iter().map(|v| v.re).collect()
}

/// Sum in index order.
pub(crate) fn sum(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in x {
        acc += v;
    }
    acc
}
