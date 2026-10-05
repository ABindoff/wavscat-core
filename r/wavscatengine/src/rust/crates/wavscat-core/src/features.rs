//! Turning coefficients into model-ready features.
//!
//! Ports of `scat_log()`, `scat_renorm()`, `scat_eps()` and the summaries in
//! `scat_features()`. These live in the core, not in each binding, because a
//! feature vector is only bit-identical across languages if the last step that
//! touches it is too. R's own `mean()` accumulates in extended precision, for
//! instance, so it would not agree with the other languages.

use crate::error::{fail, Error};
use crate::math;
use crate::scattering1d::Path1d;

/// Log compression, `sign(v) * log1p(|v| / eps)`, in place.
///
/// Behaves like `log` for coefficients well above `eps` and is linear near
/// zero, so it never produces `-inf`.
pub fn log_compress(values: &mut [f64], eps: f64) -> Result<(), Error> {
    if !(eps > 0.0) || !eps.is_finite() {
        return fail("eps must be a single positive number.");
    }
    for v in values.iter_mut() {
        let s = if *v > 0.0 {
            1.0
        } else if *v < 0.0 {
            -1.0
        } else {
            0.0
        };
        *v = s * math::log1p(v.abs() / eps);
    }
    Ok(())
}

/// Divide each second-order path by its first-order parent plus `eps`, in
/// place, so that second order measures modulation relative to the energy it
/// rides on.
///
/// `coefs` holds one vector per path, aligned with `paths`.
pub fn renorm(coefs: &mut [Vec<f64>], paths: &[Path1d], eps: f64) -> Result<(), Error> {
    if coefs.len() != paths.len() {
        return fail("coefs and paths must have the same length.");
    }
    for i in 0..paths.len() {
        if paths[i].order != 2 {
            continue;
        }
        let parent = paths
            .iter()
            .position(|p| p.order == 1 && p.n1 == paths[i].n1)
            .ok_or_else(|| Error::new("A second-order path has no first-order parent."))?;
        let (child, par) = if parent < i {
            let (a, b) = coefs.split_at_mut(i);
            (&mut b[0], &a[parent])
        } else {
            let (a, b) = coefs.split_at_mut(parent);
            (&mut a[i], &b[0])
        };
        if child.len() != par.len() {
            return fail("A second-order path and its parent differ in length.");
        }
        for (c, p) in child.iter_mut().zip(par.iter()) {
            *c /= *p + eps;
        }
    }
    Ok(())
}

/// How [`summarise`] reduces a path's time axis to one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Summary {
    Mean,
    Max,
    Sd,
    Median,
}

impl std::str::FromStr for Summary {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "mean" => Ok(Summary::Mean),
            "max" => Ok(Summary::Max),
            "sd" => Ok(Summary::Sd),
            "median" => Ok(Summary::Median),
            _ => fail(format!("Unknown summary \"{s}\".")),
        }
    }
}

/// Reduce one path to a single number.
///
/// Means are a left-to-right sum divided by the count; the standard deviation
/// is the two-pass estimate with denominator `n - 1`, `NaN` for one value; the
/// median of an even count is the mean of the middle two.
pub fn summarise(v: &[f64], how: Summary) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    match how {
        Summary::Mean => mean(v),
        Summary::Max => v.iter().copied().fold(f64::NEG_INFINITY, |a, b| {
            if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
        }),
        Summary::Sd => {
            if v.len() < 2 {
                return f64::NAN;
            }
            let m = mean(v);
            let mut ss = 0.0;
            for &x in v {
                ss += (x - m) * (x - m);
            }
            (ss / (v.len() - 1) as f64).sqrt()
        }
        Summary::Median => {
            if v.iter().any(|x| x.is_nan()) {
                return f64::NAN;
            }
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.total_cmp(b));
            let h = s.len() / 2;
            if s.len() % 2 == 1 { s[h] } else { (s[h - 1] + s[h]) / 2.0 }
        }
    }
}

fn mean(v: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &x in v {
        acc += x;
    }
    acc / v.len() as f64
}

/// A data-driven `eps` for [`log_compress`]: the `p` quantile of the positive,
/// finite coefficients, floored at machine epsilon, or `1e-6` if there are none.
///
/// The quantile is R's default (type 7) definition.
pub fn eps_quantile(values: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect();
    if v.is_empty() {
        return 1e-6;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let index = 1.0 + (v.len() - 1) as f64 * p;
    let lo = index.floor();
    let hi = index.ceil();
    let x_lo = v[lo as usize - 1];
    let x_hi = v[hi as usize - 1];
    let q = if index > lo && x_hi != x_lo {
        let h = index - lo;
        (1.0 - h) * x_lo + h * x_hi
    } else {
        x_lo
    };
    q.max(f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summaries() {
        let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        assert_eq!(summarise(&v, Summary::Mean), 23.0 / 6.0);
        assert_eq!(summarise(&v, Summary::Max), 9.0);
        assert_eq!(summarise(&v, Summary::Median), 3.5);
        assert!((summarise(&v, Summary::Sd) - 2.99443929086).abs() < 1e-10);
    }

    #[test]
    fn quantile_matches_r_type_7() {
        // quantile(c(1, 2, 3, 4, 10), 0.3) is 2.2 in R.
        assert!((eps_quantile(&[10.0, 3.0, 1.0, 4.0, 2.0, -1.0, 0.0], 0.3) - 2.2).abs() < 1e-15);
        assert_eq!(eps_quantile(&[0.0, -2.0], 0.01), 1e-6);
    }
}
