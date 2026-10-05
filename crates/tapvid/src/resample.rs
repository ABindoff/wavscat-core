//! Stage 4: resample irregularly captured samples onto a uniform grid.
//!
//! Webcam frame rates vary, and drop frames, so timestamps are irregular. The
//! SVD does not mind, but scattering and the analytic signal assume uniform
//! sampling, so the selected component is resampled here, using its true
//! capture timestamps.
//!
//! Interpolation is cubic Hermite: within each interval the curve matches the
//! two samples and a slope at each, the slope at a sample being that of the
//! parabola through it and its neighbours. That is exact for quadratics,
//! local, and needs no global solve. A gap longer than `max_gap` is never
//! interpolated across: grid points inside it are reported, and left to the
//! caller to split the trial or reject it.

use wavscat_core::Error;

/// A gap in capture longer than the interpolation limit.
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    /// Capture time of the last frame before the gap, in seconds.
    pub start: f64,
    /// Capture time of the first frame after it.
    pub end: f64,
}

/// A series on a uniform grid.
#[derive(Debug, Clone)]
pub struct Uniform {
    /// Time of the first sample, in seconds.
    pub t0: f64,
    /// Sampling rate, in Hz.
    pub fs: f64,
    /// Values at `t0 + i / fs`. Points inside a gap are `NaN`.
    pub values: Vec<f64>,
    /// Gaps that were not interpolated across.
    pub gaps: Vec<Gap>,
}

/// Resample `(timestamps, values)` onto a grid at `fs` Hz starting at the
/// first timestamp. Timestamps must be strictly increasing.
pub fn resample(timestamps: &[f64], values: &[f64], fs: f64, max_gap: f64) -> Result<Uniform, Error> {
    let n = timestamps.len();
    if values.len() != n {
        return Err(Error("timestamps and values differ in length.".into()));
    }
    if n < 3 {
        return Err(Error("At least three samples are needed to resample.".into()));
    }
    if !(fs > 0.0) || !(max_gap > 0.0) {
        return Err(Error("fs and max_gap must be positive.".into()));
    }
    if timestamps.windows(2).any(|w| !(w[1] > w[0])) {
        return Err(Error("Timestamps must be strictly increasing.".into()));
    }
    if values.iter().chain(timestamps).any(|v| !v.is_finite()) {
        return Err(Error("Timestamps and values must be finite.".into()));
    }

    let t = timestamps;
    let broken = |j: usize| t[j + 1] - t[j] > max_gap;
    let secant = |j: usize| (values[j + 1] - values[j]) / (t[j + 1] - t[j]);
    // Slope at sample i: the parabola through i and its neighbours, or the
    // one-sided secant at an end or beside a gap.
    let slopes: Vec<f64> = (0..n)
        .map(|i| {
            let left = i > 0 && !broken(i - 1);
            let right = i + 1 < n && !broken(i);
            match (left, right) {
                (true, true) => {
                    let (h0, h1) = (t[i] - t[i - 1], t[i + 1] - t[i]);
                    (h1 * secant(i - 1) + h0 * secant(i)) / (h0 + h1)
                }
                (true, false) => secant(i - 1),
                (false, true) => secant(i),
                (false, false) => 0.0,
            }
        })
        .collect();

    let gaps: Vec<Gap> = (0..n - 1).filter(|&j| broken(j)).map(|j| Gap { start: t[j], end: t[j + 1] }).collect();

    let t0 = t[0];
    let m = ((t[n - 1] - t0) * fs).floor() as usize + 1;
    let mut out = Vec::with_capacity(m);
    let mut j = 0;
    for i in 0..m {
        let ti = t0 + i as f64 / fs;
        while j + 2 < n && t[j + 1] <= ti {
            j += 1;
        }
        if broken(j) {
            out.push(f64::NAN);
            continue;
        }
        let h = t[j + 1] - t[j];
        let s = ((ti - t[j]) / h).clamp(0.0, 1.0);
        let (s2, s3) = (s * s, s * s * s);
        out.push(
            (2.0 * s3 - 3.0 * s2 + 1.0) * values[j]
                + (s3 - 2.0 * s2 + s) * h * slopes[j]
                + (-2.0 * s3 + 3.0 * s2) * values[j + 1]
                + (s3 - s2) * h * slopes[j + 1],
        );
    }
    Ok(Uniform { t0, fs, values: out, gaps })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_for_quadratics_on_irregular_samples() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.033 + 0.004 * ((i * 7 % 5) as f64 - 2.0)).collect();
        let f = |x: f64| 2.0 - 0.5 * x + 3.0 * x * x;
        let v: Vec<f64> = t.iter().map(|&x| f(x)).collect();
        let u = resample(&t, &v, 30.0, 0.15).unwrap();
        // Interior intervals, away from the one-sided slopes at the ends.
        for (i, &y) in u.values.iter().enumerate().skip(3).take(u.values.len() - 6) {
            let x = u.t0 + i as f64 / 30.0;
            assert!((y - f(x)).abs() < 1e-9, "at {x}: {y} vs {}", f(x));
        }
    }

    #[test]
    fn long_gaps_are_reported_not_bridged() {
        let mut t: Vec<f64> = (0..30).map(|i| i as f64 / 30.0).collect();
        t.extend((40..70).map(|i| i as f64 / 30.0));
        let v: Vec<f64> = t.iter().map(|&x| wavscat_core::math::sin(x)).collect();
        let u = resample(&t, &v, 30.0, 0.15).unwrap();
        assert_eq!(u.gaps.len(), 1);
        assert!(u.values.iter().any(|v| v.is_nan()));
        assert!(u.values[..29].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_non_monotonic_timestamps() {
        assert!(resample(&[0.0, 0.1, 0.1, 0.2], &[0.0; 4], 30.0, 0.15).is_err());
    }
}
