//! Stage 2: remove exposure gain and drift before the SVD.
//!
//! Webcam auto-exposure and white balance scale whole frames up and down over
//! time. Left in, that global gain would be the leading SVD component. So each
//! frame is first divided by its mean luminance, and that mean is kept as a QC
//! trace. Each pixel's series is then centred and, optionally, has slow drift
//! removed, well below the tapping band.
//!
//! Drift is removed by least squares onto cubic B-splines in time, evaluated
//! at the true capture timestamps, so irregular sampling and gaps need no
//! special handling, unlike an FFT filter, which assumes uniform sampling.
//! B-splines sum to one, so removing them also centres. With drift removal
//! off, the basis is the constant alone, and this is plain centring.
//!
//! None of this is materialised. The preprocessed matrix is
//! `A = (I - H) D^-1 X`, with `X` the frames, `D` the diagonal of frame means
//! and `H` the projection onto the orthonormalised basis, and the SVD only
//! ever needs `A x` and `A^T y`, each a product with `X` plus thin
//! corrections. That saves a second copy of the frames, 11 MB for a 30 s
//! trial.

use wavscat_core::Error;

use crate::linalg::orthonormalise;
use crate::svd::{Operator, RowSource};

/// Settings for [`preprocess`].
#[derive(Debug, Clone)]
pub struct PreprocessParams {
    /// Remove drift below this frequency, in Hz; `None` only centres.
    pub detrend_cutoff: Option<f64>,
    /// A frame whose mean luminance differs from the previous frame's by more
    /// than this fraction counts as an abrupt exposure change.
    pub abrupt_change: f64,
    /// Frames darker than this mean luminance, in 0-255 units, count as dark;
    /// their gain is floored here so that division stays finite.
    pub dark_level: f64,
}

impl Default for PreprocessParams {
    fn default() -> Self {
        PreprocessParams { detrend_cutoff: Some(0.3), abrupt_change: 0.05, dark_level: 1.0 }
    }
}

/// The exposure trace and its summary, for QC.
#[derive(Debug, Clone, PartialEq)]
pub struct GainQc {
    /// Mean luminance of each frame, in 0-255 units.
    pub trace: Vec<f64>,
    pub min: f64,
    pub max: f64,
    /// Frames whose mean jumped by more than `abrupt_change` from the last.
    pub abrupt_changes: usize,
    /// Frames darker than `dark_level`.
    pub dark_frames: usize,
}

/// The preprocessed frames, as an operator for the SVD.
pub struct Preprocessed<'a, S: RowSource> {
    src: &'a S,
    /// `1 / mean luminance` of each frame.
    inv_gain: Vec<f64>,
    /// Orthonormalised basis in time, frames by `k`, row-major; zero columns
    /// allowed.
    basis: Vec<f64>,
    k: usize,
}

/// Number of uniform knot intervals for a cubic B-spline basis that removes
/// drift below `cutoff` Hz over `duration` seconds: one interval per
/// `1 / (2 cutoff)` seconds, at least one.
pub fn knot_intervals(duration: f64, cutoff: f64) -> usize {
    ((duration * 2.0 * cutoff).ceil() as usize).max(1)
}

/// The uniform cubic B-spline with support `[0, 4)`, at `u`.
fn cubic_bspline(u: f64) -> f64 {
    if !(0.0..4.0).contains(&u) {
        0.0
    } else if u < 1.0 {
        u * u * u / 6.0
    } else if u < 2.0 {
        (-3.0 * u * u * u + 12.0 * u * u - 12.0 * u + 4.0) / 6.0
    } else if u < 3.0 {
        (3.0 * u * u * u - 24.0 * u * u + 60.0 * u - 44.0) / 6.0
    } else {
        (4.0 - u) * (4.0 - u) * (4.0 - u) / 6.0
    }
}

/// Wrap `src` (frames by pixels, captured at `timestamps` seconds) as the
/// preprocessed operator, and summarise its exposure trace.
pub fn preprocess<'a, S: RowSource>(
    src: &'a S,
    timestamps: &[f64],
    p: &PreprocessParams,
) -> Result<(Preprocessed<'a, S>, GainQc), Error> {
    let (t_len, cells) = (src.n_rows(), src.n_cols());
    if timestamps.len() != t_len {
        return Err(Error("There must be one timestamp per frame.".into()));
    }
    if t_len < 3 || cells == 0 {
        return Err(Error("At least three frames are needed.".into()));
    }
    if timestamps.windows(2).any(|w| !(w[1] > w[0])) {
        return Err(Error("Timestamps must be strictly increasing.".into()));
    }

    // Gain: the mean luminance of each frame.
    let mut trace = Vec::with_capacity(t_len);
    for r in 0..t_len {
        let mut acc = 0.0;
        for &v in src.row(r) {
            acc += v as f64;
        }
        trace.push(acc / cells as f64);
    }
    let inv_gain = trace.iter().map(|&m| 1.0 / m.max(p.dark_level)).collect();
    let mut abrupt_changes = 0;
    for w in trace.windows(2) {
        if w[0] > 0.0 && ((w[1] - w[0]) / w[0]).abs() > p.abrupt_change {
            abrupt_changes += 1;
        }
    }
    let qc = GainQc {
        min: trace.iter().copied().fold(f64::INFINITY, f64::min),
        max: trace.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        abrupt_changes,
        dark_frames: trace.iter().filter(|&&m| m < p.dark_level).count(),
        trace,
    };

    // Basis in time: cubic B-splines on uniform knots, or the constant.
    let (basis, k) = match p.detrend_cutoff {
        Some(fc) if fc > 0.0 => {
            let (a, b) = (timestamps[0], timestamps[t_len - 1]);
            let n_int = knot_intervals(b - a, fc);
            let delta = (b - a) / n_int as f64;
            let k = n_int + 3;
            let mut phi = vec![0.0; t_len * k];
            for (r, &t) in timestamps.iter().enumerate() {
                let x = (t - a) / delta;
                for i in 0..k {
                    phi[r * k + i] = cubic_bspline(x - (i as f64 - 3.0));
                }
            }
            (phi, k)
        }
        Some(_) => return Err(Error("detrend_cutoff must be positive.".into())),
        None => (vec![1.0; t_len], 1),
    };
    let mut basis = basis;
    orthonormalise(&mut basis, t_len, k);
    Ok((Preprocessed { src, inv_gain, basis, k }, qc))
}

impl<S: RowSource> Preprocessed<'_, S> {
    /// `v = (I - H) v` for `v` of size frames by `kk`, row-major: subtract
    /// the projection onto the basis, `Q (Q^T v)`.
    fn project_out(&self, v: &mut [f64], kk: usize) {
        let (t_len, k) = (self.src.n_rows(), self.k);
        let mut coef = vec![0.0; k * kk];
        for r in 0..t_len {
            for b in 0..k {
                let q = self.basis[r * k + b];
                if q != 0.0 {
                    for c in 0..kk {
                        coef[b * kk + c] += q * v[r * kk + c];
                    }
                }
            }
        }
        for r in 0..t_len {
            for b in 0..k {
                let q = self.basis[r * k + b];
                if q != 0.0 {
                    for c in 0..kk {
                        v[r * kk + c] -= q * coef[b * kk + c];
                    }
                }
            }
        }
    }

    /// The exposure gain removed from frame `r`, as `1 / mean luminance`.
    pub fn inv_gain(&self, r: usize) -> f64 {
        self.inv_gain[r]
    }
}

impl<S: RowSource> Operator for Preprocessed<'_, S> {
    fn rows(&self) -> usize {
        self.src.n_rows()
    }

    fn cols(&self) -> usize {
        self.src.n_cols()
    }

    /// `(I - H) D^-1 (X x)`.
    fn mul(&self, x: &[f64], k: usize, out: &mut [f64]) {
        self.src.mul(x, k, out);
        for (r, g) in self.inv_gain.iter().enumerate() {
            for v in &mut out[r * k..(r + 1) * k] {
                *v *= g;
            }
        }
        self.project_out(out, k);
    }

    /// `X^T (D^-1 (I - H) y)`.
    fn mul_t(&self, y: &[f64], k: usize, out: &mut [f64]) {
        let mut w = y.to_vec();
        self.project_out(&mut w, k);
        for (r, g) in self.inv_gain.iter().enumerate() {
            for v in &mut w[r * k..(r + 1) * k] {
                *v *= g;
            }
        }
        self.src.mul_t(&w, k, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bsplines_sum_to_one_inside_the_interval() {
        for i in 0..400 {
            let x = i as f64 * 0.025;
            let s: f64 = (0..13).map(|b| cubic_bspline(x - (b as f64 - 3.0))).sum();
            assert!((s - 1.0).abs() < 1e-12, "at {x}: {s}");
        }
    }
}
