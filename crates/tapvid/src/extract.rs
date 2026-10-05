//! Extracting the tapping signal: the combination of components whose power
//! lies most in the tapping band.
//!
//! The SVD of stage 3 orders components by how much the pixels vary, not by
//! rhythm. People sway, shift their arm, nod and tremble, and a large slow
//! movement can outrank the fingers; fast tapping is often spread over
//! several components. Picking one component then loses part of the taps,
//! and anything else that moves competes with them.
//!
//! So instead of one component, take the weighted sum of all of them that
//! puts the largest share of its power in narrow bands around the tapping
//! rate and its second harmonic: a generalised eigenproblem, `C_s w = l C_t
//! w`, with `C_t` the covariance of the components and `C_s` that of their
//! band-passed analytic signals (spatio-spectral decomposition, as in EEG).
//! Every component that carries the tapping contributes; motion outside the
//! band, slow sway or repositioning or a tremor at a clearly different rate,
//! is pushed down rather than competing. The combination is linear, so the
//! timing of each cycle is not distorted.
//!
//! Everything is a sum in a fixed order and a Jacobi eigendecomposition, so
//! the result is bit-identical on every target.

use wavscat_core::Error;

use crate::linalg::symmetric_eigen;
use crate::phase::analytic_band;

/// Settings for [`extract`].
#[derive(Debug, Clone)]
pub struct ExtractParams {
    /// Standard deviation of each band-pass, as a fraction of `f0`. Wide
    /// enough to hold a rate that drifts by a tenth over the trial.
    pub bandwidth: f64,
    /// Harmonics of `f0` in the signal band: 2 takes the fundamental and the
    /// second harmonic, since a fingertip crossing a pixel can change it
    /// twice a cycle.
    pub harmonics: u32,
    /// Directions of the components' covariance with less than this fraction
    /// of its largest eigenvalue are dropped, as numerically empty.
    pub min_eigen: f64,
}

impl Default for ExtractParams {
    fn default() -> Self {
        ExtractParams { bandwidth: 0.25, harmonics: 2, min_eigen: 1e-9 }
    }
}

/// The extracted tapping signal.
#[derive(Debug, Clone)]
pub struct Extraction {
    /// Weight of each component in the sum: a filter, which can be large on
    /// weak components and so says little about where the movement is.
    pub weights: Vec<f64>,
    /// The signal's pattern over the components, `C_t w`: how much of each
    /// component moves with it. This, not the weights, maps where the
    /// tapping is.
    pub pattern: Vec<f64>,
    /// The weighted sum, sign fixed to correlate positively with the
    /// reference component.
    pub signal: Vec<f64>,
    /// Share of the signal's power in the tapping bands, from 0 to about 1.
    pub band_fraction: f64,
    /// The signal's power in each harmonic band, fundamental first.
    pub harmonic_power: Vec<f64>,
}

impl Extraction {
    /// The harmonic that carries most of the signal's tapping power, to time
    /// the cycles from.
    pub fn timing_harmonic(&self) -> u32 {
        let mut best = 0;
        for (h, p) in self.harmonic_power.iter().enumerate() {
            if *p > self.harmonic_power[best] {
                best = h;
            }
        }
        best as u32 + 1
    }
}

/// Covariance `sum x_i x_j / n` of the rows of `x`, row-major `k x k`.
fn covariance(x: &[Vec<f64>]) -> Vec<f64> {
    let (k, n) = (x.len(), x[0].len());
    let mut c = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let mut acc = 0.0;
            for t in 0..n {
                acc += x[i][t] * x[j][t];
            }
            c[i * k + j] = acc / n as f64;
            c[j * k + i] = c[i * k + j];
        }
    }
    c
}

/// Extract the tapping at fundamental `f0` (Hz) from `components`, each
/// uniformly sampled at `fs` Hz, with the sign of `components[reference]`.
pub fn extract(components: &[Vec<f64>], fs: f64, f0: f64, reference: usize, p: &ExtractParams) -> Result<Extraction, Error> {
    let k = components.len();
    if k == 0 || reference >= k {
        return Err(Error("No components to extract from.".into()));
    }
    let n = components[0].len();
    if n < 8 || components.iter().any(|c| c.len() != n) {
        return Err(Error("Components must share one grid of at least eight samples.".into()));
    }
    if !(f0 > 0.0) || !(p.bandwidth > 0.0) || p.harmonics == 0 {
        return Err(Error("Need f0 > 0, bandwidth > 0 and harmonics >= 1.".into()));
    }
    // Centre each component.
    let x: Vec<Vec<f64>> = components
        .iter()
        .map(|c| {
            let m = c.iter().sum::<f64>() / n as f64;
            c.iter().map(|v| v - m).collect()
        })
        .collect();
    let ct = covariance(&x);

    // Band covariance per harmonic: Re(sum z_i conj(z_j)) / 2n, the power of
    // the real band-passed signals.
    let harmonics: Vec<u32> = (1..=p.harmonics).filter(|h| (*h as f64) * f0 < fs / 2.0).collect();
    let mut cs_h = Vec::with_capacity(harmonics.len());
    for &h in &harmonics {
        let fc = h as f64 * f0;
        let z: Vec<_> = x.iter().map(|c| analytic_band(c, fs, fc, p.bandwidth * f0, 3.0 * h as f64)).collect();
        let mut c = vec![0.0; k * k];
        for i in 0..k {
            for j in i..k {
                let mut acc = 0.0;
                for t in 0..n {
                    acc += z[i][t].re * z[j][t].re + z[i][t].im * z[j][t].im;
                }
                c[i * k + j] = acc / (2.0 * n as f64);
                c[j * k + i] = c[i * k + j];
            }
        }
        cs_h.push(c);
    }
    let mut cs = vec![0.0; k * k];
    for c in &cs_h {
        for (a, b) in cs.iter_mut().zip(c) {
            *a += b;
        }
    }

    // Whiten C_t: W = U diag(1 / sqrt(l)) over the kept directions.
    let (lt, ut) = symmetric_eigen(&ct, k);
    let top = lt[0];
    if !(top > 0.0) {
        return Err(Error("The components carry no variance.".into()));
    }
    let kept: Vec<usize> = (0..k).filter(|&i| lt[i] > p.min_eigen * top).collect();
    let r = kept.len();
    let mut w = vec![0.0; k * r];
    for (col, &e) in kept.iter().enumerate() {
        let s = 1.0 / lt[e].sqrt();
        for row in 0..k {
            w[row * r + col] = ut[row * k + e] * s;
        }
    }
    // M = W' C_s W, and its leading eigenvector.
    let mut cw = vec![0.0; k * r];
    for i in 0..k {
        for col in 0..r {
            let mut acc = 0.0;
            for j in 0..k {
                acc += cs[i * k + j] * w[j * r + col];
            }
            cw[i * r + col] = acc;
        }
    }
    let mut m = vec![0.0; r * r];
    for a in 0..r {
        for b in a..r {
            let mut acc = 0.0;
            for i in 0..k {
                acc += w[i * r + a] * cw[i * r + b];
            }
            m[a * r + b] = acc;
            m[b * r + a] = acc;
        }
    }
    let (lm, um) = symmetric_eigen(&m, r);
    let mut weights = vec![0.0; k];
    for i in 0..k {
        let mut acc = 0.0;
        for col in 0..r {
            acc += w[i * r + col] * um[col * r];
        }
        weights[i] = acc;
    }

    let mut signal = vec![0.0; n];
    for (wi, c) in weights.iter().zip(&x) {
        for (s, v) in signal.iter_mut().zip(c) {
            *s += wi * v;
        }
    }
    // Sign: positive correlation with the reference component.
    let mut dot = 0.0;
    for (s, v) in signal.iter().zip(&x[reference]) {
        dot += s * v;
    }
    if dot < 0.0 {
        for v in signal.iter_mut() {
            *v = -*v;
        }
        for v in weights.iter_mut() {
            *v = -*v;
        }
    }
    let quad = |c: &[f64]| {
        let mut acc = 0.0;
        for i in 0..k {
            for j in 0..k {
                acc += weights[i] * c[i * k + j] * weights[j];
            }
        }
        acc
    };
    let harmonic_power = cs_h.iter().map(|c| quad(c)).collect();
    let pattern = (0..k)
        .map(|i| {
            let mut acc = 0.0;
            for j in 0..k {
                acc += ct[i * k + j] * weights[j];
            }
            acc
        })
        .collect();
    Ok(Extraction { weights, pattern, signal, band_fraction: lm[0], harmonic_power })
}
