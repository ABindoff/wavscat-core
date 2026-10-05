//! Morlet filter banks, constructed directly in the Fourier domain.
//!
//! A port of `R/filter-bank.R`, itself following Kymatio's
//! `scattering1d/filter_bank.py` (BSD-3-Clause). Frequencies are in cycles per
//! sample, so 0.5 is Nyquist, and every filter is a real vector giving its
//! Fourier transform at the frequencies of `fftfreq(N)`.
//!
//! Expressions are written in the same order as the R source, because the
//! order of floating-point operations is part of the numerical definition.

use crate::backend;
use crate::fft;
use crate::math;

/// One filter, held at several resolutions.
#[derive(Debug, Clone)]
pub struct Filter {
    /// `levels[k]` is the filter periodised for a signal already subsampled by
    /// `2^k`. `levels[0]` is full resolution.
    pub levels: Vec<Vec<f64>>,
    /// Centre frequency in cycles per sample; zero for the low-pass.
    pub xi: f64,
    /// Bandwidth in cycles per sample.
    pub sigma: f64,
    /// Largest dyadic subsampling the filter admits without aliasing.
    pub j: i32,
}

impl Filter {
    /// The filter at subsampling level `k`.
    pub fn level(&self, k: usize) -> &[f64] {
        &self.levels[k]
    }
}

/// The hyperparameters shared by every bank.
#[derive(Debug, Clone, Copy)]
pub struct BankParams {
    /// Standard deviations counted as the band edge when choosing subsampling.
    pub alpha: f64,
    /// Height at which adjacent wavelets cross.
    pub r_psi: f64,
    /// Bandwidth scale; the minimum bandwidth is `sigma0 / 2^J`.
    pub sigma0: f64,
}

impl Default for BankParams {
    fn default() -> Self {
        BankParams { alpha: 5.0, r_psi: std::f64::consts::FRAC_1_SQRT_2, sigma0: 0.1 }
    }
}

/// Which schedule of centre frequencies to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Generator {
    /// Positive frequencies only: time scattering.
    Anden,
    /// Positive then negative frequencies: the spins of joint scattering.
    Spin,
}

/// `numpy.fft.fftfreq(n)`.
pub fn fftfreq(n: usize) -> Vec<f64> {
    (0..n).map(|i| fftfreq_at(n, i)).collect()
}

/// Element `i` of `fftfreq(n)`: `i / n` up to `(n - 1) / 2`, then wrapping
/// to the negative frequencies `-(n - i) / n`.
fn fftfreq_at(n: usize, i: usize) -> f64 {
    let nf = n as f64;
    if i <= (n - 1) / 2 {
        i as f64 / nf
    } else {
        -((n - i) as f64) / nf
    }
}

/// Number of periods needed to build a Morlet without a wrap discontinuity.
pub fn adaptive_choice_p(sigma: f64, eps: f64) -> usize {
    ((-2.0 * (sigma * sigma) * math::ln(eps)).sqrt() + 1.0).ceil() as usize
}

/// Fourier transform of a Morlet wavelet, or of the Gaussian low-pass when
/// `xi` is `None`, normalised to unit L1 norm in time.
pub fn morlet_1d(n: usize, xi: Option<f64>, sigma: f64) -> Vec<f64> {
    let p = adaptive_choice_p(sigma, 1e-7).min(5);
    let nf = n as f64;
    let two_sigma_sq = 2.0 * (sigma * sigma);

    // Frequencies spanning 2P - 1 periods: grid point t is (lo + t) / n, so
    // consecutive blocks of length n are successive periods. With a single
    // period the low-pass is centred on [-0.5, 0.5) to stay continuous across
    // the wrap point; the band-pass is centred on xi and does not need it.
    let lo = (1 - p as i64) * n as i64;
    let blocks = 2 * p - 1;
    let freq = |t: usize| (lo + t as i64) as f64 / nf;

    let low_pass_f = if p == 1 {
        fold_periods(n, 1, |t| {
            let f = fftfreq_at(n, t);
            math::exp(-(f * f) / two_sigma_sq)
        })
    } else {
        fold_periods(n, blocks, |t| {
            let f = freq(t);
            math::exp(-(f * f) / two_sigma_sq)
        })
    };

    let filter_f = match xi {
        Some(xi) if xi != 0.0 => {
            let gabor_f = fold_periods(n, blocks, |t| {
                let f = freq(t);
                math::exp(-((f - xi) * (f - xi)) / two_sigma_sq)
            });
            // Subtracting this multiple of the low-pass forces a zero mean.
            let kappa = gabor_f[0] / low_pass_f[0];
            gabor_f.iter().zip(&low_pass_f).map(|(g, l)| g - kappa * l).collect()
        }
        _ => low_pass_f,
    };

    let l1 = backend::sum(&backend::modulus(&fft::ifft_real(&filter_f)));
    filter_f.iter().map(|v| v / l1).collect()
}

/// Evaluate `g` on a grid of `blocks * n` points and average over the blocks,
/// since discretising in time is periodisation in frequency. Evaluating on the
/// fly avoids materialising a grid of up to nine times the signal length.
fn fold_periods(n: usize, blocks: usize, g: impl Fn(usize) -> f64) -> Vec<f64> {
    let bf = blocks as f64;
    (0..n)
        .map(|i| {
            let mut acc = 0.0;
            for b in 0..blocks {
                acc += g(b * n + i);
            }
            acc / bf
        })
        .collect()
}

/// Fourier transform of a Gaussian low-pass filter.
pub fn gauss_1d(n: usize, sigma: f64) -> Vec<f64> {
    morlet_1d(n, None, sigma)
}

/// Bandwidth at which adjacent wavelets in a bank with `q` per octave cross at
/// height `r`.
pub fn compute_sigma_psi(xi: f64, q: u32, r: f64) -> f64 {
    let factor = 1.0 / math::pow(2.0, 1.0 / q as f64);
    let term1 = (1.0 - factor) / (1.0 + factor);
    let term2 = 1.0 / (2.0 * math::ln(1.0 / r)).sqrt();
    xi * term1 * term2
}

/// Highest centre frequency of the bank.
pub fn compute_xi_max(q: u32) -> f64 {
    f64::max(1.0 / (1.0 + math::pow(2.0, 3.0 / q as f64)), 0.35)
}

/// Largest `j` such that the filter may be subsampled by `2^j`.
pub fn get_max_dyadic_subsampling(xi: f64, sigma: f64, alpha: f64) -> i32 {
    let upper_bound = f64::min(xi.abs() + alpha * sigma, 0.5);
    (-math::log2(upper_bound)).floor() as i32 - 1
}

/// Half temporal support of a filter given in Fourier: the smallest `N` such
/// that truncating it to `[-N, N]` loses at most `criterion` of its L1 mass.
///
/// Returns `None` when the filter is wider than half the transform, in which
/// case the caller warns and uses half the length.
pub fn compute_temporal_support(h_f: &[f64], criterion: f64) -> Option<usize> {
    let h = fft::ifft_real(h_f);
    let half = h_f.len() / 2;
    let mut residual = vec![0.0; half];
    let mut acc = 0.0;
    for i in (0..half).rev() {
        acc += h[i].norm();
        residual[i] = acc;
    }
    residual.iter().position(|&r| r <= criterion).map(|i| i + 1)
}

/// Centre frequencies and bandwidths of a Morlet bank, in descending
/// frequency: constant-Q above an elbow, constant bandwidth below it.
pub fn anden_generator(j: u32, q: u32, sigma0: f64, r_psi: f64) -> Vec<(f64, f64)> {
    let mut xi = compute_xi_max(q);
    let mut sigma = compute_sigma_psi(xi, q, r_psi);
    let sigma_min = sigma0 / math::pow2(j);
    let step = math::pow(2.0, 1.0 / q as f64);
    let mut out = Vec::new();

    if sigma <= sigma_min {
        xi = sigma;
    } else {
        out.push((xi, sigma));
        while sigma > sigma_min * step {
            xi /= step;
            sigma /= step;
            out.push((xi, sigma));
        }
    }

    let elbow_xi = xi;
    for _ in 1..q {
        xi -= elbow_xi / q as f64;
        out.push((xi, sigma_min));
    }
    out
}

/// The Anden bank followed by its mirror at negative frequencies.
pub fn spin_generator(j: u32, q: u32, sigma0: f64, r_psi: f64) -> Vec<(f64, f64)> {
    let base = anden_generator(j, q, sigma0, r_psi);
    let neg: Vec<(f64, f64)> = base.iter().map(|&(xi, s)| (-xi, s)).collect();
    base.into_iter().chain(neg).collect()
}

/// Periodised copies of a filter: element `k` is subsampled by `2^k`.
fn filter_levels(f_full: Vec<f64>, n_levels: i32) -> Vec<Vec<f64>> {
    let mut levels = Vec::with_capacity(n_levels.max(1) as usize);
    for k in 1..n_levels.max(1) {
        levels.push(backend::periodize_sum(&f_full, 1 << k));
    }
    levels.insert(0, f_full);
    levels
}

/// Filter banks for a scattering transform.
#[derive(Debug, Clone)]
pub struct Banks {
    pub phi: Filter,
    /// One bank per entry of `q`.
    pub banks: Vec<Vec<Filter>>,
}

/// Build the low-pass and one wavelet bank per layer.
///
/// `n` is the padded length, `q` holds wavelets per octave for each layer and
/// may be empty, and `t` is the support of the low-pass in samples.
pub fn filter_factory(
    n: usize,
    j: u32,
    q: &[u32],
    t: f64,
    params: BankParams,
    generator: Generator,
) -> Banks {
    let log2_t = math::floor_log2(t);
    let mut max_j = 0i32;
    let mut previous_j = 0i32;
    let mut banks = Vec::with_capacity(q.len());

    for &q_layer in q {
        let spec = match generator {
            Generator::Anden => anden_generator(j, q_layer, params.sigma0, params.r_psi),
            Generator::Spin => spin_generator(j, q_layer, params.sigma0, params.r_psi),
        };
        let bank = spec
            .into_iter()
            .map(|(xi, sigma)| {
                let fj = get_max_dyadic_subsampling(xi, sigma, params.alpha);
                // Coarser copies are only needed after a first-order
                // subsampling, hence previous_j; j is the aliasing limit and
                // 1 + log2_t the output stride.
                let n_levels = previous_j.min(fj).min(1 + log2_t);
                max_j = max_j.max(fj);
                Filter {
                    levels: filter_levels(morlet_1d(n, Some(xi), sigma), n_levels),
                    xi,
                    sigma,
                    j: fj,
                }
            })
            .collect();
        previous_j = max_j;
        banks.push(bank);
    }

    let sigma_low = params.sigma0 / t;
    let phi = Filter {
        levels: filter_levels(gauss_1d(n, sigma_low), previous_j.max(1 + log2_t)),
        xi: 0.0,
        sigma: sigma_low,
        j: log2_t,
    };
    Banks { phi, banks }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fftfreq_matches_numpy() {
        assert_eq!(fftfreq(4), vec![0.0, 0.25, -0.5, -0.25]);
        assert_eq!(fftfreq(5), vec![0.0, 0.2, 0.4, -0.4, -0.2]);
        assert_eq!(fftfreq(1), vec![0.0]);
    }
}

#[cfg(test)]
mod reference {
    use super::*;

    /// `morlet_1d` as first written: materialise the grid, then fold it.
    fn morlet_materialised(n: usize, xi: Option<f64>, sigma: f64) -> Vec<f64> {
        let p = adaptive_choice_p(sigma, 1e-7).min(5);
        let nf = n as f64;
        let two_sigma_sq = 2.0 * (sigma * sigma);
        let lo = (1 - p as i64) * n as i64;
        let hi = p as i64 * n as i64;
        let freqs: Vec<f64> = (lo..hi).map(|i| i as f64 / nf).collect();
        let freqs_low = if p == 1 { fftfreq(n) } else { freqs.clone() };
        let fold = |v: Vec<f64>| -> Vec<f64> {
            let blocks = v.len() / n;
            (0..n)
                .map(|i| {
                    let mut acc = 0.0;
                    for b in 0..blocks {
                        acc += v[b * n + i];
                    }
                    acc / blocks as f64
                })
                .collect()
        };
        let low = fold(freqs_low.iter().map(|&f| math::exp(-(f * f) / two_sigma_sq)).collect());
        let filt: Vec<f64> = match xi {
            Some(xi) if xi != 0.0 => {
                let g = fold(freqs.iter().map(|&f| math::exp(-((f - xi) * (f - xi)) / two_sigma_sq)).collect());
                let kappa = g[0] / low[0];
                g.iter().zip(&low).map(|(g, l)| g - kappa * l).collect()
            }
            _ => low,
        };
        let l1 = backend::sum(&backend::modulus(&fft::ifft_real(&filt)));
        filt.iter().map(|v| v / l1).collect()
    }

    #[test]
    fn on_the_fly_folding_is_bit_identical() {
        // Single-period and multi-period cases, odd and even lengths.
        for &(n, xi, sigma) in &[
            (256, Some(0.4), 0.0208),
            (512, Some(0.35), 0.15),
            (1024, Some(0.01), 0.0005),
            (128, Some(0.3), 0.25),
            (255, Some(-0.2), 0.05),
            (256, None, 0.0125),
            (1023, None, 0.3),
        ] {
            let got = morlet_1d(n, xi, sigma);
            let want = morlet_materialised(n, xi, sigma);
            assert!(
                got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()),
                "n = {n}, xi = {xi:?}, sigma = {sigma}"
            );
        }
    }
}
