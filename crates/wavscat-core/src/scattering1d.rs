//! One-dimensional time scattering: operator construction and the cascade.
//!
//! A port of `R/op-1d.R` and `R/cascade.R`. Building the operator is the
//! expensive step; apply it to as many signals of length `n` as you like.

use crate::backend::{filter_periodize, modulus, sum};
use crate::complex::C64;
use crate::error::{fail, Error};
use crate::fft;
use crate::filter_bank::{self, BankParams, Filter, Generator};
use crate::math;
use crate::pad::{compute_padding, pad_reflect, Borders};

/// The averaging support, as the R argument `T` accepts it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TSpec {
    /// `T = NULL`: average over `2^J` samples.
    Default,
    /// A support in samples. Zero disables averaging.
    Samples(f64),
    /// `T = "global"`: sum each path over the whole padded signal.
    Global,
}

/// How the cascade output is averaged in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Averaging {
    /// Low-pass filtered and subsampled to the output stride.
    Local,
    /// Summed over the padded signal, one value per path.
    Global,
    /// Not averaged: each path at the resolution of its widest wavelet.
    None,
}

/// Arguments to [`Scattering1d::new`], mirroring `scattering_1d()` in R.
#[derive(Debug, Clone)]
pub struct Params1d {
    pub n: usize,
    pub j: u32,
    /// One value, applied to order one with order two set to 1, or two values.
    pub q: Vec<u32>,
    pub t: TSpec,
    /// Averaging support in seconds; requires `sr` and overrides `t`.
    pub t_sec: Option<f64>,
    pub max_order: u8,
    /// Output subsampling factor, a power of two.
    pub stride: Option<f64>,
    /// Sampling rate in Hz, used only to interpret `t_sec`.
    pub sr: Option<f64>,
}

impl Params1d {
    pub fn new(n: usize, j: u32, q: Vec<u32>) -> Self {
        Params1d { n, j, q, t: TSpec::Default, t_sec: None, max_order: 2, stride: None, sr: None }
    }
}

/// One scattering path. Filter indices are one-based, as in R, so that labels
/// agree in every language.
#[derive(Debug, Clone, PartialEq)]
pub struct Path1d {
    pub order: u8,
    pub n1: Option<usize>,
    pub n2: Option<usize>,
    pub j1: Option<i32>,
    pub j2: Option<i32>,
    pub xi1: Option<f64>,
    pub xi2: Option<f64>,
    pub sigma1: Option<f64>,
    pub sigma2: Option<f64>,
    /// `S0`, `S1_<n1>` or `S2_<n1>_<n2>`.
    pub label: String,
}

/// A time scattering operator for signals of a fixed length.
#[derive(Debug, Clone)]
pub struct Scattering1d {
    pub n: usize,
    pub j: u32,
    pub q: [u32; 2],
    pub t: f64,
    pub max_order: u8,
    pub average: Averaging,
    pub log2_t: i32,
    pub log2_stride: i32,
    pub n_padded: usize,
    pub pad_left: usize,
    pub pad_right: usize,
    pub borders: Borders,
    pub phi: Filter,
    pub psi1: Vec<Filter>,
    pub psi2: Vec<Filter>,
    pub bank_params: BankParams,
    pub paths: Vec<Path1d>,
    /// Non-fatal problems found while building, for the caller to surface.
    pub warnings: Vec<String>,
}

pub(crate) fn check_q(q: &[u32]) -> Result<[u32; 2], Error> {
    match q {
        [q1] if *q1 >= 1 => Ok([*q1, 1]),
        [q1, q2] if *q1 >= 1 && *q2 >= 1 => Ok([*q1, *q2]),
        [_] | [_, _] => fail("Q must be at least 1."),
        _ => fail("Q must be one or two numbers."),
    }
}

/// Resolve `T` into a support in samples and an averaging mode.
pub(crate) fn parse_t(t: TSpec, j: u32, n: usize, arg: &str) -> Result<(f64, Averaging), Error> {
    let default = math::pow2(j);
    match t {
        TSpec::Default => Ok((default, Averaging::Local)),
        TSpec::Global => Ok((default, Averaging::Global)),
        TSpec::Samples(v) if v == 0.0 => Ok((default, Averaging::None)),
        TSpec::Samples(v) if !(v >= 1.0) => {
            fail(format!("{arg} must be 0 or at least 1, got {v}."))
        }
        TSpec::Samples(v) if v > n as f64 => fail(format!(
            "{arg} = {v} exceeds the signal length {n}. For averaging over the whole signal \
             use {arg} = \"global\"."
        )),
        TSpec::Samples(v) => Ok((v, Averaging::Local)),
    }
}

/// Resolve a stride (or `stride_fr`) into its base-two logarithm.
pub(crate) fn parse_stride(
    stride: Option<f64>,
    log2_t: i32,
    average: Averaging,
    arg: &str,
    support: &str,
) -> Result<i32, Error> {
    let Some(stride) = stride else { return Ok(log2_t) };
    if average != Averaging::Local {
        return fail(format!(
            "{arg} only applies when averaging is local; it is incompatible with {support} = 0 \
             and {support} = \"global\"."
        ));
    }
    if !(stride >= 1.0) || !stride.is_finite() || stride.fract() != 0.0
        || !(stride as u64).is_power_of_two()
    {
        return fail(format!("{arg} must be a power of two, got {stride}."));
    }
    let v = math::floor_log2(stride);
    if v > log2_t {
        return fail(format!(
            "{arg} = {stride} is coarser than the averaging support {support}; that would alias. \
             Use {arg} <= 2^{log2_t}."
        ));
    }
    Ok(v)
}

impl Scattering1d {
    /// Build the filter banks and the path table.
    pub fn new(p: &Params1d) -> Result<Scattering1d, Error> {
        let n = p.n;
        let j = p.j;
        if n < 1 {
            return fail("n must be a single positive whole number.");
        }
        if j < 1 {
            return fail("J must be a single positive whole number.");
        }
        if j >= 63 || math::pow2(j) > n as f64 {
            return fail(format!(
                "J = {j} implies a maximum wavelet support of 2^{j} samples, which exceeds the \
                 signal length n = {n}. Use J <= {}.",
                math::floor_log2_int(n as u64)
            ));
        }
        let q = check_q(&p.q)?;
        if !(p.max_order == 1 || p.max_order == 2) {
            return fail(format!("max_order must be 1 or 2, got {}.", p.max_order));
        }
        if let Some(sr) = p.sr {
            if !(sr > 0.0) || !sr.is_finite() {
                return fail("sr must be a single positive number, or NULL.");
            }
        }
        let t_spec = match p.t_sec {
            Some(t_sec) => {
                let Some(sr) = p.sr else {
                    return fail("T_sec needs sr, so that seconds can be converted to samples.");
                };
                // R's round() breaks ties to even.
                TSpec::Samples((t_sec * sr).round_ties_even())
            }
            None => p.t,
        };
        let (t, average) = parse_t(t_spec, j, n, "T")?;
        let log2_t = math::floor_log2(t);
        let log2_stride = parse_stride(p.stride, log2_t, average, "stride", "T")?;

        let bank_params = BankParams::default();
        let mut warnings = Vec::new();

        // Pad by three times the half support of the averaging filter, but
        // never so far that the reflection would have to repeat.
        let support = filter_bank::compute_temporal_support(
            &filter_bank::gauss_1d(n, bank_params.sigma0 / t),
            1e-3,
        )
        .unwrap_or_else(|| {
            warnings.push(
                "The averaging filter is wider than the signal, so the transform cannot be padded \
                 enough to avoid border effects. Reduce T, or use a longer signal."
                    .to_string(),
            );
            n / 2
        });
        let min_to_pad = 3 * support;
        let j_max_support = math::floor_log2_int(3 * n as u64 - 2);
        let j_pad = math::ceil_log2_int((n + 2 * min_to_pad) as u64).min(j_max_support);
        let n_padded = 1usize << j_pad;

        let (pad_left, pad_right) = compute_padding(n_padded, n)?;
        let borders = Borders::new(log2_t, j, pad_left, pad_left + n);

        let mut banks = filter_bank::filter_factory(
            n_padded, j, &q, t, bank_params, Generator::Anden,
        );
        let psi2 = banks.banks.pop().expect("two banks");
        let psi1 = banks.banks.pop().expect("two banks");

        let mut op = Scattering1d {
            n, j, q, t, max_order: p.max_order, average, log2_t, log2_stride,
            n_padded, pad_left, pad_right, borders, phi: banks.phi, psi1, psi2,
            bank_params, paths: Vec::new(), warnings,
        };
        op.paths = op.enumerate_paths();
        Ok(op)
    }

    /// The paths in output order: by order, then by first and second filter.
    fn enumerate_paths(&self) -> Vec<Path1d> {
        let mut paths = vec![Path1d {
            order: 0, n1: None, n2: None, j1: None, j2: None, xi1: None, xi2: None,
            sigma1: None, sigma2: None, label: "S0".to_string(),
        }];
        for (i1, f1) in self.psi1.iter().enumerate() {
            paths.push(Path1d {
                order: 1, n1: Some(i1 + 1), n2: None, j1: Some(f1.j), j2: None,
                xi1: Some(f1.xi), xi2: None, sigma1: Some(f1.sigma), sigma2: None,
                label: format!("S1_{}", i1 + 1),
            });
        }
        if self.max_order >= 2 {
            for (i1, f1) in self.psi1.iter().enumerate() {
                for (i2, f2) in self.psi2.iter().enumerate() {
                    if f2.j > f1.j {
                        paths.push(Path1d {
                            order: 2, n1: Some(i1 + 1), n2: Some(i2 + 1),
                            j1: Some(f1.j), j2: Some(f2.j), xi1: Some(f1.xi), xi2: Some(f2.xi),
                            sigma1: Some(f1.sigma), sigma2: Some(f2.sigma),
                            label: format!("S2_{}_{}", i1 + 1, i2 + 1),
                        });
                    }
                }
            }
        }
        paths
    }

    /// Transform one signal of length `n`.
    ///
    /// Returns one coefficient vector per path, in the order of
    /// [`Scattering1d::paths`]. With local averaging every vector has the same
    /// length; with global averaging each has length one.
    pub fn transform(&self, x: &[f64]) -> Result<Vec<Vec<f64>>, Error> {
        if x.len() != self.n {
            return fail(format!(
                "The operator was built for signals of length {}, got {}.",
                self.n,
                x.len()
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return fail("The signal contains missing or non-finite values.");
        }

        let local = self.average == Averaging::Local;
        let second = self.max_order >= 2;
        let stride = self.log2_stride;

        // Each path at its native resolution, with the level it sits at.
        let mut order1: Vec<(Vec<f64>, usize)> = Vec::with_capacity(self.psi1.len());
        let mut order2: Vec<(Vec<f64>, usize)> = Vec::new();

        let u0 = pad_reflect(x, self.pad_left, self.pad_right);
        let u0_hat = fft::fft_real(&u0);

        let s0 = if local {
            (inverse_real(filter_periodize(&u0_hat, self.phi.level(0), 1 << stride)), stride as usize)
        } else {
            (u0, 0)
        };

        for f1 in &self.psi1 {
            let j1 = f1.j;
            // Subsample as far as the bandwidth allows, but no further than
            // the output stride.
            let k1 = if local { j1.min(stride) } else { j1 };

            let u1_m = modulus(&inverse(filter_periodize(&u0_hat, f1.level(0), 1 << k1)));
            let u1_hat = if local || second { Some(fft::fft_real(&u1_m)) } else { None };

            if local {
                let s1 = inverse_real(filter_periodize(
                    u1_hat.as_ref().unwrap(),
                    self.phi.level(k1 as usize),
                    1 << (stride - k1).max(0),
                ));
                order1.push((s1, stride as usize));
            } else {
                order1.push((u1_m, j1 as usize));
            }

            if !second {
                continue;
            }
            let u1_hat = u1_hat.unwrap();
            for f2 in &self.psi2 {
                let j2 = f2.j;
                // The envelope of psi_j1 carries nothing above its own
                // bandwidth, so these second-order paths are skipped.
                if j2 <= j1 {
                    continue;
                }
                let sub2_adj = if local { j2.min(stride) } else { j2 };
                let k2 = (sub2_adj - k1).max(0);
                let u2_m = modulus(&inverse(filter_periodize(
                    &u1_hat,
                    f2.level(k1 as usize),
                    1 << k2,
                )));
                if local {
                    let s2 = inverse_real(filter_periodize(
                        &fft::fft_real(&u2_m),
                        self.phi.level((k1 + k2) as usize),
                        1 << (stride - sub2_adj).max(0),
                    ));
                    order2.push((s2, stride as usize));
                } else {
                    order2.push((u2_m, j2 as usize));
                }
            }
        }

        let finish = |(coef, level): (Vec<f64>, usize)| -> Vec<f64> {
            match self.average {
                // Sums over the padded signal, matching Kymatio.
                Averaging::Global => vec![sum(&coef)],
                _ => self.borders.unpad(&coef, level).to_vec(),
            }
        };
        let mut out = Vec::with_capacity(self.paths.len());
        out.push(finish(s0));
        out.extend(order1.into_iter().map(finish));
        out.extend(order2.into_iter().map(finish));
        debug_assert_eq!(out.len(), self.paths.len());
        Ok(out)
    }
}

fn inverse(mut x: Vec<C64>) -> Vec<C64> {
    fft::ifft(&mut x);
    x
}

fn inverse_real(x: Vec<C64>) -> Vec<f64> {
    fft::ifft_real_part(x)
}
