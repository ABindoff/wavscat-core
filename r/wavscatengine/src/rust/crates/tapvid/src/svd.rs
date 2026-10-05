//! Stage 3: the randomized SVD of the frames-by-pixels matrix.
//!
//! The dominant tapping oscillator is one of the leading temporal components
//! of the low-resolution video. The Halko-Martinsson-Tropp range finder gets
//! those components without a full SVD:
//!
//! 1. draw a Gaussian test matrix `Omega` (pixels by `rank + oversample`) from
//!    the in-crate generator, filled row by row from `seed`;
//! 2. `Y = A Omega`, orthonormalised to `Q`;
//! 3. `power_iters` times: `Q = orth(A orth(A^T Q))`, re-orthonormalising
//!    after each product, which sharpens the separation of the leading
//!    components;
//! 4. `B = Q^T A`, and the eigendecomposition `B B^T = U_b Lambda U_b^T`;
//! 5. singular values `sqrt(Lambda)`, temporal scores `Q U_b Sigma` (the left
//!    singular vectors times their singular values), spatial loadings
//!    `B^T U_b Sigma^-1`.
//!
//! The SVD does not care about time spacing, so it runs on frames as
//! captured, before any resampling.
//!
//! Each component's sign is then fixed: it is flipped so that the spatial
//! loading with the largest magnitude is positive, the first such pixel on a
//! tie. Without this, identical inputs could give components of opposite sign
//! on different runs, and every downstream phase would shift by pi.
//!
//! The matrix is reached only through the two products of [`Operator`], so
//! frames can stay in `f32` and preprocessing can be applied on the fly.
//! Products accumulate in `f64`, in a fixed order, with the test-vector index
//! innermost so the compiler can vectorise across it without changing any
//! lane's arithmetic.

use wavscat_core::Error;

use crate::linalg::{orthonormalise, symmetric_eigen};
use crate::rng::SplitMix64;

/// A `rows x cols` matrix, seen only through its products with thin
/// matrices. Thin matrices are row-major with `k` columns.
pub trait Operator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// `out = A x`, for `x` of size `cols x k`; `out` is `rows x k`.
    fn mul(&self, x: &[f64], k: usize, out: &mut [f64]);
    /// `out = A^T y`, for `y` of size `rows x k`; `out` is `cols x k`.
    fn mul_t(&self, y: &[f64], k: usize, out: &mut [f64]);
}

/// A matrix of `f32` stored as rows, such as frames of pixels. Every row
/// source is an [`Operator`], with products accumulated in `f64` in a fixed
/// order: by row, then by column, with the thin matrix's column innermost.
pub trait RowSource {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    /// Row `r`, of length `n_cols()`.
    fn row(&self, r: usize) -> &[f32];
}

impl<T: RowSource> Operator for T {
    fn rows(&self) -> usize {
        self.n_rows()
    }

    fn cols(&self) -> usize {
        self.n_cols()
    }

    fn mul(&self, x: &[f64], k: usize, out: &mut [f64]) {
        for r in 0..self.n_rows() {
            let acc = &mut out[r * k..(r + 1) * k];
            acc.fill(0.0);
            for (c, &a) in self.row(r).iter().enumerate() {
                let a = a as f64;
                for (o, xv) in acc.iter_mut().zip(&x[c * k..(c + 1) * k]) {
                    *o += a * xv;
                }
            }
        }
    }

    fn mul_t(&self, y: &[f64], k: usize, out: &mut [f64]) {
        out.fill(0.0);
        for r in 0..self.n_rows() {
            let yr = &y[r * k..(r + 1) * k];
            for (c, &a) in self.row(r).iter().enumerate() {
                let a = a as f64;
                for (o, yv) in out[c * k..(c + 1) * k].iter_mut().zip(yr) {
                    *o += a * yv;
                }
            }
        }
    }
}

/// A dense row-major matrix of `f32`, rows being frames and columns pixels.
pub struct DenseF32<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

impl RowSource for DenseF32<'_> {
    fn n_rows(&self) -> usize {
        self.rows
    }

    fn n_cols(&self) -> usize {
        self.cols
    }

    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// Settings for [`randomized_svd`]. The defaults are the brief's starting
/// point.
#[derive(Debug, Clone)]
pub struct SvdParams {
    /// Number of components returned.
    pub rank: usize,
    /// Extra test vectors beyond `rank`, which make the leading components
    /// accurate.
    pub oversample: usize,
    /// Power iterations.
    pub power_iters: usize,
    /// Seed of the Gaussian test matrix. Part of the parameters, and so of
    /// any parameter hash.
    pub seed: u64,
    /// Whether to return the spatial loadings. They show where the hand is,
    /// which helps QC, but are mildly identifying, so they are off by default.
    pub return_loadings: bool,
}

impl Default for SvdParams {
    fn default() -> Self {
        SvdParams { rank: 8, oversample: 8, power_iters: 2, seed: 0x7a9_5c47_d0e5, return_loadings: false }
    }
}

/// The leading components.
#[derive(Debug, Clone)]
pub struct Svd {
    /// Singular values, in descending order.
    pub singular_values: Vec<f64>,
    /// `scores[i]` is component `i`'s temporal score, one value per frame:
    /// the left singular vector times its singular value.
    pub scores: Vec<Vec<f64>>,
    /// `loadings[i]` is component `i`'s spatial loading, one value per pixel,
    /// with unit norm. Present only when requested.
    pub loadings: Option<Vec<Vec<f64>>>,
}

/// The `rank` leading singular triplets of `a`.
pub fn randomized_svd(a: &impl Operator, p: &SvdParams) -> Result<Svd, Error> {
    let (m, n) = (a.rows(), a.cols());
    if m == 0 || n == 0 {
        return Err(Error("The matrix is empty.".into()));
    }
    if p.rank == 0 {
        return Err(Error("rank must be at least 1.".into()));
    }
    let l = (p.rank + p.oversample).min(m).min(n);
    let k = p.rank.min(l);

    // Test matrix, filled row by row: entry (pixel j, vector c) is draw
    // j * l + c.
    let mut rng = SplitMix64::new(p.seed);
    let omega: Vec<f64> = (0..n * l).map(|_| rng.normal()).collect();

    let mut q = vec![0.0; m * l];
    a.mul(&omega, l, &mut q);
    orthonormalise(&mut q, m, l);
    let mut z = vec![0.0; n * l];
    for _ in 0..p.power_iters {
        a.mul_t(&q, l, &mut z);
        orthonormalise(&mut z, n, l);
        a.mul(&z, l, &mut q);
        orthonormalise(&mut q, m, l);
    }

    // B^T = A^T Q (n x l), and B B^T (l x l).
    let mut bt = vec![0.0; n * l];
    a.mul_t(&q, l, &mut bt);
    let mut bbt = vec![0.0; l * l];
    for j in 0..n {
        let row = &bt[j * l..(j + 1) * l];
        for x in 0..l {
            for y in 0..l {
                bbt[x * l + y] += row[x] * row[y];
            }
        }
    }
    let (lambda, ub) = symmetric_eigen(&bbt, l);
    let sigma: Vec<f64> = lambda.iter().take(k).map(|&v| v.max(0.0).sqrt()).collect();

    // Scores Q U_b Sigma, and loadings B^T U_b Sigma^-1, for the first k.
    let mut scores = vec![vec![0.0; m]; k];
    for r in 0..m {
        for (i, s) in sigma.iter().enumerate() {
            let mut acc = 0.0;
            for a_ in 0..l {
                acc += q[r * l + a_] * ub[a_ * l + i];
            }
            scores[i][r] = acc * s;
        }
    }
    let mut loadings = vec![vec![0.0; n]; k];
    for j in 0..n {
        for (i, s) in sigma.iter().enumerate() {
            if *s > 0.0 {
                let mut acc = 0.0;
                for a_ in 0..l {
                    acc += bt[j * l + a_] * ub[a_ * l + i];
                }
                loadings[i][j] = acc / s;
            }
        }
    }

    // Fix each sign: the largest-magnitude loading is positive.
    for i in 0..k {
        let mut best = 0;
        for j in 1..n {
            if loadings[i][j].abs() > loadings[i][best].abs() {
                best = j;
            }
        }
        if loadings[i][best] < 0.0 {
            for v in loadings[i].iter_mut() {
                *v = -*v;
            }
            for v in scores[i].iter_mut() {
                *v = -*v;
            }
        }
    }

    Ok(Svd {
        singular_values: sigma,
        scores,
        loadings: if p.return_loadings { Some(loadings) } else { None },
    })
}
