//! Small dense linear algebra, written out so that its arithmetic is fixed.
//!
//! The randomized SVD needs only an orthonormalisation of a tall, thin matrix
//! and a symmetric eigendecomposition of a matrix no larger than 16 x 16.
//! Both are a few dozen lines here, so they need no BLAS or LAPACK, compile to
//! wasm unchanged, and give the same bits everywhere: every operation is
//! `+`, `*`, `/` or `sqrt`, each sum runs in a fixed order, and the Jacobi
//! sweep visits the off-diagonal entries in a fixed order.
//!
//! Matrices are row-major `Vec<f64>` with explicit dimensions.

/// Orthonormalise the columns of the `rows x cols` matrix `a` in place, by
/// modified Gram-Schmidt applied twice ("twice is enough": the second pass
/// removes what rounding left behind in the first).
///
/// A column that is, to within `1e-10` of its original length, a combination
/// of the earlier ones is set to zero rather than normalised, so the result
/// has orthonormal or zero columns. That happens when the data have lower rank
/// than the number of columns.
pub fn orthonormalise(a: &mut [f64], rows: usize, cols: usize) {
    debug_assert_eq!(a.len(), rows * cols);
    for j in 0..cols {
        let mut original = 0.0;
        for r in 0..rows {
            original += a[r * cols + j] * a[r * cols + j];
        }
        let original = original.sqrt();
        for _pass in 0..2 {
            for i in 0..j {
                let mut dot = 0.0;
                for r in 0..rows {
                    dot += a[r * cols + i] * a[r * cols + j];
                }
                for r in 0..rows {
                    a[r * cols + j] -= dot * a[r * cols + i];
                }
            }
        }
        let mut norm = 0.0;
        for r in 0..rows {
            norm += a[r * cols + j] * a[r * cols + j];
        }
        let norm = norm.sqrt();
        if norm <= 1e-10 * original || norm == 0.0 {
            for r in 0..rows {
                a[r * cols + j] = 0.0;
            }
        } else {
            for r in 0..rows {
                a[r * cols + j] /= norm;
            }
        }
    }
}

/// Eigendecomposition of the symmetric `n x n` matrix `a` by the cyclic
/// Jacobi method.
///
/// Returns the eigenvalues in descending order and the matching unit
/// eigenvectors as the columns of an `n x n` row-major matrix. Ties keep
/// their original order.
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(a.len(), n * n);
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let mut scale = 0.0;
    for x in &m {
        scale += x * x;
    }

    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off <= 1e-30 * scale || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                // Rotation that zeroes m[p][q]: the smaller root of
                // t^2 + 2 theta t - 1 = 0, for stability.
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[j * n + j].total_cmp(&m[i * n + i]));
    let values = order.iter().map(|&i| m[i * n + i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (c, &i) in order.iter().enumerate() {
        for r in 0..n {
            vectors[r * n + c] = v[r * n + i];
        }
    }
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
        let mut g = SplitMix64::new(seed);
        (0..rows * cols).map(|_| g.normal()).collect()
    }

    #[test]
    fn columns_come_out_orthonormal() {
        let (rows, cols) = (200, 16);
        let mut a = gaussian(rows, cols, 1);
        orthonormalise(&mut a, rows, cols);
        for i in 0..cols {
            for j in 0..cols {
                let mut dot = 0.0;
                for r in 0..rows {
                    dot += a[r * cols + i] * a[r * cols + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-13, "({i}, {j}): {dot}");
            }
        }
    }

    #[test]
    fn dependent_columns_become_zero() {
        let (rows, cols) = (50, 3);
        let mut a = gaussian(rows, cols, 2);
        for r in 0..rows {
            a[r * cols + 2] = 2.0 * a[r * cols] - 0.5 * a[r * cols + 1];
        }
        orthonormalise(&mut a, rows, cols);
        assert!((0..rows).all(|r| a[r * cols + 2] == 0.0));
    }

    #[test]
    fn eigendecomposition_reconstructs_the_matrix() {
        let n = 16;
        let g = gaussian(n, n, 3);
        // A symmetric matrix with a spread of eigenvalues: G G^T.
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    a[i * n + j] += g[i * n + k] * g[j * n + k];
                }
            }
        }
        let (values, vectors) = symmetric_eigen(&a, n);
        assert!(values.windows(2).all(|w| w[0] >= w[1]));
        for i in 0..n {
            for j in 0..n {
                let mut rebuilt = 0.0;
                for k in 0..n {
                    rebuilt += vectors[i * n + k] * values[k] * vectors[j * n + k];
                }
                assert!((rebuilt - a[i * n + j]).abs() < 1e-10 * values[0], "({i}, {j})");
            }
        }
    }
}
