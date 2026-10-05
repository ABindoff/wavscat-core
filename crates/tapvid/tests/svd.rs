//! Acceptance tests for the randomized SVD.

use tapvid::linalg::{orthonormalise, symmetric_eigen};
use tapvid::rng::SplitMix64;
use tapvid::svd::{randomized_svd, DenseF32, Svd, SvdParams};

fn gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut g = SplitMix64::new(seed);
    (0..rows * cols).map(|_| g.normal()).collect()
}

/// `U diag(s) V^T` with random orthonormal `U` (m x r) and `V` (n x r),
/// stored as f32 like video frames, and the factors themselves.
fn low_rank(m: usize, n: usize, s: &[f64], seed: u64) -> (Vec<f32>, Vec<f64>, Vec<f64>) {
    let r = s.len();
    let mut u = gaussian(m, r, seed);
    let mut v = gaussian(n, r, seed + 1);
    orthonormalise(&mut u, m, r);
    orthonormalise(&mut v, n, r);
    let mut a = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..r {
                acc += u[i * r + k] * s[k] * v[j * r + k];
            }
            a[i * n + j] = acc as f32;
        }
    }
    (a, u, v)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn run(a: &[f32], m: usize, n: usize, p: &SvdParams) -> Svd {
    randomized_svd(&DenseF32 { data: a, rows: m, cols: n }, p).unwrap()
}

#[test]
fn recovers_an_exactly_low_rank_matrix() {
    let (m, n) = (300, 500);
    let s = [50.0, 20.0, 10.0, 5.0, 2.0];
    let (a, u, v) = low_rank(m, n, &s, 1);
    let p = SvdParams { return_loadings: true, ..SvdParams::default() };
    let svd = run(&a, m, n, &p);
    let loadings = svd.loadings.as_ref().unwrap();
    for (k, &want) in s.iter().enumerate() {
        let got = svd.singular_values[k];
        // f32 storage limits agreement to about 1e-7 of the largest value.
        assert!((got - want).abs() < 1e-5 * s[0], "sigma {k}: {got} vs {want}");
        let uk: Vec<f64> = svd.scores[k].iter().map(|x| x / got).collect();
        let ut: Vec<f64> = (0..m).map(|i| u[i * s.len() + k]).collect();
        let vt: Vec<f64> = (0..n).map(|j| v[j * s.len() + k]).collect();
        assert!(dot(&uk, &ut).abs() > 1.0 - 1e-6, "left vector {k}");
        assert!(dot(&loadings[k], &vt).abs() > 1.0 - 1e-6, "right vector {k}");
    }
    // Components beyond the rank carry nothing.
    for k in s.len()..svd.singular_values.len() {
        assert!(svd.singular_values[k] < 1e-4, "sigma {k}: {}", svd.singular_values[k]);
    }
}

#[test]
fn agrees_with_an_exact_svd_on_a_full_rank_matrix() {
    let (m, n) = (120, 80);
    // Geometric decay, the hard case for a randomized method.
    let mut s = vec![10.0];
    for i in 1..n {
        s.push(s[i - 1] * 0.8);
    }
    let (a, _, _) = low_rank(m, n, &s, 3);

    // Reference: eigenvalues of A^T A, by full Jacobi.
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for r in 0..m {
                ata[i * n + j] += a[r * n + i] as f64 * a[r * n + j] as f64;
            }
        }
    }
    let (lambda, _) = symmetric_eigen(&ata, n);
    let svd = run(&a, m, n, &SvdParams::default());
    for k in 0..8 {
        let exact = lambda[k].sqrt();
        let got = svd.singular_values[k];
        assert!((got - exact).abs() < 1e-3 * exact, "sigma {k}: {got} vs {exact}");
    }
}

#[test]
fn the_sign_convention_is_deterministic() {
    let (m, n) = (200, 300);
    let (a, _, _) = low_rank(m, n, &[30.0, 12.0, 6.0, 3.0], 5);
    let neg: Vec<f32> = a.iter().map(|x| -x).collect();
    let p = SvdParams { return_loadings: true, ..SvdParams::default() };
    let (s1, s2) = (run(&a, m, n, &p), run(&neg, m, n, &p));
    for k in 0..4 {
        let (l1, l2) = (&s1.loadings.as_ref().unwrap()[k], &s2.loadings.as_ref().unwrap()[k]);
        // The largest-magnitude loading is positive...
        let big = l1.iter().copied().fold(0.0f64, |acc, x| if x.abs() > acc.abs() { x } else { acc });
        assert!(big > 0.0, "component {k}");
        // ...so negating the data negates the scores, not the loadings.
        for (x, y) in l1.iter().zip(l2) {
            assert!((x - y).abs() < 1e-9, "component {k} loading");
        }
        for (x, y) in s1.scores[k].iter().zip(&s2.scores[k]) {
            assert!((x + y).abs() < 1e-9 * s1.singular_values[0], "component {k} score");
        }
    }
}

#[test]
fn the_seed_changes_rounding_not_the_answer() {
    let (m, n) = (200, 300);
    let (a, _, _) = low_rank(m, n, &[30.0, 12.0, 6.0, 3.0], 7);
    let p1 = SvdParams::default();
    let p2 = SvdParams { seed: 12345, ..SvdParams::default() };
    let (s1, s2) = (run(&a, m, n, &p1), run(&a, m, n, &p2));
    for k in 0..4 {
        assert!((s1.singular_values[k] - s2.singular_values[k]).abs() < 1e-9 * s1.singular_values[0]);
        for (x, y) in s1.scores[k].iter().zip(&s2.scores[k]) {
            assert!((x - y).abs() < 1e-7 * s1.singular_values[0], "component {k}");
        }
    }
}

#[test]
fn loadings_are_withheld_unless_asked_for() {
    let (a, _, _) = low_rank(50, 60, &[3.0, 1.0], 9);
    assert!(run(&a, 50, 60, &SvdParams::default()).loadings.is_none());
}

#[test]
fn a_matrix_smaller_than_the_sketch_is_handled() {
    // 10 frames, fewer than rank + oversample.
    let (a, _, _) = low_rank(10, 40, &[4.0, 2.0, 1.0], 11);
    let svd = run(&a, 10, 40, &SvdParams::default());
    assert_eq!(svd.singular_values.len(), 8);
    assert!((svd.singular_values[0] - 4.0).abs() < 1e-5);
}
