//! Reflection padding, and the bookkeeping to undo it at every resolution.

use crate::error::{fail, Error};

/// Split `n_padded - n_input` samples of padding between the two ends.
pub fn compute_padding(n_padded: usize, n_input: usize) -> Result<(usize, usize), Error> {
    if n_padded < n_input {
        return fail("Padded length must be at least the signal length.");
    }
    let to_add = n_padded - n_input;
    let left = to_add / 2;
    let right = to_add - left;
    if left.max(right) >= n_input {
        return fail(format!(
            "Signal of length {n_input} is too short for the requested scale: it would need {} \
             samples of reflection padding, which is more than the signal itself. Reduce J or T.",
            left.max(right)
        ));
    }
    Ok((left, right))
}

/// Mirror the signal at each end without repeating the edge sample, as
/// `numpy.pad(mode = "reflect")` does.
pub fn pad_reflect(x: &[f64], left: usize, right: usize) -> Vec<f64> {
    let n = x.len();
    assert!(left < n && right < n, "padding must be shorter than the signal");
    let mut out = Vec::with_capacity(n + left + right);
    out.extend((1..=left).rev().map(|i| x[i]));
    out.extend_from_slice(x);
    out.extend((0..right).map(|i| x[n - 2 - i]));
    out
}

/// Where the original signal sits inside the padded one at each subsampling
/// level `j`, as zero-based `[start, end)`. Rounding outwards after each
/// halving keeps the retained window conservative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Borders {
    pub start: Vec<usize>,
    pub end: Vec<usize>,
}

impl Borders {
    pub fn new(log2_t: i32, j: u32, i0: usize, i1: usize) -> Borders {
        let m = (log2_t.max(j as i32)).max(0) as usize;
        let mut start = vec![i0];
        let mut end = vec![i1];
        for k in 0..m {
            start.push(start[k] / 2 + start[k] % 2);
            end.push(end[k] / 2 + end[k] % 2);
        }
        Borders { start, end }
    }

    /// Cut a signal at level `j` back to the extent of the original.
    pub fn unpad<'a, T>(&self, x: &'a [T], j: usize) -> &'a [T] {
        &x[self.start[j]..self.end[j]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_matches_numpy() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(pad_reflect(&x, 2, 3), vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(pad_reflect(&x, 0, 0), x.to_vec());
    }
}
