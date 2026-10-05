//! A pseudo-random number generator whose stream is part of the definition.
//!
//! The randomized SVD draws its test matrix from here, so the stream must be
//! identical on every platform and must never change between releases. That
//! rules out the `rand` crate, which makes no such promise. SplitMix64 is a
//! few lines of integer arithmetic, and the Box-Muller transform uses only
//! `libm` functions, so both are exactly reproducible.

use wavscat_core::math;

/// SplitMix64 (Steele, Lea and Flood, 2014).
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
    /// The second normal deviate of the last Box-Muller pair.
    spare: Option<f64>,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed, spare: None }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform on `(0, 1)`, never exactly zero: the top 53 bits, offset by
    /// half a unit so that `ln` in Box-Muller is always finite.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal, by the Box-Muller transform. Deviates come in pairs;
    /// the second is kept for the next call.
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * math::ln(u1)).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * math::sin(theta));
        r * math::cos(theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_splitmix64_stream() {
        // First outputs for seed 0, from the reference C implementation.
        let mut g = SplitMix64::new(0);
        assert_eq!(g.next_u64(), 0xe220a8397b1dcdaf);
        assert_eq!(g.next_u64(), 0x6e789e6aa1b965f4);
        assert_eq!(g.next_u64(), 0x06c45d188009454f);
    }

    #[test]
    fn normals_have_unit_variance() {
        let mut g = SplitMix64::new(42);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| g.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "variance {var}");
    }
}
