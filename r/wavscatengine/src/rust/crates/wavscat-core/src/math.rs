//! Transcendental functions, routed through the pure-Rust `libm` crate.
//!
//! `f64::exp` and friends call the platform's C library, whose results differ
//! in the last bit between glibc, Apple's libm, MSVC and wasm. `libm` is plain
//! Rust compiled from the same source everywhere, so its results do not.
//! Clippy enforces the rule: see `clippy.toml`.
//!
//! Square roots and rounding are exempt. IEEE 754 requires them to be
//! correctly rounded, so every target already agrees.

/// `e^x`.
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// Natural logarithm.
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// Base-two logarithm.
#[inline]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

/// `ln(1 + x)`, accurate for small `x`.
#[inline]
pub fn log1p(x: f64) -> f64 {
    libm::log1p(x)
}

/// `x^y`.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// Sine of an angle in radians.
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// Cosine of an angle in radians.
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// The angle of the point `(x, y)` from the positive x axis, in `(-pi, pi]`.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// `2^k` for a small non-negative integer, exactly.
#[inline]
pub fn pow2(k: u32) -> f64 {
    (1u64 << k) as f64
}

/// `floor(log2(x))` for `x >= 1`, read from the exponent bits.
///
/// Exact, unlike `floor(log2(x))`, which can round up to the next integer when
/// `x` sits just below a power of two.
pub fn floor_log2(x: f64) -> i32 {
    debug_assert!(x >= 1.0 && x.is_finite());
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// `floor(log2(m))` for an integer `m >= 1`.
#[inline]
pub fn floor_log2_int(m: u64) -> u32 {
    63 - m.leading_zeros()
}

/// `ceil(log2(m))` for an integer `m >= 1`.
#[inline]
pub fn ceil_log2_int(m: u64) -> u32 {
    m.next_power_of_two().trailing_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_logs_are_exact() {
        assert_eq!(floor_log2(1.0), 0);
        assert_eq!(floor_log2(64.0), 6);
        assert_eq!(floor_log2(127.999), 6);
        assert_eq!(floor_log2(30.0), 4);
        assert_eq!(floor_log2_int(6142), 12);
        assert_eq!(ceil_log2_int(4096), 12);
        assert_eq!(ceil_log2_int(4097), 13);
        assert_eq!(ceil_log2_int(1), 0);
    }
}
