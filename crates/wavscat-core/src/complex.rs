//! A minimal complex type whose arithmetic is spelled out.
//!
//! The formulas here are part of the numerical definition, which is why the
//! crate does not borrow a complex type from elsewhere: a dependency changing
//! how it multiplies would silently change outputs.

use std::ops::{Add, Mul, Sub};

/// A complex number in double precision.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }

    #[inline]
    pub fn conj(self) -> Self {
        C64::new(self.re, -self.im)
    }

    /// Multiply by a real number.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        C64::new(self.re * s, self.im * s)
    }

    /// Divide by a real number. Division, not multiplication by `1 / s`,
    /// which would round differently.
    #[inline]
    pub fn unscale(self, s: f64) -> Self {
        C64::new(self.re / s, self.im / s)
    }

    /// The modulus, `sqrt(re^2 + im^2)`.
    ///
    /// Not `hypot`: the scattering coefficients never approach the range where
    /// squaring overflows, and `sqrt` is correctly rounded on every target.
    #[inline]
    pub fn norm(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

impl Add for C64 {
    type Output = C64;
    #[inline]
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}

impl Sub for C64 {
    type Output = C64;
    #[inline]
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
}

impl Mul for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
