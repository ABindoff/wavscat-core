//! Deterministic wavelet scattering transforms.
//!
//! This crate is the numerical core behind the `wavscat` R package and its
//! Python and JavaScript bindings. Its contract is stronger than accuracy: for a
//! given [`NUMERICS_VERSION`], the same input samples produce the same output
//! bits on every supported target, whether that is x86-64, ARM64 or
//! WebAssembly.
//!
//! Three rules make that hold.
//!
//! 1. Every transcendental function comes from the pure-Rust `libm` crate via
//!    [`math`], never from the platform C library.
//! 2. The FFT is implemented here, in scalar code with a fixed order of
//!    operations, rather than delegated to a library that selects SIMD kernels
//!    at run time.
//! 3. Every reduction runs sequentially in index order. Parallelism, where
//!    used, is across independent signals or paths, never inside a sum.
//!
//! The filter bank and cascade follow Kymatio (BSD-3-Clause) and agree with it
//! to within floating-point rounding.
#![forbid(unsafe_code)]

mod backend;
pub mod complex;
pub mod error;
pub mod features;
pub mod fft;
pub mod filter_bank;
pub mod math;
pub mod pad;
pub mod scattering1d;

pub use error::Error;

/// Version of the numerical definition.
///
/// Bumped whenever any change could alter an output bit: a different order of
/// operations, a different FFT algorithm, a `libm` upgrade. Outputs carry it so
/// that stored features can be matched to the code that produced them. Until
/// 1.0 the numerics are not frozen.
pub const NUMERICS_VERSION: &str = "0.1.0-dev";
