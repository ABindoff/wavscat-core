//! Landmark-free finger-tapping features from webcam video.
//!
//! Each video is reduced to a low-resolution pixel-by-time matrix, a
//! randomized SVD recovers the dominant tapping oscillator, and two consumers
//! read it: an analytic-signal phase estimator for sub-frame inter-tap
//! intervals, and joint time-frequency scattering (from `wavscat-core`) for
//! modulation features.
//!
//! Like `wavscat-core`, every computation is deterministic to the bit on every
//! target: transcendental functions come from `wavscat_core::math`, random
//! numbers from the in-crate [`rng`], and every reduction runs in a fixed
//! order.
//!
//! Stages, as numbered in the design brief:
//!
//! - 1: [`ingest`], streaming frames into a ring buffer of small grids
//! - 3: [`svd`], the leading temporal components, by randomized SVD
//! - 4: [`resample`], irregular capture times to a uniform grid
//! - 5: [`select`], the tapping component and its fundamental, from a
//!   [`spectrum`]
//! - 6: [`phase`], inter-tap intervals and amplitude from the analytic signal
//!
//! [`synth`] generates recordings with known ground truth for testing.
#![forbid(unsafe_code)]

pub mod ingest;
pub mod linalg;
pub mod phase;
pub mod resample;
pub mod rng;
pub mod select;
pub mod spectrum;
pub mod svd;
pub mod synth;
