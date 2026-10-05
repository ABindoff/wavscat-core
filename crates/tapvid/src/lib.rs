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
//! - 2: [`preprocess`], exposure gain, centring and drift, applied on the fly
//! - 3: [`svd`], the leading temporal components, by randomized SVD
//! - 4: [`resample`], irregular capture times to a uniform grid
//! - 5: [`select`], the tapping component and its fundamental, from a
//!   [`spectrum`]
//! - 6: [`phase`], inter-tap intervals and amplitude from the analytic signal
//! - 8: [`qc`], quality control and gating of whole trials
//!
//! [`pipeline`] runs stages 2 to 6 over a captured trial. [`synth`] and
//! [`synth_video`] generate signals and videos with known ground truth.
#![forbid(unsafe_code)]

pub mod ingest;
pub mod linalg;
pub mod phase;
pub mod pipeline;
pub mod preprocess;
pub mod qc;
pub mod resample;
pub mod rng;
pub mod select;
pub mod spectrum;
pub mod svd;
pub mod synth;
pub mod synth_video;
