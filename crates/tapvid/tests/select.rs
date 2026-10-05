//! Acceptance tests for stage 5: component selection and the fundamental.
//! Thresholds sit inside what the `f0_sweep` example measures.

use std::f64::consts::PI;

use tapvid::phase::{analyse_at, PhaseParams};
use tapvid::resample::resample;
use tapvid::rng::SplitMix64;
use tapvid::select::{periodicity, select, BandParams, F0Case};
use tapvid::spectrum::WelchParams;
use tapvid::synth::{render, FrameClock, TapSpec};
use wavscat_core::math;

const FS: f64 = 30.0;

/// A tapping component, resampled to 30 Hz from a rough capture clock.
fn tapping(rate: f64, fundamental: f64, second: f64, noise: f64, seed: u64) -> Vec<f64> {
    let spec = TapSpec {
        mean_iti: 1.0 / rate,
        iti_sd: 0.1 / rate,
        fundamental,
        second_harmonic: second,
        noise_sd: noise,
        seed,
        ..TapSpec::default()
    };
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 50, ..FrameClock::default() };
    let rec = render(&spec, &clock);
    resample(&rec.timestamps, &rec.values, FS, 0.15).unwrap().values
}

fn score(x: &[f64]) -> tapvid::select::Periodicity {
    periodicity(x, FS, &WelchParams::default(), &BandParams::default()).unwrap()
}

#[test]
fn a_dominant_fundamental_is_taken_as_is() {
    for rate in [1.5, 3.0, 4.0, 6.0] {
        for seed in 1..=5 {
            let p = score(&tapping(rate, 1.0, 0.5, 0.1, seed));
            assert_eq!(p.case, F0Case::Fundamental, "rate {rate}, seed {seed}");
            // Each 30 s sample's own mean rate differs from the nominal by about
            // 1.5% at this jitter, and stage 6 tolerates 10%.
            assert!((p.f0_hz - rate).abs() < 0.05 * rate, "rate {rate}: f0 {}", p.f0_hz);
        }
    }
}

#[test]
fn a_dominant_second_harmonic_is_recognised() {
    // Opening and closing look alike: the fundamental at 0.3 of the harmonic.
    // At 4 Hz the harmonic is above the 1-7 Hz band.
    for rate in [1.5, 3.0, 4.0] {
        for seed in 1..=5 {
            let p = score(&tapping(rate, 0.3, 1.0, 0.05, seed));
            assert_eq!(p.case, F0Case::Harmonic, "rate {rate}, seed {seed}: {p:?}");
            assert!((p.f0_hz - rate).abs() < 0.05 * rate, "rate {rate}: f0 {}", p.f0_hz);
            assert!(p.half_locking > 0.6, "rate {rate}: locking {}", p.half_locking);
        }
    }
}

#[test]
fn fast_tapping_is_not_halved() {
    for noise in [0.05, 0.3] {
        for seed in 1..=5 {
            let p = score(&tapping(6.0, 1.0, 0.0, noise, seed));
            assert_eq!(p.case, F0Case::Fundamental, "noise {noise}, seed {seed}: {p:?}");
            assert!((p.f0_hz - 6.0).abs() < 0.2, "f0 {}", p.f0_hz);
        }
    }
}

#[test]
fn the_tapping_component_is_selected_over_drift_noise_and_breathing() {
    let tap = tapping(3.0, 1.0, 0.5, 0.2, 4);
    let n = tap.len();
    let mut rng = SplitMix64::new(9);
    let mut walk = 0.0;
    let drift: Vec<f64> = (0..n)
        .map(|_| {
            walk += 0.05 * rng.normal();
            walk
        })
        .collect();
    let noise: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let breathing: Vec<f64> = (0..n).map(|i| math::sin(2.0 * PI * 0.25 * i as f64 / FS)).collect();
    let comps = vec![drift, noise, tap, breathing];
    let s = select(&comps, FS, &WelchParams::default(), &BandParams::default()).unwrap();
    assert_eq!(s.best().0, 2, "{:?}", s.ranking);
    let runner = s.runner_up().unwrap().1.score;
    assert!(runner < 0.3 * s.best().1.score, "runner-up {runner} vs {}", s.best().1.score);
}

#[test]
fn a_second_oscillator_shows_as_a_close_runner_up() {
    // A distractor moving rhythmically in the tapping band, as a second
    // person or a pet might: QC should see that the choice was close.
    let tap = tapping(3.0, 1.0, 0.5, 0.2, 5);
    let other = tapping(4.5, 1.0, 0.3, 0.2, 6);
    let s = select(&[tap, other], FS, &WelchParams::default(), &BandParams::default()).unwrap();
    let (best, runner) = (s.best().1.score, s.runner_up().unwrap().1.score);
    assert!(runner > 0.5 * best, "runner-up {runner} vs {best}");
}

#[test]
fn stage_five_feeds_stage_six() {
    // The harmonic case end to end: f0 from the spectrum, then intervals.
    for seed in 1..=5 {
        let spec = TapSpec {
            iti_sd: 0.010,
            fundamental: 0.3,
            second_harmonic: 1.0,
            noise_sd: 0.05,
            seed,
            ..TapSpec::default()
        };
        let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: seed + 100, ..FrameClock::default() };
        let rec = render(&spec, &clock);
        let u = resample(&rec.timestamps, &rec.values, FS, 0.15).unwrap();
        let p = score(&u.values);
        assert_eq!(p.case, F0Case::Harmonic);
        // The second harmonic dominates, so time the cycles from it.
        let a = analyse_at(&u.values, u.fs, u.t0, p.f0_hz, p.timing_harmonic(), &PhaseParams::default()).unwrap();
        assert!((a.summary.mean - 1.0 / 3.0).abs() < 0.002, "seed {seed}: mean ITI {}", a.summary.mean);
        let kept = &a.boundaries[1..a.boundaries.len() - 1];
        let truth = rec.itis_between(kept[0] - 0.05, kept[kept.len() - 1] + 0.05);
        let m = truth.iter().sum::<f64>() / truth.len() as f64;
        let true_sd = (truth.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (truth.len() - 1) as f64).sqrt();
        assert!((a.summary.sd - true_sd).abs() < 0.004, "seed {seed}: SD {} vs {true_sd}", a.summary.sd);
    }
}
