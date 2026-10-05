//! Locating the tapping, window by window, on synthetic videos whose hand
//! position is known: still, drifting, relocated, and beside a second
//! moving object.

use tapvid::ingest::{Ingest, IngestParams};
use tapvid::locate::{locate, LocateParams, Locations};
use tapvid::synth::{frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};

const SECONDS: f64 = 10.0;

/// Render `SECONDS` of `spec` (320 x 240, so 5 pixels per grid cell) at a
/// rough 30 fps clock, and locate the 3 Hz tapping. Returns the locations and
/// the time of the first frame.
fn run(spec: VideoSpec) -> (Locations, f64) {
    let video = VideoSynth::new(spec.clone());
    let clock = FrameClock { jitter_sd: 0.004, drop_prob: 0.05, seed: 77, ..FrameClock::default() };
    let times = frame_times(&clock, SECONDS);
    let mut ing = Ingest::new(IngestParams::default()).unwrap();
    let (w, h) = (spec.width, spec.height);
    let (mut frame, mut work) = (vec![0u8; w * h], vec![0f32; w * h]);
    for &t in &times {
        video.render(t, &mut frame, &mut work);
        ing.push_frame(&frame, w, w, h, (t * 1e6).round() as i64).unwrap();
    }
    (locate(&ing, 3.0, &LocateParams::default()).unwrap(), times[0])
}

fn spec() -> VideoSpec {
    VideoSpec { tap: TapSpec { iti_sd: 0.01, ..TapSpec::default() }, exposure_step: Some((5.0, 1.3)), ..VideoSpec::default() }
}

/// The hand's rest position at time `t`, in grid cells.
fn truth(s: &VideoSpec, t: f64) -> (f64, f64) {
    let (mut fx, mut fy) = match s.relocate {
        Some((at, c)) if t >= at => c,
        _ => s.centre,
    };
    if let Some((vx, vy)) = s.drift {
        fx += vx * t;
        fy += vy * t;
    }
    (fx * 64.0, fy * 48.0)
}

fn assert_follows(s: &VideoSpec, loc: &Locations, t_first: f64, tolerance: f64, skip: impl Fn(f64, f64) -> bool) {
    assert!(loc.windows.len() >= 8, "{} windows", loc.windows.len());
    for w in &loc.windows {
        let (a, b) = (w.t_start + t_first, w.t_end + t_first);
        if skip(a, b) {
            continue;
        }
        let (tx, ty) = truth(s, 0.5 * (a + b));
        let (dx, dy) = (w.centroid_x - tx, w.centroid_y - ty);
        let d = (dx * dx + dy * dy).sqrt();
        assert!(d < tolerance, "window {a:.1}-{b:.1} s: centroid ({:.1}, {:.1}), hand at ({tx:.1}, {ty:.1})", w.centroid_x, w.centroid_y);
        assert!(
            w.box_x0 as f64 <= tx && tx <= w.box_x1 as f64 + 1.0 && w.box_y0 as f64 <= ty && ty <= w.box_y1 as f64 + 1.0,
            "window {a:.1}-{b:.1} s: box ({}, {})-({}, {}) misses the hand at ({tx:.1}, {ty:.1})",
            w.box_x0, w.box_y0, w.box_x1, w.box_y1
        );
        assert!(w.concentration > 0.5, "window {a:.1}-{b:.1} s: concentration {}", w.concentration);
    }
}

#[test]
fn a_still_hand_is_found_in_every_window() {
    let s = spec();
    let (loc, t0) = run(s.clone());
    assert_follows(&s, &loc, t0, 2.0, |_, _| false);
    assert!(loc.travel < 1.5, "travel {}", loc.travel);
}

#[test]
fn a_drifting_hand_is_followed() {
    // 0.02 of the width per second: 13 cells over the trial.
    let s = VideoSpec { centre: (0.4, 0.5), drift: Some((0.02, 0.0)), ..spec() };
    let (loc, t0) = run(s.clone());
    assert_follows(&s, &loc, t0, 2.5, |_, _| false);
    assert!(loc.travel > 4.0, "travel {}", loc.travel);
    assert!(loc.max_step < 3.0, "max step {}", loc.max_step);
}

#[test]
fn a_relocated_hand_shows_as_a_jump() {
    let s = VideoSpec { centre: (0.3, 0.5), relocate: Some((5.0, (0.7, 0.5))), ..spec() };
    let (loc, t0) = run(s.clone());
    // Windows that straddle the move hold both positions.
    assert_follows(&s, &loc, t0, 2.0, |a, b| a < 5.0 && b > 5.0);
    assert!(loc.max_step > 10.0, "max step {}", loc.max_step);
}

#[test]
fn a_second_object_at_another_rate_is_ignored() {
    let s = VideoSpec {
        centre: (0.35, 0.5),
        distractor: Some(Distractor { rate_hz: 4.5, displacement: 15.0, radius: 10.0, contrast: 90.0, centre: (0.8, 0.3) }),
        ..spec()
    };
    let (loc, t0) = run(s.clone());
    assert_follows(&s, &loc, t0, 2.0, |_, _| false);
}
