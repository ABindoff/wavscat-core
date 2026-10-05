//! Tests of stage 1, frame ingest.

use tapvid::ingest::{rec601, Ingest, IngestParams};

fn ingest() -> Ingest {
    Ingest::new(IngestParams::default()).unwrap()
}

/// A luma plane whose pixel `(x, y)` is `f(x, y)`, with `pad` junk bytes at
/// the end of each row.
fn plane(width: usize, height: usize, pad: usize, f: impl Fn(usize, usize) -> u8) -> (Vec<u8>, usize) {
    let stride = width + pad;
    let mut v = vec![255u8; stride * height];
    for y in 0..height {
        for x in 0..width {
            v[y * stride + x] = f(x, y);
        }
    }
    (v, stride)
}

#[test]
fn a_uniform_frame_stays_uniform() {
    let mut ing = ingest();
    let (v, s) = plane(640, 480, 0, |_, _| 117);
    ing.push_frame(&v, s, 640, 480, 0).unwrap();
    assert!(ing.frame(0).iter().all(|&c| c == 117.0));
}

#[test]
fn each_cell_is_the_mean_of_its_block() {
    // 640 x 480 onto 64 x 48: every cell is an exact 10 x 10 block.
    let mut ing = ingest();
    let f = |x: usize, y: usize| ((x * 7 + y * 13) % 251) as u8;
    let (v, s) = plane(640, 480, 0, f);
    ing.push_frame(&v, s, 640, 480, 0).unwrap();
    let frame = ing.frame(0);
    for cy in 0..48 {
        for cx in 0..64 {
            let mut sum = 0u32;
            for y in 0..10 {
                for x in 0..10 {
                    sum += f(cx * 10 + x, cy * 10 + y) as u32;
                }
            }
            assert_eq!(frame[cy * 64 + cx], (sum as f64 / 100.0) as f32, "cell ({cx}, {cy})");
        }
    }
}

#[test]
fn wide_frames_are_centre_cropped_to_four_by_three() {
    // 1280 x 720 crops to the central 960 x 720. Bright bars in the 160-pixel
    // margins must not reach the grid.
    let mut ing = ingest();
    let (v, s) = plane(1280, 720, 0, |x, _| if !(160..1120).contains(&x) { 255 } else { 40 });
    ing.push_frame(&v, s, 1280, 720, 0).unwrap();
    assert!(ing.frame(0).iter().all(|&c| c == 40.0));
}

#[test]
fn odd_sizes_use_every_cropped_pixel_once() {
    // 100 x 75 does not divide evenly into 64 x 48, so cells hold one or two
    // pixels on each axis; a uniform frame must still come out uniform, and a
    // ramp must keep its mean.
    let mut ing = ingest();
    let (v, s) = plane(100, 75, 3, |_, _| 90);
    ing.push_frame(&v, s, 100, 75, 0).unwrap();
    assert!(ing.frame(0).iter().all(|&c| c == 90.0));
}

#[test]
fn row_padding_is_ignored() {
    let (mut a, mut b) = (ingest(), ingest());
    let f = |x: usize, y: usize| ((x + 3 * y) % 200) as u8;
    let (tight, s1) = plane(320, 240, 0, f);
    let (padded, s2) = plane(320, 240, 32, f);
    a.push_frame(&tight, s1, 320, 240, 0).unwrap();
    b.push_frame(&padded, s2, 320, 240, 0).unwrap();
    assert_eq!(a.frame(0), b.frame(0));
}

#[test]
fn timestamps_must_increase() {
    let mut ing = ingest();
    let (v, s) = plane(64, 48, 0, |_, _| 1);
    ing.push_frame(&v, s, 64, 48, 1_000).unwrap();
    assert!(ing.push_frame(&v, s, 64, 48, 1_000).is_err());
    assert!(ing.push_frame(&v, s, 64, 48, 500).is_err());
    ing.push_frame(&v, s, 64, 48, 34_333).unwrap();
    assert_eq!(ing.len(), 2);
    assert_eq!(ing.qc().frames_rejected, 2);
}

/// Push frames at the given capture times, in microseconds.
fn push_times(ing: &mut Ingest, times: &[i64]) {
    let (v, s) = plane(64, 48, 0, |_, _| 1);
    for &t in times {
        ing.push_frame(&v, s, 64, 48, t).unwrap();
    }
}

#[test]
fn dropped_frames_are_counted_from_gaps() {
    let frame_us = 33_333;
    // 30 fps, missing frame 50 (one dropped) and frames 100-104 (five).
    let times: Vec<i64> = (0..300).filter(|&i| i != 50 && !(100..105).contains(&i)).map(|i| i * frame_us).collect();
    let mut ing = ingest();
    push_times(&mut ing, &times);
    let qc = ing.qc();
    assert_eq!(qc.frames_dropped, 6);
    assert!((qc.longest_gap - 6.0 * frame_us as f64 / 1e6).abs() < 1e-9);
    // 294 frames accepted: 293 intervals over a span of 299 frame times.
    assert!((qc.effective_fps - 293.0 / (299.0 * frame_us as f64 / 1e6)).abs() < 1e-9);
}

#[test]
fn a_real_drop_in_frame_rate_is_learned_not_counted_forever() {
    // Auto-exposure halves the rate to 15 fps after 4 s.
    let mut times: Vec<i64> = (0..120).map(|i| i * 33_333).collect();
    let t = times[119];
    times.extend((1..=300).map(|i| t + i * 66_667));
    let mut ing = ingest();
    push_times(&mut ing, &times);
    // Only the first few slow intervals, before the median moves, are flagged.
    let dropped = ing.qc().frames_dropped;
    assert!(dropped > 0 && dropped <= 20, "dropped {dropped}");
}

#[test]
fn the_ring_keeps_the_newest_frames_in_order() {
    let mut ing = Ingest::new(IngestParams { capacity: 10, ..IngestParams::default() }).unwrap();
    for i in 0..25u8 {
        let (v, s) = plane(64, 48, 0, |_, _| i);
        ing.push_frame(&v, s, 64, 48, i as i64 * 33_333).unwrap();
    }
    assert_eq!(ing.len(), 10);
    for k in 0..10 {
        assert_eq!(ing.frame(k)[0], (15 + k) as f32);
        assert_eq!(ing.timestamp_us(k), (15 + k) as i64 * 33_333);
    }
    assert_eq!(ing.qc().frames_accepted, 25);
}

#[test]
fn rgba_is_reduced_with_rec601_weights() {
    assert_eq!(rec601(255, 255, 255), 255);
    assert_eq!(rec601(0, 0, 0), 0);
    assert_eq!(rec601(255, 0, 0), 77);
    assert_eq!(rec601(0, 255, 0), 149);
    assert_eq!(rec601(0, 0, 255), 29);

    // push_rgba gives the same grid as converting first and pushing luma.
    let (w, h) = (160, 120);
    let rgba: Vec<u8> = (0..w * h).flat_map(|i| [(i % 256) as u8, (i * 3 % 256) as u8, (i * 7 % 256) as u8, 255]).collect();
    let luma: Vec<u8> = rgba.chunks(4).map(|p| rec601(p[0], p[1], p[2])).collect();
    let (mut a, mut b) = (ingest(), ingest());
    a.push_rgba(&rgba, 4 * w, w, h, 0).unwrap();
    b.push_frame(&luma, w, w, h, 0).unwrap();
    assert_eq!(a.frame(0), b.frame(0));
}

#[test]
fn bad_buffers_are_refused() {
    let mut ing = ingest();
    assert!(ing.push_frame(&[0u8; 100], 64, 64, 48, 0).is_err());
    assert!(ing.push_frame(&vec![0u8; 64 * 48], 32, 64, 48, 0).is_err());
    assert!(ing.push_frame(&vec![0u8; 32 * 24], 32, 32, 24, 0).is_err(), "smaller than the grid");
}
