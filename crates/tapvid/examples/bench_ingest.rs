//! Time to push one frame, at common webcam sizes.
//!
//!     cargo run --release -p tapvid --example bench_ingest
use std::time::Instant;
use tapvid::ingest::{Ingest, IngestParams};

fn main() {
    for (w, h) in [(640usize, 480usize), (1280, 720)] {
        let luma: Vec<u8> = (0..w * h).map(|i| (i % 251) as u8).collect();
        let rgba: Vec<u8> = (0..4 * w * h).map(|i| (i % 253) as u8).collect();
        let mut ing = Ingest::new(IngestParams::default()).unwrap();
        let reps = 300i64;
        let t = Instant::now();
        for i in 0..reps {
            ing.push_frame(&luma, w, w, h, i * 33_333).unwrap();
        }
        let luma_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let t = Instant::now();
        for i in reps..2 * reps {
            ing.push_rgba(&rgba, 4 * w, w, h, i * 33_333).unwrap();
        }
        let rgba_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("{w} x {h}: luma {luma_ms:.3} ms/frame, rgba {rgba_ms:.3} ms/frame");
    }
}
