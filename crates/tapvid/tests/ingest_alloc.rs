//! The ingest hot path must not allocate: on a phone, garbage and heap
//! traffic at 30 frames a second are a cost the brief rules out. This binary
//! counts allocations, so it holds only this test.
//!
//! Only allocations made on the measuring thread are counted. The test
//! harness's own threads allocate at unpredictable moments (to capture
//! output, for instance), and counting those made the test flaky.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use tapvid::ingest::{Ingest, IngestParams};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    // Const-initialised and without a destructor, so reading it never
    // allocates, which matters inside an allocator.
    static ARMED: Cell<bool> = const { Cell::new(false) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.with(|a| a.get()) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn pushing_frames_allocates_nothing() {
    let (w, h) = (640, 480);
    let luma = vec![100u8; w * h];
    let rgba = vec![100u8; 4 * w * h];
    let mut ing = Ingest::new(IngestParams { capacity: 64, ..IngestParams::default() }).unwrap();
    // The first frame of a size builds the cell maps.
    ing.push_frame(&luma, w, w, h, 0).unwrap();

    // Enough frames to wrap the ring, fill the interval window, and take
    // the RGBA path too.
    ARMED.with(|a| a.set(true));
    for i in 1..200i64 {
        if i % 2 == 0 {
            ing.push_frame(&luma, w, w, h, i * 33_333).unwrap();
        } else {
            ing.push_rgba(&rgba, 4 * w, w, h, i * 33_333).unwrap();
        }
    }
    ARMED.with(|a| a.set(false));
    let n = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(n, 0, "the hot path allocated {n} times");
}
