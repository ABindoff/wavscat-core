//! Stage 1: streaming frame ingest.
//!
//! Frames arrive one at a time, as luma planes, with their capture
//! timestamps. Each is centre-cropped to the grid's aspect ratio and
//! box-averaged down to a small grid (64 x 48 by default), and only that grid
//! is kept, in a preallocated ring buffer.
//!
//! **The raw frame is never retained or copied.** It is read once, during the
//! averaging, and nothing derived from it at full resolution survives the
//! call. This is a privacy property the application relies on, not an
//! optimisation. Pushing a frame also allocates nothing, once the first frame
//! of a given size has been seen.
//!
//! Box averaging assigns each source pixel wholly to the grid cell that
//! contains its centre, sums in integers, and divides once per cell, so the
//! result is exact and deterministic, with no aliasing from point sampling.
//!
//! Timestamps are mandatory, because webcam frame rates vary and fall in low
//! light. A timestamp that does not increase is rejected. A gap longer than
//! `gap_factor` times the running median frame interval counts as dropped
//! frames, as many as would fit in it at the median rate.

use wavscat_core::Error;

use crate::svd::RowSource;

/// Settings for [`Ingest`].
#[derive(Debug, Clone)]
pub struct IngestParams {
    /// Grid width and height, in cells. Frames are centre-cropped to this
    /// aspect ratio first.
    pub grid_width: usize,
    pub grid_height: usize,
    /// Frames kept; once full, the oldest is overwritten. 1024 frames of
    /// 64 x 48 cells is 12.6 MB.
    pub capacity: usize,
    /// Number of recent frame intervals the running median is taken over.
    pub median_window: usize,
    /// A gap longer than this many median intervals counts as dropped frames.
    pub gap_factor: f64,
}

impl Default for IngestParams {
    fn default() -> Self {
        IngestParams { grid_width: 64, grid_height: 48, capacity: 1024, median_window: 31, gap_factor: 1.5 }
    }
}

/// Capture quality, for the QC output.
#[derive(Debug, Clone, PartialEq)]
pub struct IngestQc {
    /// Frames accepted, including any since overwritten in the ring.
    pub frames_accepted: u64,
    /// Frames rejected for a timestamp that did not increase.
    pub frames_rejected: u64,
    /// Frames estimated to have been dropped, from gaps in the timestamps.
    pub frames_dropped: u64,
    /// Accepted frames per second of capture time.
    pub effective_fps: f64,
    /// Longest interval between accepted frames, in seconds.
    pub longest_gap: f64,
}

/// Where a frame of a given size maps onto the grid.
#[derive(Debug, Clone)]
struct Geometry {
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    /// Grid column of each cropped column, and grid row of each cropped row.
    col_cell: Vec<u16>,
    row_cell: Vec<u16>,
    /// Source pixels in each cell.
    counts: Vec<u32>,
}

/// The ring buffer of downsampled frames.
pub struct Ingest {
    p: IngestParams,
    cells: usize,
    frames: Vec<f32>,
    stamps: Vec<i64>,
    /// Slot of the oldest frame.
    head: usize,
    len: usize,
    geometry: Option<Geometry>,
    sums: Vec<u32>,
    intervals: Vec<i64>,
    intervals_seen: usize,
    scratch: Vec<i64>,
    accepted: u64,
    rejected: u64,
    dropped: u64,
    first_us: Option<i64>,
    last_us: Option<i64>,
    longest_gap_us: i64,
}

impl Ingest {
    pub fn new(p: IngestParams) -> Result<Ingest, Error> {
        if p.grid_width == 0 || p.grid_height == 0 || p.grid_width > u16::MAX as usize || p.grid_height > u16::MAX as usize {
            return Err(Error("The grid must have between 1 and 65535 cells on each side.".into()));
        }
        if p.capacity < 2 || p.median_window == 0 || !(p.gap_factor > 1.0) {
            return Err(Error("Need capacity >= 2, median_window >= 1 and gap_factor > 1.".into()));
        }
        let cells = p.grid_width * p.grid_height;
        Ok(Ingest {
            frames: vec![0.0; p.capacity * cells],
            stamps: vec![0; p.capacity],
            sums: vec![0; cells],
            intervals: vec![0; p.median_window],
            scratch: vec![0; p.median_window],
            cells,
            head: 0,
            len: 0,
            geometry: None,
            intervals_seen: 0,
            accepted: 0,
            rejected: 0,
            dropped: 0,
            first_us: None,
            last_us: None,
            longest_gap_us: 0,
            p,
        })
    }

    /// Add a luma (Y) plane: `height` rows of `width` bytes, each row starting
    /// `stride` bytes after the last, captured at `timestamp_us` microseconds.
    pub fn push_frame(&mut self, luma: &[u8], stride: usize, width: usize, height: usize, timestamp_us: i64) -> Result<(), Error> {
        check_plane(luma.len(), stride, width, height, 1)?;
        self.push_with(width, height, timestamp_us, |x, y| luma[y * stride + x] as u32)
    }

    /// Add an RGBA image (as from a canvas), converted to luma with the
    /// Rec. 601 weights in integer arithmetic,
    /// `(77 R + 150 G + 29 B + 128) >> 8`, during the averaging, so that no
    /// luma plane is ever formed. `stride` is in bytes.
    pub fn push_rgba(&mut self, rgba: &[u8], stride: usize, width: usize, height: usize, timestamp_us: i64) -> Result<(), Error> {
        check_plane(rgba.len(), stride, width, height, 4)?;
        self.push_with(width, height, timestamp_us, |x, y| {
            let i = y * stride + 4 * x;
            rec601(rgba[i], rgba[i + 1], rgba[i + 2]) as u32
        })
    }

    fn push_with(&mut self, width: usize, height: usize, ts: i64, pixel: impl Fn(usize, usize) -> u32) -> Result<(), Error> {
        if let Some(last) = self.last_us {
            if ts <= last {
                self.rejected += 1;
                return Err(Error(format!(
                    "Timestamp {ts} us does not follow the previous frame's {last} us; frame rejected."
                )));
            }
        }
        let fits = matches!(&self.geometry, Some(g) if g.width == width && g.height == height);
        if !fits {
            self.geometry = Some(geometry(width, height, self.p.grid_width, self.p.grid_height)?);
        }
        let g = self.geometry.as_ref().unwrap();

        // Box average: integer sums per cell, one division each.
        self.sums.fill(0);
        let gw = self.p.grid_width;
        for (yy, &cy) in g.row_cell.iter().enumerate() {
            let base = cy as usize * gw;
            let y = g.y0 + yy;
            for (xx, &cx) in g.col_cell.iter().enumerate() {
                self.sums[base + cx as usize] += pixel(g.x0 + xx, y);
            }
        }
        let slot = (self.head + self.len) % self.p.capacity;
        let out = &mut self.frames[slot * self.cells..(slot + 1) * self.cells];
        for ((o, &s), &n) in out.iter_mut().zip(&self.sums).zip(&g.counts) {
            *o = (s as f64 / n as f64) as f32;
        }
        self.stamps[slot] = ts;
        if self.len < self.p.capacity {
            self.len += 1;
        } else {
            self.head = (self.head + 1) % self.p.capacity;
        }

        // Timing bookkeeping.
        if let Some(last) = self.last_us {
            let gap = ts - last;
            self.longest_gap_us = self.longest_gap_us.max(gap);
            let seen = self.intervals_seen.min(self.p.median_window);
            if seen >= 3 {
                let w = &mut self.scratch[..seen];
                w.copy_from_slice(&self.intervals[..seen]);
                let mid = seen / 2;
                let median = *w.select_nth_unstable(mid).1;
                if median > 0 && gap as f64 > self.p.gap_factor * median as f64 {
                    let missing = ((gap as f64 / median as f64).round() as u64).saturating_sub(1).max(1);
                    self.dropped += missing;
                }
            }
            // Every interval enters the window, so the median follows a real
            // change of frame rate within half a window.
            self.intervals[self.intervals_seen % self.p.median_window] = gap;
            self.intervals_seen += 1;
        } else {
            self.first_us = Some(ts);
        }
        self.last_us = Some(ts);
        self.accepted += 1;
        Ok(())
    }

    /// Frames currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Grid width and height, in cells.
    pub fn grid(&self) -> (usize, usize) {
        (self.p.grid_width, self.p.grid_height)
    }

    /// Cells per frame.
    pub fn cells(&self) -> usize {
        self.cells
    }

    /// Frame `i`, oldest first: `grid_height` rows of `grid_width` cells.
    pub fn frame(&self, i: usize) -> &[f32] {
        assert!(i < self.len, "frame {i} of {}", self.len);
        let slot = (self.head + i) % self.p.capacity;
        &self.frames[slot * self.cells..(slot + 1) * self.cells]
    }

    /// Capture time of frame `i`, in microseconds.
    pub fn timestamp_us(&self, i: usize) -> i64 {
        assert!(i < self.len);
        self.stamps[(self.head + i) % self.p.capacity]
    }

    /// Capture times of the held frames in seconds, relative to the oldest.
    pub fn timestamps(&self) -> Vec<f64> {
        let t0 = if self.len > 0 { self.timestamp_us(0) } else { 0 };
        (0..self.len).map(|i| (self.timestamp_us(i) - t0) as f64 / 1e6).collect()
    }

    pub fn qc(&self) -> IngestQc {
        let span = match (self.first_us, self.last_us) {
            (Some(a), Some(b)) if b > a => (b - a) as f64 / 1e6,
            _ => 0.0,
        };
        IngestQc {
            frames_accepted: self.accepted,
            frames_rejected: self.rejected,
            frames_dropped: self.dropped,
            effective_fps: if span > 0.0 { (self.accepted - 1) as f64 / span } else { 0.0 },
            longest_gap: self.longest_gap_us as f64 / 1e6,
        }
    }

    /// Discard every frame and statistic, keeping the buffers.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.intervals_seen = 0;
        self.accepted = 0;
        self.rejected = 0;
        self.dropped = 0;
        self.first_us = None;
        self.last_us = None;
        self.longest_gap_us = 0;
    }
}

/// The held frames as a frames-by-cells matrix, oldest first, for the SVD.
impl RowSource for Ingest {
    fn n_rows(&self) -> usize {
        self.len
    }

    fn n_cols(&self) -> usize {
        self.cells
    }

    fn row(&self, r: usize) -> &[f32] {
        self.frame(r)
    }
}

/// Rec. 601 luma in 8-bit integer arithmetic: the weights 0.299, 0.587 and
/// 0.114, scaled to sum to 256, with rounding.
pub fn rec601(r: u8, g: u8, b: u8) -> u8 {
    ((77 * r as u32 + 150 * g as u32 + 29 * b as u32 + 128) >> 8) as u8
}

fn check_plane(len: usize, stride: usize, width: usize, height: usize, bytes: usize) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(Error("The frame is empty.".into()));
    }
    if stride < width * bytes {
        return Err(Error(format!("stride {stride} is shorter than a row of {width} pixels.")));
    }
    if len < (height - 1) * stride + width * bytes {
        return Err(Error(format!("The buffer of {len} bytes is too small for {width} x {height} pixels.")));
    }
    Ok(())
}

/// Centre crop to the grid's aspect ratio, and the cell of every cropped row
/// and column: the cell containing the pixel's centre.
fn geometry(width: usize, height: usize, gw: usize, gh: usize) -> Result<Geometry, Error> {
    let (cw, ch) = if width * gh > height * gw {
        (height * gw / gh, height)
    } else {
        (width, width * gh / gw)
    };
    if cw < gw || ch < gh {
        return Err(Error(format!(
            "A {width} x {height} frame is smaller than the {gw} x {gh} grid after cropping."
        )));
    }
    let x0 = (width - cw) / 2;
    let y0 = (height - ch) / 2;
    let col_cell: Vec<u16> = (0..cw).map(|x| ((2 * x + 1) * gw / (2 * cw)) as u16).collect();
    let row_cell: Vec<u16> = (0..ch).map(|y| ((2 * y + 1) * gh / (2 * ch)) as u16).collect();
    let mut counts = vec![0u32; gw * gh];
    for &cy in &row_cell {
        for &cx in &col_cell {
            counts[cy as usize * gw + cx as usize] += 1;
        }
    }
    Ok(Geometry { width, height, x0, y0, col_cell, row_cell, counts })
}
