//! Where the tapping is, window by window: the region of the frame whose
//! brightness oscillates at the tapping rate, and how it moves.
//!
//! The SVD of stage 3 finds one spatial pattern for the whole trial. A hand
//! that drifts across the frame leaves that pattern, and the selected
//! component then loses the taps. This stage looks at short windows instead:
//! in each, every grid cell's brightness is demodulated at the fundamental
//! and its second harmonic (a fingertip crossing a cell darkens it twice a
//! cycle), on the true capture times, after dividing out each frame's mean
//! luminance as stage 2 does. The cells that oscillate most strongly mark the
//! moving fingers, not just a hand that is present, so a still hand or a
//! still person is ignored.
//!
//! From each window's map come the peak cell, a box around the connected
//! region of strong cells, its energy-weighted centroid, and how much of the
//! window's oscillation the box holds. Across windows, the travel of the
//! centroid says whether the hand stayed put, drifted or moved.
//!
//! Everything is a direct sum in a fixed order, with `cos` and `sin` from
//! `wavscat_core::math`, so the result is bit-identical on every target.

use std::f64::consts::PI;

use wavscat_core::{math, Error};

use crate::ingest::Ingest;

/// Settings for [`locate`].
#[derive(Debug, Clone)]
pub struct LocateParams {
    /// Window length and step, in seconds.
    pub window_sec: f64,
    pub step_sec: f64,
    /// Harmonics of `f0` whose energy is summed: 2 takes the fundamental and
    /// the second harmonic.
    pub harmonics: u32,
    /// A cell belongs to the region if its energy is at least this fraction
    /// of the window's peak.
    pub threshold: f64,
}

impl Default for LocateParams {
    fn default() -> Self {
        LocateParams { window_sec: 2.0, step_sec: 1.0, harmonics: 2, threshold: 0.25 }
    }
}

/// The tapping region in one window. Positions are in grid cells, `x` from
/// the left and `y` from the top of the (unmirrored) camera image; a cell's
/// centre is at `+0.5`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Location {
    /// Window start and end, in seconds from the oldest frame.
    pub t_start: f64,
    pub t_end: f64,
    pub frames: usize,
    /// The cell with the most oscillation energy.
    pub peak_x: usize,
    pub peak_y: usize,
    /// Bounding box of the connected region around the peak, inclusive.
    pub box_x0: usize,
    pub box_y0: usize,
    pub box_x1: usize,
    pub box_y1: usize,
    /// Energy-weighted centroid of the region.
    pub centroid_x: f64,
    pub centroid_y: f64,
    /// Share of the window's oscillation energy inside the region.
    pub concentration: f64,
    /// Total oscillation energy, in units of relative brightness squared.
    pub energy: f64,
}

/// The tapping region over a whole trial.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Locations {
    pub grid_width: usize,
    pub grid_height: usize,
    pub windows: Vec<Location>,
    /// Largest distance, in cells, of any window's centroid from the median
    /// centroid: how far the hand travelled.
    pub travel: f64,
    /// Largest distance, in cells, between consecutive windows' centroids:
    /// the biggest sudden move.
    pub max_step: f64,
}

/// Locate the tapping, at fundamental `f0_hz`, in the frames `ing` holds.
pub fn locate(ing: &Ingest, f0_hz: f64, p: &LocateParams) -> Result<Locations, Error> {
    let (gw, gh) = ing.grid();
    let cells = gw * gh;
    let ts = ing.timestamps();
    let n = ts.len();
    if n < 3 {
        return Err(Error("At least three frames are needed to locate the tapping.".into()));
    }
    if !(f0_hz > 0.0) || !(p.window_sec > 0.0) || !(p.step_sec > 0.0) || p.harmonics == 0 {
        return Err(Error("Need f0 > 0, window_sec > 0, step_sec > 0 and harmonics >= 1.".into()));
    }
    let duration = ts[n - 1];
    let window = p.window_sec.min(duration);

    // Each frame's mean luminance, divided out as in stage 2.
    let inv_gain: Vec<f64> = (0..n)
        .map(|r| {
            let mut acc = 0.0;
            for &v in ing.frame(r) {
                acc += v as f64;
            }
            1.0 / (acc / cells as f64).max(1.0)
        })
        .collect();

    let h = p.harmonics as usize;
    let mut windows = Vec::new();
    let mut start = 0.0;
    loop {
        let end = start + window;
        let rows: Vec<usize> = (0..n).filter(|&r| ts[r] >= start && ts[r] <= end).collect();
        if rows.len() >= 3 {
            windows.push(window_location(ing, &ts, &inv_gain, &rows, start, end, f0_hz, h, p.threshold, gw, gh));
        }
        if end >= duration - 1e-9 {
            break;
        }
        start = (start + p.step_sec).min(duration - window);
    }

    // The path of the centroid.
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        let k = v.len() / 2;
        if v.len() % 2 == 1 { v[k] } else { 0.5 * (v[k - 1] + v[k]) }
    };
    let mx = median(windows.iter().map(|w| w.centroid_x).collect());
    let my = median(windows.iter().map(|w| w.centroid_y).collect());
    let dist = |ax: f64, ay: f64, bx: f64, by: f64| ((ax - bx) * (ax - bx) + (ay - by) * (ay - by)).sqrt();
    let travel = windows.iter().map(|w| dist(w.centroid_x, w.centroid_y, mx, my)).fold(0.0, f64::max);
    let max_step = windows
        .windows(2)
        .map(|w| dist(w[0].centroid_x, w[0].centroid_y, w[1].centroid_x, w[1].centroid_y))
        .fold(0.0, f64::max);
    Ok(Locations { grid_width: gw, grid_height: gh, windows, travel, max_step })
}

/// A frame's Hann weight, and its weighted phasor at each harmonic.
type Weighted = (f64, Vec<(f64, f64)>);

#[allow(clippy::too_many_arguments)]
fn window_location(
    ing: &Ingest,
    ts: &[f64],
    inv_gain: &[f64],
    rows: &[usize],
    start: f64,
    end: f64,
    f0: f64,
    h: usize,
    threshold: f64,
    gw: usize,
    gh: usize,
) -> Location {
    let cells = gw * gh;
    let span = end - start;
    // Per frame: Hann weight, and the weighted phasors of each harmonic.
    let mut wsum = 0.0;
    let mut phasor_sum = vec![(0.0f64, 0.0f64); h];
    let weights: Vec<Weighted> = rows
        .iter()
        .map(|&r| {
            let u = (ts[r] - start) / span;
            let w = 0.5 - 0.5 * math::cos(2.0 * PI * u);
            let ph: Vec<(f64, f64)> = (1..=h)
                .map(|k| {
                    let a = 2.0 * PI * k as f64 * f0 * (ts[r] - start);
                    (w * math::cos(a), -w * math::sin(a))
                })
                .collect();
            wsum += w;
            for (acc, z) in phasor_sum.iter_mut().zip(&ph) {
                acc.0 += z.0;
                acc.1 += z.1;
            }
            (w, ph)
        })
        .collect();

    // Per cell: the weighted sum, and the weighted phasor sums.
    let mut s0 = vec![0.0f64; cells];
    let mut sz = vec![(0.0f64, 0.0f64); cells * h];
    for (&r, (w, ph)) in rows.iter().zip(&weights) {
        let g = inv_gain[r];
        for (c, &v) in ing.frame(r).iter().enumerate() {
            let x = v as f64 * g;
            s0[c] += w * x;
            for (k, z) in ph.iter().enumerate() {
                let acc = &mut sz[c * h + k];
                acc.0 += x * z.0;
                acc.1 += x * z.1;
            }
        }
    }
    // Energy: |sum w (x - mean) e^{-i a}|^2 over harmonics, the mean being
    // the weighted mean.
    let energy: Vec<f64> = (0..cells)
        .map(|c| {
            let m = if wsum > 0.0 { s0[c] / wsum } else { 0.0 };
            let mut e = 0.0;
            for k in 0..h {
                let (re, im) = sz[c * h + k];
                let (re, im) = (re - m * phasor_sum[k].0, im - m * phasor_sum[k].1);
                e += re * re + im * im;
            }
            e
        })
        .collect();

    let mut peak = 0;
    for c in 1..cells {
        if energy[c] > energy[peak] {
            peak = c;
        }
    }
    let cut = threshold * energy[peak];
    // The connected region around the peak (4-neighbours), by flood fill.
    let mut inside = vec![false; cells];
    let mut stack = vec![peak];
    inside[peak] = true;
    while let Some(c) = stack.pop() {
        let (x, y) = (c % gw, c / gw);
        let mut visit = |nx: usize, ny: usize| {
            let nc = ny * gw + nx;
            if !inside[nc] && energy[nc] >= cut {
                inside[nc] = true;
                stack.push(nc);
            }
        };
        if x > 0 {
            visit(x - 1, y);
        }
        if x + 1 < gw {
            visit(x + 1, y);
        }
        if y > 0 {
            visit(x, y - 1);
        }
        if y + 1 < gh {
            visit(x, y + 1);
        }
    }
    let (mut x0, mut y0, mut x1, mut y1) = (gw, gh, 0, 0);
    let (mut ein, mut etot, mut cx, mut cy) = (0.0, 0.0, 0.0, 0.0);
    for c in 0..cells {
        etot += energy[c];
        if inside[c] {
            let (x, y) = (c % gw, c / gw);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            ein += energy[c];
            cx += energy[c] * (x as f64 + 0.5);
            cy += energy[c] * (y as f64 + 0.5);
        }
    }
    let (centroid_x, centroid_y) = if ein > 0.0 { (cx / ein, cy / ein) } else { ((peak % gw) as f64 + 0.5, (peak / gw) as f64 + 0.5) };
    Location {
        t_start: start,
        t_end: end,
        frames: rows.len(),
        peak_x: peak % gw,
        peak_y: peak / gw,
        box_x0: x0,
        box_y0: y0,
        box_x1: x1,
        box_y1: y1,
        centroid_x,
        centroid_y,
        concentration: if etot > 0.0 { ein / etot } else { 0.0 },
        energy: etot,
    }
}
