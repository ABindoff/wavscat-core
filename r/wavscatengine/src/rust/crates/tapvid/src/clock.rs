//! Frame times from the camera's regular clock.
//!
//! A webcam's sensor captures frames at a fixed period, but the timestamps
//! that reach the browser are often coarser than the frames: on Windows they
//! fall on a grid of about 16 ms, so a 30 fps camera reads as intervals of
//! 32 and 48 ms, and a 20 fps one as 48 and 64 ms. That is up to 16 ms of
//! error in every frame time, and so in every tap time.
//!
//! The period is regular, though, so it can be recovered: give each frame
//! its slot on the camera's clock (a dropped frame leaves a slot empty), fit
//! a straight line of time against slot by least squares, and read each
//! frame's time off the line. Over a few hundred frames the line is far
//! finer than the timestamps.
//!
//! The period is not regular across a whole trial, though. Auto-exposure can
//! halve or third the frame rate when the light dims, part way through. So
//! the trial is first split wherever the local period, the mean interval
//! over a short window with drops trimmed out, shifts by more than a set
//! fraction and stays shifted; each segment gets its own line. A segment
//! that does not fit a regular clock keeps its raw timestamps, so a camera
//! that really is irregular is never forced onto a grid.
//!
//! Every step is a sum or a median in a fixed order, so the result is
//! bit-identical on every target.

use wavscat_core::Error;

/// Settings for [`regularise`].
#[derive(Debug, Clone)]
pub struct ClockParams {
    /// Intervals each side of a frame in the local period estimate.
    pub half_window: usize,
    /// An interval more than this many times the window's median counts as
    /// a drop and is left out of the local period.
    pub drop_factor: f64,
    /// The local period must differ from the segment's by more than this
    /// fraction...
    pub change: f64,
    /// ...for this many consecutive frames to start a new segment.
    pub change_frames: usize,
    /// A segment of fewer frames keeps its raw timestamps.
    pub min_segment: usize,
    /// A segment whose fit leaves an RMS residual above this fraction of its
    /// period keeps its raw timestamps. Timestamps at random phases leave
    /// about `1 / sqrt(12) = 0.29` of a period; a 16 ms timestamp grid leaves
    /// 0.14 at 30 fps and 0.09 at 20 fps...
    pub max_rms: f64,
    /// ...as does one with more than this fraction of its slots empty: a
    /// fine enough grid fits any timestamps, so a fit that needs many empty
    /// slots is not a camera clock.
    pub max_missing: f64,
}

impl Default for ClockParams {
    fn default() -> Self {
        ClockParams { half_window: 7, drop_factor: 1.6, change: 0.15, change_frames: 5, min_segment: 15, max_rms: 0.2, max_missing: 0.25 }
    }
}

/// One stretch of steady frame rate.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ClockSegment {
    /// First frame and one past the last.
    pub start: usize,
    pub end: usize,
    /// Fitted frame period, in seconds.
    pub period: f64,
    /// RMS and largest distance of the raw timestamps from the fitted line,
    /// in seconds.
    pub rms_residual: f64,
    pub max_residual: f64,
    /// Empty slots inside the segment: frames the camera captured but that
    /// never arrived.
    pub missing: usize,
    /// Whether the fit was used. If not, the segment keeps its raw times.
    pub regular: bool,
}

/// What [`regularise`] found.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ClockQc {
    pub segments: Vec<ClockSegment>,
    /// Segment boundaries where the frame rate changed.
    pub rate_changes: usize,
    /// Frames missing inside regular segments.
    pub missing: usize,
    /// Fraction of all frames, captured and missing, that went missing.
    pub missing_fraction: f64,
    /// RMS residual over the regular segments, in seconds: roughly the
    /// timestamp error that was removed.
    pub rms_residual: f64,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let k = v.len() / 2;
    if v.len() % 2 == 1 { v[k] } else { 0.5 * (v[k - 1] + v[k]) }
}

/// Mean of the intervals in `d`, leaving out those over `factor` times their
/// median: an estimate of the frame period that drops do not inflate and
/// timestamp quantisation does not bias, unlike the median of intervals.
fn trimmed_period(d: &[f64], factor: f64) -> f64 {
    let mut s = d.to_vec();
    let m = median(&mut s);
    let (mut acc, mut n) = (0.0, 0usize);
    for &x in d {
        if x <= factor * m {
            acc += x;
            n += 1;
        }
    }
    if n > 0 { acc / n as f64 } else { m }
}

/// Least-squares line `t = a + period * slot`.
fn fit(slots: &[f64], t: &[f64]) -> (f64, f64) {
    let n = slots.len() as f64;
    let (mut sx, mut sy) = (0.0, 0.0);
    for (x, y) in slots.iter().zip(t) {
        sx += x;
        sy += y;
    }
    let (mx, my) = (sx / n, sy / n);
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for (x, y) in slots.iter().zip(t) {
        sxx += (x - mx) * (x - mx);
        sxy += (x - mx) * (y - my);
    }
    let period = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (my - period * mx, period)
}

/// Fit one segment of timestamps `t`, starting from period `p0`. Slots are
/// first counted interval by interval, each advancing by its rounded number
/// of periods, so that a small error in `p0` never accumulates; then the
/// line is fitted, and the slots re-read against it until they settle.
/// Returns the corrected times and the segment summary (start and end are
/// left for the caller).
fn fit_segment(t: &[f64], p0: f64, p: &ClockParams) -> (Vec<f64>, ClockSegment) {
    let n = t.len();
    let mut slots = vec![0.0; n];
    for i in 1..n {
        slots[i] = slots[i - 1] + ((t[i] - t[i - 1]) / p0).round().max(1.0);
    }
    let (mut a, mut period) = fit(&slots, t);
    for _ in 0..3 {
        if !(period > 0.0) {
            break;
        }
        let mut next = slots.clone();
        let mut prev = f64::NEG_INFINITY;
        for i in 0..n {
            let s = ((t[i] - a) / period).round().max(prev + 1.0);
            next[i] = s;
            prev = s;
        }
        if next == slots {
            break;
        }
        slots = next;
        let (na, np) = fit(&slots, t);
        a = na;
        period = np;
    }
    let corrected: Vec<f64> = slots.iter().map(|s| a + period * s).collect();
    let (mut ss, mut worst) = (0.0f64, 0.0f64);
    for (c, r) in corrected.iter().zip(t) {
        let e = (r - c).abs();
        ss += e * e;
        worst = worst.max(e);
    }
    let rms = (ss / n as f64).sqrt();
    let missing = (slots[n - 1] - slots[0] + 1.0) as usize - n;
    let regular = n >= p.min_segment
        && period > 0.0
        && rms <= p.max_rms * period
        && (missing as f64) <= p.max_missing * (n + missing) as f64;
    let seg = ClockSegment { start: 0, end: n, period, rms_residual: rms, max_residual: worst, missing, regular };
    (corrected, seg)
}

/// Re-estimate frame times `ts` (seconds, strictly increasing) from the
/// camera's regular clock. Returns the corrected times, in the same order
/// and also strictly increasing, and what was found.
pub fn regularise(ts: &[f64], p: &ClockParams) -> Result<(Vec<f64>, ClockQc), Error> {
    let n = ts.len();
    if ts.windows(2).any(|w| !(w[1] > w[0])) {
        return Err(Error("Timestamps must be strictly increasing.".into()));
    }
    if n < 3 {
        let qc = ClockQc { segments: Vec::new(), rate_changes: 0, missing: 0, missing_fraction: 0.0, rms_residual: 0.0 };
        return Ok((ts.to_vec(), qc));
    }
    let d: Vec<f64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
    // Local period at each interval.
    let local: Vec<f64> = (0..d.len())
        .map(|i| {
            let lo = i.saturating_sub(p.half_window);
            let hi = (i + p.half_window + 1).min(d.len());
            trimmed_period(&d[lo..hi], p.drop_factor)
        })
        .collect();

    // Split where the local period leaves the segment's for long enough.
    let mut bounds = vec![0usize];
    let mut seg_period = local[0];
    let mut run = 0usize;
    for i in 0..local.len() {
        if (local[i] / seg_period - 1.0).abs() > p.change {
            run += 1;
            if run >= p.change_frames {
                // The change began where the run did; interval i - run + 1
                // ends at frame i - run + 2, so the new segment starts there.
                let start = i + 2 - run;
                if start > *bounds.last().unwrap() {
                    bounds.push(start);
                }
                seg_period = local[i];
                run = 0;
            }
        } else {
            run = 0;
        }
    }
    bounds.push(n);

    let mut out = Vec::with_capacity(n);
    let mut segments = Vec::new();
    for w in bounds.windows(2) {
        let (s, e) = (w[0], w[1]);
        let seg_t = &ts[s..e];
        let p0 = if e - s >= 2 { trimmed_period(&d[s..e - 1], p.drop_factor) } else { local[s.min(local.len() - 1)] };
        let (corrected, mut seg) = fit_segment(seg_t, p0, p);
        seg.start = s;
        seg.end = e;
        if seg.regular {
            out.extend(corrected);
        } else {
            out.extend_from_slice(seg_t);
        }
        segments.push(seg);
    }
    // Keep times strictly increasing across segment boundaries.
    for i in 1..n {
        if !(out[i] > out[i - 1]) {
            out[i] = ts[i].max(out[i - 1] + 1e-6);
        }
    }

    let rate_changes = segments.windows(2).filter(|w| (w[1].period / w[0].period - 1.0).abs() > p.change).count();
    let regular: Vec<&ClockSegment> = segments.iter().filter(|s| s.regular).collect();
    let missing: usize = regular.iter().map(|s| s.missing).sum();
    let frames: usize = regular.iter().map(|s| s.end - s.start).sum();
    let (mut ss, mut m) = (0.0, 0usize);
    for s in &regular {
        ss += s.rms_residual * s.rms_residual * (s.end - s.start) as f64;
        m += s.end - s.start;
    }
    let qc = ClockQc {
        rate_changes,
        missing,
        missing_fraction: if frames + missing > 0 { missing as f64 / (frames + missing) as f64 } else { 0.0 },
        rms_residual: if m > 0 { (ss / m as f64).sqrt() } else { 0.0 },
        segments,
    };
    Ok((out, qc))
}
