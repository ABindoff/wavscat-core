//! Self-check of the bit-identity guarantee on the machine it runs on.
//!
//! [`golden_report`] transforms a fixed set of signals, generated from integer
//! arithmetic so that the input is identical everywhere, and hashes the exact
//! output bits. [`verify`] compares the report with the record embedded at
//! build time, which was produced on the reference machine.
//!
//! CI runs this on every supported target and browser. An application can run
//! it too, on the participant's own device, before trusting the features it
//! computes there: a mismatch means that device's arithmetic differs from the
//! reference, which the crate promises cannot happen.
//!
//! After a deliberate change to the numerics, which must come with a bump to
//! [`NUMERICS_VERSION`], regenerate the record with
//!
//! ```text
//! WAVSCAT_UPDATE_GOLDEN=1 cargo test --test golden
//! ```

use std::fmt::Write as _;

use crate::features::{self, Summary};
use crate::jtfs::{Format, OutType, ParamsJtfs, ScatteringJtfs};
use crate::scattering1d::{Averaging, Params1d, Scattering1d, TSpec};
use crate::NUMERICS_VERSION;

/// The golden record for this build's numerics.
pub const GOLDEN_RECORD: &str = include_str!("../golden.tsv");

/// FNV-1a over the little-endian bytes of each value. Not cryptographic, and
/// it does not need to be: it detects any change to any bit.
pub fn hash(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// A test signal built from integer arithmetic, so it is the same on every
/// target: a linear congruential generator, scaled exactly to `[-1, 1)`, plus a
/// slow integer-period sawtooth for some low-frequency structure.
pub fn signal(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((s >> 11) as f64) / (1u64 << 52) as f64 - 1.0;
            let saw = ((i % 97) as f64 - 48.0) / 64.0;
            noise + saw
        })
        .collect()
}

fn cases_1d() -> Vec<(&'static str, Params1d, u64)> {
    let p = |n, j, q: Vec<u32>| Params1d::new(n, j, q);
    let mut kinematic = p(1800, 8, vec![8, 1]);
    kinematic.t_sec = Some(1.0);
    kinematic.sr = Some(30.0);
    let mut global = p(2048, 6, vec![8, 1]);
    global.t = TSpec::Global;
    let mut unaveraged = p(1500, 6, vec![4, 2]);
    unaveraged.t = TSpec::Samples(0.0);
    let mut strided = p(4096, 7, vec![8, 2]);
    strided.stride = Some(16.0);
    let mut first = p(3000, 6, vec![12]);
    first.max_order = 1;
    vec![
        ("kinematic-1800-J8-Tsec1", kinematic, 1),
        ("audio-32000-J9-Q8", p(32000, 9, vec![8]), 2),
        ("global-2048-J6", global, 3),
        ("unaveraged-1500-J6", unaveraged, 4),
        ("strided-4096-J7", strided, 5),
        ("order1-3000-J6-Q12", first, 6),
    ]
}

/// Joint operators whose frequential axes have 120 and 152 rows, so that the
/// mixed-radix and paired prime DFT paths are covered as well as powers of two.
fn cases_jtfs() -> Vec<(&'static str, ParamsJtfs, u64)> {
    let time = ParamsJtfs::new(4096, 6, vec![8]);
    let mut joint = ParamsJtfs::new(4096, 6, vec![8]);
    joint.format = Format::Joint;
    joint.out_type = OutType::List;
    let mut unaveraged_fr = ParamsJtfs::new(4096, 6, vec![8]);
    unaveraged_fr.f = TSpec::Samples(0.0);
    let speech = ParamsJtfs::new(16000, 10, vec![8]);
    vec![
        ("jtfs-time-4096-J6", time, 7),
        ("jtfs-joint-4096-J6", joint, 8),
        ("jtfs-F0-4096-J6", unaveraged_fr, 9),
        ("jtfs-speech-16000-J10", speech, 10),
    ]
}

/// Hash the outputs of every golden case, one line per quantity.
pub fn golden_report() -> String {
    let mut out = String::new();
    writeln!(out, "# wavscat-core golden hashes, numerics {NUMERICS_VERSION}").unwrap();
    for (name, params, seed) in cases_1d() {
        let op = Scattering1d::new(&params).expect("golden operator");
        let coefs = op.transform(&signal(params.n, seed)).expect("golden transform");
        writeln!(out, "{name}\tcoefs\t{:016x}", hash(coefs.iter().flatten().copied())).unwrap();

        // The usual feature pipeline: renormalise, log-compress, summarise.
        if op.average != Averaging::Local || op.paths.iter().all(|p| p.order < 2) {
            continue;
        }
        let mut feats = coefs;
        features::renorm(&mut feats, &op.paths, 1e-12).expect("golden renorm");
        let eps = features::eps_quantile(&feats.concat(), 0.01);
        for v in feats.iter_mut() {
            features::log_compress(v, eps).expect("golden log");
        }
        for how in [Summary::Mean, Summary::Sd, Summary::Median, Summary::Max] {
            let s = feats.iter().map(|v| features::summarise(v, how));
            writeln!(out, "{name}\t{how:?}\t{:016x}", hash(s)).unwrap();
        }
    }
    for (name, params, seed) in cases_jtfs() {
        let op = ScatteringJtfs::new(&params).expect("golden operator");
        let coefs = op.transform(&signal(params.time.n, seed)).expect("golden transform");
        let shape = coefs.iter().flat_map(|m| [m.rows as f64, m.cols as f64]);
        writeln!(out, "{name}\tshape\t{:016x}", hash(shape)).unwrap();
        let values = coefs.iter().flat_map(|m| m.data.iter().copied());
        writeln!(out, "{name}\tcoefs\t{:016x}", hash(values)).unwrap();
    }

    // A 30 Hz movement configuration, renormalised by S1: 30 s at 30 Hz.
    let mut video = ParamsJtfs::new(900, 7, vec![8, 1]);
    video.time.t_sec = Some(6.0);
    video.time.sr = Some(30.0);
    let op = ScatteringJtfs::new(&video).expect("golden operator");
    let (mut coefs, s1) = op.transform_with_s1(&signal(900, 15)).expect("golden transform");
    writeln!(out, "jtfs-video-900-J7\ts1\t{:016x}", hash(s1.data.iter().copied())).unwrap();
    op.renorm(&mut coefs, &s1, 1e-12).expect("golden renorm");
    let values = coefs.iter().flat_map(|m| m.data.iter().copied());
    writeln!(out, "jtfs-video-900-J7\trenorm\t{:016x}", hash(values)).unwrap();
    out
}

/// Lines of `report` that differ from `record`, as `got` / `want` pairs.
pub fn differences(report: &str, record: &str) -> Vec<String> {
    let got: Vec<&str> = report.lines().collect();
    let want: Vec<&str> = record.lines().collect();
    let mut diffs: Vec<String> = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| g != w)
        .map(|(g, w)| format!("got  {g}\nwant {w}"))
        .collect();
    if got.len() != want.len() {
        diffs.push(format!("got {} lines, want {}", got.len(), want.len()));
    }
    diffs
}

/// Run the golden cases on this machine and compare with the embedded
/// record. Returns the mismatches, empty when every output bit agrees.
pub fn verify() -> Vec<String> {
    differences(&golden_report(), GOLDEN_RECORD)
}
