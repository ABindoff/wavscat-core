//! Bit-for-bit reproducibility across targets.
//!
//! Each case transforms a signal generated from integers alone, so the input
//! is identical everywhere, and hashes the exact bits of the output. The hashes
//! are checked in under `golden/`, keyed by `NUMERICS_VERSION`. CI runs this
//! test on every supported target; any mismatch means the numerics differ
//! somewhere, which is exactly what the crate promises cannot happen.
//!
//! Regenerate after a deliberate change to the numerics, which must come with
//! a bump to `NUMERICS_VERSION`:
//!
//!     WAVSCAT_UPDATE_GOLDEN=1 cargo test --test golden

use std::fmt::Write as _;
use std::path::PathBuf;

use wavscat_core::features::{self, Summary};
use wavscat_core::jtfs::{Format, OutType, ParamsJtfs, ScatteringJtfs};
use wavscat_core::scattering1d::{Averaging, Params1d, Scattering1d, TSpec};
use wavscat_core::NUMERICS_VERSION;

/// FNV-1a over the little-endian bytes of each value. Not cryptographic, and
/// it does not need to be: it detects any change to any bit.
fn hash(values: impl IntoIterator<Item = f64>) -> u64 {
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
fn signal(n: usize, seed: u64) -> Vec<f64> {
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

struct Case {
    name: &'static str,
    params: Params1d,
    seed: u64,
}

fn cases() -> Vec<Case> {
    let p = |n, j, q: Vec<u32>| Params1d::new(n, j, q);
    let mut tapping = p(1800, 8, vec![8, 1]);
    tapping.t_sec = Some(1.0);
    tapping.sr = Some(30.0);
    let mut global = p(2048, 6, vec![8, 1]);
    global.t = TSpec::Global;
    let mut unaveraged = p(1500, 6, vec![4, 2]);
    unaveraged.t = TSpec::Samples(0.0);
    let mut strided = p(4096, 7, vec![8, 2]);
    strided.stride = Some(16.0);
    let mut first = p(3000, 6, vec![12]);
    first.max_order = 1;
    vec![
        Case { name: "tapping-1800-J8-Tsec1", params: tapping, seed: 1 },
        Case { name: "audio-32000-J9-Q8", params: p(32000, 9, vec![8]), seed: 2 },
        Case { name: "global-2048-J6", params: global, seed: 3 },
        Case { name: "unaveraged-1500-J6", params: unaveraged, seed: 4 },
        Case { name: "strided-4096-J7", params: strided, seed: 5 },
        Case { name: "order1-3000-J6-Q12", params: first, seed: 6 },
    ]
}

fn compute() -> String {
    let mut out = String::new();
    writeln!(out, "# wavscat-core golden hashes, numerics {NUMERICS_VERSION}").unwrap();
    for c in cases() {
        let op = Scattering1d::new(&c.params).unwrap();
        let x = signal(c.params.n, c.seed);
        let coefs = op.transform(&x).unwrap();
        writeln!(out, "{}\tcoefs\t{:016x}", c.name, hash(coefs.iter().flatten().copied())).unwrap();

        // The usual feature pipeline: renormalise, log-compress, summarise.
        if op.average != Averaging::Local || op.paths.iter().all(|p| p.order < 2) {
            continue;
        }
        let mut feats = coefs.clone();
        features::renorm(&mut feats, &op.paths, 1e-12).unwrap();
        let eps = features::eps_quantile(&feats.concat(), 0.01);
        for v in feats.iter_mut() {
            features::log_compress(v, eps).unwrap();
        }
        for how in [Summary::Mean, Summary::Sd, Summary::Median, Summary::Max] {
            let s = feats.iter().map(|v| features::summarise(v, how));
            writeln!(out, "{}\t{:?}\t{:016x}", c.name, how, hash(s)).unwrap();
        }
    }

    // Joint time-frequency scattering. The frequential axis of these
    // operators has 120 and 152 rows, so the mixed-radix and paired prime
    // DFT paths are covered as well as the power-of-two ones.
    for (name, p, seed) in jtfs_cases() {
        let op = ScatteringJtfs::new(&p).unwrap();
        let x = signal(p.time.n, seed);
        let coefs = op.transform(&x).unwrap();
        let shape = coefs.iter().flat_map(|m| [m.rows as f64, m.cols as f64]);
        writeln!(out, "{name}\tshape\t{:016x}", hash(shape)).unwrap();
        writeln!(out, "{name}\tcoefs\t{:016x}", hash(coefs.iter().flat_map(|m| m.data.iter().copied()))).unwrap();
    }
    out
}

fn jtfs_cases() -> Vec<(&'static str, ParamsJtfs, u64)> {
    let mut time = ParamsJtfs::new(4096, 6, vec![8]);
    time.j_fr = 3;
    let mut joint = ParamsJtfs::new(4096, 6, vec![8]);
    joint.format = Format::Joint;
    joint.out_type = OutType::List;
    let mut unaveraged_fr = ParamsJtfs::new(4096, 6, vec![8]);
    unaveraged_fr.f = TSpec::Samples(0.0);
    let mut speech = ParamsJtfs::new(16000, 10, vec![8]);
    speech.j_fr = 3;
    vec![
        ("jtfs-time-4096-J6", time, 7),
        ("jtfs-joint-4096-J6", joint, 8),
        ("jtfs-F0-4096-J6", unaveraged_fr, 9),
        ("jtfs-speech-16000-J10", speech, 10),
    ]
}

#[test]
fn outputs_are_bit_identical_to_the_golden_record() {
    // Overridable because a sandboxed target, such as wasm under WASI, cannot
    // see the path baked in at compile time.
    let dir = std::env::var_os("WAVSCAT_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../golden"));
    let path = dir.join(format!("numerics-{NUMERICS_VERSION}.tsv"));
    let got = compute();
    if std::env::var_os("WAVSCAT_UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("no golden record at {}; generate one with WAVSCAT_UPDATE_GOLDEN=1", path.display())
    });
    let diffs: Vec<String> = got
        .lines()
        .zip(want.lines())
        .filter(|(g, w)| g != w)
        .map(|(g, w)| format!("  got  {g}\n  want {w}"))
        .collect();
    assert!(
        diffs.is_empty() && got.lines().count() == want.lines().count(),
        "outputs differ from the golden record on this target:\n{}",
        diffs.join("\n")
    );
}
