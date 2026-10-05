//! Reader for the fixtures written by `tools/export-fixtures.R`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

pub enum Leaf {
    Num { dims: Vec<usize>, values: Vec<f64> },
    Str(Vec<String>),
    Null,
}

pub struct Fixture {
    pub leaves: HashMap<String, Leaf>,
}

impl Fixture {
    pub fn load(stem: &str) -> Fixture {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
        let manifest = std::fs::read_to_string(dir.join(format!("{stem}.tsv")))
            .expect("fixture manifest; run tools/export-fixtures.R");
        let bytes = std::fs::read(dir.join(format!("{stem}.bin"))).expect("fixture data");
        let data: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut leaves = HashMap::new();
        for line in manifest.lines().filter(|l| !l.is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            let leaf = match f[1] {
                "f64" => {
                    let off: usize = f[3].parse().unwrap();
                    let len: usize = f[4].parse().unwrap();
                    Leaf::Num {
                        dims: f[2].split(',').map(|d| d.parse().unwrap()).collect(),
                        values: data[off..off + len].to_vec(),
                    }
                }
                "str" => Leaf::Str(f[5].split('|').map(String::from).collect()),
                _ => Leaf::Null,
            };
            leaves.insert(f[0].to_string(), leaf);
        }
        Fixture { leaves }
    }

    pub fn has(&self, key: &str) -> bool {
        self.leaves.contains_key(key)
    }

    pub fn num(&self, key: &str) -> &[f64] {
        match self.leaves.get(key) {
            Some(Leaf::Num { values, .. }) => values,
            _ => panic!("no numeric leaf {key}"),
        }
    }

    pub fn dims(&self, key: &str) -> &[usize] {
        match self.leaves.get(key) {
            Some(Leaf::Num { dims, .. }) => dims,
            _ => panic!("no numeric leaf {key}"),
        }
    }

    pub fn scalar(&self, key: &str) -> f64 {
        self.num(key)[0]
    }

    pub fn opt_scalar(&self, key: &str) -> Option<f64> {
        match self.leaves.get(key) {
            Some(Leaf::Num { values, .. }) => Some(values[0]),
            _ => None,
        }
    }

    pub fn string(&self, key: &str) -> Option<&str> {
        match self.leaves.get(key) {
            Some(Leaf::Str(v)) => Some(&v[0]),
            _ => None,
        }
    }

    pub fn strings(&self, key: &str) -> &[String] {
        match self.leaves.get(key) {
            Some(Leaf::Str(v)) => v,
            _ => panic!("no string leaf {key}"),
        }
    }

    /// Column `c` of a column-major matrix.
    pub fn col(&self, key: &str, c: usize) -> Vec<f64> {
        let nr = self.dims(key)[0];
        self.num(key)[c * nr..(c + 1) * nr].to_vec()
    }

    /// Row `r` of a column-major matrix.
    pub fn row(&self, key: &str, r: usize) -> Vec<f64> {
        let d = self.dims(key);
        let v = self.num(key);
        (0..d[1]).map(|c| v[c * d[0] + r]).collect()
    }
}

/// Maximum absolute error relative to the largest reference magnitude, the
/// same measure as `expect_close()` in the R parity tests.
pub fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let scale = want.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    let err = got.iter().zip(want).fold(0.0f64, |a, (g, w)| a.max((g - w).abs()));
    if scale > 0.0 { err / scale } else { err }
}

#[track_caller]
pub fn assert_close(got: &[f64], want: &[f64], tol: f64, label: &str) {
    let e = rel_err(got, want);
    if std::env::var_os("WAVSCAT_SHOW_ERR").is_some() {
        eprintln!("{label}: {e:.2e}");
    }
    assert!(e <= tol, "{label}: relative error {e:e} exceeds {tol:e}");
}
