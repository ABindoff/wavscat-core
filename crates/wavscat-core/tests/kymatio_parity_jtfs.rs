//! Joint time-frequency scattering against Kymatio, using the reference values
//! from the R package. Mirrors `tests/testthat/test-kymatio-parity-jtfs.R`.

mod common;

use std::collections::HashMap;

use common::{Fixture, Leaf};
use wavscat_core::jtfs::{Format, OutType, ParamsJtfs, ScatteringJtfs};
use wavscat_core::scattering1d::TSpec;

fn spec(f: &Fixture, key: &str) -> TSpec {
    match f.leaves.get(key) {
        Some(Leaf::Num { values, .. }) => TSpec::Samples(values[0]),
        Some(Leaf::Str(s)) if s[0] == "global" => TSpec::Global,
        _ => TSpec::Default,
    }
}

#[test]
fn joint_transforms_match_kymatio() {
    let f = Fixture::load("kymatio-jtfs");
    let show = std::env::var_os("WAVSCAT_SHOW_ERR").is_some();
    for c in 1..=12 {
        let k = |s: &str| format!("{c}/{s}");
        let q: Vec<u32> = f.num(&k("Q")).iter().map(|&q| q as u32).collect();
        let mut p = ParamsJtfs::new(f.scalar(&k("n")) as usize, f.scalar(&k("J")) as u32, q);
        p.j_fr = f.scalar(&k("J_fr")) as u32;
        p.q_fr = f.scalar(&k("Q_fr")) as u32;
        p.time.t = spec(&f, &k("T"));
        p.f = spec(&f, &k("F"));
        p.format = if f.string(&k("format")) == Some("joint") { Format::Joint } else { Format::Time };
        p.out_type = OutType::List;
        let tag = format!("case {c} ({:?}, T {:?}, F {:?}, J_fr {}, Q_fr {})", p.format, p.time.t, p.f, p.j_fr, p.q_fr);

        let op = ScatteringJtfs::new(&p).unwrap_or_else(|e| panic!("{tag}: {e}"));
        let out = op.transform(f.num(&k("x"))).unwrap();

        // Path keys, zero-based: n1 (time format), n2 (order two), n_fr.
        let keys = f.strings(&k("keys"));
        assert_eq!(op.paths.len(), keys.len(), "path count, {tag}");
        let index: HashMap<&str, usize> = keys.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

        let scale = (1..=keys.len())
            .flat_map(|i| f.num(&format!("{c}/coefs/{i}")).iter())
            .fold(0.0f64, |a, v| a.max(v.abs()));
        let mut worst = 0.0f64;
        for (path, coef) in op.paths.iter().zip(&out) {
            let mut parts = Vec::new();
            if path.order > 0 {
                if p.format == Format::Time {
                    parts.push(path.n1.unwrap() - 1);
                }
                if path.order == 2 {
                    parts.push(path.n2.unwrap() - 1);
                }
                parts.push(path.n_fr.unwrap() - 1);
            }
            let key = parts.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            let i = *index.get(key.as_str()).unwrap_or_else(|| panic!("{tag}: no reference for path {key}"));
            let want = f.num(&format!("{c}/coefs/{}", i + 1));

            // References are column-major [band, time]; ours are by rows.
            let got: Vec<f64> = (0..coef.cols)
                .flat_map(|t| (0..coef.rows).map(move |r| coef.data[r * coef.cols + t]))
                .collect();
            assert_eq!(got.len(), want.len(), "size of path {key} ({}), {tag}", path.label);
            let err = got.iter().zip(want).fold(0.0f64, |a, (g, w)| a.max((g - w).abs()));
            worst = worst.max(err);
        }
        if show {
            eprintln!("{tag}: {:.2e}", worst / scale);
        }
        assert!(worst / scale < 1e-10, "coefficients, {tag}: relative error {:e}", worst / scale);
    }
}
