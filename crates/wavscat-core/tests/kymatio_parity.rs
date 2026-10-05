//! Agreement with Kymatio, using the reference values from the R package.
//! Tolerances are those of `tests/testthat/test-kymatio-parity.R`.

mod common;

use common::{assert_close, Fixture, Leaf};
use wavscat_core::filter_bank::{self as fb, BankParams, Generator};
use wavscat_core::scattering1d::{Params1d, Scattering1d, TSpec};

const ALPHA: f64 = 5.0;
const SIGMA0: f64 = 0.1;
const R_PSI: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[test]
fn scalar_helpers() {
    let f = Fixture::load("kymatio-filter-bank");
    let p: Vec<f64> = f
        .num("scalars/sigmas")
        .iter()
        .map(|&s| fb::adaptive_choice_p(s, 1e-7) as f64)
        .collect();
    assert_close(&p, f.num("scalars/P"), 0.0, "P");
    let qs: Vec<u32> = f.num("scalars/Q").iter().map(|&q| q as u32).collect();
    let xm: Vec<f64> = qs.iter().map(|&q| fb::compute_xi_max(q)).collect();
    assert_close(&xm, f.num("scalars/xi_max"), 1e-14, "xi_max");
    let sp: Vec<f64> = qs.iter().map(|&q| fb::compute_sigma_psi(0.4, q, R_PSI)).collect();
    assert_close(&sp, f.num("scalars/sigma_psi"), 1e-14, "sigma_psi");
    let ms: Vec<f64> = f
        .num("scalars/pair_xi")
        .iter()
        .zip(f.num("scalars/pair_sigma"))
        .map(|(&x, &s)| fb::get_max_dyadic_subsampling(x, s, ALPHA) as f64)
        .collect();
    assert_close(&ms, f.num("scalars/maxsub"), 0.0, "maxsub");
}

#[test]
fn morlet_and_gauss_filters() {
    let f = Fixture::load("kymatio-filter-bank");
    for i in 0..5 {
        let m = f.row("morlet/meta", i);
        let got = fb::morlet_1d(m[0] as usize, Some(m[1]), m[2]);
        let want = f.num(&format!("morlet/values/{}", i + 1));
        assert_close(&got, want, 1e-12, &format!("morlet {m:?}"));
    }
    for i in 0..4 {
        let m = f.row("gauss/meta", i);
        let got = fb::gauss_1d(m[0] as usize, m[1]);
        let want = f.num(&format!("gauss/values/{}", i + 1));
        assert_close(&got, want, 1e-12, &format!("gauss {m:?}"));
    }
    for i in 0..3 {
        let m = f.row("tsupp/meta", i);
        let got = fb::compute_temporal_support(&fb::gauss_1d(m[0] as usize, m[1]), 1e-3).unwrap();
        assert_eq!(got as f64, f.num("tsupp/values")[i], "temporal support {m:?}");
    }
}

#[test]
fn generator_schedule() {
    let f = Fixture::load("kymatio-filter-bank");
    for i in 0..6 {
        let m = f.row("generator/meta", i);
        let spec = fb::anden_generator(m[0] as u32, m[1] as u32, SIGMA0, R_PSI);
        let xi: Vec<f64> = spec.iter().map(|s| s.0).collect();
        let sg: Vec<f64> = spec.iter().map(|s| s.1).collect();
        assert_close(&xi, f.num(&format!("generator/xi/{}", i + 1)), 1e-13, &format!("xi {m:?}"));
        assert_close(&sg, f.num(&format!("generator/sigma/{}", i + 1)), 1e-13, &format!("sigma {m:?}"));
    }
}

/// The levels stored for one filter: a bare vector when there is one level,
/// otherwise a list.
fn stored_levels<'a>(f: &'a Fixture, key: &str) -> Vec<&'a [f64]> {
    if f.has(key) {
        vec![f.num(key)]
    } else {
        (1..)
            .map(|l| format!("{key}/{l}"))
            .take_while(|k| f.has(k))
            .map(|k| f.num(&k))
            .collect()
    }
}

#[test]
fn assembled_banks_at_every_resolution() {
    let f = Fixture::load("kymatio-filter-bank");
    for e in 1..=4 {
        let cfg = f.num(&format!("factory/{e}/cfg")).to_vec();
        let tag = format!("{cfg:?}");
        let banks = fb::filter_factory(
            cfg[0] as usize,
            cfg[1] as u32,
            &[cfg[2] as u32, cfg[3] as u32],
            cfg[4],
            BankParams::default(),
            Generator::Anden,
        );
        assert_eq!(banks.phi.j as f64, f.scalar(&format!("factory/{e}/phi/j")), "phi j {tag}");
        assert_close(&[banks.phi.sigma], f.num(&format!("factory/{e}/phi/sigma")), 1e-14, "phi sigma");
        let want_phi = stored_levels(&f, &format!("factory/{e}/phi/levels"));
        assert_eq!(banks.phi.levels.len(), want_phi.len(), "phi levels {tag}");
        for (lev, w) in want_phi.iter().enumerate() {
            assert_close(&banks.phi.levels[lev], w, 1e-12, &format!("phi level {lev} {tag}"));
        }
        for ord in 1..=2 {
            let bank = &banks.banks[ord - 1];
            let key = format!("factory/{e}/psi{ord}");
            assert_eq!(bank.len(), f.num(&format!("{key}/xi")).len(), "psi{ord} count {tag}");
            let xi: Vec<f64> = bank.iter().map(|p| p.xi).collect();
            let sg: Vec<f64> = bank.iter().map(|p| p.sigma).collect();
            let jj: Vec<f64> = bank.iter().map(|p| p.j as f64).collect();
            assert_close(&xi, f.num(&format!("{key}/xi")), 1e-13, "xi");
            assert_close(&sg, f.num(&format!("{key}/sigma")), 1e-13, "sigma");
            assert_close(&jj, f.num(&format!("{key}/j")), 0.0, "j");
            for (n, p) in bank.iter().enumerate() {
                let want = stored_levels(&f, &format!("{key}/levels/{}", n + 1));
                assert_eq!(p.levels.len(), want.len(), "psi{ord}[{n}] levels {tag}");
                for (lev, w) in want.iter().enumerate() {
                    assert_close(&p.levels[lev], w, 1e-12, &format!("psi{ord}[{n}] level {lev} {tag}"));
                }
            }
        }
    }
}

fn params_for(f: &Fixture, c: usize) -> Params1d {
    let k = |s: &str| format!("{c}/{s}");
    let q: Vec<u32> = f.num(&k("Q")).iter().map(|&q| q as u32).collect();
    let mut p = Params1d::new(f.scalar(&k("n")) as usize, f.scalar(&k("J")) as u32, q);
    p.max_order = f.scalar(&k("max_order")) as u8;
    p.stride = f.opt_scalar(&k("stride"));
    p.t = match f.leaves.get(&k("T")) {
        Some(Leaf::Num { values, .. }) => TSpec::Samples(values[0]),
        Some(Leaf::Str(s)) if s[0] == "global" => TSpec::Global,
        _ => TSpec::Default,
    };
    p
}

#[test]
fn full_transforms() {
    let f = Fixture::load("kymatio-transforms");
    for c in 1..=12 {
        let p = params_for(&f, c);
        let tag = format!("case {c}: {p:?}");
        let op = Scattering1d::new(&p).unwrap_or_else(|e| panic!("{tag}: {e}"));
        let out = op.transform(f.num(&format!("{c}/x"))).unwrap();

        let order: Vec<f64> = op.paths.iter().map(|p| p.order as f64).collect();
        assert_close(&order, f.num(&format!("{c}/meta/order")), 0.0, &format!("order, {tag}"));
        let n1: Vec<f64> = op.paths.iter().map(|p| p.n1.map_or(-1.0, |v| v as f64 - 1.0)).collect();
        assert_close(&n1, &f.col(&format!("{c}/meta/n"), 0), 0.0, &format!("n1, {tag}"));
        let xi1: Vec<f64> = op.paths.iter().map(|p| p.xi1.unwrap_or(-1.0)).collect();
        assert_close(&xi1, &f.col(&format!("{c}/meta/xi"), 0), 1e-13, &format!("xi1, {tag}"));
        if p.max_order >= 2 {
            let n2: Vec<f64> = op.paths.iter().map(|p| p.n2.map_or(-1.0, |v| v as f64 - 1.0)).collect();
            assert_close(&n2, &f.col(&format!("{c}/meta/n"), 1), 0.0, &format!("n2, {tag}"));
            let j2: Vec<f64> = op.paths.iter().map(|p| p.j2.map_or(-1.0, |v| v as f64)).collect();
            assert_close(&j2, &f.col(&format!("{c}/meta/j"), 1), 0.0, &format!("j2, {tag}"));
        }

        let out_key = format!("{c}/out");
        if f.has(&out_key) {
            // Array output: a [path, time] matrix, column-major.
            let d = f.dims(&out_key).to_vec();
            let n_time = if d.len() == 2 { d[1] } else { f.num(&out_key).len() / out.len() };
            assert_eq!(out.len() * n_time, f.num(&out_key).len(), "size, {tag}");
            assert!(out.iter().all(|p| p.len() == n_time), "time, {tag}");
            let flat: Vec<f64> = (0..n_time).flat_map(|t| out.iter().map(move |p| p[t])).collect();
            assert_close(&flat, f.num(&out_key), 1e-10, &format!("coefs, {tag}"));
        } else {
            for (i, path) in out.iter().enumerate() {
                let want = f.num(&format!("{out_key}/{}", i + 1));
                assert_close(path, want, 1e-10, &format!("path {i}, {tag}"));
            }
        }
    }
}
