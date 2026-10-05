//! Renormalisation of joint scattering by first-order energy.

use wavscat_core::jtfs::{Format, OutType, ParamsJtfs, ScatteringJtfs};
use wavscat_core::scattering1d::{Scattering1d, TSpec};
use wavscat_core::verify::signal;

fn params(f: TSpec, format: Format) -> ParamsJtfs {
    // The video tapping configuration: 30 s at 30 Hz, 6 s averaging.
    let mut p = ParamsJtfs::new(900, 7, vec![8, 1]);
    p.time.t_sec = Some(6.0);
    p.time.sr = Some(30.0);
    p.f = f;
    p.format = format;
    p.out_type = OutType::List;
    p
}

#[test]
fn joint_s1_is_bit_identical_to_time_scattering_s1() {
    let p = params(TSpec::Default, Format::Time);
    let jtfs = ScatteringJtfs::new(&p).unwrap();
    let time = Scattering1d::new(&p.time).unwrap();
    let x = signal(900, 11);
    let (_, s1) = jtfs.transform_with_s1(&x).unwrap();
    let t = time.transform(&x).unwrap();
    let first: Vec<&Vec<f64>> = time.paths.iter().zip(&t).filter(|(p, _)| p.order == 1).map(|(_, v)| v).collect();
    assert_eq!(s1.rows, first.len());
    for (b, want) in first.iter().enumerate() {
        let got = s1.row(b);
        assert!(got.iter().zip(want.iter()).all(|(g, w)| g.to_bits() == w.to_bits()), "band {}", b + 1);
    }
}

#[test]
fn without_frequential_averaging_each_spun_row_divides_by_its_own_band() {
    let p = params(TSpec::Samples(0.0), Format::Time);
    let op = ScatteringJtfs::new(&p).unwrap();
    let x = signal(900, 12);
    let (coefs, s1) = op.transform_with_s1(&x).unwrap();
    let mut renormed = coefs.clone();
    let eps = 1e-9;
    op.renorm(&mut renormed, &s1, eps).unwrap();
    let mut checked = 0;
    for ((path, raw), got) in op.paths.iter().zip(&coefs).zip(&renormed) {
        if path.order != 2 {
            assert_eq!(raw, got, "{} should be unchanged", path.label);
            continue;
        }
        // Unspun paths came from the J_fr low-pass, so they divide by S1
        // through that low-pass instead; the amplitude test covers them.
        if path.spin == Some(0) {
            continue;
        }
        let parent = s1.row(path.n1.unwrap() - 1);
        for ((r, g), d) in raw.data.iter().zip(&got.data).zip(parent) {
            assert_eq!(g.to_bits(), (r / (d + eps)).to_bits(), "{}", path.label);
        }
        checked += 1;
    }
    assert!(checked > 0);
}

/// The property that motivates renormalisation: scaling the signal, as a
/// nearer camera or a larger hand would, leaves the features unchanged.
#[test]
fn renormalised_features_do_not_depend_on_amplitude() {
    let cases = [
        ("time, local F", TSpec::Default, Format::Time),
        ("time, F = 0", TSpec::Samples(0.0), Format::Time),
        ("time, global F", TSpec::Global, Format::Time),
        ("joint, local F", TSpec::Default, Format::Joint),
    ];
    let x = signal(900, 13);
    for (name, f, format) in cases {
        let op = ScatteringJtfs::new(&params(f, format)).unwrap();
        let features = |scale: f64| {
            let xs: Vec<f64> = x.iter().map(|v| v * scale).collect();
            let (mut c, s1) = op.transform_with_s1(&xs).unwrap();
            op.renorm(&mut c, &s1, 1e-300).unwrap();
            c
        };
        let base = features(1.0);
        for scale in [1e-3, 37.0, 1e4] {
            let scaled = features(scale);
            for ((path, a), b) in op.paths.iter().zip(&base).zip(&scaled) {
                if path.order != 2 {
                    continue;
                }
                for (u, v) in a.data.iter().zip(&b.data) {
                    assert!(u.is_finite() && *u >= 0.0, "{name}, {}: {u}", path.label);
                    let rel = (u - v).abs() / u.abs().max(1e-300);
                    assert!(rel < 1e-9, "{name}, {}, scale {scale}: {u} vs {v}", path.label);
                }
            }
        }
    }
}

#[test]
fn global_time_averaging_is_refused() {
    let mut p = params(TSpec::Default, Format::Time);
    p.time.t_sec = None;
    p.time.t = TSpec::Global;
    let op = ScatteringJtfs::new(&p).unwrap();
    let (mut c, s1) = op.transform_with_s1(&signal(900, 14)).unwrap();
    let err = op.renorm(&mut c, &s1, 1e-12).unwrap_err();
    assert!(err.0.contains("local time averaging"), "{err}");
}
