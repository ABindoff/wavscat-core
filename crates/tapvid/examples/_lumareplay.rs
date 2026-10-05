use tapvid::features::{trial_features, FeatureParams};
use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::{analyse_trial, PipelineParams};
use tapvid::qc::{run_trial, QcParams};
fn main() {
    for path in std::env::args().skip(1) {
        let b = std::fs::read(&path).unwrap();
        let u32at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        let (n, w, h) = (u32at(0), u32at(4), u32at(8));
        let mut ing = Ingest::new(IngestParams { capacity: n.max(1024), ..IngestParams::default() }).unwrap();
        let mut o = 12;
        for _ in 0..n {
            let us = i64::from_le_bytes(b[o..o + 8].try_into().unwrap());
            ing.push_frame(&b[o + 8..o + 8 + w * h], w, w, h, us).unwrap();
            o += 8 + w * h;
        }
        let name = path.rsplit('/').next().unwrap();
        for (label, p) in [("single", PipelineParams { extract: None, ..PipelineParams::default() }), ("extract", PipelineParams::default())] {
            let r = analyse_trial(&ing, &p).unwrap();
            let q = run_trial(&ing, &p, &QcParams::default());
            let f = trial_features(&r, &p, &QcParams::default(), &FeatureParams::default()).unwrap();
            let get = |k: &str| f.values[f.names.iter().position(|x| x == k).unwrap()];
            println!("{name:12} {label:8} f0 {:.3} cycles {:2} usable {:.2} cv {:.3} median {:.0} ms hesit {} score {:.2} band {:?} competitor {:.2} spread {:.3} harm {} | QC {}",
                r.f0_hz, r.cycles.itis.len(), get("usable_fraction"), get("iti_cv"), 1000.0 * get("iti_median"), get("hesitations"), r.score,
                r.band_fraction.map(|x| (x * 100.0).round() / 100.0), r.competitor_ratio, r.loading_spread, r.timing_harmonic,
                if q.qc.accepted() { "accepted".to_string() } else { q.qc.reasons.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("; ") });
        }
    }
}
