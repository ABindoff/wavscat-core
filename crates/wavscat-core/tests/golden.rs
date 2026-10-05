//! Bit-for-bit reproducibility: the outputs on this target must match the
//! golden record exactly. See `wavscat_core::verify`.
//!
//! Regenerate after a deliberate change to the numerics, which must come with
//! a bump to `NUMERICS_VERSION`:
//!
//!     WAVSCAT_UPDATE_GOLDEN=1 cargo test --test golden

use wavscat_core::verify;

#[test]
fn outputs_are_bit_identical_to_the_golden_record() {
    let report = verify::golden_report();
    if std::env::var_os("WAVSCAT_UPDATE_GOLDEN").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/golden.tsv");
        std::fs::write(path, &report).expect("write golden record");
        return;
    }
    let diffs = verify::differences(&report, verify::GOLDEN_RECORD);
    assert!(
        diffs.is_empty(),
        "outputs differ from the golden record on this target:\n{}",
        diffs.join("\n")
    );
}
