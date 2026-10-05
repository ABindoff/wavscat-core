//! Bit-for-bit reproducibility of the video pipeline: the outputs on this
//! target must match the golden record exactly. See `tapvid::verify`.
//!
//! Regenerate after a deliberate change to the numerics:
//!
//! ```text
//! WAVSCAT_UPDATE_GOLDEN=1 cargo test -p tapvid --test golden
//! ```

use tapvid::verify::{golden_report, GOLDEN_RECORD};
use wavscat_core::verify::differences;

#[test]
fn outputs_are_bit_identical_to_the_golden_record() {
    let got = golden_report();
    if std::env::var_os("WAVSCAT_UPDATE_GOLDEN").is_some() {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/golden.tsv"), &got).unwrap();
        return;
    }
    let diffs = differences(&got, GOLDEN_RECORD);
    assert!(diffs.is_empty(), "outputs differ on this target:\n{}", diffs.join("\n"));
}
