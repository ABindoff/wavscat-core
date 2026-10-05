# wavscat-core

The numerical core of [wavscat](https://github.com/ABindoff/wavscat): wavelet
scattering and joint time-frequency scattering, in Rust, with one guarantee the
R package alone cannot give. For a given `NUMERICS_VERSION`, the same input
samples produce **bit-for-bit identical** coefficients and features on every
supported target: x86-64 and ARM64 under Linux, macOS and Windows, and
WebAssembly in Node and in browsers. R, Python and JavaScript all call this one
implementation, so features computed on a phone in a web app match those
computed on a server.

## How identity is kept

| Source of drift | Rule |
|---|---|
| Platform libm (`exp`, `cos`, `log`) differs in the last bit | All transcendental functions come from the pure-Rust `libm` crate, pinned exactly, with `arch` off. `clippy.toml` bans the `std` versions. |
| FFT libraries choose SIMD kernels at run time | The FFT is our own, in scalar code: radix-2, plus Bluestein for other lengths. Twiddle angles are reduced in integers. |
| Fused multiply-add, fast-math | Rust never contracts or reassociates floating-point operations by default. Shipped builds use no `target-cpu=native` and no relaxed SIMD. |
| Parallel reductions | Every sum runs sequentially in index order. Parallelism is only ever across signals or paths. |

The guarantee covers samples in, features out. Audio decoding and resampling
are outside it: browsers resample differently, so a web app must supply samples
at the operator's rate.

## Verification

- `tests/kymatio_parity.rs` checks agreement with Kymatio, using the reference
  values from the R package, at the R package's tolerances. Observed relative
  errors are between 7e-16 and 6e-14.
- `tests/golden.rs` hashes the exact output bits of a fixed set of transforms
  and feature pipelines, and compares them with `golden/numerics-<version>.tsv`.
  CI runs it on every target. To run it locally as wasm under Node:

      cargo test --release --target wasm32-wasip1 --test golden --no-run
      node tools/run-wasi.mjs target/wasm32-wasip1/release/deps/golden-*.wasm

To refresh the fixtures from the R package, run
`Rscript tools/export-fixtures.R ../wavscat/tests/testthat/fixtures fixtures`.

## Status

The numerics are not frozen until 1.0. Changing the FFT algorithm now, for
speed, is expected and free. After 1.0, any change that alters an output bit
needs a `NUMERICS_VERSION` bump and a new golden record.

- [x] Filter banks, padding, 1D scattering (orders 1 and 2; local, global and
      no averaging), feature post-processing
- [ ] Faster FFT (radix-4 or split-radix, real-input transforms) before freezing
- [ ] Joint time-frequency scattering
- [ ] wasm-bindgen binding, and CI across targets and browsers
- [ ] R binding through extendr; Python binding through PyO3

## Licence

BSD-3-Clause. The filter bank and cascade design follow Kymatio (BSD-3-Clause).
