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
| FFT libraries choose SIMD kernels at run time | The FFT is our own, in scalar code: textbook radix-2 decimation in time, run in a cache-blocked order that tests prove bit-identical to the textbook order, with Bluestein for other lengths. Real signals go through a half-length complex FFT. Twiddle angles are reduced in integers. |
| Fused multiply-add, fast-math | Rust never contracts or reassociates floating-point operations by default. Shipped builds use no `target-cpu=native` and no relaxed SIMD. |
| Parallel reductions | Every sum runs sequentially in index order. Parallelism is only ever across signals or paths. |

The guarantee covers samples in, features out. Audio decoding and resampling
are outside it: browsers resample differently, so a web app must supply samples
at the operator's rate.

## Verification

- `tests/kymatio_parity.rs` and `tests/kymatio_parity_jtfs.rs` check agreement
  with Kymatio, using the reference values from the R package, at the R
  package's tolerances. Observed relative errors are between 3e-16 and 6e-14.
- `wavscat_core::verify` hashes the exact output bits of a fixed set of
  transforms and feature pipelines, and compares them with the record in
  `crates/wavscat-core/golden.tsv`, which is embedded in the library. CI runs it
  natively on x86-64 and ARM64 under Linux, macOS and Windows, under WASI, and
  through the wasm binding in Chromium, Firefox and WebKit. To run it locally as
  wasm under Node:

      cargo test --release --target wasm32-wasip1 --test golden --no-run
      node tools/run-wasi.mjs target/wasm32-wasip1/release/deps/golden-*.wasm

## Checking a device

Because the record is embedded, an application can call `verify()` on a
participant's own phone or tablet and confirm that it computes exactly what
every other platform computes, before trusting its features. To check a device
by hand, build the wasm module with `sh tools/build-wasm.sh`, serve
`crates/wavscat-wasm/` over HTTP, and open `js/browser.html` on the device.
`node crates/wavscat-wasm/js/test-browser.mjs` does the same in Playwright's
browsers; set `BROWSERS=chrome,msedge` to use installed ones instead.

## JavaScript

```js
import init, { Scattering1d, ScatteringJtfs, verify } from "./wavscat_wasm.js";
await init();
const op = new ScatteringJtfs({ n: 900, J: 7, J_fr: 3, Q: 8, T_sec: 6, sr: 30 });
const S = op.transform(signal);   // signal: Float64Array of length 900
op.paths();                       // one entry per path, R's labels
S.path(5);                        // one path's coefficients
```

Parameter names are the R argument names, so one parameter set means the same
thing in every language. Unknown names are rejected.

To refresh the fixtures from the R package, run
`Rscript tools/export-fixtures.R ../wavscat/tests/testthat/fixtures fixtures`.

## Status

The numerics are not frozen until 1.0. Changing the FFT algorithm now, for
speed, is expected and free. After 1.0, any change that alters an output bit
needs a `NUMERICS_VERSION` bump and a new golden record.

- [x] Filter banks, padding, 1D scattering (orders 1 and 2; local, global and
      no averaging), feature post-processing
- [x] Faster FFT: cache-blocked radix-2, real-input transforms, fused filter-and-subsample
- [ ] Optional before freezing: radix-4 butterflies (fewer multiplications, different rounding)
- [x] Joint time-frequency scattering (both formats; local, global and no
      averaging in time and frequency)
- [x] Mixed-radix FFT with paired prime DFTs, and batched column transforms
      for the frequential axis
- [x] wasm-bindgen binding, a device self-check, and CI across targets and browsers
- [ ] JTFS S2/S1 renormalisation (by the time-scattering S1 of band n1)
- [ ] Video tapping pipeline, as a separate crate
- [ ] R binding through extendr; Python binding through PyO3

## Licence

BSD-3-Clause. The filter bank and cascade design follow Kymatio (BSD-3-Clause).
