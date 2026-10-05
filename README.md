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

## R

The R package [wavscat](https://github.com/ABindoff/wavscat) computes
everything in base R and needs no compiler. The optional companion package
`r/wavscatengine` adds this engine: with it installed, wavscat computes the
coefficients in Rust, bit-for-bit identically to Python and the browser
(`options(wavscat.engine = "auto")`, the default; `"r"` opts out). It also
runs the webcam tapping pipeline from R:

```r
remotes::install_github("ABindoff/wavscat-core", subdir = "r/wavscatengine")
wavscatengine::engine_verify()   # character(0): this machine has the reference bits
session <- wavscatengine::tapping_session()
wavscatengine::tapping_push(session, frame, width, height, timestamp_us)
report <- wavscatengine::tapping_finish(session)
```

It needs Rust (`cargo`) and, on Windows, Rtools and the GNU target
(`rustup target add x86_64-pc-windows-gnu`). The binding is plain C and
`.Call`, without extendr: Rust exports a few C functions, and `src/init.c`
turns their results into R lists. The package carries its own copy of the
crates it needs, refreshed by `tools/sync-r-engine.sh`; CI fails if the copy
drifts.

## Python

`crates/wavscat-py` builds a `wavscat` module with maturin
(`maturin develop --release` from that directory). It has the same classes
and the same R argument names as keyword arguments:

```python
import numpy as np, wavscat
assert wavscat.verify() == []          # this machine computes the reference bits
op = wavscat.ScatteringJtfs(n=900, J=7, J_fr=3, Q=(8, 1), T_sec=6, sr=30)
coefs, s1 = op.transform_with_s1(x)    # x: float64 array of length 900
session = wavscat.TappingSession()     # push_frame(luma_uint8_2d, timestamp_us)
```

The same synthetic trial analysed in Python (native) and in JavaScript (wasm,
SIMD) gives every feature value and inter-tap interval bit for bit equal.

### Webcam finger tapping

`TappingSession` runs the whole video pipeline (`crates/tapvid`) on the
participant's device. It keeps only a 64 x 48 grid of each frame:

```js
import init, { TappingSession, wasmMemory, verify } from "./wavscat_wasm.js";
await init();
const session = new TappingSession();

// Each frame, with its capture time in microseconds. Either pass the luma
// plane, or write it straight into wasm memory with no intermediate copy:
const ptr = session.stagingPointer(byteLength);
await videoFrame.copyTo(new Uint8Array(wasmMemory().buffer, ptr, byteLength), { rect, layout });
session.pushStaged(stride, width, height, timestampUs, /* rgba */ false);

// At the end of the trial:
const report = session.finish();
// { accepted, reasons, qc, features: { names, values, params_hash, ... }, itis }
```

The staging buffer is zeroed after every push, so no raw frame stays even in
wasm memory. On a laptop under V8, pushing a 320 x 240 frame takes 0.06 ms
and `finish()` on a 30 s trial about 100 ms; peak wasm memory for 30 s of
HD frames is under 27 MB. The build uses WebAssembly SIMD, which changes no
output bit; set `WAVSCAT_WASM_SIMD=0` when running `tools/build-wasm.sh` to
support browsers from before 2023.

### Checking the video against hand landmarks

`apps/tap-compare` is a page for validating the landmark-free pipeline on a
real hand. It records three 10 s trials (left hand, right hand, and both hands
tapping in anti-phase) and sends every frame both to a `TappingSession` and to
MediaPipe Hands. Each hand's thumb–index aperture then goes through
`analyseTrace`, which applies the same drift removal, resampling, cycle timing
and features as the video's selected component. The page compares the two:
the fundamental, waveform correlation, tap timing, inter-tap intervals and
the JTFS features, with the component's spatial loading drawn under the
landmark positions. `TappingSession.diagnose()` supplies the ungated analysis
behind each verdict.

    sh tools/build-wasm.sh
    node apps/tap-compare/serve.mjs     # then open http://localhost:8000/apps/tap-compare/

The page loads MediaPipe from jsDelivr and its hand model from Google; no
frame leaves the browser. The JSON download holds the reports, the traces and
the landmark coordinates, but no images.

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
- [x] JTFS renormalisation: each second-order path divided by S1 of the bands
      it spans, through the same frequential low-pass
- [x] Video tapping pipeline (`crates/tapvid`), every stage of the brief:
  - 1, streaming ingest into a box-averaged 64 x 48 ring buffer; no raw frame
    is retained and no frame allocates
  - 2, exposure gain, centring and B-spline drift removal on true
    timestamps, applied on the fly (no copy of the frames)
  - 3, randomized SVD: seeded, sign-fixed, f32 frames with f64 accumulation
  - 4, resampling to a uniform grid on true timestamps
  - 5, component selection and f0, with phase locking across components to
    recognise harmonics of the hand and of slow motion such as a sway
  - 6, sub-frame inter-tap intervals from the analytic signal
  - 7, the feature vector: interval and amplitude features plus JTFS of the
    unit-RMS component, stamped with versions and a hash of every parameter
  - 8, QC gating: a report for every trial, with rejection reasons for
    capture, lighting, interruption, weak or competing rhythm and diffuse
    motion, every threshold a parameter
  - `analyse_trial` and `run_trial` end to end, synthetic signals and videos
    with known ground truth, and the wasm `TappingSession`
- [x] Python binding through PyO3 (`crates/wavscat-py`), tested in CI on Linux
      (x86-64 and ARM64), macOS and Windows
- [x] R: the optional `wavscatengine` package (plain C and `.Call`, no
      extendr), used automatically by `wavscat` when installed; tested in CI
      on Linux, macOS and Windows
- [ ] CRAN: vendor the crates.io dependencies (`cargo vendor`) for an
      offline build, then the `cran-rust` checklist

## Licence

BSD-3-Clause. The filter bank and cascade design follow Kymatio (BSD-3-Clause).
