//! C interface between the R package wavscatengine and the Rust core.
//!
//! Deliberately plain: no extendr, nothing generated. Every entry point
//! returns a [`WseResult`], a list of numeric arrays with their shapes plus a
//! list of strings, or an error message; one small C function (in
//! `src/init.c`) turns any result into an R list. Numbers cross as doubles in
//! fixed positions, so every value arrives exactly, and matrices are laid out
//! column-major, as R stores them.
//!
//! Panics are caught at the boundary and returned as errors, because
//! unwinding into C is undefined behaviour.

use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use wavscat_core::jtfs::{Format, Mat, OutType, ParamsJtfs, ScatteringJtfs};
use wavscat_core::scattering1d::{Params1d, Scattering1d, TSpec};

/// Numeric arrays, each with `(rows, cols)`; `cols == 0` marks a plain
/// vector of length `rows`. Plus strings, or an error.
pub struct WseResult {
    arrays: Vec<(Vec<f64>, usize, usize)>,
    strings: Vec<CString>,
    error: Option<CString>,
}

impl WseResult {
    fn empty() -> Self {
        WseResult { arrays: Vec::new(), strings: Vec::new(), error: None }
    }

    fn vector(&mut self, v: Vec<f64>) {
        let n = v.len();
        self.arrays.push((v, n, 0));
    }

    /// A row-major matrix, stored column-major for R.
    fn matrix(&mut self, m: &Mat) {
        let mut v = Vec::with_capacity(m.rows * m.cols);
        for c in 0..m.cols {
            for r in 0..m.rows {
                v.push(m.data[r * m.cols + c]);
            }
        }
        self.arrays.push((v, m.rows, m.cols));
    }

    fn string(&mut self, s: &str) {
        self.strings.push(CString::new(s.replace('\0', " ")).unwrap());
    }
}

fn boxed(r: WseResult) -> *mut WseResult {
    Box::into_raw(Box::new(r))
}

fn failure(msg: impl std::fmt::Display) -> *mut WseResult {
    let mut r = WseResult::empty();
    r.error = Some(CString::new(msg.to_string().replace('\0', " ")).unwrap());
    boxed(r)
}

/// Run `f`, turning errors and panics into an error result.
fn guarded(f: impl FnOnce() -> Result<WseResult, String>) -> *mut WseResult {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(r)) => boxed(r),
        Ok(Err(e)) => failure(e),
        Err(_) => failure("internal error in wavscatengine (a Rust panic); please report it"),
    }
}

/// # Safety
/// `p` must point to `n` doubles, or be null with `n == 0`.
unsafe fn doubles<'a>(p: *const f64, n: usize) -> &'a [f64] {
    if n == 0 { &[] } else { unsafe { slice::from_raw_parts(p, n) } }
}

fn opt(v: f64) -> Option<f64> {
    if v.is_nan() { None } else { Some(v) }
}

/// `T` or `F` from a kind code (0 default, 1 samples, 2 global) and a value.
fn support(kind: f64, value: f64) -> TSpec {
    match kind as i64 {
        1 => TSpec::Samples(value),
        2 => TSpec::Global,
        _ => TSpec::Default,
    }
}

fn q_pair(q1: f64, q2: f64) -> Vec<u32> {
    if q2.is_nan() { vec![q1 as u32] } else { vec![q1 as u32, q2 as u32] }
}

// --- Results -----------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wse_result_error(r: *const WseResult) -> *const c_char {
    match unsafe { &*r }.error.as_ref() {
        Some(e) => e.as_ptr(),
        None => ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn wse_result_n_arrays(r: *const WseResult) -> usize {
    unsafe { &*r }.arrays.len()
}

/// Array `i`, with its shape written to `rows` and `cols`.
#[no_mangle]
pub extern "C" fn wse_result_array(r: *const WseResult, i: usize, rows: *mut usize, cols: *mut usize) -> *const f64 {
    let (v, nr, nc) = &unsafe { &*r }.arrays[i];
    unsafe {
        *rows = *nr;
        *cols = *nc;
    }
    v.as_ptr()
}

#[no_mangle]
pub extern "C" fn wse_result_n_strings(r: *const WseResult) -> usize {
    unsafe { &*r }.strings.len()
}

#[no_mangle]
pub extern "C" fn wse_result_string(r: *const WseResult, i: usize) -> *const c_char {
    unsafe { &*r }.strings[i].as_ptr()
}

#[no_mangle]
pub extern "C" fn wse_result_free(r: *mut WseResult) {
    if !r.is_null() {
        drop(unsafe { Box::from_raw(r) });
    }
}

// --- Version and self-check ----------------------------------------------------

/// Strings: numerics version, crate version.
#[no_mangle]
pub extern "C" fn wse_version() -> *mut WseResult {
    guarded(|| {
        let mut r = WseResult::empty();
        r.string(wavscat_core::NUMERICS_VERSION);
        r.string(env!("CARGO_PKG_VERSION"));
        Ok(r)
    })
}

/// Strings: every golden-record mismatch on this machine; none is a pass.
#[no_mangle]
pub extern "C" fn wse_verify() -> *mut WseResult {
    guarded(|| {
        let mut r = WseResult::empty();
        for d in wavscat_core::verify::verify() {
            r.string(&format!("wavscat-core: {d}"));
        }
        Ok(r)
    })
}

// --- Time scattering --------------------------------------------------------------

/// Time scattering of `channels` signals of length `n`, stored one after
/// another in `x`. Parameters: `[n, J, Q1, Q2 (NaN for one), T kind, T,
/// max_order, stride (NaN for default)]`.
///
/// Arrays: for each channel, each path's coefficients. Strings: path labels.
#[no_mangle]
pub extern "C" fn wse_s1d(p: *const f64, np: usize, x: *const f64, n: usize, channels: usize) -> *mut WseResult {
    guarded(|| {
        let p = unsafe { doubles(p, np) };
        if p.len() != 8 {
            return Err("wse_s1d needs 8 parameters".into());
        }
        let mut params = Params1d::new(p[0] as usize, p[1] as u32, q_pair(p[2], p[3]));
        params.t = support(p[4], p[5]);
        params.max_order = p[6] as u8;
        params.stride = opt(p[7]);
        let op = Scattering1d::new(&params).map_err(|e| e.0)?;
        let x = unsafe { doubles(x, n * channels) };
        let mut r = WseResult::empty();
        for c in 0..channels {
            for v in op.transform(&x[c * n..(c + 1) * n]).map_err(|e| e.0)? {
                r.vector(v);
            }
        }
        for path in &op.paths {
            r.string(&path.label);
        }
        Ok(r)
    })
}

// --- Joint time-frequency scattering --------------------------------------------

fn jtfs_op(p: &[f64]) -> Result<ScatteringJtfs, String> {
    if p.len() != 14 {
        return Err("joint scattering needs 14 parameters".into());
    }
    let mut params = ParamsJtfs::new(p[0] as usize, p[1] as u32, q_pair(p[2], p[3]));
    params.time.t = support(p[4], p[5]);
    params.time.stride = opt(p[6]);
    params.j_fr = p[7] as u32;
    params.q_fr = p[8] as u32;
    params.f = support(p[9], p[10]);
    params.stride_fr = opt(p[11]);
    params.format = if p[12] == 1.0 { Format::Joint } else { Format::Time };
    params.out_type = if p[13] == 1.0 { OutType::List } else { OutType::Array };
    ScatteringJtfs::new(&params).map_err(|e| e.0)
}

/// Joint scattering of `channels` signals. Parameters: `[n, J, Q1, Q2, T
/// kind, T, stride, J_fr, Q_fr, F kind, F, stride_fr, format (0 time, 1
/// joint), out_type (0 array, 1 list)]`.
///
/// Arrays: for each channel, each path's `[band, time]` matrix, then that
/// channel's S1 `[band, time]`. Strings: path labels.
#[no_mangle]
pub extern "C" fn wse_jtfs(p: *const f64, np: usize, x: *const f64, n: usize, channels: usize) -> *mut WseResult {
    guarded(|| {
        let op = jtfs_op(unsafe { doubles(p, np) })?;
        let x = unsafe { doubles(x, n * channels) };
        let mut r = WseResult::empty();
        for c in 0..channels {
            let (coefs, s1) = op.transform_with_s1(&x[c * n..(c + 1) * n]).map_err(|e| e.0)?;
            for m in &coefs {
                r.matrix(m);
            }
            r.matrix(&s1);
        }
        for path in &op.paths {
            r.string(&path.label);
        }
        Ok(r)
    })
}

/// Renormalise one channel of joint coefficients by its S1. `data` holds
/// every path's matrix column-major, one after another, with `dims` giving
/// `(rows, cols)` for each of `n_paths` paths. Arrays: the renormalised
/// paths.
#[no_mangle]
pub extern "C" fn wse_jtfs_renorm(
    p: *const f64,
    np: usize,
    data: *const f64,
    dims: *const i32,
    n_paths: usize,
    s1: *const f64,
    s1_rows: usize,
    s1_cols: usize,
    eps: f64,
) -> *mut WseResult {
    guarded(|| {
        let op = jtfs_op(unsafe { doubles(p, np) })?;
        let dims = if n_paths == 0 { &[][..] } else { unsafe { slice::from_raw_parts(dims, 2 * n_paths) } };
        let total: usize = dims.chunks(2).map(|d| d[0] as usize * d[1] as usize).sum();
        let data = unsafe { doubles(data, total) };
        // Column-major from R to row-major for the core.
        let from_r = |v: &[f64], rows: usize, cols: usize| {
            let mut m = vec![0.0; rows * cols];
            for c in 0..cols {
                for rr in 0..rows {
                    m[rr * cols + c] = v[c * rows + rr];
                }
            }
            Mat { rows, cols, data: m }
        };
        let mut offset = 0;
        let mut mats = Vec::with_capacity(n_paths);
        for d in dims.chunks(2) {
            let (rows, cols) = (d[0] as usize, d[1] as usize);
            mats.push(from_r(&data[offset..offset + rows * cols], rows, cols));
            offset += rows * cols;
        }
        let s1 = from_r(unsafe { doubles(s1, s1_rows * s1_cols) }, s1_rows, s1_cols);
        op.renorm(&mut mats, &s1, eps).map_err(|e| e.0)?;
        let mut r = WseResult::empty();
        for m in &mats {
            r.matrix(m);
        }
        Ok(r)
    })
}
