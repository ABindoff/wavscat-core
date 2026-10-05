//! WebAssembly bindings for `wavscat-core`.
//!
//! A thin layer: every number is computed by the core, so features computed
//! here are bit-identical to those computed from R, Python or native Rust with
//! the same `numericsVersion()`.
//!
//! Parameters are plain objects whose keys are the R argument names (`n`, `J`,
//! `Q`, `T`, `T_sec`, `J_fr`, `F`, ...), so one parameter set means the same
//! thing in every language. Unknown keys are rejected rather than ignored, so
//! a typo cannot silently fall back to a default.
//!
//! ```js
//! import init, { Scattering1d, verify } from "./wavscat_wasm.js";
//! await init();
//! console.assert(verify().length === 0, "this device's arithmetic differs");
//! const op = new Scattering1d({ n: 900, J: 7, Q: [8, 1], T_sec: 4, sr: 30 });
//! const S = op.transform(signal);        // Float64Array of length n
//! const row = S.path(3);                 // one path's coefficients
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use wavscat_core::features::{self, Summary};
use wavscat_core::jtfs::{self, Format, OutType, ParamsJtfs};
use wavscat_core::scattering1d::{self as s1d, Params1d, TSpec};

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Version of the numerical definition. Features computed under different
/// versions are not comparable.
#[wasm_bindgen(js_name = numericsVersion)]
pub fn numerics_version() -> String {
    wavscat_core::NUMERICS_VERSION.to_string()
}

/// Run the golden cases on this device and compare every output bit with the
/// reference record. Returns the mismatches: an empty array means this
/// device computes exactly what every other supported platform computes.
#[wasm_bindgen]
pub fn verify() -> Vec<String> {
    wavscat_core::verify::verify()
}

/// The golden report itself, for diagnostics when `verify()` fails.
#[wasm_bindgen(js_name = goldenReport)]
pub fn golden_report() -> String {
    wavscat_core::verify::golden_report()
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// `Q` as one number or a pair.
#[derive(Deserialize)]
#[serde(untagged)]
enum QArg {
    One(u32),
    Two([u32; 2]),
}

impl QArg {
    fn into_vec(self) -> Vec<u32> {
        match self {
            QArg::One(q) => vec![q],
            QArg::Two(q) => q.to_vec(),
        }
    }
}

/// `T` or `F`: a number of samples (0 for none), or `"global"`.
#[derive(Deserialize)]
#[serde(untagged)]
enum SupportArg {
    Samples(f64),
    Named(String),
}

fn support(arg: Option<SupportArg>, name: &str) -> Result<TSpec, JsError> {
    match arg {
        None => Ok(TSpec::Default),
        Some(SupportArg::Samples(v)) => Ok(TSpec::Samples(v)),
        Some(SupportArg::Named(s)) if s == "global" => Ok(TSpec::Global),
        Some(SupportArg::Named(s)) => Err(JsError::new(&format!(
            "{name} must be a number, 0, \"global\", or omitted; got \"{s}\"."
        ))),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimeArgs {
    n: usize,
    #[serde(rename = "J")]
    j: u32,
    #[serde(rename = "Q", default)]
    q: Option<QArg>,
    #[serde(rename = "T", default)]
    t: Option<SupportArg>,
    #[serde(rename = "T_sec", default)]
    t_sec: Option<f64>,
    #[serde(default)]
    max_order: Option<u8>,
    #[serde(default)]
    stride: Option<f64>,
    #[serde(default)]
    sr: Option<f64>,
}

impl TimeArgs {
    fn into_params(self) -> Result<Params1d, JsError> {
        let mut p = Params1d::new(self.n, self.j, self.q.map_or(vec![8], QArg::into_vec));
        p.t = support(self.t, "T")?;
        p.t_sec = self.t_sec;
        p.max_order = self.max_order.unwrap_or(2);
        p.stride = self.stride;
        p.sr = self.sr;
        Ok(p)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JtfsArgs {
    n: usize,
    #[serde(rename = "J")]
    j: u32,
    #[serde(rename = "J_fr", default)]
    j_fr: Option<u32>,
    #[serde(rename = "Q", default)]
    q: Option<QArg>,
    #[serde(rename = "Q_fr", default)]
    q_fr: Option<u32>,
    #[serde(rename = "T", default)]
    t: Option<SupportArg>,
    #[serde(rename = "T_sec", default)]
    t_sec: Option<f64>,
    #[serde(rename = "F", default)]
    f: Option<SupportArg>,
    #[serde(default)]
    stride: Option<f64>,
    #[serde(default)]
    stride_fr: Option<f64>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    sr: Option<f64>,
    #[serde(default)]
    out_type: Option<String>,
}

impl JtfsArgs {
    fn into_params(self) -> Result<ParamsJtfs, JsError> {
        let mut p = ParamsJtfs::new(self.n, self.j, self.q.map_or(vec![8], QArg::into_vec));
        p.time.t = support(self.t, "T")?;
        p.time.t_sec = self.t_sec;
        p.time.stride = self.stride;
        p.time.sr = self.sr;
        if let Some(v) = self.j_fr {
            p.j_fr = v;
        }
        if let Some(v) = self.q_fr {
            p.q_fr = v;
        }
        p.f = support(self.f, "F")?;
        p.stride_fr = self.stride_fr;
        p.format = match self.format.as_deref() {
            None | Some("time") => Format::Time,
            Some("joint") => Format::Joint,
            Some(s) => return Err(JsError::new(&format!("format must be \"time\" or \"joint\", got \"{s}\"."))),
        };
        p.out_type = match self.out_type.as_deref() {
            None | Some("array") => OutType::Array,
            Some("list") => OutType::List,
            Some(s) => return Err(JsError::new(&format!("out_type must be \"array\" or \"list\", got \"{s}\"."))),
        };
        Ok(p)
    }
}

const TIME_KEYS: &[&str] = &["n", "J", "Q", "T", "T_sec", "max_order", "stride", "sr"];
const JTFS_KEYS: &[&str] = &[
    "n", "J", "J_fr", "Q", "Q_fr", "T", "T_sec", "F", "stride", "stride_fr", "format", "sr", "out_type",
];

/// Deserialise parameters, rejecting keys not in `allowed`.
///
/// serde-wasm-bindgen only ever reads the fields it expects, so serde's
/// `deny_unknown_fields` never sees a misspelt key; check them here instead.
fn parse<T: for<'de> Deserialize<'de>>(params: JsValue, allowed: &[&str]) -> Result<T, JsError> {
    if let Some(obj) = params.dyn_ref::<js_sys::Object>() {
        for key in js_sys::Object::keys(obj).iter() {
            let key = key.as_string().unwrap_or_default();
            if !allowed.contains(&key.as_str()) {
                return Err(JsError::new(&format!(
                    "Invalid parameters: unknown field `{key}`, expected one of {}",
                    allowed.join(", ")
                )));
            }
        }
    }
    serde_wasm_bindgen::from_value(params).map_err(|e| JsError::new(&format!("Invalid parameters: {e}")))
}

fn check_signal(x: &[f64], n: usize) -> Result<(), JsError> {
    if x.len() != n {
        return Err(JsError::new(&format!(
            "The operator was built for signals of length {n}, got {}.",
            x.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

/// Scattering coefficients: one block per path, in the operator's path order.
///
/// Every block is a `rows x cols` matrix stored by rows, bands by time. Time
/// scattering and time-format joint paths have one row.
#[wasm_bindgen]
pub struct Coefficients {
    data: Vec<f64>,
    rows: Vec<u32>,
    cols: Vec<u32>,
    offsets: Vec<usize>,
}

impl Coefficients {
    fn from_blocks(blocks: impl IntoIterator<Item = (usize, usize, Vec<f64>)>) -> Coefficients {
        let mut c = Coefficients { data: Vec::new(), rows: Vec::new(), cols: Vec::new(), offsets: vec![0] };
        for (r, k, d) in blocks {
            c.rows.push(r as u32);
            c.cols.push(k as u32);
            c.data.extend(d);
            c.offsets.push(c.data.len());
        }
        c
    }

    fn block(&self, i: usize) -> &[f64] {
        &self.data[self.offsets[i]..self.offsets[i + 1]]
    }
}

#[wasm_bindgen]
impl Coefficients {
    /// Number of paths.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.rows.len()
    }

    /// Every coefficient, path after path.
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Rows of each path's block.
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> Vec<u32> {
        self.rows.clone()
    }

    /// Columns (time samples) of each path's block.
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> Vec<u32> {
        self.cols.clone()
    }

    /// The coefficients of path `i`.
    pub fn path(&self, i: usize) -> Result<Vec<f64>, JsError> {
        if i >= self.length() {
            return Err(JsError::new(&format!("path {i} out of range; there are {}", self.length())));
        }
        Ok(self.block(i).to_vec())
    }
}

// ---------------------------------------------------------------------------
// Time scattering
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PathInfo1d {
    order: u8,
    n1: Option<usize>,
    n2: Option<usize>,
    j1: Option<i32>,
    j2: Option<i32>,
    xi1: Option<f64>,
    xi2: Option<f64>,
    sigma1: Option<f64>,
    sigma2: Option<f64>,
    path: String,
}

/// A time scattering operator for signals of a fixed length, as
/// `scattering_1d()` in R.
#[wasm_bindgen]
pub struct Scattering1d {
    op: s1d::Scattering1d,
}

#[wasm_bindgen]
impl Scattering1d {
    /// Build the filter banks: `{ n, J, Q, T, T_sec, max_order, stride, sr }`.
    #[wasm_bindgen(constructor)]
    pub fn new(params: JsValue) -> Result<Scattering1d, JsError> {
        let p = parse::<TimeArgs>(params, TIME_KEYS)?.into_params()?;
        Ok(Scattering1d { op: s1d::Scattering1d::new(&p).map_err(js_err)? })
    }

    /// One entry per path, in output order. Filter indices are one-based.
    pub fn paths(&self) -> Result<JsValue, JsError> {
        let rows: Vec<PathInfo1d> = self
            .op
            .paths
            .iter()
            .map(|p| PathInfo1d {
                order: p.order, n1: p.n1, n2: p.n2, j1: p.j1, j2: p.j2, xi1: p.xi1, xi2: p.xi2,
                sigma1: p.sigma1, sigma2: p.sigma2, path: p.label.clone(),
            })
            .collect();
        serde_wasm_bindgen::to_value(&rows).map_err(js_err)
    }

    /// Non-fatal problems found while building the operator.
    pub fn warnings(&self) -> Vec<String> {
        self.op.warnings.clone()
    }

    /// Transform one signal of length `n`.
    pub fn transform(&self, x: &[f64]) -> Result<Coefficients, JsError> {
        check_signal(x, self.op.n)?;
        let coefs = self.op.transform(x).map_err(js_err)?;
        Ok(Coefficients::from_blocks(coefs.into_iter().map(|v| (1, v.len(), v))))
    }

    /// Divide second-order paths by their first-order parent plus `eps`, as
    /// `scat_renorm()` in R.
    pub fn renorm(&self, coefs: &Coefficients, eps: f64) -> Result<Coefficients, JsError> {
        let mut paths: Vec<Vec<f64>> = (0..coefs.length()).map(|i| coefs.block(i).to_vec()).collect();
        features::renorm(&mut paths, &self.op.paths, eps).map_err(js_err)?;
        Ok(Coefficients::from_blocks(paths.into_iter().map(|v| (1, v.len(), v))))
    }
}

// ---------------------------------------------------------------------------
// Joint time-frequency scattering
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PathInfoJtfs {
    order: u8,
    n1: Option<usize>,
    n2: Option<usize>,
    n_fr: Option<usize>,
    j1: Option<i32>,
    j2: Option<i32>,
    j_fr: Option<i32>,
    xi1: Option<f64>,
    xi2: Option<f64>,
    xi_fr: Option<f64>,
    sigma1: Option<f64>,
    sigma2: Option<f64>,
    sigma_fr: Option<f64>,
    spin: Option<i8>,
    path: String,
}

/// A joint time-frequency scattering operator, as `scattering_jtfs()` in R.
#[wasm_bindgen]
pub struct ScatteringJtfs {
    op: jtfs::ScatteringJtfs,
}

#[wasm_bindgen]
impl ScatteringJtfs {
    /// Build the filter banks: `{ n, J, J_fr, Q, Q_fr, T, T_sec, F, stride,
    /// stride_fr, format, sr, out_type }`.
    #[wasm_bindgen(constructor)]
    pub fn new(params: JsValue) -> Result<ScatteringJtfs, JsError> {
        let p = parse::<JtfsArgs>(params, JTFS_KEYS)?.into_params()?;
        Ok(ScatteringJtfs { op: jtfs::ScatteringJtfs::new(&p).map_err(js_err)? })
    }

    /// One entry per path, in output order. Filter indices are one-based.
    pub fn paths(&self) -> Result<JsValue, JsError> {
        let rows: Vec<PathInfoJtfs> = self
            .op
            .paths
            .iter()
            .map(|p| PathInfoJtfs {
                order: p.order, n1: p.n1, n2: p.n2, n_fr: p.n_fr, j1: p.j1, j2: p.j2, j_fr: p.j_fr,
                xi1: p.xi1, xi2: p.xi2, xi_fr: p.xi_fr, sigma1: p.sigma1, sigma2: p.sigma2,
                sigma_fr: p.sigma_fr, spin: p.spin, path: p.label.clone(),
            })
            .collect();
        serde_wasm_bindgen::to_value(&rows).map_err(js_err)
    }

    /// Non-fatal problems found while building the operator.
    pub fn warnings(&self) -> Vec<String> {
        self.op.time.warnings.clone()
    }

    /// Transform one signal of length `n`.
    pub fn transform(&self, x: &[f64]) -> Result<Coefficients, JsError> {
        check_signal(x, self.op.time.n)?;
        let coefs = self.op.transform(x).map_err(js_err)?;
        Ok(Coefficients::from_blocks(coefs.into_iter().map(|m| (m.rows, m.cols, m.data))))
    }
}

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

/// `sign(v) * log1p(|v| / eps)` for every value, as `scat_log()` in R.
#[wasm_bindgen(js_name = logCompress)]
pub fn log_compress(values: &[f64], eps: f64) -> Result<Vec<f64>, JsError> {
    let mut v = values.to_vec();
    features::log_compress(&mut v, eps).map_err(js_err)?;
    Ok(v)
}

/// A data-driven `eps` for [`log_compress`], as `scat_eps()` in R.
#[wasm_bindgen(js_name = epsQuantile)]
pub fn eps_quantile(values: &[f64], quantile: f64) -> f64 {
    features::eps_quantile(values, quantile)
}

/// Reduce one path to a number: `"mean"`, `"max"`, `"sd"` or `"median"`.
#[wasm_bindgen]
pub fn summarise(values: &[f64], how: &str) -> Result<f64, JsError> {
    let how: Summary = how.parse().map_err(js_err)?;
    Ok(features::summarise(values, how))
}
