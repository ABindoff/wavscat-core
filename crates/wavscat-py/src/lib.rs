//! Python bindings for `wavscat-core` and `tapvid`.
//!
//! A thin layer: every number is computed by the Rust core, so features
//! computed here are bit-identical to those computed in R, in the browser or
//! natively, for the same `numerics_version()`. Keyword arguments use the R
//! argument names (`J`, `Q`, `T`, `T_sec`, `J_fr`, `F`, ...).
//!
//! ```python
//! import numpy as np, wavscat
//! assert wavscat.verify() == []          # this machine computes the reference bits
//! op = wavscat.ScatteringJtfs(n=900, J=7, J_fr=3, Q=(8, 1), T_sec=6, sr=30)
//! coefs = op.transform(x)                # list of 2-D arrays, one per path
//! ```
#![allow(non_snake_case)]

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use tapvid::features::FeatureParams;
use tapvid::ingest::{Ingest, IngestParams};
use tapvid::pipeline::PipelineParams;
use tapvid::qc::QcParams;
use tapvid::report::trial_report;
use tapvid::synth::{frame_times as synth_frame_times, FrameClock, TapSpec};
use tapvid::synth_video::{Distractor, VideoSpec, VideoSynth};
use wavscat_core::features::{self, Summary};
use wavscat_core::jtfs::{self, Format, OutType, ParamsJtfs};
use wavscat_core::scattering1d::{self as s1d, Params1d, TSpec};

fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Version of the numerical definition; features from different versions are
/// not comparable.
#[pyfunction]
fn numerics_version() -> &'static str {
    wavscat_core::NUMERICS_VERSION
}

/// Run the golden cases on this machine and compare every output bit with
/// the reference record. An empty list means this machine computes exactly
/// what every other supported platform computes.
#[pyfunction]
fn verify() -> Vec<String> {
    let mut d: Vec<String> = wavscat_core::verify::verify().into_iter().map(|x| format!("wavscat-core: {x}")).collect();
    d.extend(tapvid::verify::verify().into_iter().map(|x| format!("tapvid: {x}")));
    d
}

/// `T` or `F`: None, a number of samples (0 for none), or "global".
fn support(v: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<TSpec> {
    match v {
        None => Ok(TSpec::Default),
        Some(v) if v.is_none() => Ok(TSpec::Default),
        Some(v) => {
            if let Ok(x) = v.extract::<f64>() {
                Ok(TSpec::Samples(x))
            } else if v.extract::<String>().is_ok_and(|s| s == "global") {
                Ok(TSpec::Global)
            } else {
                Err(PyValueError::new_err(format!("{name} must be None, a number, or \"global\".")))
            }
        }
    }
}

/// `Q` as one integer or a pair.
fn q_arg(v: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<u32>> {
    match v {
        None => Ok(vec![8]),
        Some(v) if v.is_none() => Ok(vec![8]),
        Some(v) => {
            if let Ok(q) = v.extract::<u32>() {
                Ok(vec![q])
            } else if let Ok((a, b)) = v.extract::<(u32, u32)>() {
                Ok(vec![a, b])
            } else {
                Err(PyValueError::new_err("Q must be an integer or a pair of integers."))
            }
        }
    }
}

fn f64_slice<'a>(x: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    x.as_slice().map_err(|_| PyValueError::new_err("The signal must be a contiguous float64 array."))
}

/// A time scattering operator, as `scattering_1d()` in R.
#[pyclass(module = "wavscat")]
struct Scattering1d {
    op: s1d::Scattering1d,
}

#[pymethods]
impl Scattering1d {
    #[new]
    #[pyo3(signature = (n, J, Q=None, T=None, T_sec=None, max_order=2, stride=None, sr=None))]
    fn new(
        n: usize,
        J: u32,
        Q: Option<&Bound<'_, PyAny>>,
        T: Option<&Bound<'_, PyAny>>,
        T_sec: Option<f64>,
        max_order: u8,
        stride: Option<f64>,
        sr: Option<f64>,
    ) -> PyResult<Self> {
        let mut p = Params1d::new(n, J, q_arg(Q)?);
        p.t = support(T, "T")?;
        p.t_sec = T_sec;
        p.max_order = max_order;
        p.stride = stride;
        p.sr = sr;
        Ok(Scattering1d { op: s1d::Scattering1d::new(&p).map_err(err)? })
    }

    /// One dict per path, in output order. Filter indices are one-based.
    #[getter]
    fn paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for p in &self.op.paths {
            let d = PyDict::new(py);
            d.set_item("order", p.order)?;
            d.set_item("n1", p.n1)?;
            d.set_item("n2", p.n2)?;
            d.set_item("j1", p.j1)?;
            d.set_item("j2", p.j2)?;
            d.set_item("xi1", p.xi1)?;
            d.set_item("xi2", p.xi2)?;
            d.set_item("sigma1", p.sigma1)?;
            d.set_item("sigma2", p.sigma2)?;
            d.set_item("path", &p.label)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Transform one signal of length `n`: one array per path.
    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        let coefs = self.op.transform(f64_slice(&x)?).map_err(err)?;
        Ok(coefs.into_iter().map(|v| v.into_pyarray(py)).collect())
    }

    /// Divide order-2 paths by their order-1 parent plus `eps`.
    #[pyo3(signature = (coefs, eps=1e-12))]
    fn renorm<'py>(&self, py: Python<'py>, coefs: Vec<PyReadonlyArray1<'py, f64>>, eps: f64) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        let mut paths: Vec<Vec<f64>> = coefs.iter().map(|c| f64_slice(c).map(|s| s.to_vec())).collect::<PyResult<_>>()?;
        features::renorm(&mut paths, &self.op.paths, eps).map_err(err)?;
        Ok(paths.into_iter().map(|v| v.into_pyarray(py)).collect())
    }
}

/// A joint time-frequency scattering operator, as `scattering_jtfs()` in R.
#[pyclass(module = "wavscat")]
struct ScatteringJtfs {
    op: jtfs::ScatteringJtfs,
}

fn mats_to_py<'py>(py: Python<'py>, mats: Vec<jtfs::Mat>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
    mats.into_iter()
        .map(|m| m.data.into_pyarray(py).reshape([m.rows, m.cols]))
        .collect()
}

fn py_to_mats(coefs: &[PyReadonlyArrayDyn<'_, f64>]) -> PyResult<Vec<jtfs::Mat>> {
    coefs
        .iter()
        .map(|c| {
            let shape = c.shape();
            let (rows, cols) = match shape {
                [r, k] => (*r, *k),
                [k] => (1, *k),
                _ => return Err(PyValueError::new_err("Each path must be a 1-D or 2-D array.")),
            };
            let data = c.as_slice().map_err(|_| PyValueError::new_err("Arrays must be contiguous."))?.to_vec();
            Ok(jtfs::Mat { rows, cols, data })
        })
        .collect()
}

#[pymethods]
impl ScatteringJtfs {
    #[new]
    #[pyo3(signature = (n, J, J_fr=3, Q=None, Q_fr=1, T=None, T_sec=None, F=None, stride=None, stride_fr=None, format="time", sr=None, out_type="array"))]
    fn new(
        n: usize,
        J: u32,
        J_fr: u32,
        Q: Option<&Bound<'_, PyAny>>,
        Q_fr: u32,
        T: Option<&Bound<'_, PyAny>>,
        T_sec: Option<f64>,
        F: Option<&Bound<'_, PyAny>>,
        stride: Option<f64>,
        stride_fr: Option<f64>,
        format: &str,
        sr: Option<f64>,
        out_type: &str,
    ) -> PyResult<Self> {
        let mut p = ParamsJtfs::new(n, J, q_arg(Q)?);
        p.j_fr = J_fr;
        p.q_fr = Q_fr;
        p.time.t = support(T, "T")?;
        p.time.t_sec = T_sec;
        p.time.stride = stride;
        p.time.sr = sr;
        p.f = support(F, "F")?;
        p.stride_fr = stride_fr;
        p.format = match format {
            "time" => Format::Time,
            "joint" => Format::Joint,
            _ => return Err(PyValueError::new_err("format must be \"time\" or \"joint\".")),
        };
        p.out_type = match out_type {
            "array" => OutType::Array,
            "list" => OutType::List,
            _ => return Err(PyValueError::new_err("out_type must be \"array\" or \"list\".")),
        };
        Ok(ScatteringJtfs { op: jtfs::ScatteringJtfs::new(&p).map_err(err)? })
    }

    /// One dict per path, in output order. Filter indices are one-based.
    #[getter]
    fn paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for p in &self.op.paths {
            let d = PyDict::new(py);
            d.set_item("order", p.order)?;
            d.set_item("n1", p.n1)?;
            d.set_item("n2", p.n2)?;
            d.set_item("n_fr", p.n_fr)?;
            d.set_item("j1", p.j1)?;
            d.set_item("j2", p.j2)?;
            d.set_item("j_fr", p.j_fr)?;
            d.set_item("xi1", p.xi1)?;
            d.set_item("xi2", p.xi2)?;
            d.set_item("xi_fr", p.xi_fr)?;
            d.set_item("spin", p.spin)?;
            d.set_item("path", &p.label)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Transform one signal: one `[band, time]` array per path.
    fn transform<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        mats_to_py(py, self.op.transform(f64_slice(&x)?).map_err(err)?)
    }

    /// Transform one signal, also returning the first-order time scattering
    /// coefficients S1, `[band, time]`, that `renorm` divides by.
    fn transform_with_s1<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Vec<Bound<'py, PyArray2<f64>>>, Bound<'py, PyArray2<f64>>)> {
        let (coefs, s1) = self.op.transform_with_s1(f64_slice(&x)?).map_err(err)?;
        let s1 = s1.data.clone().into_pyarray(py).reshape([s1.rows, s1.cols])?;
        Ok((mats_to_py(py, coefs)?, s1))
    }

    /// Divide second-order paths by S1 of the bands they span, through the
    /// same frequential low-pass, plus `eps`. Needs local time averaging.
    #[pyo3(signature = (coefs, s1, eps=1e-12))]
    fn renorm<'py>(
        &self,
        py: Python<'py>,
        coefs: Vec<PyReadonlyArrayDyn<'py, f64>>,
        s1: PyReadonlyArrayDyn<'py, f64>,
        eps: f64,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        let mut mats = py_to_mats(&coefs)?;
        let s1 = py_to_mats(&[s1])?.pop().unwrap();
        self.op.renorm(&mut mats, &s1, eps).map_err(err)?;
        mats_to_py(py, mats)
    }
}

/// `sign(v) * log1p(|v| / eps)`, as `scat_log()` in R.
#[pyfunction]
fn log_compress<'py>(py: Python<'py>, values: PyReadonlyArray1<'py, f64>, eps: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut v = f64_slice(&values)?.to_vec();
    features::log_compress(&mut v, eps).map_err(err)?;
    Ok(v.into_pyarray(py))
}

/// A data-driven `eps` for `log_compress`, as `scat_eps()` in R.
#[pyfunction]
fn eps_quantile(values: PyReadonlyArray1<'_, f64>, quantile: f64) -> PyResult<f64> {
    Ok(features::eps_quantile(f64_slice(&values)?, quantile))
}

/// Reduce one path to a number: "mean", "max", "sd" or "median".
#[pyfunction]
fn summarise(values: PyReadonlyArray1<'_, f64>, how: &str) -> PyResult<f64> {
    let how: Summary = how.parse().map_err(err)?;
    Ok(features::summarise(f64_slice(&values)?, how))
}

/// One trial of webcam finger tapping, captured frame by frame. Only a small
/// grid of each frame is kept; the raw frame is never stored.
#[pyclass(module = "wavscat")]
struct TappingSession {
    ingest: Ingest,
}

/// The layout of a frame array: `(stride, width, height)` in elements of
/// `channels` bytes.
fn frame_layout(a: &PyReadonlyArrayDyn<'_, u8>, channels: usize) -> PyResult<(usize, usize, usize)> {
    match (a.shape(), channels) {
        ([h, w], 1) => Ok((*w, *w, *h)),
        ([h, w, 4], 4) => Ok((4 * w, *w, *h)),
        _ => Err(PyValueError::new_err(if channels == 1 {
            "luma must be a 2-D uint8 array of height x width."
        } else {
            "rgba must be a uint8 array of height x width x 4."
        })),
    }
}

#[pymethods]
impl TappingSession {
    #[new]
    #[pyo3(signature = (capacity=None, grid_width=None, grid_height=None))]
    fn new(capacity: Option<usize>, grid_width: Option<usize>, grid_height: Option<usize>) -> PyResult<Self> {
        let d = IngestParams::default();
        let p = IngestParams {
            capacity: capacity.unwrap_or(d.capacity),
            grid_width: grid_width.unwrap_or(d.grid_width),
            grid_height: grid_height.unwrap_or(d.grid_height),
            ..d
        };
        Ok(TappingSession { ingest: Ingest::new(p).map_err(err)? })
    }

    /// Add a luma frame, a C-contiguous `height x width` uint8 array,
    /// captured at `timestamp_us` microseconds.
    fn push_frame(&mut self, luma: PyReadonlyArrayDyn<'_, u8>, timestamp_us: i64) -> PyResult<()> {
        let (stride, w, h) = frame_layout(&luma, 1)?;
        let data = luma.as_slice().map_err(|_| PyValueError::new_err("luma must be C-contiguous."))?;
        self.ingest.push_frame(data, stride, w, h, timestamp_us).map_err(err)
    }

    /// Add an RGBA frame, a C-contiguous `height x width x 4` uint8 array.
    fn push_rgba(&mut self, rgba: PyReadonlyArrayDyn<'_, u8>, timestamp_us: i64) -> PyResult<()> {
        let (stride, w, h) = frame_layout(&rgba, 4)?;
        let data = rgba.as_slice().map_err(|_| PyValueError::new_err("rgba must be C-contiguous."))?;
        self.ingest.push_rgba(data, stride, w, h, timestamp_us).map_err(err)
    }

    /// Frames held.
    #[getter]
    fn frames(&self) -> usize {
        self.ingest.len()
    }

    /// Analyse and gate the trial: a dict with `accepted`, `reasons`, `qc`,
    /// `features` and `itis`, the last two only when accepted.
    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let report = trial_report(&self.ingest, &PipelineParams::default(), &QcParams::default(), &FeatureParams::default())
            .map_err(err)?;
        pythonize::pythonize(py, &report).map_err(err)
    }

    /// Discard every frame, ready for the next trial.
    fn reset(&mut self) {
        self.ingest.clear();
    }
}

/// A synthetic tapping video with known taps, for demonstrations and tests.
#[pyclass(module = "wavscat")]
struct SyntheticVideo {
    synth: VideoSynth,
    work: Vec<f32>,
}

#[pymethods]
impl SyntheticVideo {
    #[new]
    #[pyo3(signature = (width=320, height=240, iti_sd=0.0, seed=1, distractor_hz=None))]
    fn new(width: usize, height: usize, iti_sd: f64, seed: u64, distractor_hz: Option<f64>) -> Self {
        let spec = VideoSpec {
            width,
            height,
            tap: TapSpec { iti_sd, seed, ..TapSpec::default() },
            distractor: distractor_hz.map(|rate_hz| Distractor { rate_hz, displacement: 15.0, radius: 10.0, contrast: 90.0, centre: (0.8, 0.3) }),
            seed,
            ..VideoSpec::default()
        };
        SyntheticVideo { work: vec![0f32; width * height], synth: VideoSynth::new(spec) }
    }

    /// The frame captured at `t` seconds, a `height x width` uint8 array.
    fn render<'py>(&mut self, py: Python<'py>, t: f64) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let (w, h) = (self.synth.spec().width, self.synth.spec().height);
        let mut out = vec![0u8; w * h];
        self.synth.render(t, &mut out, &mut self.work);
        out.into_pyarray(py).reshape([h, w])
    }

    /// True tap times, in seconds.
    fn taps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.synth.taps().to_vec().into_pyarray(py)
    }
}

/// Capture times, in seconds, of a webcam clock with jitter and dropped frames.
#[pyfunction]
#[pyo3(signature = (duration, fps=30.0, jitter_sd=0.0, drop_prob=0.0, seed=2))]
fn frame_times<'py>(py: Python<'py>, duration: f64, fps: f64, jitter_sd: f64, drop_prob: f64, seed: u64) -> Bound<'py, PyArray1<f64>> {
    synth_frame_times(&FrameClock { fps, jitter_sd, drop_prob, seed }, duration).into_pyarray(py)
}

#[pymodule]
fn wavscat(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(numerics_version, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(log_compress, m)?)?;
    m.add_function(wrap_pyfunction!(eps_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(summarise, m)?)?;
    m.add_function(wrap_pyfunction!(frame_times, m)?)?;
    m.add_class::<Scattering1d>()?;
    m.add_class::<ScatteringJtfs>()?;
    m.add_class::<TappingSession>()?;
    m.add_class::<SyntheticVideo>()?;
    Ok(())
}
