//! Joint time-frequency scattering: operator construction and the cascade.
//!
//! A port of `R/op-jtfs.R` and `R/cascade-jtfs.R`, following Kymatio's
//! `TimeFrequencyScattering`. Time scattering treats each first-order band
//! separately; joint scattering adds a filter bank along log-frequency, so the
//! second order convolves jointly in time and frequency. The frequential
//! wavelets come in two spins, which respond to spectral patterns drifting up
//! and down in frequency: a rising formant transition and a falling one have
//! identical time-scattering coefficients and opposite spins here.
//!
//! The cascade runs in two modes. Given a signal it computes coefficients;
//! given none it walks the same code for its structure alone, which is how the
//! path table is produced. The table and the coefficients therefore cannot
//! drift apart.

use crate::backend::{filter_periodize, filter_periodize_rows, modulus, sum};
use crate::complex::C64;
use crate::error::{fail, Error};
use crate::fft;
use crate::filter_bank::{self, Filter, Generator};
use crate::math;
use crate::pad::pad_reflect;
use crate::scattering1d::{parse_stride, parse_t, Averaging, Params1d, Scattering1d, TSpec};

/// How joint coefficients are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// One series per `(band, modulation, spin)` path, shaped like time
    /// scattering output.
    Time,
    /// The frequency axis kept intact: a `[band, time]` image per path.
    Joint,
}

/// Whether the caller will stack paths into one array.
///
/// With `Joint` format and `Array` output, paths keep every padded frequency
/// row so that they stack; otherwise rows beyond the input bands are trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutType {
    Array,
    List,
}

/// Arguments to [`ScatteringJtfs::new`], mirroring `scattering_jtfs()` in R.
#[derive(Debug, Clone)]
pub struct ParamsJtfs {
    /// The temporal half: `n`, `J`, `Q`, `T`, `T_sec`, `stride` and `sr`.
    /// `max_order` is ignored; joint scattering is always of order two.
    pub time: Params1d,
    /// Log-scale of the frequential transform, in first-order bands.
    pub j_fr: u32,
    /// Wavelets per octave along log-frequency.
    pub q_fr: u32,
    /// Frequential averaging support, in first-order bands.
    pub f: TSpec,
    /// Frequential output subsampling, a power of two.
    pub stride_fr: Option<f64>,
    pub format: Format,
    pub out_type: OutType,
}

impl ParamsJtfs {
    /// The R defaults: `J_fr = 3`, `Q_fr = 1`, `format = "time"`.
    pub fn new(n: usize, j: u32, q: Vec<u32>) -> Self {
        ParamsJtfs {
            time: Params1d::new(n, j, q),
            j_fr: 3,
            q_fr: 1,
            f: TSpec::Default,
            stride_fr: None,
            format: Format::Time,
            out_type: OutType::Array,
        }
    }
}

/// A real matrix stored by rows: rows are frequency bands, columns are time.
/// Time-format paths are single rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    fn from_rows(rows: Vec<Vec<f64>>) -> Result<Mat, Error> {
        let cols = rows.first().map_or(0, |r| r.len());
        if rows.iter().any(|r| r.len() != cols) {
            return fail("Internal error: rows of a joint path differ in length.");
        }
        let n = rows.len();
        Ok(Mat {
            rows: n,
            cols,
            data: rows.concat(),
        })
    }

    fn row_vec(v: Vec<f64>) -> Mat {
        Mat {
            rows: 1,
            cols: v.len(),
            data: v,
        }
    }

    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    fn col(&self, c: usize) -> Vec<f64> {
        (0..self.rows)
            .map(|r| self.data[r * self.cols + c])
            .collect()
    }

    fn from_cols(cols: Vec<Vec<f64>>) -> Mat {
        let nc = cols.len();
        let nr = cols.first().map_or(0, |c| c.len());
        let mut data = vec![0.0; nr * nc];
        for (c, col) in cols.iter().enumerate() {
            for (r, v) in col.iter().enumerate() {
                data[r * nc + c] = *v;
            }
        }
        Mat {
            rows: nr,
            cols: nc,
            data,
        }
    }

    fn keep_rows(&mut self, n: usize) {
        self.rows = self.rows.min(n);
        self.data.truncate(self.rows * self.cols);
    }

    fn keep_cols(&self, start: usize, end: usize) -> Mat {
        let mut data = Vec::with_capacity(self.rows * (end - start));
        for r in 0..self.rows {
            data.extend_from_slice(&self.row(r)[start..end]);
        }
        Mat {
            rows: self.rows,
            cols: end - start,
            data,
        }
    }
}

/// A complex matrix, used between the frequential convolution and the modulus.
struct CMat {
    rows: usize,
    cols: usize,
    data: Vec<C64>,
}

/// One joint scattering path. Filter indices are one-based, as in R.
#[derive(Debug, Clone, PartialEq)]
pub struct JtfsPath {
    pub order: u8,
    /// First-order band; time format only.
    pub n1: Option<usize>,
    pub n2: Option<usize>,
    /// Frequential filter. Index 1 is the unspun low-pass.
    pub n_fr: Option<usize>,
    pub j1: Option<i32>,
    pub j2: Option<i32>,
    pub j_fr: Option<i32>,
    pub xi1: Option<f64>,
    pub xi2: Option<f64>,
    pub xi_fr: Option<f64>,
    pub sigma1: Option<f64>,
    pub sigma2: Option<f64>,
    pub sigma_fr: Option<f64>,
    /// 1 for up, -1 for down, 0 for the unspun path.
    pub spin: Option<i8>,
    /// `S0`, `J1_...` or `J2_...`, with a `u`, `d` or `n` spin suffix.
    pub label: String,
}

/// Structure of a path while the cascade runs.
#[derive(Debug, Clone)]
struct Meta {
    /// Filter indices along the path: the sort key, and its length the order.
    n: Vec<usize>,
    order: u8,
    n1: Option<usize>,
    n2: Option<usize>,
    j2: Option<i32>,
    n_fr: Option<usize>,
    j_fr: Option<i32>,
    spin: Option<i8>,
    /// The first-order bands this path spans, one-based.
    n1_index: Vec<usize>,
    n1_max: usize,
    /// How many input bands each output row summarises.
    n1_stride: usize,
}

impl Meta {
    fn zeroth() -> Meta {
        Meta {
            n: Vec::new(),
            order: 0,
            n1: None,
            n2: None,
            j2: None,
            n_fr: None,
            j_fr: None,
            spin: None,
            n1_index: Vec::new(),
            n1_max: 0,
            n1_stride: 1,
        }
    }
}

/// The input to a frequential convolution: first-order rows are real,
/// second-order rows complex.
enum FreqInput {
    Real(Mat),
    Complex(CMat),
}

/// A joint time-frequency scattering operator for signals of a fixed length.
#[derive(Debug, Clone)]
pub struct ScatteringJtfs {
    /// The temporal half, exactly a time scattering operator.
    pub time: Scattering1d,
    pub j_fr: u32,
    pub q_fr: u32,
    pub f: f64,
    pub average_fr: Averaging,
    pub log2_f: i32,
    pub log2_stride_fr: i32,
    /// Nominal number of first-order bands, `(J + 1) Q1`.
    pub n_input_fr: usize,
    pub n_padded_fr: usize,
    pub format: Format,
    pub out_type: OutType,
    /// Frequential low-pass, of support `F`.
    pub phi_fr: Filter,
    /// Frequential filters: the unspun low-pass of support `2^J_fr`, then the
    /// spun wavelets, positive spins first.
    pub psis_fr: Vec<Filter>,
    pub paths: Vec<JtfsPath>,
}

/// `2^k` as a subsampling factor; zero or negative `k` means none, as R's
/// `as.integer(2^k)` falls below two.
fn factor(k: i32) -> usize {
    if k <= 0 {
        1
    } else {
        1usize << k
    }
}

fn level(f: &Filter, k: i32) -> Result<&[f64], Error> {
    usize::try_from(k)
        .ok()
        .and_then(|k| f.levels.get(k))
        .map(|v| v.as_slice())
        .ok_or_else(|| {
            Error::new(format!(
                "Internal error: no filter at subsampling level {k}."
            ))
        })
}

fn inverse(mut x: Vec<C64>) -> Vec<C64> {
    fft::ifft(&mut x);
    x
}

/// Columns processed together along frequency; see [`freq_filter_modulus`].
const COLUMN_BLOCK: usize = 64;

/// `|ifft(filter_periodize(column, f, k))|` for every column of the
/// frequency-by-time spectrum `x_hat`, with `cols` columns.
///
/// Done one block of columns at a time, so that filtering, subsampling, the
/// inverse transform and the modulus all run on data in cache, rather than as
/// four passes over matrices of tens of megabytes. Columns never mix, so the
/// blocking is invisible in the result.
fn freq_filter_modulus(x_hat: &[C64], cols: usize, f: &[f64], k: usize) -> Mat {
    let n = f.len();
    let rows = n / k;
    let mut out = vec![0.0; rows * cols];
    let mut block = vec![C64::ZERO; n * COLUMN_BLOCK.min(cols)];
    let mut c0 = 0;
    while c0 < cols {
        let w = COLUMN_BLOCK.min(cols - c0);
        let blk = &mut block[..n * w];
        for r in 0..n {
            blk[r * w..(r + 1) * w].copy_from_slice(&x_hat[r * cols + c0..r * cols + c0 + w]);
        }
        let mut y = filter_periodize_rows(blk, w, f, k);
        fft::ifft_columns(&mut y, w);
        for r in 0..rows {
            for (o, v) in out[r * cols + c0..r * cols + c0 + w]
                .iter_mut()
                .zip(&y[r * w..(r + 1) * w])
            {
                *o = v.norm();
            }
        }
        c0 += w;
    }
    Mat {
        rows,
        cols,
        data: out,
    }
}

impl ScatteringJtfs {
    /// Build the temporal and frequential filter banks and the path table.
    pub fn new(p: &ParamsJtfs) -> Result<ScatteringJtfs, Error> {
        let mut tp = p.time.clone();
        tp.max_order = 2;
        let time = Scattering1d::new(&tp)?;

        let j_fr = p.j_fr;
        let q_fr = p.q_fr;
        if j_fr < 1 {
            return fail("J_fr must be a single positive whole number.");
        }
        if q_fr < 1 {
            return fail("Q_fr must be a single positive whole number.");
        }
        if p.format == Format::Joint && p.out_type == OutType::Array && p.f == TSpec::Samples(0.0) {
            return fail(
                "format = \"joint\" with out_type = \"array\" needs frequential averaging, because \
                 unaveraged paths have differing numbers of bands. Set F to a positive value, or \
                 use out_type = \"list\".",
            );
        }

        // Sized from the nominal band count rather than the realised one, as
        // Kymatio does, so that F and the padding do not shift when the bank
        // gains or loses a filter at the edge.
        let n_input_fr = (time.j as usize + 1) * time.q[0] as usize;
        if j_fr >= 63 || (1usize << j_fr) > n_input_fr {
            return fail(format!(
                "J_fr = {j_fr} asks for a frequential wavelet spanning 2^{j_fr} bands, but this \
                 operator has only {n_input_fr} first-order bands. Use J_fr <= {}, or raise J or \
                 Q to widen the frequency axis.",
                math::floor_log2_int(n_input_fr as u64)
            ));
        }
        let (f, average_fr) = parse_t(p.f, j_fr, n_input_fr, "F")?;
        let log2_f = math::floor_log2(f);
        let log2_stride_fr = parse_stride(p.stride_fr, log2_f, average_fr, "stride_fr", "F")?;

        // Room enough that the frequential convolution does not wrap, and a
        // multiple of every subsampling factor it will use.
        let min_to_pad_fr = 8.0 * f.min(math::pow2(j_fr));
        let k = math::pow2(j_fr);
        let n_padded_fr = (((n_input_fr as f64 + min_to_pad_fr) / k).floor() * k) as usize;

        let params = time.bank_params;
        let phi_fr =
            filter_bank::filter_factory(n_padded_fr, j_fr, &[], f, params, Generator::Spin).phi;
        let spun = filter_bank::filter_factory(
            n_padded_fr,
            j_fr,
            &[q_fr],
            math::pow2(j_fr),
            params,
            Generator::Spin,
        );
        if spun.banks[0]
            .iter()
            .any(|w| w.xi.abs() >= 0.5 / math::pow2(w.j.max(0) as u32))
        {
            return fail(format!(
                "The frequential filter bank would alias with J_fr = {j_fr} and Q_fr = {q_fr}. \
                 Reduce J_fr."
            ));
        }
        let mut psis_fr = vec![spun.phi];
        psis_fr.extend(spun.banks.into_iter().next().unwrap());

        let mut op = ScatteringJtfs {
            time,
            j_fr,
            q_fr,
            f,
            average_fr,
            log2_f,
            log2_stride_fr,
            n_input_fr,
            n_padded_fr,
            format: p.format,
            out_type: p.out_type,
            phi_fr,
            psis_fr,
            paths: Vec::new(),
        };
        let structure = op.run(None)?;
        op.paths = structure.iter().map(|(_, m)| op.describe(m)).collect();
        Ok(op)
    }

    /// Transform one signal of length `n`.
    ///
    /// Returns one matrix per path, in the order of [`ScatteringJtfs::paths`].
    /// Time-format paths are single rows; joint-format paths are
    /// `[band, time]`. With global time averaging each has one column.
    pub fn transform(&self, x: &[f64]) -> Result<Vec<Mat>, Error> {
        if x.len() != self.time.n {
            return fail(format!(
                "The operator was built for signals of length {}, got {}.",
                self.time.n,
                x.len()
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return fail("The signal contains missing or non-finite values.");
        }
        let paths = self.run(Some(x))?;
        paths
            .into_iter()
            .map(|(coef, meta)| Ok(self.unpad(coef.expect("computed"), &meta)))
            .collect()
    }

    /// The cascade, with or without data, sorted into output order.
    fn run(&self, x: Option<&[f64]>) -> Result<Vec<(Option<Mat>, Meta)>, Error> {
        let op = &self.time;
        let local = op.average == Averaging::Local;
        let stride = op.log2_stride;

        let u0 = x.map(|x| pad_reflect(x, op.pad_left, op.pad_right));
        let u0_hat = u0.as_ref().map(|u| fft::fft_real(u));

        // Order zero.
        let coef0 = match (&u0, &u0_hat) {
            (Some(u0), Some(h)) => Some(Mat::row_vec(match op.average {
                Averaging::Local => {
                    fft::ifft_real_part(filter_periodize(h, level(&op.phi, 0)?, factor(stride)))
                }
                Averaging::Global => vec![sum(u0)],
                Averaging::None => u0.clone(),
            })),
            _ => None,
        };

        // First order, breadth first: the frequential convolution needs every
        // band at once.
        let k1s: Vec<i32> = op
            .psi1
            .iter()
            .map(|f| if local { f.j.min(stride) } else { f.j })
            .collect();
        let mut u1_hats = Vec::new();
        let mut s1 = None;
        if let Some(h) = &u0_hat {
            let mut rows = Vec::with_capacity(op.psi1.len());
            for (f, &k1) in op.psi1.iter().zip(&k1s) {
                let u1 = modulus(&inverse(filter_periodize(h, level(f, 0)?, factor(k1))));
                let u1_hat = fft::fft_real(&u1);
                rows.push(fft::ifft_real_part(filter_periodize(
                    &u1_hat,
                    level(&op.phi, k1)?,
                    factor(stride - k1),
                )));
                u1_hats.push(u1_hat);
            }
            s1 = Some(Mat::from_rows(rows)?);
        }

        // S1 is real, so only the non-negative spins carry information: the
        // negative ones would be its complex conjugates.
        let base1 = Meta {
            n1_index: (1..=op.psi1.len()).collect(),
            ..Meta::zeroth()
        };
        let mut joint = self.freq_scatter(s1.map(FreqInput::Real), &base1, false)?;

        for (i2, f2) in op.psi2.iter().enumerate() {
            let j2 = f2.j;
            let mut n1_index = Vec::new();
            let mut rows: Vec<Vec<C64>> = Vec::new();
            for (i1, f1) in op.psi1.iter().enumerate() {
                if j2 <= f1.j {
                    continue;
                }
                n1_index.push(i1 + 1);
                if u0_hat.is_none() {
                    continue;
                }
                let k1 = k1s[i1];
                let k2 = if local {
                    (j2 - k1).min(stride)
                } else {
                    j2 - k1
                };
                rows.push(inverse(filter_periodize(
                    &u1_hats[i1],
                    level(f2, k1)?,
                    factor(k2),
                )));
            }
            if n1_index.is_empty() {
                continue;
            }
            let y2 = if u0_hat.is_some() {
                let cols = rows[0].len();
                if rows.iter().any(|r| r.len() != cols) {
                    return fail("Internal error: second-order rows differ in length.");
                }
                Some(FreqInput::Complex(CMat {
                    rows: rows.len(),
                    cols,
                    data: rows.concat(),
                }))
            } else {
                None
            };
            let base = Meta {
                n: vec![i2 + 1],
                n2: Some(i2 + 1),
                j2: Some(j2),
                n1_index,
                ..Meta::zeroth()
            };
            joint.extend(self.freq_scatter(y2, &base, true)?);
        }

        let mut out = vec![(coef0, Meta::zeroth())];
        out.extend(joint);
        // Stable, as R's order() is: by path length, then by filter indices.
        out.sort_by(|a, b| (a.1.n.len(), &a.1.n).cmp(&(b.1.n.len(), &b.1.n)));
        Ok(out)
    }

    /// Convolve along log-frequency with every frequential filter, finishing
    /// each path before starting the next.
    ///
    /// All time columns are transformed along frequency as one batch, which is
    /// bit-identical to transforming them one by one. Finishing each path in
    /// turn, rather than holding every frequential output at once, bounds the
    /// peak memory by one path.
    fn freq_scatter(
        &self,
        x: Option<FreqInput>,
        base: &Meta,
        spinned: bool,
    ) -> Result<Vec<(Option<Mat>, Meta)>, Error> {
        let n_pad = self.n_padded_fr;
        let local_fr = self.average_fr == Averaging::Local;
        let n1_max = base.n1_index.len();

        // Spectrum along frequency of every time column, zero-padded: the
        // input rows first, then zero rows up to n_pad.
        let x_hat: Option<(usize, Vec<C64>)> = match x {
            None => None,
            Some(x) => {
                let (rows, cols, mut data) = match x {
                    FreqInput::Real(m) => (
                        m.rows,
                        m.cols,
                        m.data.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>(),
                    ),
                    FreqInput::Complex(m) => (m.rows, m.cols, m.data),
                };
                if rows > n_pad {
                    return fail(format!(
                        "Internal error: {rows} bands exceed the padded frequency axis of {n_pad}."
                    ));
                }
                data.resize(n_pad * cols, C64::ZERO);
                fft::fft_columns(&mut data, cols);
                Some((cols, data))
            }
        };

        let mut out = Vec::new();
        for (i, psi) in self.psis_fr.iter().enumerate() {
            if !spinned && psi.xi < 0.0 {
                continue;
            }
            let j_fr = psi.j;
            let k_fr = if local_fr {
                j_fr.min(self.log2_stride_fr)
            } else {
                j_fr
            };
            if n_pad % factor(k_fr) != 0 {
                return fail(format!(
                    "Internal error: cannot subsample {n_pad} frequency rows by 2^{k_fr}."
                ));
            }
            let coef = match &x_hat {
                Some((cols, data)) => Some(freq_filter_modulus(
                    data,
                    *cols,
                    level(psi, 0)?,
                    factor(k_fr),
                )),
                None => None,
            };
            let mut n = base.n.clone();
            n.push(i + 1);
            let spin = if psi.xi > 0.0 {
                1
            } else if psi.xi < 0.0 {
                -1
            } else {
                0
            };
            let meta = Meta {
                n,
                n_fr: Some(i + 1),
                j_fr: Some(j_fr),
                spin: Some(spin),
                n1_max,
                n1_stride: factor(j_fr),
                ..base.clone()
            };
            out.extend(self.finalise(coef, meta)?);
        }
        Ok(out)
    }

    /// Average a path's moduli in time and frequency, and lay it out.
    fn finalise(&self, coef: Option<Mat>, mut m: Meta) -> Result<Vec<(Option<Mat>, Meta)>, Error> {
        let keep_freq_axis = self.format == Format::Joint;
        let trim = !keep_freq_axis || self.out_type != OutType::Array;
        let mut coef = coef;

        // The unspun path came from a frequential low-pass already.
        let spin = m.spin.unwrap_or(0);
        let average_fr = if spin != 0 {
            self.average_fr
        } else {
            Averaging::None
        };
        m.n1_stride = match average_fr {
            Averaging::Global => m.n1_max,
            Averaging::Local => factor(self.log2_stride_fr),
            Averaging::None if self.average_fr == Averaging::None => {
                factor(m.j_fr.unwrap_or(0).max(0))
            }
            Averaging::None => m.n1_stride,
        };
        let n_keep = 1 + m.n1_max / m.n1_stride;

        // Temporal averaging works row by row, so when no frequential
        // averaging follows, rows that will be discarded can be dropped first
        // without changing a bit of the rows kept.
        if trim && average_fr == Averaging::None {
            if let Some(c) = coef.as_mut() {
                c.keep_rows(n_keep);
            }
        }

        // First-order paths descend from S1, which is already averaged in
        // time, so only the second order needs it here.
        match self.time.average {
            Averaging::Global => {
                coef = coef
                    .map(|c| Mat::from_rows((0..c.rows).map(|r| vec![sum(c.row(r))]).collect()))
                    .transpose()?;
            }
            Averaging::Local if m.n.len() > 1 => {
                coef = coef
                    .map(|c| self.time_average(&c, m.j2.unwrap()))
                    .transpose()?;
            }
            _ => {}
        }

        match average_fr {
            Averaging::Global => {
                coef = coef.map(|c| {
                    let sums = (0..c.cols).map(|j| sum(&c.col(j))).collect();
                    Mat::row_vec(sums)
                });
            }
            Averaging::Local => {
                coef = coef
                    .map(|c| self.freq_average(&c, m.j_fr.unwrap()))
                    .transpose()?;
            }
            Averaging::None => {}
        }

        if trim {
            if let Some(c) = coef.as_mut() {
                c.keep_rows(n_keep);
            }
        }

        if keep_freq_axis {
            m.order = m.n.len() as u8;
            Ok(vec![(coef, m)])
        } else {
            self.split_bands(coef, &m)
        }
    }

    /// Low-pass each band along time and subsample to the output stride.
    fn time_average(&self, c: &Mat, j2: i32) -> Result<Mat, Error> {
        let k_j = self.time.log2_stride - j2;
        if k_j < 0 {
            return fail(
                "The averaging support T is shorter than the widest second-order wavelet, which \
                 joint scattering cannot accommodate. Increase T, or reduce J.",
            );
        }
        let phi = level(&self.time.phi, j2)?;
        let rows = (0..c.rows)
            .map(|r| {
                let row = c.row(r);
                if row.len() != phi.len() {
                    return fail("Internal error: a joint path and the low-pass differ in length.");
                }
                Ok(fft::ifft_real_part(filter_periodize(
                    &fft::fft_real(row),
                    phi,
                    factor(k_j),
                )))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Mat::from_rows(rows)
    }

    /// Low-pass each time column along frequency.
    fn freq_average(&self, c: &Mat, j_fr: i32) -> Result<Mat, Error> {
        let k_in = j_fr.min(self.log2_f);
        let k_j = self.log2_stride_fr - k_in;
        let phi = level(&self.phi_fr, k_in)?;
        if c.rows != phi.len() {
            return fail(
                "The frequential subsampling does not match the frequential averaging. Use the \
                 default stride_fr.",
            );
        }
        let cols = (0..c.cols)
            .map(|j| {
                fft::ifft_real_part(filter_periodize(
                    &fft::fft_real(&c.col(j)),
                    phi,
                    factor(k_j),
                ))
            })
            .collect();
        Ok(Mat::from_cols(cols))
    }

    /// Split a joint path into one series per output band (`format = "time"`).
    fn split_bands(&self, coef: Option<Mat>, m: &Meta) -> Result<Vec<(Option<Mat>, Meta)>, Error> {
        let starts: Vec<usize> = (0..m.n1_max).step_by(m.n1_stride.max(1)).collect();
        starts
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let row = match &coef {
                    Some(c) if i < c.rows => Some(Mat::row_vec(c.row(i).to_vec())),
                    Some(_) => return fail("Internal error: a joint path has too few bands."),
                    None => None,
                };
                let mut n = vec![s];
                n.extend_from_slice(&m.n);
                Ok((
                    row,
                    Meta {
                        n,
                        // The first band of the group this row summarises.
                        n1: Some(m.n1_index[s]),
                        order: m.n.len() as u8,
                        ..m.clone()
                    },
                ))
            })
            .collect()
    }

    /// Trim along time back to the extent of the input signal.
    fn unpad(&self, coef: Mat, m: &Meta) -> Mat {
        let op = &self.time;
        if op.average == Averaging::Global {
            return coef;
        }
        let res = if m.order == 0 {
            if op.average == Averaging::None {
                0
            } else {
                op.log2_stride
            }
        } else if op.average == Averaging::None && m.n.len() > 1 {
            // Unaveraged second-order paths sit at the resolution of their
            // widest wavelet.
            m.j2.unwrap_or(-1).max(0)
        } else {
            op.log2_stride.max(0)
        } as usize;
        coef.keep_cols(op.borders.start[res], op.borders.end[res])
    }

    fn describe(&self, m: &Meta) -> JtfsPath {
        let op = &self.time;
        let f1 = m.n1.map(|i| &op.psi1[i - 1]);
        let f2 = m.n2.map(|i| &op.psi2[i - 1]);
        let ff = m.n_fr.map(|i| &self.psis_fr[i - 1]);
        let tag = match m.spin {
            None => "",
            Some(s) if s > 0 => "u",
            Some(s) if s < 0 => "d",
            Some(_) => "n",
        };
        let n_fr = m.n_fr.unwrap_or(0);
        let label = match (m.order, self.format) {
            (0, _) => "S0".to_string(),
            (1, Format::Joint) => format!("J1_{n_fr}{tag}"),
            (_, Format::Joint) => format!("J2_{}_{n_fr}{tag}", m.n2.unwrap_or(0)),
            (1, Format::Time) => format!("J1_{}_{n_fr}{tag}", m.n1.unwrap_or(0)),
            (_, Format::Time) => {
                format!("J2_{}_{}_{n_fr}{tag}", m.n1.unwrap_or(0), m.n2.unwrap_or(0))
            }
        };
        JtfsPath {
            order: m.order,
            n1: m.n1,
            n2: m.n2,
            n_fr: m.n_fr,
            j1: f1.map(|f| f.j),
            j2: f2.map(|f| f.j),
            j_fr: ff.map(|f| f.j),
            xi1: f1.map(|f| f.xi),
            xi2: f2.map(|f| f.xi),
            xi_fr: ff.map(|f| f.xi),
            sigma1: f1.map(|f| f.sigma),
            sigma2: f2.map(|f| f.sigma),
            sigma_fr: ff.map(|f| f.sigma),
            spin: m.spin,
            label,
        }
    }
}
