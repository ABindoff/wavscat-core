/*
 * The C glue between R and the Rust library (src/rust).
 *
 * Every Rust entry point returns an opaque result holding numeric arrays and
 * strings, or an error; wse_unwrap() turns one into the R list
 * list(arrays = list(...), strings = character(...)) and frees it. The rest
 * is thin .Call wrappers.
 */
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdint.h>
#include <string.h>

typedef struct WseResult WseResult;

extern const char *wse_result_error(const WseResult *r);
extern size_t wse_result_n_arrays(const WseResult *r);
extern const double *wse_result_array(const WseResult *r, size_t i, size_t *rows, size_t *cols);
extern size_t wse_result_n_strings(const WseResult *r);
extern const char *wse_result_string(const WseResult *r, size_t i);
extern void wse_result_free(WseResult *r);

extern WseResult *wse_version(void);
extern WseResult *wse_verify(void);
extern WseResult *wse_s1d(const double *p, size_t np, const double *x, size_t n, size_t channels);
extern WseResult *wse_jtfs(const double *p, size_t np, const double *x, size_t n, size_t channels);
extern WseResult *wse_jtfs_renorm(const double *p, size_t np, const double *data, const int *dims,
                                  size_t n_paths, const double *s1, size_t s1_rows, size_t s1_cols,
                                  double eps);

/* Convert a result to an R list and free it, or raise its error. */
static SEXP wse_unwrap(WseResult *r) {
    const char *e = wse_result_error(r);
    if (e != NULL) {
        char msg[2048];
        strncpy(msg, e, sizeof msg - 1);
        msg[sizeof msg - 1] = '\0';
        wse_result_free(r);
        Rf_error("%s", msg);
    }
    size_t na = wse_result_n_arrays(r), ns = wse_result_n_strings(r);
    SEXP arrays = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t) na));
    for (size_t i = 0; i < na; i++) {
        size_t rows, cols;
        const double *data = wse_result_array(r, i, &rows, &cols);
        size_t len = cols == 0 ? rows : rows * cols;
        SEXP v = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t) len));
        if (len > 0) memcpy(REAL(v), data, len * sizeof(double));
        if (cols > 0) {
            SEXP dim = PROTECT(Rf_allocVector(INTSXP, 2));
            INTEGER(dim)[0] = (int) rows;
            INTEGER(dim)[1] = (int) cols;
            Rf_setAttrib(v, R_DimSymbol, dim);
            UNPROTECT(1);
        }
        SET_VECTOR_ELT(arrays, (R_xlen_t) i, v);
        UNPROTECT(1);
    }
    SEXP strings = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t) ns));
    for (size_t i = 0; i < ns; i++) {
        SET_STRING_ELT(strings, (R_xlen_t) i, Rf_mkCharCE(wse_result_string(r, i), CE_UTF8));
    }
    wse_result_free(r);
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, arrays);
    SET_VECTOR_ELT(out, 1, strings);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("arrays"));
    SET_STRING_ELT(names, 1, Rf_mkChar("strings"));
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(4);
    return out;
}

static SEXP c_version(void) { return wse_unwrap(wse_version()); }
static SEXP c_verify(void) { return wse_unwrap(wse_verify()); }

static SEXP c_s1d(SEXP p, SEXP x, SEXP n, SEXP channels) {
    return wse_unwrap(wse_s1d(REAL(p), (size_t) XLENGTH(p), REAL(x), (size_t) Rf_asInteger(n),
                              (size_t) Rf_asInteger(channels)));
}

static SEXP c_jtfs(SEXP p, SEXP x, SEXP n, SEXP channels) {
    return wse_unwrap(wse_jtfs(REAL(p), (size_t) XLENGTH(p), REAL(x), (size_t) Rf_asInteger(n),
                               (size_t) Rf_asInteger(channels)));
}

static SEXP c_jtfs_renorm(SEXP p, SEXP data, SEXP dims, SEXP s1, SEXP eps) {
    SEXP d = Rf_getAttrib(s1, R_DimSymbol);
    return wse_unwrap(wse_jtfs_renorm(REAL(p), (size_t) XLENGTH(p), REAL(data), INTEGER(dims),
                                      (size_t) XLENGTH(dims) / 2, REAL(s1),
                                      (size_t) INTEGER(d)[0], (size_t) INTEGER(d)[1],
                                      Rf_asReal(eps)));
}

/* Handles: external pointers whose finalisers free the Rust object. */
static const R_CallMethodDef calls[] = {
    {"c_version", (DL_FUNC) &c_version, 0},
    {"c_verify", (DL_FUNC) &c_verify, 0},
    {"c_s1d", (DL_FUNC) &c_s1d, 4},
    {"c_jtfs", (DL_FUNC) &c_jtfs, 4},
    {"c_jtfs_renorm", (DL_FUNC) &c_jtfs_renorm, 5},
    {NULL, NULL, 0}};

void R_init_wavscatengine(DllInfo *dll) {
    R_registerRoutines(dll, NULL, calls, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
