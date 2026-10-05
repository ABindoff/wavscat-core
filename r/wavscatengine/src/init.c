/*
 * The C glue between R and the Rust library (src/rust).
 *
 * Every Rust entry point returns an opaque result holding numeric arrays and
 * strings, or an error; wse_unwrap() turns one into the R list
 * list(arrays = list(...), strings = character(...)) and frees it. The rest
 * is thin .Call wrappers and finalisers for the two handle types.
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
extern void *wse_session_new(size_t capacity, size_t grid_width, size_t grid_height);
extern void wse_session_free(void *h);
extern WseResult *wse_session_push(void *h, const uint8_t *pixels, size_t len, size_t stride,
                                   size_t width, size_t height, double timestamp_us, int rgba);
extern size_t wse_session_len(const void *h);
extern void wse_session_reset(void *h);
extern WseResult *wse_session_finish(const void *h);
extern void *wse_video_new(size_t width, size_t height, double iti_sd, double seed, double distractor_hz);
extern void wse_video_free(void *v);
extern int wse_video_render(const void *v, double t, uint8_t *out, size_t len);
extern WseResult *wse_video_info(const void *v);
extern WseResult *wse_frame_times(double duration, double fps, double jitter_sd, double drop_prob, double seed);

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
static void session_finalise(SEXP ptr) {
    void *h = R_ExternalPtrAddr(ptr);
    if (h != NULL) {
        wse_session_free(h);
        R_ClearExternalPtr(ptr);
    }
}

static void video_finalise(SEXP ptr) {
    void *v = R_ExternalPtrAddr(ptr);
    if (v != NULL) {
        wse_video_free(v);
        R_ClearExternalPtr(ptr);
    }
}

static void *handle(SEXP ptr, const char *what) {
    void *h = R_ExternalPtrAddr(ptr);
    if (h == NULL) Rf_error("This %s is no longer valid (was it saved and reloaded?).", what);
    return h;
}

static SEXP c_session_new(SEXP capacity, SEXP gw, SEXP gh) {
    void *h = wse_session_new((size_t) Rf_asInteger(capacity), (size_t) Rf_asInteger(gw),
                              (size_t) Rf_asInteger(gh));
    if (h == NULL) Rf_error("Invalid session settings.");
    SEXP ptr = PROTECT(R_MakeExternalPtr(h, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, session_finalise, TRUE);
    UNPROTECT(1);
    return ptr;
}

static SEXP c_session_push(SEXP ptr, SEXP pixels, SEXP stride, SEXP width, SEXP height, SEXP ts, SEXP rgba) {
    void *h = handle(ptr, "tapping session");
    return wse_unwrap(wse_session_push(h, RAW(pixels), (size_t) XLENGTH(pixels), (size_t) Rf_asInteger(stride),
                                       (size_t) Rf_asInteger(width), (size_t) Rf_asInteger(height),
                                       Rf_asReal(ts), Rf_asLogical(rgba)));
}

static SEXP c_session_len(SEXP ptr) {
    return Rf_ScalarInteger((int) wse_session_len(handle(ptr, "tapping session")));
}

static SEXP c_session_reset(SEXP ptr) {
    wse_session_reset(handle(ptr, "tapping session"));
    return R_NilValue;
}

static SEXP c_session_finish(SEXP ptr) {
    return wse_unwrap(wse_session_finish(handle(ptr, "tapping session")));
}

static SEXP c_video_new(SEXP width, SEXP height, SEXP iti_sd, SEXP seed, SEXP distractor) {
    void *v = wse_video_new((size_t) Rf_asInteger(width), (size_t) Rf_asInteger(height), Rf_asReal(iti_sd),
                            Rf_asReal(seed), Rf_asReal(distractor));
    if (v == NULL) Rf_error("Invalid video settings.");
    SEXP ptr = PROTECT(R_MakeExternalPtr(v, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, video_finalise, TRUE);
    UNPROTECT(1);
    return ptr;
}

static SEXP c_video_render(SEXP ptr, SEXP t, SEXP len) {
    void *v = handle(ptr, "synthetic video");
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t) Rf_asInteger(len)));
    int status = wse_video_render(v, Rf_asReal(t), RAW(out), (size_t) XLENGTH(out));
    if (status != 0) {
        UNPROTECT(1);
        Rf_error("Rendering failed (status %d).", status);
    }
    UNPROTECT(1);
    return out;
}

static SEXP c_video_info(SEXP ptr) { return wse_unwrap(wse_video_info(handle(ptr, "synthetic video"))); }

static SEXP c_frame_times(SEXP duration, SEXP fps, SEXP jitter, SEXP drop, SEXP seed) {
    return wse_unwrap(wse_frame_times(Rf_asReal(duration), Rf_asReal(fps), Rf_asReal(jitter),
                                      Rf_asReal(drop), Rf_asReal(seed)));
}

static const R_CallMethodDef calls[] = {
    {"c_version", (DL_FUNC) &c_version, 0},
    {"c_verify", (DL_FUNC) &c_verify, 0},
    {"c_s1d", (DL_FUNC) &c_s1d, 4},
    {"c_jtfs", (DL_FUNC) &c_jtfs, 4},
    {"c_jtfs_renorm", (DL_FUNC) &c_jtfs_renorm, 5},
    {"c_session_new", (DL_FUNC) &c_session_new, 3},
    {"c_session_push", (DL_FUNC) &c_session_push, 7},
    {"c_session_len", (DL_FUNC) &c_session_len, 1},
    {"c_session_reset", (DL_FUNC) &c_session_reset, 1},
    {"c_session_finish", (DL_FUNC) &c_session_finish, 1},
    {"c_video_new", (DL_FUNC) &c_video_new, 5},
    {"c_video_render", (DL_FUNC) &c_video_render, 3},
    {"c_video_info", (DL_FUNC) &c_video_info, 1},
    {"c_frame_times", (DL_FUNC) &c_frame_times, 5},
    {NULL, NULL, 0}};

void R_init_wavscatengine(DllInfo *dll) {
    R_registerRoutines(dll, NULL, calls, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
