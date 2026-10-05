#' @useDynLib wavscatengine, .registration = TRUE
#' @keywords internal
"_PACKAGE"

#' Version of the engine
#'
#' @return A list with `numerics`, the version of the numerical definition
#'   (features from different numerics versions are not comparable), and
#'   `crate`, the version of the Rust code.
#' @export
engine_version <- function() {
  r <- .Call(c_version)
  list(numerics = r$strings[[1L]], crate = r$strings[[2L]])
}

#' Check that this machine computes the reference bits
#'
#' Runs a fixed set of transforms, feature pipelines and a synthetic video
#' through the engine, and compares every output bit with the record built in.
#' The same check runs in CI on Linux, macOS and Windows, x86-64 and ARM64,
#' and in browsers; passing it means features computed here equal, bit for
#' bit, those computed on any of them.
#'
#' @return A character vector of mismatches, empty on success.
#' @export
engine_verify <- function() {
  .Call(c_verify)$strings
}

#' Wavelet scattering through the Rust engine
#'
#' Low-level entry points used by 'wavscat'. Parameters arrive as numeric
#' vectors in fixed positions, already resolved by 'wavscat', so that every
#' value reaches the engine exactly.
#'
#' @param params Numeric vector of resolved operator parameters.
#' @param x Numeric matrix, one signal per column.
#' @return `engine_scattering_1d()`: a list with `labels`, the path labels, and
#'   `coef`, one list per channel of one numeric vector per path.
#'   `engine_jtfs()`: as before, with one matrix per path, plus `s1`, one
#'   first-order `[band, time]` matrix per channel.
#' @export
#' @keywords internal
engine_scattering_1d <- function(params, x) {
  x <- as.matrix(x)
  r <- .Call(c_s1d, as.double(params), as.double(x), nrow(x), ncol(x))
  labels <- r$strings
  np <- length(labels)
  coef <- lapply(seq_len(ncol(x)), function(ch) r$arrays[(ch - 1L) * np + seq_len(np)])
  list(labels = labels, coef = coef)
}

#' @rdname engine_scattering_1d
#' @export
engine_jtfs <- function(params, x) {
  x <- as.matrix(x)
  r <- .Call(c_jtfs, as.double(params), as.double(x), nrow(x), ncol(x))
  labels <- r$strings
  np <- length(labels)
  per <- np + 1L
  coef <- lapply(seq_len(ncol(x)), function(ch) r$arrays[(ch - 1L) * per + seq_len(np)])
  s1 <- lapply(seq_len(ncol(x)), function(ch) r$arrays[[ch * per]])
  list(labels = labels, coef = coef, s1 = s1)
}

#' @rdname engine_scattering_1d
#' @param coef One channel of joint coefficients: a list of matrices.
#' @param s1 That channel's first-order matrix.
#' @param eps Floor added to the first-order energy.
#' @return `engine_jtfs_renorm()`: the renormalised list of matrices.
#' @export
engine_jtfs_renorm <- function(params, coef, s1, eps) {
  coef <- lapply(coef, as.matrix)
  dims <- as.integer(unlist(lapply(coef, dim)))
  data <- unlist(lapply(coef, as.double), use.names = FALSE)
  .Call(c_jtfs_renorm, as.double(params), as.double(data), dims, as.matrix(s1), as.double(eps))$arrays
}
