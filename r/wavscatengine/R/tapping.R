#' Webcam finger tapping: a capture session
#'
#' Feed a trial's video frames to the engine one at a time, then analyse it.
#' The engine keeps only a small grid of each frame (64 x 48 by default), never
#' the frame itself. [tapping_finish()] returns quality control for every trial,
#' and features and inter-tap intervals for trials that pass it.
#'
#' Frames are luma (greyscale) or RGBA images. Pass them as a raw or integer
#' vector in row-major order, as image and video libraries produce them, or as
#' a `height x width` matrix, which is transposed to row-major for you.
#'
#' @param capacity Frames kept; once full, the oldest is overwritten.
#' @param grid_width,grid_height Size of the grid each frame is averaged onto.
#' @return A `wavscatengine_session`.
#'
#' @examples
#' video <- synthetic_video(width = 160, height = 120, iti_sd = 0.01)
#' session <- tapping_session()
#' for (t in frame_times(30)) {
#'   tapping_push(session, render_frame(video, t), 160, 120, round(t * 1e6))
#' }
#' report <- tapping_finish(session)
#' report$accepted
#' report$qc$f0_hz
#' @export
tapping_session <- function(capacity = 1024, grid_width = 64, grid_height = 48) {
  structure(
    list(ptr = .Call(c_session_new, as.integer(capacity), as.integer(grid_width), as.integer(grid_height))),
    class = "wavscatengine_session"
  )
}

#' @export
print.wavscatengine_session <- function(x, ...) {
  cat("<wavscatengine tapping session:", tapping_frames(x), "frames>\n")
  invisible(x)
}

#' @rdname tapping_session
#' @param session A `wavscatengine_session`.
#' @param pixels The frame: a raw or integer vector (row-major), or a
#'   `height x width` matrix of values 0 to 255.
#' @param width,height Frame size in pixels.
#' @param timestamp_us Capture time in microseconds. Must increase.
#' @param format `"luma"` or `"rgba"`.
#' @export
tapping_push <- function(session, pixels, width, height, timestamp_us, format = c("luma", "rgba")) {
  format <- match.arg(format)
  channels <- if (format == "rgba") 4L else 1L
  if (is.matrix(pixels)) {
    if (channels != 1L) stop("Pass RGBA frames as a row-major raw vector.", call. = FALSE)
    pixels <- t(pixels)
  }
  if (!is.raw(pixels)) {
    pixels <- as.raw(pmin(pmax(round(as.numeric(pixels)), 0), 255))
  }
  .Call(c_session_push, session$ptr, pixels, as.integer(width * channels),
        as.integer(width), as.integer(height), as.double(timestamp_us), channels == 4L)
  invisible(session)
}

#' @rdname tapping_session
#' @export
tapping_frames <- function(session) {
  .Call(c_session_len, session$ptr)
}

#' @rdname tapping_session
#' @export
tapping_reset <- function(session) {
  .Call(c_session_reset, session$ptr)
  invisible(session)
}

QC_FIELDS <- c(
  "accepted", "frames_accepted", "frames_rejected", "frames_dropped",
  "effective_fps", "longest_gap", "dropped_fraction", "gain_min", "gain_max",
  "abrupt_changes", "dark_frames", "score", "competitor_ratio",
  "from_harmonic", "f0_hz", "usable_cycles", "loading_spread"
)

#' @rdname tapping_session
#' @return `tapping_finish()`: a list with `accepted`, `reasons` (why it was
#'   rejected, in words), `qc` (the measurements, `NA` where the analysis did
#'   not get that far), `features` (for an accepted trial: `values` named by
#'   feature, and the `params_hash`, `crate_version`, `numerics_version` and
#'   `schema_version` that make them comparable) and `itis` (inter-tap
#'   intervals in seconds, for an accepted trial).
#' @export
tapping_finish <- function(session) {
  r <- .Call(c_session_finish, session$ptr)
  qc <- as.list(r$arrays[[1L]])
  names(qc) <- QC_FIELDS
  qc <- lapply(qc, function(v) if (is.nan(v)) NA_real_ else v)
  accepted <- isTRUE(qc$accepted == 1)
  qc$accepted <- NULL
  qc$from_harmonic <- if (is.na(qc$from_harmonic)) NA else qc$from_harmonic == 1
  counts <- r$arrays[[4L]]
  n_reasons <- counts[[1L]]
  n_features <- counts[[2L]]
  reasons <- r$strings[seq_len(n_reasons)]
  names_f <- r$strings[n_reasons + seq_len(n_features)]
  stamp <- r$strings[n_reasons + n_features + 1:4]
  features <- if (accepted) {
    list(
      values = stats::setNames(r$arrays[[2L]], names_f),
      params_hash = stamp[[1L]],
      crate_version = stamp[[2L]],
      numerics_version = stamp[[3L]],
      schema_version = as.integer(stamp[[4L]])
    )
  }
  list(
    accepted = accepted,
    reasons = reasons,
    qc = qc,
    features = features,
    itis = if (accepted) r$arrays[[3L]]
  )
}

#' Synthetic tapping videos with known taps
#'
#' A bright blob tapping at 3 Hz on a textured background, with sensor noise,
#' gain drift and an exposure step, rendered identically on every platform.
#' For trying the pipeline and testing it.
#'
#' @param width,height Frame size in pixels.
#' @param iti_sd Standard deviation of the inter-tap intervals, in seconds.
#' @param seed Random seed.
#' @param distractor_hz Rate of a second moving object, in Hz, or `NULL`.
#' @return `synthetic_video()`: a `wavscatengine_video`.
#' @export
synthetic_video <- function(width = 320, height = 240, iti_sd = 0, seed = 1, distractor_hz = NULL) {
  ptr <- .Call(c_video_new, as.integer(width), as.integer(height), as.double(iti_sd),
               as.double(seed), if (is.null(distractor_hz)) NaN else as.double(distractor_hz))
  info <- .Call(c_video_info, ptr)$arrays
  structure(list(ptr = ptr, taps = info[[1L]], width = width, height = height),
            class = "wavscatengine_video")
}

#' @rdname synthetic_video
#' @param video A `wavscatengine_video`.
#' @param t Capture time in seconds.
#' @return `render_frame()`: the frame as a row-major raw vector of
#'   `width * height` luma values.
#' @export
render_frame <- function(video, t) {
  .Call(c_video_render, video$ptr, as.double(t), as.integer(video$width * video$height))
}

#' @rdname synthetic_video
#' @param duration Length in seconds.
#' @param fps Nominal frame rate.
#' @param jitter_sd Timing jitter, in seconds.
#' @param drop_prob Probability of dropping each frame.
#' @return `frame_times()`: capture times in seconds.
#' @export
frame_times <- function(duration, fps = 30, jitter_sd = 0.004, drop_prob = 0.05, seed = 41) {
  .Call(c_frame_times, as.double(duration), as.double(fps), as.double(jitter_sd),
        as.double(drop_prob), as.double(seed))$arrays[[1L]]
}
