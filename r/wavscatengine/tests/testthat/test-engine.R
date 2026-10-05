test_that("this machine reproduces every reference bit", {
  expect_length(engine_verify(), 0L)
  expect_match(engine_version()$numerics, "^0\\.")
})

test_that("time and joint scattering run through the engine", {
  x <- cbind(sin(seq_len(1024) / 7), cos(seq_len(1024) / 3))
  # n, J, Q1, Q2, T kind, T, max_order, stride
  s <- engine_scattering_1d(c(1024, 5, 8, 1, 1, 32, 2, 32), x)
  expect_length(s$coef, 2L)
  expect_length(s$coef[[1L]], length(s$labels))
  expect_identical(s$labels[1:2], c("S0", "S1_1"))
  # n, J, Q1, Q2, T kind, T, stride, J_fr, Q_fr, F kind, F, stride_fr, format, out_type
  j <- engine_jtfs(c(1024, 5, 8, 1, 1, 32, 32, 3, 1, 1, 8, 8, 0, 0), x)
  expect_length(j$s1, 2L)
  expect_true(is.matrix(j$coef[[1L]][[2L]]))
  r <- engine_jtfs_renorm(c(1024, 5, 8, 1, 1, 32, 32, 3, 1, 1, 8, 8, 0, 0), j$coef[[1L]], j$s1[[1L]], 1e-12)
  expect_length(r, length(j$labels))
})

test_that("a synthetic tapping trial is accepted with features", {
  video <- synthetic_video(width = 160, height = 120, iti_sd = 0.01)
  session <- tapping_session()
  for (t in frame_times(30)) {
    tapping_push(session, render_frame(video, t), 160, 120, round(t * 1e6))
  }
  r <- tapping_finish(session)
  expect_true(r$accepted, info = paste(r$reasons, collapse = "; "))
  expect_lt(abs(r$qc$f0_hz - 3), 0.06)
  expect_identical(names(r$features$values)[1L], "f0_hz")
  expect_match(r$features$params_hash, "^[0-9a-f]{16}$")
  expect_lt(abs(mean(r$itis) - 1 / 3), 0.003)
})

test_that("a matrix frame is read as an image, and bad frames are refused", {
  session <- tapping_session(capacity = 8)
  img <- matrix(as.integer(outer(1:120, 1:160, function(y, x) (x + y) %% 256)), 120, 160)
  tapping_push(session, img, 160, 120, 1000)
  expect_identical(tapping_frames(session), 1L)
  expect_error(tapping_push(session, img, 160, 120, 1000), "does not follow")
  expect_error(tapping_push(session, raw(10), 160, 120, 2000), "too small")
  tapping_reset(session)
  expect_identical(tapping_frames(session), 0L)
})

test_that("a second rhythmic object is rejected with a reason", {
  video <- synthetic_video(width = 160, height = 120, iti_sd = 0.01, distractor_hz = 4.5)
  session <- tapping_session()
  for (t in frame_times(30)) {
    tapping_push(session, render_frame(video, t), 160, 120, round(t * 1e6))
  }
  r <- tapping_finish(session)
  expect_false(r$accepted)
  expect_null(r$features)
  expect_true(any(grepl("another rhythmic movement", r$reasons)))
})
