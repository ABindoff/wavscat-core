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
