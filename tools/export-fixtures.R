# Convert the wavscat Kymatio reference fixtures into a form Rust can read.
#
# The fixtures live in the R package as nested lists in .rds files. Each one is
# flattened here into two files: `<name>.tsv`, a manifest with one line per leaf,
# and `<name>.bin`, every numeric leaf concatenated as little-endian float64.
#
# Manifest columns: key, type ("f64", "str" or "null"), dims (comma-separated,
# column-major as in R), offset and length in values, and the string value.
# Keys join list names with "/", using one-based positions for unnamed elements.
#
# Usage, from the wavscat-core root:
#
#     Rscript tools/export-fixtures.R ../wavscat/tests/testthat/fixtures fixtures

args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args) >= 1L) args[[1L]] else "../wavscat/tests/testthat/fixtures"
dst <- if (length(args) >= 2L) args[[2L]] else "fixtures"
dir.create(dst, showWarnings = FALSE, recursive = TRUE)

export_one <- function(path, dst) {
  obj <- readRDS(path)
  stem <- sub("\\.rds$", "", basename(path))
  bin <- file(file.path(dst, paste0(stem, ".bin")), "wb")
  on.exit(close(bin))
  rows <- character(0)
  offset <- 0

  walk <- function(x, key) {
    if (is.null(x)) {
      rows <<- c(rows, paste(key, "null", "", 0, 0, "", sep = "\t"))
    } else if (is.list(x)) {
      nms <- names(x)
      for (i in seq_along(x)) {
        nm <- if (is.null(nms) || !nzchar(nms[i])) as.character(i) else nms[i]
        walk(x[[i]], if (nzchar(key)) paste0(key, "/", nm) else nm)
      }
    } else if (is.character(x)) {
      # String vectors are joined with "|", so no element may contain one.
      if (any(grepl("[|\t\n]", x))) stop("Unsupported character in: ", key)
      rows <<- c(rows, paste(key, "str", length(x), 0, length(x),
                             paste(x, collapse = "|"), sep = "\t"))
    } else if (is.numeric(x) || is.logical(x)) {
      v <- as.double(x)
      if (anyNA(v)) stop("NA in numeric leaf: ", key)
      dims <- if (is.null(dim(x))) length(v) else dim(x)
      writeBin(v, bin, size = 8L, endian = "little")
      rows <<- c(rows, paste(key, "f64", paste(dims, collapse = ","),
                             format(offset, scientific = FALSE),
                             length(v), "", sep = "\t"))
      offset <<- offset + length(v)
    } else {
      stop("Unsupported leaf type ", class(x)[1L], " at ", key)
    }
  }

  walk(obj, "")
  # A binary connection, so that Windows does not write CRLF line endings.
  tsv <- file(file.path(dst, paste0(stem, ".tsv")), "wb")
  writeLines(rows, tsv, sep = "\n", useBytes = TRUE)
  close(tsv)
  cat(sprintf("%s: %d leaves, %d values\n", stem, length(rows), offset))
}

for (f in list.files(src, pattern = "\\.rds$", full.names = TRUE)) {
  export_one(f, dst)
}
