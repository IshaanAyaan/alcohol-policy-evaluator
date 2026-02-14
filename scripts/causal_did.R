#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: causal_did.R <input_csv> <output_csv> <meta_json>")
}

input_csv <- args[[1]]
output_csv <- args[[2]]
meta_json <- args[[3]]

write_meta <- function(status, msg = NULL) {
  lines <- c(
    "{",
    sprintf('  "status": "%s"%s', status, ifelse(is.null(msg), "", ","))
  )
  if (!is.null(msg)) {
    safe <- gsub('"', "'", msg)
    lines <- c(lines, sprintf('  "message": "%s"', safe))
  }
  lines <- c(lines, "}")
  writeLines(lines, con = meta_json)
}

if (!file.exists(input_csv)) {
  write_meta("failed", "input file missing")
  quit(save = "no", status = 0)
}

if (!requireNamespace("did", quietly = TRUE)) {
  write_meta("skipped", "R package did is not installed")
  quit(save = "no", status = 0)
}

suppressPackageStartupMessages(library(did))

tryCatch({
  df <- read.csv(input_csv)
  needed <- c("state_abbrev", "year", "rate_impaired_per100k", "treatment_year", "unemployment_rate", "pcpi_nominal", "vmt_per_capita")
  missing_cols <- setdiff(needed, names(df))
  if (length(missing_cols) > 0) {
    write_meta("failed", paste("missing columns:", paste(missing_cols, collapse = ", ")))
    quit(save = "no", status = 0)
  }

  df <- df[complete.cases(df[, needed]), ]
  if (nrow(df) == 0) {
    write_meta("failed", "no complete cases")
    quit(save = "no", status = 0)
  }

  df$gname <- ifelse(is.na(df$treatment_year), 0, df$treatment_year)

  att <- did::att_gt(
    yname = "rate_impaired_per100k",
    tname = "year",
    idname = "state_abbrev",
    gname = "gname",
    data = df,
    xformla = ~ unemployment_rate + pcpi_nominal + vmt_per_capita,
    est_method = "dr",
    panel = TRUE,
    bstrap = TRUE,
    cband = TRUE
  )

  dyn <- did::aggte(att, type = "dynamic")

  out <- data.frame(
    event_time = dyn$egt,
    att = dyn$att.egt,
    se = dyn$se.egt,
    crit_val = dyn$crit.val.egt
  )
  write.csv(out, output_csv, row.names = FALSE)
  write_meta("ok")
}, error = function(e) {
  write_meta("failed", conditionMessage(e))
})
