#!/usr/bin/env Rscript
# Run one checkpointed R/MixSIAR/JAGS numerical-agreement fit.

suppressPackageStartupMessages({
  library(MixSIAR)
  library(posterior)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
value <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  args[[i + 1]]
}
unit <- value("--unit")
config_file <- value("--config")
output_dir <- value("--output")
chains <- as.integer(value("--chains", "3"))
chain_length <- as.integer(value("--chain-length", "50000"))
burn <- as.integer(value("--burn", "25000"))
thin <- as.integer(value("--thin", "25"))
seed <- as.integer(value("--seed", "20260814"))
force <- "--force" %in% args
if (is.null(unit) || is.null(config_file) || is.null(output_dir)) {
  stop("--unit, --config and --output are required")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
done_file <- file.path(output_dir, "DONE")
failed_file <- file.path(output_dir, "FAILED")
if (file.exists(done_file) && !force) {
  cat("SKIP complete:", output_dir, "\n")
  quit(status = 0)
}
if (file.exists(failed_file)) file.remove(failed_file)

cfg <- fromJSON(config_file, simplifyVector = TRUE)
model <- cfg$model
root <- normalizePath(file.path(dirname(config_file), "..", "..", "..", ".."),
                      winslash = "/", mustWork = TRUE)
data_dir <- file.path(root, "mixsiarpy", "data")

write_json(list(unit=unit, sampling=list(chains=chains,
  chainLength=chain_length, burn=burn, thin=thin), seed=seed, model=model),
  file.path(output_dir, "config.json"), auto_unbox=TRUE, pretty=TRUE)

tryCatch({
  factors <- model$factors
  if (is.null(factors)) factors <- NULL
  fac_random <- model$random
  if (is.null(fac_random)) fac_random <- NULL
  fac_nested <- model$nested
  if (is.null(fac_nested)) fac_nested <- NULL
  continuous <- model$continuous
  if (is.null(continuous)) continuous <- NULL
  # Python CSV headers use ':' where R's check.names converts spaces/dots.
  if (!is.null(continuous)) continuous <- gsub(":", ".", continuous, fixed=TRUE)

  mix <- load_mix_data(file.path(data_dir, model$mix), model$iso,
    factors, fac_random, fac_nested, continuous)
  source_factor <- model$source_factor
  if (is.null(source_factor)) source_factor <- NULL
  source <- load_source_data(file.path(data_dir, model$source), source_factor,
    isTRUE(model$conc_dep), model$source_type, mix)
  discr <- load_discr_data(file.path(data_dir, model$discr), mix)
  model_file <- file.path(output_dir, "MixSIAR_model.txt")
  write_JAGS_model(model_file, isTRUE(model$resid_err), isTRUE(model$process_err),
                   mix, source)
  alpha <- model$alpha_prior
  settings <- list(chainLength=chain_length, burn=burn, thin=thin,
                   chains=chains, calcDIC=TRUE)
  set.seed(seed)
  started <- proc.time()[["elapsed"]]
  fit <- run_model(settings, mix, source, discr, model_file, alpha.prior=alpha)
  elapsed <- proc.time()[["elapsed"]] - started
  saveRDS(fit, file.path(output_dir, "fit.rds"), compress=FALSE)

  # R2jags returns retained draws in BUGSoutput$sims.array with dimensions
  # iteration x chain x variable (there is no fit$samples member).
  draws <- posterior::as_draws_array(fit$BUGSoutput$sims.array)
  saveRDS(draws, file.path(output_dir, "draws_array.rds"), compress=FALSE)
  draws_df <- posterior::as_draws_df(draws)
  write.csv(draws_df, file.path(output_dir, "posterior_draws.csv"), row.names=FALSE)
  sm <- posterior::summarise_draws(draws,
    mean, sd, median,
    q2.5=~quantile(.x, .025), q97.5=~quantile(.x, .975),
    posterior::rhat, posterior::ess_bulk, posterior::ess_tail,
    posterior::mcse_mean, posterior::mcse_sd)
  sm <- as.data.frame(sm)
  names(sm)[names(sm) == "posterior::rhat"] <- "rhat"
  names(sm)[names(sm) == "posterior::ess_bulk"] <- "ess_bulk"
  names(sm)[names(sm) == "posterior::ess_tail"] <- "ess_tail"
  names(sm)[names(sm) == "posterior::mcse_mean"] <- "mcse_mean"
  names(sm)[names(sm) == "posterior::mcse_sd"] <- "mcse_sd"
  write.csv(sm, file.path(output_dir, "summary.csv"), row.names=FALSE)
  monitor_pattern <- "^(p\\.global|p\\.fac|p\\.both|fac[12]\\.sig|resid\\.prop|ilr\\.|beta|Sigma)"
  if (unit == "alligator_length_ind") monitor_pattern <- paste0(monitor_pattern, "|^p\\.ind")
  key <- subset(sm, grepl(monitor_pattern, variable))
  use <- if (nrow(key)) key else as.data.frame(sm)
  use <- subset(use, is.finite(rhat) & is.finite(ess_bulk) & is.finite(ess_tail))
  max_rhat <- max(use$rhat)
  min_bulk <- min(use$ess_bulk)
  min_tail <- min(use$ess_tail)
  ratios <- use$mcse_mean[is.finite(use$sd) & use$sd > 0] /
            use$sd[is.finite(use$sd) & use$sd > 0]
  max_mcse_sd <- max(ratios, na.rm=TRUE)
  strict <- max_rhat <= 1.01 && min_bulk >= 400 && min_tail >= 400 && max_mcse_sd <= 0.05
  longer <- !strict && max_rhat <= 1.05 && min_bulk >= 100 && min_tail >= 100
  convergence_status <- if (strict) "CONVERGED" else if (longer) "NEEDS_LONGER_RUN" else "NOT_CONVERGED"
  metadata <- list(
    unit=unit, R=R.version.string,
    MixSIAR=as.character(packageVersion("MixSIAR")),
    rjags=as.character(packageVersion("rjags")),
    elapsed_seconds=elapsed, chains=chains, chain_length=chain_length,
    burn=burn, thin=thin,
    retained_draws_per_chain=(chain_length-burn)/thin,
    monitored_parameter_count=nrow(use),
    max_rhat=max_rhat, min_ess_bulk=min_bulk, min_ess_tail=min_tail,
    max_mcse_over_sd=max_mcse_sd,
    min_bulk_ess_per_second=min_bulk/elapsed,
    convergence_status=convergence_status)
  write_json(metadata, file.path(output_dir, "metadata.json"),
             auto_unbox=TRUE, pretty=TRUE)
  write_json(metadata, done_file, auto_unbox=TRUE, pretty=TRUE)
  for (marker in c("CONVERGED", "NEEDS_LONGER_RUN", "NOT_CONVERGED")) {
    path <- file.path(output_dir, marker)
    if (file.exists(path)) file.remove(path)
  }
  write_json(metadata, file.path(output_dir, convergence_status),
             auto_unbox=TRUE, pretty=TRUE)
  cat(toJSON(metadata, auto_unbox=TRUE, pretty=TRUE), "\n")
}, error=function(e) {
  report <- paste(conditionMessage(e), paste(capture.output(traceback()), collapse="\n"), sep="\n")
  writeLines(report, failed_file)
  writeLines(report, file.path(output_dir, "traceback.txt"))
  stop(e)
})
