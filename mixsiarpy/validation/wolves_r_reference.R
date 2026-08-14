# Reproducible MixSIAR 3.1.12 / JAGS reference run for cross-language validation.
library(MixSIAR)
library(coda)

root <- normalizePath(file.path(getwd()), winslash = "/")
data_dir <- file.path(root, "mixsiarpy", "data")
output_dir <- file.path(root, "validation", "results", "wolves_r")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

mix <- load_mix_data(
  filename = file.path(data_dir, "wolves_consumer.csv"),
  iso_names = c("d13C", "d15N"),
  factors = c("Region", "Pack"),
  fac_random = c(TRUE, TRUE),
  fac_nested = c(FALSE, TRUE),
  cont_effects = NULL
)
source <- load_source_data(
  filename = file.path(data_dir, "wolves_sources.csv"),
  source_factors = "Region", conc_dep = FALSE, data_type = "means", mix
)
discr <- load_discr_data(file.path(data_dir, "wolves_discrimination.csv"), mix)

model_file <- file.path(output_dir, "MixSIAR_model.txt")
write_JAGS_model(model_file, resid_err = TRUE, process_err = TRUE, mix, source)
set.seed(20260813)
run_settings <- list(chainLength = 100000, burn = 50000, thin = 50,
                     chains = 3, calcDIC = TRUE)
fit <- run_model(run_settings, mix, source, discr, model_file, alpha.prior = 1)

draws <- as.matrix(coda::as.mcmc(fit))
keep <- grep("^(p\\.global|p\\.fac1|p\\.fac2|fac1\\.sig|fac2\\.sig|resid\\.prop)",
             colnames(draws), value = TRUE)
write.csv(draws[, keep, drop = FALSE], file.path(output_dir, "posterior_draws.csv"),
          row.names = FALSE)

summary_table <- data.frame(
  parameter = keep,
  mean = colMeans(draws[, keep, drop = FALSE]),
  sd = apply(draws[, keep, drop = FALSE], 2, sd),
  q025 = apply(draws[, keep, drop = FALSE], 2, quantile, 0.025),
  median = apply(draws[, keep, drop = FALSE], 2, quantile, 0.5),
  q975 = apply(draws[, keep, drop = FALSE], 2, quantile, 0.975),
  row.names = NULL
)
write.csv(summary_table, file.path(output_dir, "summary.csv"), row.names = FALSE)

chains <- fit$samples
gelman <- coda::gelman.diag(chains, multivariate = FALSE)$psrf
gelman_table <- data.frame(parameter = rownames(gelman), rhat = gelman[, "Point est."],
                           rhat_upper = gelman[, "Upper C.I."], row.names = NULL)
write.csv(gelman_table, file.path(output_dir, "gelman.csv"), row.names = FALSE)

metadata <- c(
  paste("R", R.version.string),
  paste("MixSIAR", as.character(packageVersion("MixSIAR"))),
  paste("JAGS", as.character(rjags::jags.version())),
  paste("chains", run_settings$chains), paste("chainLength", run_settings$chainLength),
  paste("burn", run_settings$burn), paste("thin", run_settings$thin),
  paste("saved_draws_per_chain", (run_settings$chainLength-run_settings$burn)/run_settings$thin),
  paste("DIC", fit$BUGSoutput$DIC)
)
writeLines(metadata, file.path(output_dir, "metadata.txt"))
saveRDS(fit, file.path(output_dir, "fit.rds"))

cat("R reference complete. Max R-hat:", max(gelman_table$rhat, na.rm = TRUE), "\n")
