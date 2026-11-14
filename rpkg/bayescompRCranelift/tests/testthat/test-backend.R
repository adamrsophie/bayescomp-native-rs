models <- list(
  lin = list(code = "data { int N; vector[N] x; vector[N] y; } parameters { real a; real b; real<lower=0> s; }
    model { a ~ normal(0, 5); b ~ normal(0, 5); s ~ exponential(1); y ~ normal(a + b * x, s); }",
    data = list(N = 20, x = seq(-1, 1, length.out = 20), y = sin(seq(-1, 1, length.out = 20)))),
  loops = list(code = "data { int N; array[N] int y; } parameters { real<lower=0, upper=1> p; vector[3] z; }
    model { real acc = 0; for (i in 1:N) { if (y[i] == 1) acc += log(p); else acc += log1m(p); }
      target += acc; for (k in 1:3) { if (k == 2) continue; z[k] ~ normal(0, 1); } z[2] ~ normal(1, 2);
      int j = 0; while (1) { j += 1; if (j > 3) break; } }",
    data = list(N = 6, y = c(1L, 0L, 1L, 1L, 0L, 1L))))

for (engine in c("jit", "interpreter")) {
  test_that(sprintf("the %s engine matches the R backend", engine), {
    old <- cranelift_engine(engine)
    on.exit(options(bayescompRCranelift.engine = NULL))
    for (nm in names(models)) {
      m <- bayescompR::bayescomp_model(models[[nm]]$code, data = models[[nm]]$data, backend = "cranelift", allow_fallback = TRUE)
      expect_equal(m$provenance$log_density, "cranelift")
      v <- bayescompR::bayescomp_validate_backend(m, n = 10)
      expect_lt(max(v$lp_rel), 1e-12, label = nm)
      expect_lt(max(v$grad_rel), 1e-12, label = nm)
    }
  })
}

test_that("errors keep their classes and unsupported programs are refused", {
  m <- bayescompR::bayescomp_model("parameters { real x; } model { if (x > 1) reject(\"too big\"); x ~ normal(0, 1); }",
                                   backend = "cranelift", allow_fallback = TRUE)
  expect_error(bayescompR::bayescomp_log_prob(m, 2), "too big", class = "bayescomp_reject_error")
  expect_error(bayescompR::bayescomp_model("functions { real f(real x) { return x; } } parameters { real x; } model { x ~ normal(f(0), 1); }",
                                           backend = "cranelift", allow_fallback = TRUE),
               class = "bayescomp_unsupported_error")
})

test_that("sampling with the native backend runs", {
  m <- bayescompR::bayescomp_model(models$lin$code, data = models$lin$data, backend = "cranelift", allow_fallback = TRUE)
  f <- bayescompR::bayescomp_sample(m, chains = 1, iter_warmup = 100, iter_sampling = 100, seed = 1, refresh = 0)
  expect_equal(dim(f$draws)[1:2], c(100L, 1L))
})
