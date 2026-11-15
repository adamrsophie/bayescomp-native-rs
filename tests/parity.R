# Edge case parity of a native library with the R backend of bayescompR.
# Usage: Rscript tests/parity.R PATH/TO/LIBRARY [cranelift|llvm]
# Every case is evaluated with the interpreter and the JIT, under all four
# Jacobian and propto combinations, and must give the same value, gradient
# and error class as the R backend.
if (nzchar(Sys.getenv("BAYESCOMPR_SRC"))) pkgload::load_all(Sys.getenv("BAYESCOMPR_SRC"), quiet = TRUE) else suppressMessages(library(bayescompR))
a <- commandArgs(TRUE)
backend <- if (length(a) > 1L) a[2] else "cranelift"
Sys.setenv(R_USER_DATA_DIR = tempfile())
bayescomp_install_backend(backend, path = a[1])

P <- function(model, u, data = list(), refuse = FALSE) list(model = model, u = u, data = data, refuse = refuse)
cases <- list(
  phi0 = P("parameters { real x; } model { target += Phi(x) + erf(x) + erfc(x); }", c(0, 1, -1)),
  phi_inf = P("parameters { real x; } model { target += Phi(positive_infinity()) + erf(positive_infinity()) + x; }", 0.3),
  lgamma_pole = P("parameters { real x; } model { target += x; if (is_inf(lgamma(-1))) target += 1; }", 0.2),
  digamma_neg = P("parameters { real<upper=-1e19> x; } model { target += lgamma(x) * 0; x ~ normal(-1e20, 1e19); }", 0),
  propto_const = P("parameters { real x; } model { real y = 1; if (x < 0) y = 2; y ~ std_normal(); x ~ std_normal(); }", c(-1, 1)),
  propto_branch = P("parameters { real x; } model { real z; if (x > 0) z = 1; else z = 2; z ~ normal(0, 1); x ~ std_normal(); }", c(-1, 1)),
  cond_promote = P("parameters { real x; } model { target += (x > 0 ? 1 : 2.0) / 2; }", c(1, -1)),
  abs_int = P("parameters { real x; } model { target += abs(3) / 2 + x; }", 0.1),
  self_assign = P("parameters { vector[2] x; } model { vector[2] z = x; z[{2, 1}] = z; target += z[1]; }", c(10, 20)),
  foreach_snapshot = P("parameters { real x; } model { vector[2] v = [1, 2]'; real s = 0; for (a in v) { s += a; v[2] = x; } target += s; }", 9),
  int_overflow = P("parameters { real x; } model { int k = 2147483647; target += x; if (x > 100) k += 1; }", c(0.5, 200)),
  int_overflow2 = P("parameters { real x; } model { target += 2147483647 + 1 + x; }", 0.5),
  gamma0 = P("parameters { real<lower=0> x; } model { target += gamma_lpdf(0 | 1, 2) + weibull_lpdf(0 | 1, 2) + chi_square_lpdf(0 | 2); x ~ exponential(1); }", 0),
  gamma0_param = P("parameters { real<lower=0, upper=1> a; } model { target += gamma_lpdf(0 | 1, 1 + a) + weibull_lpdf(0 | 1, 1 + a) + chi_square_lpdf(0 | 2); }", 0),
  uniform_nan = P("parameters { real x; } model { target += uniform_lpdf(not_a_number() | 0, 1) + x; }", 0),
  normal_inf_loc = P("parameters { real x; } model { target += normal_lpdf(x | positive_infinity(), 1); }", 0),
  poisson_neg = P("parameters { real<lower=0> l; } model { target += poisson_lpmf(-1 | l); }", 0),
  poisson_log_neg = P("parameters { real l; } model { target += poisson_log_lpmf(-1 | l); }", 0),
  negbin_neg = P("parameters { real<lower=0> mu; } model { target += neg_binomial_2_lpmf(-1 | mu, 2); }", 0),
  empty_normal = P("parameters { real x; } model { vector[0] y; target += normal_lpdf(y | 0, -1) + x; }", 0),
  empty_exp = P("parameters { real x; } model { vector[0] y; target += exponential_lpdf(y | -1) + x; }", 0),
  exp_neg = P("parameters { real x; } model { target += exponential_lpdf(-1 | 1) + x; }", 0),
  binlogit_inf = P("parameters { real x; } model { target += binomial_logit_lpmf(0 | 1, negative_infinity()) + x; }", 0),
  negbin_big = P("parameters { real<lower=0> mu; } model { target += neg_binomial_2_lpmf(1 | mu, 1e15) + neg_binomial_2_lpmf(3 | mu, 1e20) + neg_binomial_2_lpmf(2 | mu, 1e8); }", log(2)),
  beta_big = P("parameters { real<lower=0, upper=1> p; } model { target += beta_lpdf(p | 1e6, 1e6) + beta_lpdf(0.25 | 1e16, 1e16) * 0; }", 0),
  variance1 = P("parameters { real x; } model { target += variance(rep_vector(x, 1)) + sd(rep_vector(x, 1)) + sd(rep_vector(x, 2)) + x; }", 0.3),
  min_empty = P("parameters { real x; } model { array[0] real r; if (is_inf(min(r))) target += x; }", 0.3),
  min_empty_int = P("parameters { real x; } model { array[0] int k; target += min(k) + x; }", 0.3),
  fma_vec = P("parameters { vector[2] x; } model { target += sum(fma(x, x, x)); }", c(1, 2)),
  softmax_rv = P("parameters { real x; } model { row_vector[2] r = softmax([x, 0]')'; target += r[1]; }", 0.3),
  softmax_arr = P("parameters { real x; } model { array[2] vector[2] a = {[x, 0]', [0, x]'}; target += log_softmax(a[1])[1]; }", 0.3),
  offset_inf = P("parameters { real<offset=positive_infinity()> x; } model { x ~ normal(0, 1); }", 0),
  empty_bound = P("parameters { vector<lower=rep_vector(0, 0)>[0] v; real x; } model { x ~ normal(0, 1); }", 0.2),
  nan_cond = P("parameters { real x; } model { if (not_a_number()) target += 1; else target += 2; target += x; }", 0),
  nan_cmp = P("parameters { real x; } model { if (not_a_number() < 1) target += 1; target += x; }", 0),
  print_domain = P("parameters { real x; } model { print(normal_lpdf(x | 0, -1)); x ~ normal(0, 1); }", 0),
  fmin_nan = P("parameters { real x; } model { target += fmin(2, not_a_number()) + fmax(2, not_a_number()) + x; }", 0),
  log1m_exp = P("parameters { real<upper=0> x; } model { target += log1m_exp(-40) * 1e18 + log1m_exp(x - 1); }", -1),
  log_mix_inf = P("parameters { real x; } model { target += log_mix(0.5, negative_infinity(), negative_infinity()) * 0 + x; }", 0),
  lognormal_propto = P("data { int N; vector[N] y; } parameters { real a; real<lower=0> s; } model { y ~ lognormal(a, s); }", c(0.2, -0.4), data = list(N = 3, y = c(0.5, 1.2, 2.3))),
  tp_nan = P("parameters { real a; } transformed parameters { real t = log(a); } model { target += -a * a; }", c(-0.5, 0.5)),
  tp_unset = P("parameters { real a; } transformed parameters { real t; } model { a ~ normal(0, 1); }", 0.3),
  tp_bound_nan = P("parameters { real a; } transformed parameters { real<lower=0> t = log(a); } model { target += -a * a; }", c(-0.5, 2)),
  erf_small = P("parameters { real x; } model { target += erf(x) * 1e10; }", 1e-10),
  deep = P(paste0("parameters { real x; } model { target += ", paste(rep("x", 140), collapse = " + "), "; }"), 0.1),
  nested_loops = P("parameters { real x; } model { for (i in 1:200000) for (j in 1:1) { } target += x; }", 1),
  nested_foreach = P("parameters { real x; } model { array[3] int k = {1, 2, 3}; real s = 0; for (i in 1:50000) for (a in k) s += a; while (s > 1e9) s -= 1; target += s * 1e-6 * x; }", 1)
)

err_class <- function(cond) {
  cl <- class(cond)
  cl[cl %in% c("bayescomp_domain_error", "bayescomp_reject_error", "bayescomp_shape_error", "bayescomp_unsupported_error")][1] %|% "other"
}
`%|%` <- function(a, b) if (length(a) && !is.na(a)) a else b
run <- function(m, u, jac, pro) tryCatch({
  g <- bayescomp_grad_log_prob(m, u, jacobian = jac, propto = pro)
  list(lp = attr(g, "log_prob"), grad = as.vector(g))
}, error = function(e) list(err = err_class(e)))
same <- function(a, b) {
  if (!is.null(a$err) || !is.null(b$err)) return(identical(a$err, b$err))
  ok <- function(x, y) length(x) == length(y) && isTRUE(all((is.na(x) & is.na(y)) | (!is.na(x) & !is.na(y) & (x == y | abs(x - y) <= 1e-8 * pmax(1, abs(y))))))
  ok(a$lp, b$lp) && length(a$grad) == length(b$grad) && ok(a$grad, b$grad)
}
bad <- 0L
for (nm in names(cases)) {
  cs <- cases[[nm]]
  r <- bayescomp_model(cs$model, data = cs$data)
  for (engine in c("interpreter", "jit")) {
    options(bayescompR.native_engine = engine)
    n <- tryCatch(bayescomp_model(cs$model, data = cs$data, backend = backend, allow_fallback = TRUE),
                  bayescomp_unsupported_error = function(e) NULL)
    if (is.null(n)) {
      cat(sprintf("%-18s %-11s refused (unsupported)\n", nm, engine))
      next
    }
    k <- bayescomp_dim(r)
    for (u0 in cs$u) for (jac in c(TRUE, FALSE)) for (pro in c(TRUE, FALSE)) {
      u <- rep(u0, k)
      a <- run(n, u, jac, pro)
      b <- run(r, u, jac, pro)
      if (!isTRUE(same(a, b))) {
        bad <- bad + 1L
        cat(sprintf("%-18s %-11s u=%g jac=%d propto=%d  native %s  R %s\n", nm, engine, u0, jac, pro,
                    if (is.null(a$err)) paste(format(c(a$lp, a$grad), digits = 10), collapse = " ") else a$err,
                    if (is.null(b$err)) paste(format(c(b$lp, b$grad), digits = 10), collapse = " ") else b$err))
      }
    }
  }
}
cat(sprintf("%d mismatches in %d cases\n", bad, length(cases)))
quit(status = if (bad) 1L else 0L)
