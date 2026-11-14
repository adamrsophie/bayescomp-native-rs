# Cross-check the native interpreter against the R backend of bayescompR.
# Usage: Rscript tests/crosscheck.R OUTDIR PATH/TO/eval
suppressMessages(library(bayescompR))
model_lpg <- bayescompR:::model_lpg
S <- commandArgs(TRUE)[1]
dir.create(S, recursive = TRUE, showWarnings = FALSE)
bin <- commandArgs(TRUE)[2]
models <- list(
  lin = list(code = "data { int N; vector[N] x; vector[N] y; } parameters { real a; real b; real<lower=0> s; } model { a ~ normal(0, 5); b ~ student_t(3, 0, 2); s ~ exponential(1); y ~ normal(a + b * x, s); }",
             data = list(N = 5, x = c(1, 2, 3, 4, 5), y = c(1.1, 2.3, 2.9, 4.2, 5.1))),
  ex = list(code = "data { int N; array[N] int k; vector[N] x; } parameters { real a; vector<lower=0>[2] s; } transformed parameters { real t = a * 2; } model { vector[N] eta; for (n in 1:N) { eta[n] = a + x[n]; } if (a > 0) target += -a; else a ~ normal(0, 1); k ~ poisson_log(eta); target += log_sum_exp(eta[1:2]) + normal_lpdf(a | 0, s[1]); s ~ exponential(1); }",
            data = list(N = 2, k = c(1L, 2L), x = c(1, 2))),
  glm = list(code = "data { int N; array[N] int y; array[N] int n; matrix[N, 2] X; } parameters { vector[2] beta; real<lower=0,upper=1> p; real<offset=1,multiplier=2> z; } model { beta ~ std_normal(); p ~ beta(2, 3); z ~ logistic(1, 2); y ~ binomial_logit(n, X * beta); y ~ binomial(n, p); }",
             data = list(N = 3, y = c(1L, 3L, 4L), n = c(5L, 5L, 6L), X = matrix(c(1, 1, 1, -1, 0, 2), 3))),
  hier = list(code = "data { int J; array[J] real y; array[J] real<lower=0> sigma; } parameters { real mu; real<lower=0> tau; array[J] real theta_raw; } transformed parameters { array[J] real theta; for (j in 1:J) theta[j] = mu + tau * theta_raw[j]; } model { mu ~ normal(0, 5); tau ~ cauchy(0, 5); theta_raw ~ std_normal(); y ~ normal(theta, sigma); }",
              data = list(J = 3, y = c(28, 8, -3), sigma = c(15, 10, 16))),
  misc = list(code = "data { int N; vector[N] y; } parameters { real<lower=-1,upper=2> r; real<upper=3> q; vector[N] v; real<lower=0> g; } model { matrix[2,2] M = [[1, r], [q, 2]]; row_vector[2] rv = [r, q]; vector[2] w = M * v[1:2]; target += -0.5 * dot_self(w) + sum(log1p_exp(v)) + log_mix(0.3, normal_lpdf(r | 0, 1), normal_lpdf(r | 1, 1)); target += rv * w; g ~ gamma(2, 1); y ~ lognormal(r, g); q ~ double_exponential(0, 1); v ~ weibull(2, 1.5) ; target += fmax(r, q) + inv_logit(q) + (r > 0 ? r^2 : -r); for (i in 1:N) { if (i > 2) break; target += pow(g, 0.5) * i; } y ~ gamma(g, 1); y ~ inv_gamma(2, g); }",
              data = list(N = 3, y = c(0.5, 1.2, 2.0))),
  ctl = list(code = "data { array[2, 3] real A; array[3] int ks; } parameters { array[2] vector<lower=-2>[3] V; matrix<lower=0>[2, 2] M; array[2] vector<lower=[-1, -2, -3]', upper=5>[3] W; } transformed parameters { real<upper=1> tp = V[1][1]; } model { int i = 0; real acc = 0; while (i < 5) { i += 1; if (i == 2) continue; acc += i * 0.1; } for (x in ks) acc += x; for (r in 1:2) for (c in 1:3) target += A[r, c] * V[r][c]; vector[3] z = rep_vector(0, 3); z[2:3] = V[2][1:2]; z[{1}] = [M[1, 2]]'; target += -0.5 * dot_self(z) - acc * 0.01; target += -sum(M); M[1] ~ normal(0, 1); target += target() * 0; int q = -7 / 2; int rr = -7 % 3; target += q + rr; for (k in 1:2) W[k] ~ normal(0, 2); if (V[1][2] > 3) reject(\"V too big: \", V[1][2]); }",
             data = list(A = matrix(c(1, 2, 3, 4, 5, 6), 2, byrow = TRUE), ks = c(1L, 2L, 3L)))
)
model_err_class <- function(cond) {
  cl <- class(cond)
  cl[cl %in% c("bayescomp_domain_error", "bayescomp_reject_error", "bayescomp_shape_error", "bayescomp_unsupported_error")][1]
}
code_class <- c(E_DOMAIN = "bayescomp_domain_error", E_REJECT = "bayescomp_reject_error", E_FATAL = "bayescomp_reject_error",
                E_SHAPE = "bayescomp_shape_error", E_UNSUPPORTED = "bayescomp_unsupported_error")
bad <- FALSE
set.seed(3)
for (nm in names(models)) {
  mm <- models[[nm]]
  m <- bayescomp_model(mm$code, data = mm$data)
  inp <- bayescomp_native_inputs(m)
  f_ir <- file.path(S, paste0(nm, "_ir.json")); writeLines(inp$ir, f_ir)
  f_d <- file.path(S, paste0(nm, "_d.json")); writeLines(inp$data, f_d)
  U <- matrix(runif(5 * m$dim, -1.5, 1.5), 5, m$dim)
  if (nm == "misc") U[, 3:5] <- abs(U[, 3:5]) + 0.2
  if (nm == "ctl") { U <- rbind(U, U); U[6:10, 1] <- 2.5; U[8:10, 2] <- 4 }
  f_u <- file.path(S, paste0(nm, "_u.txt")); write.table(U, f_u, row.names = FALSE, col.names = FALSE)
  out <- system2(bin, c(f_ir, f_d, m$dim, f_u), stdout = TRUE)
  if (startsWith(out[1], "LOWER")) { bad <- TRUE; cat(nm, out[1], "\n"); next }
  combos <- list(c(TRUE, FALSE), c(FALSE, FALSE), c(TRUE, TRUE), c(FALSE, TRUE))
  worst <- 0; k <- 0; nerr <- 0
  for (i in seq_len(nrow(U))) for (cb in combos) {
    k <- k + 1
    r <- tryCatch(model_lpg(m, cb[1], cb[2])(U[i, ]), error = function(e) list(err = conditionMessage(e), cls = model_err_class(e)))
    o <- out[k]
    if (startsWith(o, "ERR")) {
      code <- sub("^ERR ([A-Z_]+):.*$", "\\1", o)
      if (is.null(r$err) || !identical(unname(code_class[code]), r$cls)) { bad <- TRUE; cat(nm, "mismatch:", o, "| R:", if (!is.null(r$err)) paste(r$cls, r$err) else r$value, "\n") }
      else nerr <- nerr + 1
      next
    }
    if (!is.null(r$err)) { bad <- TRUE; cat(nm, "R error only:", r$err, "\n"); next }
    v <- as.numeric(strsplit(o, " ")[[1]])
    d <- max(abs(v[1] - r$value) / max(1, abs(r$value)), abs(v[-1] - r$grad) / pmax(1, abs(r$grad)))
    worst <- max(worst, d)
  }
  if (worst > 1e-10) bad <- TRUE
  cat(sprintf("%-5s dim=%d worst rel diff %.3g, matching errors %d\n", nm, m$dim, worst, nerr))
}
if (bad) quit(status = 1)
