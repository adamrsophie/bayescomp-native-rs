# bayescomp-native-rs

Native code backends for [bayescompR](https://github.com/adamrsophie/bayescompR).
bayescompR itself is pure R; these optional packages evaluate the log density
and its gradient of a model in compiled code. They read the typed, versioned
intermediate representation (IR 1.0.0) that bayescompR emits, so the R
package stays free of native dependencies and the two implementations are
never mixed silently: a model uses a native backend only when it is built
with `backend = "cranelift"`, and its `provenance` records which backend
computed each capability.

## Layout

| Path | Contents |
|------|----------|
| `crates/bcn-core` | IR lowering, reverse mode AD tape, math and distribution kernels, reference interpreter |
| `crates/bcn-cranelift` | Cranelift JIT: compiles the statement tree to one machine code function |
| `crates/bcn-ffi` | C ABI linked into the R packages |
| `rpkg/bayescompRCranelift` | R package registering the `"cranelift"` backend |
| `tests/crosscheck.R` | Compares the native engines with the R backend over all Jacobian and `propto` combinations |

The LLVM backend (`bayescompRLLVM`) will share `bcn-core` and `bcn-ffi`; it
is not released yet.

## Install

Needs Rust (`rustc` 1.85 or newer, with Cargo) and bayescompR.

```r
remotes::install_github("adamrsophie/bayescompR")
remotes::install_github("adamrsophie/bayescomp-native-rs", subdir = "rpkg/bayescompRCranelift")
```

## Use

```r
library(bayescompR)
library(bayescompRCranelift)
m <- bayescomp_model(code, data = data, backend = "cranelift", allow_fallback = TRUE)
m$provenance                      # log density and gradient: cranelift; the rest: r
bayescomp_validate_backend(m)     # compare with the R backend at random points
fit <- bayescomp_sample(m)
```

`allow_fallback = TRUE` is required because constraining, generated
quantities and random numbers still run in R in this version.
`cranelift_engine("interpreter")` switches to the reference interpreter.
Programs that use features the native lowering does not cover yet (user
defined functions, truncation, matrix division and a few others) are refused
with `bayescomp_unsupported_error` when the model is built; use the R
backend for them.

## Correctness

The JIT calls the same runtime as the interpreter for every leaf operation,
so the two are bit identical; `eval_jit` checks this on every point. Both
agree with the R backend to a relative difference below 4e-14 on the
cross check models, and raise the same error classes (reject, domain,
shape, unsupported).

## Speed

Gradient of a normal linear regression with N = 5000, on an Apple silicon
laptop (milliseconds per evaluation):

| Model | R backend | Interpreter | Cranelift JIT |
|-------|-----------|-------------|---------------|
| Vectorized likelihood | 0.56 | 0.23 | 0.23 |
| Loop over observations | 939 | 2.8 | 2.7 |

NUTS with one chain of 500 warmup and 500 sampling iterations on the
vectorized model takes 2.7 s with the R backend and 1.6 s with Cranelift;
the rest of the time is the sampler, which runs in R. The JIT currently
compiles only control flow, so it is no faster than the interpreter; the
gain over R comes from the native runtime. Fusing scalar arithmetic into
machine code is the next step.

## License

GPL-3.
