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
| `crates/bcn-llvm` | LLVM JIT (through inkwell), same design as the Cranelift backend |
| `crates/bcn-ffi` | C ABI linked into the R packages |
| `rpkg/bayescompRCranelift` | R package registering the `"cranelift"` backend (Linux, macOS, Windows) |
| `rpkg/bayescompRLLVM` | R package registering the `"llvm"` backend (Linux, macOS; needs LLVM 18, 20, 21 or 22) |
| `tools/sync-rust.sh` | Copies the crates into both R packages so each builds on its own, and vendors the crates.io dependencies of bayescompRCranelift (`tools/vendor.py`) so it builds offline as CRAN requires |
| `tests/crosscheck.R` | Compares the native engines with the R backend over all Jacobian and `propto` combinations |

## Install

Needs Rust (`rustc` 1.92 or newer, with Cargo) and bayescompR.

```r
remotes::install_github("adamrsophie/bayescompR")
remotes::install_github("adamrsophie/bayescomp-native-rs", subdir = "rpkg/bayescompRCranelift")
# LLVM: install its development files first (for example `brew install llvm@20`
# or `apt install llvm-20-dev libpolly-20-dev`); set LLVM_CONFIG if needed
remotes::install_github("adamrsophie/bayescomp-native-rs", subdir = "rpkg/bayescompRLLVM")
```

The LLVM package does not build on Windows in this version.

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
`cranelift_engine("interpreter")` (or `llvm_engine("interpreter")`)
switches to the reference interpreter. `backend = "llvm"` works the same way.
Programs that use features the native lowering does not cover yet (user
defined functions, truncation, matrix division and a few others) are refused
with `bayescomp_unsupported_error` when the model is built; use the R
backend for them.

## Correctness

Both JITs call the same runtime as the interpreter for every leaf operation,
so all three engines are bit identical; `eval_jit` and `eval_llvm` check
this on every point. All of them
agree with the R backend to a relative difference below 4e-14 on the
cross check models, and raise the same error classes (reject, domain,
shape, unsupported).

## Speed

Gradient of a normal linear regression with N = 5000, on an Apple silicon
laptop (milliseconds per evaluation):

| Model | R backend | Interpreter | Cranelift JIT | LLVM JIT |
|-------|-----------|-------------|---------------|----------|
| Vectorized likelihood | 0.56 | 0.23 | 0.23 | 0.27 |
| Loop over observations | 939 | 2.8 | 2.7 | 3.2 |

Building the LLVM model takes about 0.2 s.

NUTS with one chain of 500 warmup and 500 sampling iterations on the
vectorized model takes 2.7 s with the R backend and 1.6 s with Cranelift;
the rest of the time is the sampler, which runs in R. The JITs currently
compile only control flow, so they are no faster than the interpreter; the
gain over R comes from the native runtime. Fusing scalar arithmetic into
machine code is the next step.

## License

GPL-3.
