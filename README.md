# bayescomp-native-rs

Native code backends for [bayescompR](https://github.com/adamrsophie/bayescompR).
bayescompR itself is pure R; the shared library built here evaluates the
log density and its gradient of a model in compiled code. It reads the
typed, versioned intermediate representation (IR 1.0.0) that bayescompR
emits, so the two implementations are never mixed silently: a model uses a
native backend only when it is built with `backend = "cranelift"` or
`"llvm"`, and its `provenance` records which backend computed each
capability.

## Layout

| Path | Contents |
|------|----------|
| `crates/bcn-core` | IR lowering, reverse mode AD tape, math and distribution kernels, reference interpreter |
| `crates/bcn-cranelift` | Cranelift JIT: compiles the statement tree to one machine code function |
| `crates/bcn-llvm` | LLVM JIT (through inkwell), same design as the Cranelift backend |
| `crates/bcn-ffi` | the shared library: entry points with the calling convention of R's `.C()`, so bayescompR needs no compiled code of its own |
| `.github/workflows/release.yml` | builds the library for each platform and attaches it, with checksums, to a release |
| `tests/crosscheck.R` | Compares the native engines with the R backend over all Jacobian and `propto` combinations |

## Install

bayescompR downloads a prebuilt library from the releases of this
repository:

```r
library(bayescompR)
bayescomp_install_backend("cranelift")   # or "llvm"
```

To build the library yourself (Rust 1.92 or newer; for LLVM, LLVM 18, 20,
21 or 22 and the matching `llvm18` to `llvm22` feature):

```sh
cargo build --release -p bcn-ffi --features cranelift
```

```r
bayescomp_install_backend("cranelift", path = "target/release/libbcn_ffi.so")  # .dylib on macOS, .dll on Windows
```

## Use

```r
m <- bayescomp_model(code, data = data, backend = "cranelift", allow_fallback = TRUE)
m$provenance                      # log density and gradient: cranelift; the rest: r
bayescomp_validate_backend(m)     # compare with the R backend at random points
fit <- bayescomp_sample(m)
```

`allow_fallback = TRUE` is required because constraining, generated
quantities and random numbers still run in R in this version.
`options(bayescompR.native_engine = "interpreter")` switches to the
reference interpreter. Programs that use features the native lowering does
not cover yet (user defined functions, truncation, matrix division and a
few others) are refused with `bayescomp_unsupported_error` when the model
is built; use the R backend for them.

## Correctness

Both JITs call the same runtime as the interpreter for every leaf operation,
so all three engines are bit identical; `eval_jit` and `eval_llvm` check
this on every point. Two test scripts compare them with the R backend of
bayescompR, value, gradient and error class (reject, domain, shape,
unsupported), under all four Jacobian and `propto` combinations:
`tests/crosscheck.R` on six models covering the supported language, where
the relative difference stays below 4e-14, and `tests/parity.R` on edge
cases (special functions at poles and infinities, distribution domains and
boundaries, integer overflow, `propto` with arguments that are constant at
run time, transformed parameter checks, loops). These are tests of the
cases they cover; `bayescomp_validate_backend()` checks a model of your own.

The native backend cannot be interrupted while it evaluates: a Stan `while`
loop that never ends blocks the R session. Programs outside the supported
subset are refused with `bayescomp_unsupported_error` when the model is
built; such a program runs with the default R backend.

Prebuilt libraries: the Cranelift libraries need glibc 2.28 or newer on
Linux and macOS 11 or newer. The LLVM libraries link LLVM statically; on
Linux they need glibc 2.35 or newer, libstdc++, zlib, zstd, libffi and
libxml2 (`libxml2.so.2`), and on macOS they are built and tested on macOS 15
(Apple silicon).

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
