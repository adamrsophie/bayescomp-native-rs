## Submission

This is the first submission of bayescompRCranelift. It depends on
bayescompR, which is submitted first.

## Rust

* The package follows the CRAN policy for packages using Rust: all crates.io
  dependencies are vendored in `src/rust/vendor.tar.xz` and the build runs
  `cargo build --offline --locked -j 2`, with `CARGO_HOME` inside the build
  directory, so nothing is downloaded or written to the user's home.
* The versions of `rustc` and `cargo` are printed in the installation log;
  the minimum supported version is rustc 1.85.
* The authors and licenses of the vendored crates are listed in
  `inst/AUTHORS`; they are MIT, Apache-2.0 (some with the LLVM exception),
  BSD-2-Clause, Unlicense or Zlib, all compatible with GPL-3.
* Import libraries for Windows targets CRAN does not build on, the crates'
  tests and unused Windows API bindings are removed from the vendored copy to
  keep the tarball at about 3.7 MB.

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission.
