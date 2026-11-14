#!/bin/sh
# Copy the shared crates into each R package so it builds on its own.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
for pkg in bayescompRCranelift bayescompRLLVM; do
  dst="$root/rpkg/$pkg/src/rust"
  rm -rf "$dst"
  mkdir -p "$dst/crates"
  for c in bcn-core bcn-cranelift bcn-llvm bcn-ffi; do
    cp -R "$root/crates/$c" "$dst/crates/$c"
    rm -rf "$dst/crates/$c/target" "$dst/crates/$c/examples"
  done
  cp "$root/Cargo.lock" "$dst/Cargo.lock"
  cp "$root/Cargo.toml" "$dst/Cargo.toml"
  printf '\n[profile.release]\nopt-level = 3\ncodegen-units = 1\nlto = "thin"\npanic = "unwind"\n' >> "$dst/Cargo.toml"
done
