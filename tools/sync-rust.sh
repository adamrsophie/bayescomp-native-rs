#!/bin/sh
# Copy the shared crates into each R package so it builds on its own.
# bayescompRCranelift (the CRAN package) gets only the crates it links and a
# vendored copy of its crates.io dependencies, so it builds offline.
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
profile='\n[profile.release]\nopt-level = 3\ncodegen-units = 1\nlto = "thin"\npanic = "unwind"\n'

copy() { # package, crates...
  pkg=$1; shift
  dst="$root/rpkg/$pkg/src/rust"
  rm -rf "$dst"
  mkdir -p "$dst/crates"
  for c in "$@"; do
    cp -R "$root/crates/$c" "$dst/crates/$c"
    rm -rf "$dst/crates/$c/target" "$dst/crates/$c/examples"
  done
  # the R packages link the static library; the shared one is for bayescompR's downloads
  if [ -f "$dst/crates/bcn-ffi/Cargo.toml" ]; then
    python3 - "$dst/crates/bcn-ffi/Cargo.toml" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
open(p, "w").write(s.replace('"staticlib", "cdylib", "rlib"', '"staticlib", "rlib"'))
PY
  fi
  cp "$root/Cargo.lock" "$dst/Cargo.lock"
  cp "$root/Cargo.toml" "$dst/Cargo.toml"
  printf "$profile" >> "$dst/Cargo.toml"
}

copy bayescompRLLVM bcn-core bcn-cranelift bcn-llvm bcn-ffi

copy bayescompRCranelift bcn-core bcn-cranelift bcn-ffi
dst="$root/rpkg/bayescompRCranelift/src/rust"
# drop the LLVM backend from the C ABI crate; the llvm feature stays, empty
python3 - "$dst/crates/bcn-ffi/Cargo.toml" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'llvm[0-9]+ = \[.*\]\n', '', s)
s = s.replace('llvm = ["dep:bcn-llvm"]', 'llvm = []')
s = s.replace('bcn-llvm = { path = "../bcn-llvm", optional = true }\n', '')
open(p, "w").write(s)
PY
# prune the lockfile to the remaining crates without changing any version
cargo metadata --format-version 1 --manifest-path "$dst/Cargo.toml" > /dev/null
python3 "$root/tools/vendor.py" "$dst"
