#!/bin/sh
# Copy the shared crates into the package so it builds on its own.
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
root=$(cd "$here/../.." && pwd)
dst="$here/src/rust"
rm -rf "$dst"
mkdir -p "$dst/crates"
for c in bcn-core bcn-cranelift bcn-ffi; do
  cp -R "$root/crates/$c" "$dst/crates/$c"
  rm -rf "$dst/crates/$c/target"
done
cp "$root/Cargo.lock" "$dst/Cargo.lock"
sed -e 's|members = \["crates/\*"\]|members = ["crates/bcn-core", "crates/bcn-cranelift", "crates/bcn-ffi"]|' "$root/Cargo.toml" > "$dst/Cargo.toml"
printf '\n[profile.release]\nopt-level = 3\ncodegen-units = 1\nlto = "thin"\npanic = "unwind"\n' >> "$dst/Cargo.toml"
