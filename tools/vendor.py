"""Vendor the crates.io dependencies of an R package's Rust workspace.

CRAN builds packages offline, so every dependency ships in
src/rust/vendor.tar.xz. Import libraries for targets CRAN does not build on,
tests and unused Windows API bindings are removed to keep the tarball small.
The archive is deterministic so CI can check that it is up to date.
"""
import io, lzma, os, shutil, subprocess, sys, tarfile, tempfile

rust = os.path.abspath(sys.argv[1])
tmp = tempfile.mkdtemp()
vendor = os.path.join(tmp, "vendor")
r = subprocess.run(["cargo", "vendor", "--locked", "--versioned-dirs", "--manifest-path",
                    os.path.join(rust, "Cargo.toml"), vendor], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
if r.returncode:
    sys.exit(r.stderr.decode())
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "trim_vendor.py"), vendor], check=True)

# inst/AUTHORS: authors and licenses of every vendored crate, as CRAN asks
import tomllib
lines = ["The package bundles the following Rust crates in src/rust/vendor.tar.xz.",
         "Their authors hold the copyright; each crate is under the license shown.", ""]
for d in sorted(os.listdir(vendor)):
    meta = tomllib.load(open(os.path.join(vendor, d, "Cargo.toml"), "rb")).get("package", {})
    authors = ", ".join(meta.get("authors", [])) or "the " + meta["name"] + " developers"
    lines.append("%s %s (%s): %s" % (meta["name"], meta["version"], meta.get("license", "see the crate"), authors))
inst = os.path.join(rust, "..", "..", "inst")
os.makedirs(inst, exist_ok=True)
open(os.path.join(inst, "AUTHORS"), "w").write("\n".join(lines) + "\n")

buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tar:
    for dirpath, dirnames, filenames in os.walk(vendor):
        dirnames.sort()
        for name in sorted(dirnames) + sorted(filenames):
            path = os.path.join(dirpath, name)
            info = tar.gettarinfo(path, os.path.relpath(path, tmp))
            info.mtime = 0; info.uid = info.gid = 0; info.uname = info.gname = ""
            info.mode = 0o755 if info.isdir() else 0o644
            if info.isfile():
                with open(path, "rb") as f: tar.addfile(info, f)
            else:
                tar.addfile(info)
with open(os.path.join(rust, "vendor.tar.xz"), "wb") as out:
    out.write(lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME))
shutil.rmtree(tmp)
