import json, os, sys, shutil, glob
# trims a vendored crate directory in place (see vendor.py)
v = sys.argv[1]
for d in sorted(os.listdir(v)):
    p = os.path.join(v, d)
    ck = os.path.join(p, '.cargo-checksum.json')
    if not os.path.exists(ck): continue
    # import libraries for targets CRAN does not build on
    if d.startswith(('windows_i686', 'windows_aarch64', 'windows_x86_64_msvc', 'windows_x86_64_gnullvm')):
        shutil.rmtree(os.path.join(p, 'lib'), ignore_errors=True)
    for sub in ('tests', 'benches', 'examples', 'docs', 'doc'):
        shutil.rmtree(os.path.join(p, sub), ignore_errors=True)
    j = json.load(open(ck)); j['files'] = {}
    json.dump(j, open(ck, 'w'))

# windows-sys ships bindings for every Windows API; keep the modules of the enabled features
KEEP = {'Win32', 'Win32/Foundation', 'Win32/System', 'Win32/System/Diagnostics',
        'Win32/System/Diagnostics/Debug', 'Win32/System/Memory', 'Win32/System/SystemInformation',
        'Win32/System/LibraryLoader', 'Win32/System/Threading'}
for p in glob.glob(os.path.join(v, 'windows-sys-*')):
    root = os.path.join(p, 'src', 'Windows')
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        rel = os.path.relpath(dirpath, root)
        for dn in list(dirnames):
            r = dn if rel == '.' else rel + '/' + dn
            if r not in KEEP:
                shutil.rmtree(os.path.join(dirpath, dn)); dirnames.remove(dn)
