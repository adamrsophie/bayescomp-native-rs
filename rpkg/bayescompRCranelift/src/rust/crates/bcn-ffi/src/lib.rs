//! C ABI used by the R packages. A handle owns the lowered program, its
//! runtime and, for the compiled engines, the machine code. Every entry
//! point catches panics. Errors are written to the caller's buffer as
//! `E_CODE: message`, which bayescompR maps to its condition classes.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use bcn_core::{exec::Runtime, lower::lower};

pub const ENGINE_INTERPRETER: c_int = 0;
pub const ENGINE_CRANELIFT: c_int = 1;
pub const ENGINE_LLVM: c_int = 2;

struct Handle {
    rt: Runtime,
    dim: usize,
    #[cfg(feature = "cranelift")]
    jit: Option<bcn_cranelift::Compiled>,
    #[cfg(feature = "llvm")]
    llvm: Option<bcn_llvm::Compiled>,
}

fn write_err(buf: *mut c_char, len: usize, msg: &str) {
    if buf.is_null() || len == 0 {
        return;
    }
    let b = msg.as_bytes();
    let n = b.len().min(len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
}

fn cstr(p: *const c_char) -> Result<String, String> {
    if p.is_null() {
        return Err("E_INTERNAL: null string".into());
    }
    unsafe { CStr::from_ptr(p) }.to_str().map(|s| s.to_owned()).map_err(|_| "E_INTERNAL: invalid UTF-8".into())
}

fn compile(ir: *const c_char, data: *const c_char, dim: usize, engine: c_int) -> Result<Handle, String> {
    let ir: serde_json::Value = serde_json::from_str(&cstr(ir)?).map_err(|e| format!("E_INTERNAL: IR JSON: {e}"))?;
    let data: serde_json::Value = serde_json::from_str(&cstr(data)?).map_err(|e| format!("E_INTERNAL: data JSON: {e}"))?;
    let prog = lower(&ir, &data, dim)?;
    let mut h = Handle {
        rt: Runtime::new(prog),
        dim,
        #[cfg(feature = "cranelift")]
        jit: None,
        #[cfg(feature = "llvm")]
        llvm: None,
    };
    match engine {
        ENGINE_INTERPRETER => {}
        #[cfg(feature = "cranelift")]
        ENGINE_CRANELIFT => h.jit = Some(bcn_cranelift::Compiled::new(&h.rt.prog)?),
        #[cfg(feature = "llvm")]
        ENGINE_LLVM => h.llvm = Some(bcn_llvm::Compiled::new(&h.rt.prog)?),
        _ => return Err(format!("E_UNSUPPORTED: engine {engine} is not built into this library")),
    }
    Ok(h)
}

/// Lower and compile a model. Returns null and fills `err` on failure.
#[no_mangle]
pub extern "C" fn bcn_compile(ir: *const c_char, data: *const c_char, dim: usize, engine: c_int, err: *mut c_char, errlen: usize) -> *mut c_void {
    match catch_unwind(AssertUnwindSafe(|| compile(ir, data, dim, engine))) {
        Ok(Ok(h)) => Box::into_raw(Box::new(h)) as *mut c_void,
        Ok(Err(e)) => {
            write_err(err, errlen, &e);
            std::ptr::null_mut()
        }
        Err(_) => {
            write_err(err, errlen, "E_INTERNAL: panic while compiling the model");
            std::ptr::null_mut()
        }
    }
}

/// Log density and gradient at `u` (length `n`). Returns 0 on success.
#[no_mangle]
pub extern "C" fn bcn_eval(h: *mut c_void, u: *const f64, n: usize, jacobian: c_int, propto: c_int, lp: *mut f64, grad: *mut f64, err: *mut c_char, errlen: usize) -> c_int {
    if h.is_null() {
        write_err(err, errlen, "E_INTERNAL: the native model was released");
        return 1;
    }
    let h = unsafe { &mut *(h as *mut Handle) };
    if n != h.dim {
        write_err(err, errlen, &format!("E_INTERNAL: expected {} unconstrained parameters, got {n}", h.dim));
        return 1;
    }
    let u = unsafe { std::slice::from_raw_parts(u, n) };
    let g = unsafe { std::slice::from_raw_parts_mut(grad, n) };
    let r = catch_unwind(AssertUnwindSafe(|| {
        #[cfg(feature = "cranelift")]
        if let Some(j) = &h.jit {
            return j.eval(&mut h.rt, u, jacobian != 0, propto != 0, Some(g));
        }
        #[cfg(feature = "llvm")]
        if let Some(j) = &h.llvm {
            return j.eval(&mut h.rt, u, jacobian != 0, propto != 0, Some(g));
        }
        h.rt.interpret(u, jacobian != 0, propto != 0, Some(g))
    }));
    match r {
        Ok(Ok(v)) => {
            unsafe { *lp = v };
            0
        }
        Ok(Err(e)) => {
            write_err(err, errlen, &e);
            1
        }
        Err(_) => {
            write_err(err, errlen, "E_INTERNAL: panic in the native runtime");
            1
        }
    }
}

/// Release a handle (null is allowed).
#[no_mangle]
pub extern "C" fn bcn_free(h: *mut c_void) {
    if !h.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(unsafe { Box::from_raw(h as *mut Handle) })));
    }
}

// Entry points with the calling convention of R's `.C()`: every argument is
// a pointer and nothing is returned. bayescompR loads the shared library
// built from this crate with `dyn.load()` and calls them without any
// compiled code of its own, so the library does not link R. Models are kept
// in a table and named by integer ids; each id is tied to the process that
// created it, so a copy of a model in another process recompiles.

use std::collections::HashMap;
use std::sync::Mutex;

/// Version of the `.C` interface; bayescompR refuses other versions.
pub const RC_ABI_VERSION: c_int = 1;

struct Table {
    next: c_int,
    models: HashMap<c_int, Handle>,
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);

fn with_table<T>(f: impl FnOnce(&mut Table) -> T) -> T {
    let mut g = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let t = g.get_or_insert_with(|| Table { next: 1, models: HashMap::new() });
    f(t)
}

/// `.C("bcn_rc_info", abi = integer(1), engines = integer(1))`: the
/// interface version and a bit mask of the engines built in (1 interpreter,
/// 2 cranelift, 4 llvm).
#[no_mangle]
pub extern "C" fn bcn_rc_info(abi: *mut c_int, engines: *mut c_int) {
    let mut m = 1;
    #[cfg(feature = "cranelift")]
    {
        m |= 2;
    }
    #[cfg(feature = "llvm")]
    {
        m |= 4;
    }
    unsafe {
        *abi = RC_ABI_VERSION;
        *engines = m;
    }
}

/// Compile a model; `id` receives a positive id, or 0 with the message in
/// `err` (a preallocated string of `errlen` bytes).
#[no_mangle]
pub extern "C" fn bcn_rc_compile(ir: *const *const c_char, data: *const *const c_char, dim: *const c_int, engine: *const c_int,
                                 id: *mut c_int, err: *const *mut c_char, errlen: *const c_int) {
    let (ir, data, dim, engine, err, errlen) = unsafe { (*ir, *data, *dim as usize, *engine, *err, *errlen as usize) };
    let r = match catch_unwind(AssertUnwindSafe(|| compile(ir, data, dim, engine))) {
        Ok(r) => r,
        Err(_) => Err("E_INTERNAL: panic while compiling the model".into()),
    };
    let out = match r {
        Ok(h) => with_table(|t| {
            let k = t.next;
            t.next += 1;
            t.models.insert(k, h);
            k
        }),
        Err(e) => {
            write_err(err, errlen, &e);
            0
        }
    };
    unsafe { *id = out };
}

/// Log density and gradient. `status` is 0 on success, 2 when the id is
/// unknown in this process, and 1 for an error described in `err`.
#[no_mangle]
pub extern "C" fn bcn_rc_eval(id: *const c_int, u: *const f64, n: *const c_int, jacobian: *const c_int, propto: *const c_int,
                              lp: *mut f64, grad: *mut f64, status: *mut c_int, err: *const *mut c_char, errlen: *const c_int) {
    let (id, n, jacobian, propto, err, errlen) = unsafe { (*id, *n as usize, *jacobian, *propto, *err, *errlen as usize) };
    let mut g = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    let h = match g.as_mut().and_then(|t| t.models.get_mut(&id)) {
        Some(h) => h as *mut Handle as *mut c_void,
        None => {
            unsafe { *status = 2 };
            return;
        }
    };
    let st = bcn_eval(h, u, n, jacobian, propto, lp, grad, err, errlen);
    unsafe { *status = st };
}

/// Release a model (unknown ids are ignored).
#[no_mangle]
pub extern "C" fn bcn_rc_free(id: *const c_int) {
    let id = unsafe { *id };
    let h = with_table(|t| t.models.remove(&id));
    let _ = catch_unwind(AssertUnwindSafe(|| drop(h)));
}
