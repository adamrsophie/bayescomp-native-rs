//! C ABI loaded by bayescompR with dyn.load(). A handle owns the lowered program, its
//! runtime and, for the compiled engines, the machine code. The entry
//! points validate their pointers and lengths and catch panics; R must still
//! pass buffers of the stated lengths. Errors are written to the caller's buffer as
//! `E_CODE: message`, which bayescompR maps to its condition classes.

use std::ffi::{c_char, c_int, CStr};
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
    let mut n = b.len().min(len - 1);
    // cut at a character boundary, so R receives valid UTF-8
    while n > 0 && !msg.is_char_boundary(n) {
        n -= 1;
    }
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
    // a program nested deeper than serde_json's recursion limit falls back to the R backend
    let ir: serde_json::Value = serde_json::from_str(&cstr(ir)?).map_err(|e| {
        if e.to_string().contains("recursion limit") {
            "E_UNSUPPORTED: expressions nested too deeply for the native backend".to_string()
        } else {
            format!("E_INTERNAL: IR JSON: {e}")
        }
    })?;
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

/// Log density and gradient of a model at `u` (length `n`, as `grad`).
/// Returns 0 on success and 1 with the message in `err`.
fn eval_handle(h: &mut Handle, u: *const f64, n: usize, jacobian: c_int, propto: c_int, lp: *mut f64, grad: *mut f64, err: *mut c_char, errlen: usize) -> c_int {
    if n != h.dim {
        write_err(err, errlen, &format!("E_INTERNAL: expected {} unconstrained parameters, got {n}", h.dim));
        return 1;
    }
    let (u, g): (&[f64], &mut [f64]) = if n == 0 {
        (&[], &mut [])
    } else {
        unsafe { (std::slice::from_raw_parts(u, n), std::slice::from_raw_parts_mut(grad, n)) }
    };
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

// Entry points with the calling convention of R's `.C()`: every argument is
// a pointer and nothing is returned. bayescompR loads the shared library
// built from this crate with `dyn.load()` and calls them without any
// compiled code of its own, so the library does not link R.
//
// Models live in a table owned by the calling thread (R calls from its main
// thread only), so there is no lock to inherit across fork() and the JIT
// objects never move between threads. A model is named by an id and the
// generation of the loaded library: a library loaded again gets a new
// generation, and ids of the old one are reported as unknown.

use std::cell::RefCell;
use std::collections::HashMap;

/// Version of the `.C` interface; bayescompR refuses other versions.
pub const RC_ABI_VERSION: c_int = 2;

struct Table {
    next: c_int,
    models: HashMap<c_int, Handle>,
}

thread_local! {
    static TABLE: RefCell<Table> = RefCell::new(Table { next: 1, models: HashMap::new() });
}

fn generation() -> c_int {
    use std::sync::OnceLock;
    static GEN: OnceLock<c_int> = OnceLock::new();
    *GEN.get_or_init(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let a = &GEN as *const _ as usize as u128;
        let x = (t ^ (a << 7) ^ (std::process::id() as u128) << 32) as u64;
        // positive and non zero
        ((x ^ (x >> 31)) as u32 & 0x3fff_ffff) as c_int + 1
    })
}

/// Read a non negative length; None for a null pointer, NA or a negative value.
fn len_arg(p: *const c_int) -> Option<usize> {
    if p.is_null() {
        return None;
    }
    let v = unsafe { *p };
    if v < 0 { None } else { Some(v as usize) }
}

fn err_buf(err: *const *mut c_char, errlen: *const c_int) -> (*mut c_char, usize) {
    match (err.is_null(), len_arg(errlen)) {
        (false, Some(n)) => (unsafe { *err }, n),
        _ => (std::ptr::null_mut(), 0),
    }
}

/// `.C("bcn_rc_info", abi = integer(1), engines = integer(1), generation = integer(1))`:
/// the interface version, a bit mask of the engines built in (1
/// interpreter, 2 cranelift, 4 llvm) and the generation of this load of the
/// library.
///
/// This signature is frozen: it is called before the interface version is
/// known, so every future version keeps these three `int *` arguments.
#[no_mangle]
pub extern "C" fn bcn_rc_info(abi: *mut c_int, engines: *mut c_int, gen: *mut c_int) {
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
        if !abi.is_null() {
            *abi = RC_ABI_VERSION;
        }
        if !engines.is_null() {
            *engines = m;
        }
        if !gen.is_null() {
            *gen = generation();
        }
    }
}

/// Compile a model; `id` receives a positive id, or 0 with the message in
/// `err` (a preallocated string of `errlen` bytes including the final NUL).
#[no_mangle]
pub extern "C" fn bcn_rc_compile(ir: *const *const c_char, data: *const *const c_char, dim: *const c_int, engine: *const c_int,
                                 id: *mut c_int, err: *const *mut c_char, errlen: *const c_int) {
    if id.is_null() {
        return;
    }
    unsafe { *id = 0 };
    let (err, errlen) = err_buf(err, errlen);
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<c_int, String> {
        if ir.is_null() || data.is_null() || engine.is_null() {
            return Err("E_INTERNAL: null argument".into());
        }
        let dim = len_arg(dim).ok_or("E_INTERNAL: invalid dimension")?;
        let h = compile(unsafe { *ir }, unsafe { *data }, dim, unsafe { *engine })?;
        TABLE.with(|t| {
            let mut t = t.borrow_mut();
            let k = t.next;
            t.next = k.checked_add(1).ok_or("E_INTERNAL: too many native models in this session")?;
            t.models.insert(k, h);
            Ok(k)
        })
    }));
    match r {
        Ok(Ok(k)) => unsafe { *id = k },
        Ok(Err(e)) => write_err(err, errlen, &e),
        Err(_) => write_err(err, errlen, "E_INTERNAL: panic while compiling the model"),
    }
}

/// Log density and gradient. `status` is 0 on success, 2 when the id or the
/// generation is unknown in this process, and 1 for an error described in
/// `err`. `u` and `grad` must hold `n` values.
#[no_mangle]
pub extern "C" fn bcn_rc_eval(gen: *const c_int, id: *const c_int, u: *const f64, n: *const c_int, jacobian: *const c_int,
                              propto: *const c_int, lp: *mut f64, grad: *mut f64, status: *mut c_int,
                              err: *const *mut c_char, errlen: *const c_int) {
    if status.is_null() {
        return;
    }
    let (err, errlen) = err_buf(err, errlen);
    let n = match len_arg(n) {
        Some(n) if !gen.is_null() && !id.is_null() && !jacobian.is_null() && !propto.is_null() && !lp.is_null()
            && (n == 0 || (!u.is_null() && !grad.is_null())) => n,
        _ => {
            write_err(err, errlen, "E_INTERNAL: invalid argument");
            unsafe { *status = 1 };
            return;
        }
    };
    let (g, id, jacobian, propto) = unsafe { (*gen, *id, *jacobian, *propto) };
    if g != generation() {
        unsafe { *status = 2 };
        return;
    }
    let st = TABLE
        .try_with(|t| {
            let mut t = t.borrow_mut();
            match t.models.get_mut(&id) {
                Some(h) => eval_handle(h, u, n, jacobian, propto, lp, grad, err, errlen),
                None => 2,
            }
        })
        .unwrap_or(2);
    unsafe { *status = st };
}

/// Release a model (unknown ids and other generations are ignored).
#[no_mangle]
pub extern "C" fn bcn_rc_free(gen: *const c_int, id: *const c_int) {
    if gen.is_null() || id.is_null() || unsafe { *gen } != generation() {
        return;
    }
    let id = unsafe { *id };
    // try_with: a panic here (thread local already destroyed) would abort the process
    let h = TABLE.try_with(|t| t.borrow_mut().models.remove(&id)).ok().flatten();
    let _ = catch_unwind(AssertUnwindSafe(|| drop(h)));
}
