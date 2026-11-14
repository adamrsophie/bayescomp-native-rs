//! C ABI used by the R packages. A handle owns the lowered program, its
//! runtime and, for the compiled engines, the machine code. Every entry
//! point catches panics. Errors are written to the caller's buffer as
//! `E_CODE: message`, which bayescompR maps to its condition classes.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use bcn_core::{exec::Runtime, lower::lower};

pub const ENGINE_INTERPRETER: c_int = 0;
pub const ENGINE_CRANELIFT: c_int = 1;

struct Handle {
    rt: Runtime,
    dim: usize,
    #[cfg(feature = "cranelift")]
    jit: Option<bcn_cranelift::Compiled>,
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
    match engine {
        ENGINE_INTERPRETER => Ok(Handle {
            rt: Runtime::new(prog),
            dim,
            #[cfg(feature = "cranelift")]
            jit: None,
        }),
        #[cfg(feature = "cranelift")]
        ENGINE_CRANELIFT => {
            let jit = bcn_cranelift::Compiled::new(&prog)?;
            Ok(Handle { rt: Runtime::new(prog), dim, jit: Some(jit) })
        }
        _ => Err(format!("E_UNSUPPORTED: engine {engine} is not built into this library")),
    }
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
