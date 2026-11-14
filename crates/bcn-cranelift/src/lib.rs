//! Cranelift backend: the statement tree of a lowered program (loops,
//! branches, `break`, `continue`) is compiled to one machine code function
//! `fn(*mut Runtime) -> i32`. Leaf operations call the shared runtime of
//! `bcn-core`, so values, gradients and error messages are bit identical to
//! the reference interpreter. Runtime entry points are embedded as constant
//! addresses and catch panics; a nonzero status means `Runtime::error` holds
//! the message.

use std::panic::{catch_unwind, AssertUnwindSafe};

use bcn_core::exec::Runtime;
use bcn_core::kernel::{Program, Stmt};
use cranelift_codegen::ir::{
    condcodes::IntCC, types, AbiParam, Block, InstBuilder, SigRef, Signature, StackSlotData, StackSlotKind,
    UserFuncName, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

type R<T> = Result<T, String>;

fn tag(e: String) -> String {
    if e.starts_with("E_") { e } else { format!("E_INTERNAL: {e}") }
}

fn guard<T>(rt: *mut Runtime, f: impl FnOnce(&mut Runtime) -> R<T>, out: impl FnOnce(T)) -> i32 {
    let r = catch_unwind(AssertUnwindSafe(|| f(unsafe { &mut *rt })));
    let rt = unsafe { &mut *rt };
    match r {
        Ok(Ok(v)) => {
            out(v);
            0
        }
        Ok(Err(e)) => {
            rt.error = Some(tag(e));
            1
        }
        Err(_) => {
            rt.error = Some("E_INTERNAL: panic in the native runtime".into());
            1
        }
    }
}

extern "C" fn rt_op(rt: *mut Runtime, i: u64) -> i32 {
    guard(rt, |r| r.op(i as usize), |_| ())
}
extern "C" fn rt_get_int(rt: *mut Runtime, k: u64, out: *mut i64) -> i32 {
    guard(rt, |r| r.get_int(k as usize), |v| unsafe { *out = v })
}
extern "C" fn rt_truthy(rt: *mut Runtime, k: u64, out: *mut i64) -> i32 {
    guard(rt, |r| r.truthy(k as usize), |v| unsafe { *out = v as i64 })
}
extern "C" fn rt_iter_len(rt: *mut Runtime, k: u64, out: *mut i64) -> i32 {
    guard(rt, |r| r.iter_len(k as usize), |v| unsafe { *out = v as i64 })
}
extern "C" fn rt_set_int(rt: *mut Runtime, k: u64, v: i64) -> i32 {
    guard(rt, |r| { r.set_int(k as usize, v); Ok(()) }, |_| ())
}
extern "C" fn rt_iter_set(rt: *mut Runtime, var: u64, k: u64, i: u64) -> i32 {
    guard(rt, |r| { r.iter_set(var as usize, k as usize, i as usize); Ok(()) }, |_| ())
}

struct Ctx {
    rt: Value,
    err: Block,
    slot: Value,
    ss: cranelift_codegen::ir::StackSlot,
    s_op: SigRef,
    s_out: SigRef,
    s_set: SigRef,
    s_iset: SigRef,
    ptr: types::Type,
}

/// Call a runtime entry point; on a nonzero status jump to the error block.
fn call(b: &mut FunctionBuilder, c: &Ctx, sig: SigRef, f: usize, args: &[Value]) {
    let callee = b.ins().iconst(c.ptr, f as i64);
    let inst = b.ins().call_indirect(sig, callee, args);
    let st = b.inst_results(inst)[0];
    let ok = b.create_block();
    b.ins().brif(st, c.err, &[], ok, &[]);
    b.switch_to_block(ok);
}

fn k(b: &mut FunctionBuilder, x: usize) -> Value {
    b.ins().iconst(types::I64, x as i64)
}

/// Query an i64 through the scratch stack slot.
fn query(b: &mut FunctionBuilder, c: &Ctx, f: usize, slot: usize) -> Value {
    let s = k(b, slot);
    call(b, c, c.s_out, f, &[c.rt, s, c.slot]);
    b.ins().stack_load(types::I64, c.ss, 0)
}

fn emit(b: &mut FunctionBuilder, c: &Ctx, s: &Stmt, lp: Option<(Block, Block)>) {
    match s {
        Stmt::Op(i) => {
            let i = k(b, *i);
            call(b, c, c.s_op, rt_op as *const () as usize, &[c.rt, i]);
        }
        Stmt::Block(v) => {
            for x in v {
                emit(b, c, x, lp);
            }
        }
        Stmt::For { var, lo, hi, body } => {
            let l = query(b, c, rt_get_int as *const () as usize, *lo);
            let h = query(b, c, rt_get_int as *const () as usize, *hi);
            let head = b.create_block();
            let bodyb = b.create_block();
            let latch = b.create_block();
            let exit = b.create_block();
            b.append_block_param(head, types::I64);
            b.append_block_param(latch, types::I64);
            b.ins().jump(head, &[l.into()]);
            b.switch_to_block(head);
            let i = b.block_params(head)[0];
            let more = b.ins().icmp(IntCC::SignedLessThanOrEqual, i, h);
            b.ins().brif(more, bodyb, &[], exit, &[]);
            b.switch_to_block(bodyb);
            let vs = k(b, *var);
            call(b, c, c.s_set, rt_set_int as *const () as usize, &[c.rt, vs, i]);
            let cont = b.create_block();
            emit(b, c, body, Some((exit, cont)));
            b.ins().jump(cont, &[]);
            b.switch_to_block(cont);
            b.ins().jump(latch, &[i.into()]);
            b.switch_to_block(latch);
            let li = b.block_params(latch)[0];
            let n = b.ins().iadd_imm(li, 1);
            b.ins().jump(head, &[n.into()]);
            b.switch_to_block(exit);
        }
        Stmt::ForEach { var, container, body } => {
            let n = query(b, c, rt_iter_len as *const () as usize, *container);
            let head = b.create_block();
            let bodyb = b.create_block();
            let exit = b.create_block();
            b.append_block_param(head, types::I64);
            let zero = k(b, 0);
            b.ins().jump(head, &[zero.into()]);
            b.switch_to_block(head);
            let i = b.block_params(head)[0];
            let more = b.ins().icmp(IntCC::SignedLessThan, i, n);
            b.ins().brif(more, bodyb, &[], exit, &[]);
            b.switch_to_block(bodyb);
            let vs = k(b, *var);
            let cs = k(b, *container);
            call(b, c, c.s_iset, rt_iter_set as *const () as usize, &[c.rt, vs, cs, i]);
            let cont = b.create_block();
            emit(b, c, body, Some((exit, cont)));
            b.ins().jump(cont, &[]);
            b.switch_to_block(cont);
            let nx = b.ins().iadd_imm(i, 1);
            b.ins().jump(head, &[nx.into()]);
            b.switch_to_block(exit);
        }
        Stmt::While { pre, cond, body } => {
            let head = b.create_block();
            let bodyb = b.create_block();
            let exit = b.create_block();
            b.ins().jump(head, &[]);
            b.switch_to_block(head);
            emit(b, c, pre, lp);
            let t = query(b, c, rt_truthy as *const () as usize, *cond);
            b.ins().brif(t, bodyb, &[], exit, &[]);
            b.switch_to_block(bodyb);
            emit(b, c, body, Some((exit, head)));
            b.ins().jump(head, &[]);
            b.switch_to_block(exit);
        }
        Stmt::If { cond, then, els } => {
            let t = query(b, c, rt_truthy as *const () as usize, *cond);
            let tb = b.create_block();
            let eb = b.create_block();
            let merge = b.create_block();
            b.ins().brif(t, tb, &[], eb, &[]);
            b.switch_to_block(tb);
            emit(b, c, then, lp);
            b.ins().jump(merge, &[]);
            b.switch_to_block(eb);
            emit(b, c, els, lp);
            b.ins().jump(merge, &[]);
            b.switch_to_block(merge);
        }
        Stmt::Break | Stmt::Continue => {
            let (brk, cont) = lp.expect("break or continue outside a loop");
            b.ins().jump(if matches!(s, Stmt::Break) { brk } else { cont }, &[]);
            // code after the jump is unreachable but still needs a block
            let dead = b.create_block();
            b.switch_to_block(dead);
        }
    }
}

/// A program compiled to machine code.
pub struct Compiled {
    module: Option<JITModule>,
    f: extern "C" fn(*mut Runtime) -> i32,
}

impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            unsafe { m.free_memory() };
        }
    }
}

impl Compiled {
    pub fn new(prog: &Program) -> R<Compiled> {
        let mut fb = settings::builder();
        let e = |x: cranelift_codegen::settings::SetError| format!("E_INTERNAL: cranelift settings: {x}");
        fb.set("opt_level", "speed").map_err(e)?;
        fb.set("is_pic", "false").map_err(e)?;
        let isa = cranelift_native::builder()
            .map_err(|x| format!("E_UNSUPPORTED: cranelift does not support this host: {x}"))?
            .finish(settings::Flags::new(fb))
            .map_err(|x| format!("E_INTERNAL: cranelift: {x}"))?;
        let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));
        let tcfg = module.target_config();
        let ptr = tcfg.pointer_type();
        let cc = module.target_config().default_call_conv;
        let sig = |params: &[types::Type], ret: bool| {
            let mut s = Signature::new(cc);
            for &p in params {
                s.params.push(AbiParam::new(p));
            }
            if ret {
                s.returns.push(AbiParam::new(types::I32));
            }
            s
        };
        let mut ctx = module.make_context();
        ctx.func.signature = sig(&[ptr], true);
        ctx.func.name = UserFuncName::user(0, 0);
        let mut fctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            let rt = b.block_params(entry)[0];
            let ss = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let slot = b.ins().stack_addr(ptr, ss, 0);
            let err = b.create_block();
            let c = Ctx {
                rt,
                err,
                slot,
                ss,
                s_op: b.import_signature(sig(&[ptr, types::I64], true)),
                s_out: b.import_signature(sig(&[ptr, types::I64, ptr], true)),
                s_set: b.import_signature(sig(&[ptr, types::I64, types::I64], true)),
                s_iset: b.import_signature(sig(&[ptr, types::I64, types::I64, types::I64], true)),
                ptr,
            };
            emit(&mut b, &c, &prog.body, None);
            let z = b.ins().iconst(types::I32, 0);
            b.ins().return_(&[z]);
            b.switch_to_block(err);
            let one = b.ins().iconst(types::I32, 1);
            b.ins().return_(&[one]);
            b.seal_all_blocks();
            b.finalize();
        }
        let id = module
            .declare_function("bcn_model", Linkage::Local, &ctx.func.signature)
            .map_err(|x| format!("E_INTERNAL: cranelift: {x}"))?;
        module.define_function(id, &mut ctx).map_err(|x| format!("E_INTERNAL: cranelift: {x:?}"))?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|x| format!("E_INTERNAL: cranelift: {x}"))?;
        let code = module.get_finalized_function(id);
        let f = unsafe { std::mem::transmute::<*const u8, extern "C" fn(*mut Runtime) -> i32>(code) };
        Ok(Compiled { module: Some(module), f })
    }

    /// Log density and, if requested, its gradient at `u`.
    pub fn eval(&self, rt: &mut Runtime, u: &[f64], jacobian: bool, propto: bool, grad: Option<&mut [f64]>) -> R<f64> {
        rt.begin(u, jacobian, propto);
        if (self.f)(rt as *mut Runtime) != 0 {
            return Err(rt.error.take().unwrap_or_else(|| "E_INTERNAL: native code failed without a message".into()));
        }
        rt.finish(grad)
    }
}
