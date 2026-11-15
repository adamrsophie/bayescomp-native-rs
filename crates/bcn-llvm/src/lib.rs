//! LLVM backend: the statement tree of a lowered program is compiled through
//! LLVM's MCJIT to one function `fn(*mut Runtime) -> i32`. As in the
//! Cranelift backend, leaf operations call the shared `bcn-core` runtime
//! through constant addresses, so results are bit identical to the
//! interpreter; a nonzero status means `Runtime::error` holds the message.

use std::mem::ManuallyDrop;
use std::panic::{catch_unwind, AssertUnwindSafe};

use bcn_core::exec::Runtime;
use bcn_core::kernel::{Program, Stmt};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::targets::{InitializationConfig, Target};
use inkwell::types::FunctionType;
use inkwell::values::{BasicMetadataValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

type R<T> = Result<T, String>;
type ModelFn = unsafe extern "C" fn(*mut Runtime) -> i32;

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

fn be<E: std::fmt::Display>(e: E) -> String {
    format!("E_INTERNAL: llvm: {e}")
}

struct Em<'c> {
    cx: &'c Context,
    b: Builder<'c>,
    f: FunctionValue<'c>,
    rt: PointerValue<'c>,
    slot: PointerValue<'c>,
    err: BasicBlock<'c>,
    t_op: FunctionType<'c>,
    t_out: FunctionType<'c>,
    t_set: FunctionType<'c>,
    t_iset: FunctionType<'c>,
}

impl<'c> Em<'c> {
    fn k(&self, x: usize) -> IntValue<'c> {
        self.cx.i64_type().const_int(x as u64, false)
    }

    fn call(&self, ty: FunctionType<'c>, addr: usize, args: &[BasicMetadataValueEnum<'c>]) -> R<()> {
        let p = self.b.build_int_to_ptr(self.k(addr), self.cx.ptr_type(AddressSpace::default()), "fp").map_err(be)?;
        let st = self.b.build_indirect_call(ty, p, args, "st").map_err(be)?;
        let st = match st.try_as_basic_value() {
            inkwell::values::ValueKind::Basic(v) => v.into_int_value(),
            _ => return Err("E_INTERNAL: llvm: call without a result".into()),
        };
        let bad = self.b.build_int_compare(IntPredicate::NE, st, self.cx.i32_type().const_zero(), "bad").map_err(be)?;
        let ok = self.cx.append_basic_block(self.f, "ok");
        self.b.build_conditional_branch(bad, self.err, ok).map_err(be)?;
        self.b.position_at_end(ok);
        Ok(())
    }

    fn query(&self, addr: usize, slot: usize) -> R<IntValue<'c>> {
        self.call(self.t_out, addr, &[self.rt.into(), self.k(slot).into(), self.slot.into()])?;
        Ok(self.b.build_load(self.cx.i64_type(), self.slot, "q").map_err(be)?.into_int_value())
    }

    /// A loop counter in the entry block: an alloca inside the loop body would
    /// grow the stack on every iteration of an enclosing loop.
    fn entry_alloca(&self) -> R<inkwell::values::PointerValue<'c>> {
        let eb = self.cx.create_builder();
        let entry = self.f.get_first_basic_block().ok_or("E_INTERNAL: function without an entry block")?;
        match entry.get_first_instruction() {
            Some(i) => eb.position_before(&i),
            None => eb.position_at_end(entry),
        }
        eb.build_alloca(self.cx.i64_type(), "i").map_err(be)
    }

    fn jump(&self, to: BasicBlock<'c>) -> R<()> {
        self.b.build_unconditional_branch(to).map_err(be)?;
        Ok(())
    }

    fn emit(&self, s: &Stmt, lp: Option<(BasicBlock<'c>, BasicBlock<'c>)>) -> R<()> {
        let i64t = self.cx.i64_type();
        match s {
            Stmt::Op(i) => self.call(self.t_op, rt_op as *const () as usize, &[self.rt.into(), self.k(*i).into()])?,
            Stmt::Block(v) => {
                for x in v {
                    self.emit(x, lp)?;
                }
            }
            Stmt::For { var, lo, hi, body } => {
                let l = self.query(rt_get_int as *const () as usize, *lo)?;
                let h = self.query(rt_get_int as *const () as usize, *hi)?;
                let iv = self.entry_alloca()?;
                self.b.build_store(iv, l).map_err(be)?;
                let head = self.cx.append_basic_block(self.f, "for_head");
                let bodyb = self.cx.append_basic_block(self.f, "for_body");
                let cont = self.cx.append_basic_block(self.f, "for_next");
                let exit = self.cx.append_basic_block(self.f, "for_exit");
                self.jump(head)?;
                self.b.position_at_end(head);
                let i = self.b.build_load(i64t, iv, "i").map_err(be)?.into_int_value();
                let more = self.b.build_int_compare(IntPredicate::SLE, i, h, "more").map_err(be)?;
                self.b.build_conditional_branch(more, bodyb, exit).map_err(be)?;
                self.b.position_at_end(bodyb);
                self.call(self.t_set, rt_set_int as *const () as usize, &[self.rt.into(), self.k(*var).into(), i.into()])?;
                self.emit(body, Some((exit, cont)))?;
                self.jump(cont)?;
                self.b.position_at_end(cont);
                let i = self.b.build_load(i64t, iv, "i").map_err(be)?.into_int_value();
                let n = self.b.build_int_add(i, i64t.const_int(1, false), "n").map_err(be)?;
                self.b.build_store(iv, n).map_err(be)?;
                self.jump(head)?;
                self.b.position_at_end(exit);
            }
            Stmt::ForEach { var, container, body } => {
                let n = self.query(rt_iter_len as *const () as usize, *container)?;
                let iv = self.entry_alloca()?;
                self.b.build_store(iv, i64t.const_zero()).map_err(be)?;
                let head = self.cx.append_basic_block(self.f, "each_head");
                let bodyb = self.cx.append_basic_block(self.f, "each_body");
                let cont = self.cx.append_basic_block(self.f, "each_next");
                let exit = self.cx.append_basic_block(self.f, "each_exit");
                self.jump(head)?;
                self.b.position_at_end(head);
                let i = self.b.build_load(i64t, iv, "i").map_err(be)?.into_int_value();
                let more = self.b.build_int_compare(IntPredicate::SLT, i, n, "more").map_err(be)?;
                self.b.build_conditional_branch(more, bodyb, exit).map_err(be)?;
                self.b.position_at_end(bodyb);
                self.call(self.t_iset, rt_iter_set as *const () as usize,
                          &[self.rt.into(), self.k(*var).into(), self.k(*container).into(), i.into()])?;
                self.emit(body, Some((exit, cont)))?;
                self.jump(cont)?;
                self.b.position_at_end(cont);
                let i = self.b.build_load(i64t, iv, "i").map_err(be)?.into_int_value();
                let nx = self.b.build_int_add(i, i64t.const_int(1, false), "n").map_err(be)?;
                self.b.build_store(iv, nx).map_err(be)?;
                self.jump(head)?;
                self.b.position_at_end(exit);
            }
            Stmt::While { pre, cond, body } => {
                let head = self.cx.append_basic_block(self.f, "while_head");
                let bodyb = self.cx.append_basic_block(self.f, "while_body");
                let exit = self.cx.append_basic_block(self.f, "while_exit");
                self.jump(head)?;
                self.b.position_at_end(head);
                self.emit(pre, lp)?;
                let t = self.query(rt_truthy as *const () as usize, *cond)?;
                let t = self.b.build_int_compare(IntPredicate::NE, t, i64t.const_zero(), "t").map_err(be)?;
                self.b.build_conditional_branch(t, bodyb, exit).map_err(be)?;
                self.b.position_at_end(bodyb);
                self.emit(body, Some((exit, head)))?;
                self.jump(head)?;
                self.b.position_at_end(exit);
            }
            Stmt::If { cond, then, els } => {
                let t = self.query(rt_truthy as *const () as usize, *cond)?;
                let t = self.b.build_int_compare(IntPredicate::NE, t, i64t.const_zero(), "t").map_err(be)?;
                let tb = self.cx.append_basic_block(self.f, "then");
                let eb = self.cx.append_basic_block(self.f, "else");
                let merge = self.cx.append_basic_block(self.f, "merge");
                self.b.build_conditional_branch(t, tb, eb).map_err(be)?;
                self.b.position_at_end(tb);
                self.emit(then, lp)?;
                self.jump(merge)?;
                self.b.position_at_end(eb);
                self.emit(els, lp)?;
                self.jump(merge)?;
                self.b.position_at_end(merge);
            }
            Stmt::Break | Stmt::Continue => {
                let (brk, cont) = lp.ok_or("E_INTERNAL: break or continue outside a loop")?;
                self.jump(if matches!(s, Stmt::Break) { brk } else { cont })?;
                // code after the jump is unreachable but still needs a block
                let dead = self.cx.append_basic_block(self.f, "dead");
                self.b.position_at_end(dead);
            }
        }
        Ok(())
    }
}

/// A program compiled to machine code. The LLVM context is owned here and
/// released after the engine and module that borrow it.
pub struct Compiled {
    f: ManuallyDrop<JitFunction<'static, ModelFn>>,
    ee: ManuallyDrop<ExecutionEngine<'static>>,
    cx: *mut Context,
}

impl Drop for Compiled {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.f);
            ManuallyDrop::drop(&mut self.ee);
            drop(Box::from_raw(self.cx));
        }
    }
}

impl Compiled {
    pub fn new(prog: &Program) -> R<Compiled> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| format!("E_UNSUPPORTED: LLVM cannot target this host: {e}"))?;
        let cxp = Box::into_raw(Box::new(Context::create()));
        let cx: &'static Context = unsafe { &*cxp };
        let r = (|| -> R<(JitFunction<'static, ModelFn>, ExecutionEngine<'static>)> {
            let module: Module<'static> = cx.create_module("bcn");
            let ptr = cx.ptr_type(AddressSpace::default());
            let (i32t, i64t) = (cx.i32_type(), cx.i64_type());
            let f = module.add_function("bcn_model", i32t.fn_type(&[ptr.into()], false), None);
            let entry = cx.append_basic_block(f, "entry");
            let b = cx.create_builder();
            b.position_at_end(entry);
            let slot = b.build_alloca(i64t, "slot").map_err(be)?;
            let em = Em {
                cx,
                b,
                f,
                rt: f.get_nth_param(0).ok_or("E_INTERNAL: llvm: no parameter")?.into_pointer_value(),
                slot,
                err: cx.append_basic_block(f, "err"),
                t_op: i32t.fn_type(&[ptr.into(), i64t.into()], false),
                t_out: i32t.fn_type(&[ptr.into(), i64t.into(), ptr.into()], false),
                t_set: i32t.fn_type(&[ptr.into(), i64t.into(), i64t.into()], false),
                t_iset: i32t.fn_type(&[ptr.into(), i64t.into(), i64t.into(), i64t.into()], false),
            };
            em.emit(&prog.body, None)?;
            em.b.build_return(Some(&i32t.const_zero())).map_err(be)?;
            em.b.position_at_end(em.err);
            em.b.build_return(Some(&i32t.const_int(1, false))).map_err(be)?;
            if !f.verify(false) {
                return Err("E_INTERNAL: llvm: generated function failed verification".into());
            }
            let ee = module.create_jit_execution_engine(OptimizationLevel::Aggressive).map_err(be)?;
            let jf = unsafe { ee.get_function::<ModelFn>("bcn_model") }.map_err(be)?;
            Ok((jf, ee))
        })();
        match r {
            Ok((f, ee)) => Ok(Compiled { f: ManuallyDrop::new(f), ee: ManuallyDrop::new(ee), cx: cxp }),
            Err(e) => {
                unsafe { drop(Box::from_raw(cxp)) };
                Err(e)
            }
        }
    }

    /// Log density and, if requested, its gradient at `u`.
    pub fn eval(&self, rt: &mut Runtime, u: &[f64], jacobian: bool, propto: bool, grad: Option<&mut [f64]>) -> R<f64> {
        rt.begin(u, jacobian, propto);
        if unsafe { self.f.call(rt as *mut Runtime) } != 0 {
            return Err(rt.error.take().unwrap_or_else(|| "E_INTERNAL: native code failed without a message".into()));
        }
        rt.finish(grad)
    }
}
