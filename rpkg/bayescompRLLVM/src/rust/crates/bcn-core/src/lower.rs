//! Lowering of the bayescompR IR (JSON, schema 1.x) to a [`Program`].
//!
//! Data and transformed data come in as values computed by the R model, so
//! only the parameters, transformed parameters and model blocks are lowered.
//! Anything the native runtime cannot execute is rejected here, before any
//! sampling starts, with an error naming the IR node.

use crate::calls;
use crate::dist;
use crate::kernel::*;
use crate::math;
use crate::val::Val;
use serde_json::Value as J;
use std::collections::HashMap;

pub const SUPPORTED_IR_MAJOR: u64 = 1;

/// Errors raised by lowering start with `E_UNSUPPORTED:` (the IR uses a
/// feature the native runtime lacks) or `E_IR:` (malformed input).
fn unsupported<T>(what: impl AsRef<str>) -> Result<T, String> {
    Err(format!("E_UNSUPPORTED: {}", what.as_ref()))
}

fn bad<T>(what: impl AsRef<str>) -> Result<T, String> {
    Err(format!("E_IR: {}", what.as_ref()))
}

fn s<'a>(j: &'a J, k: &str) -> Result<&'a str, String> {
    j.get(k).and_then(|v| v.as_str()).ok_or_else(|| format!("E_IR: missing string field '{k}'"))
}

fn arr<'a>(j: &'a J, k: &str) -> &'a [J] {
    j.get(k).and_then(|v| v.as_array()).map(|v| v.as_slice()).unwrap_or(&[])
}

/// A number, allowing the strings the R JSON writer uses for non finite values.
fn num(j: &J) -> f64 {
    match j {
        J::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        J::String(x) => match x.as_str() {
            "Infinity" | "Inf" => f64::INFINITY,
            "-Infinity" | "-Inf" => f64::NEG_INFINITY,
            other => other.parse().unwrap_or(f64::NAN),
        },
        _ => f64::NAN,
    }
}

/// A list of numbers; a scalar is a list of one (the R writer unboxes).
fn nums(j: Option<&J>) -> Vec<f64> {
    match j {
        Some(J::Array(v)) => v.iter().map(num).collect(),
        Some(J::Null) | None => vec![],
        Some(x) => vec![num(x)],
    }
}

fn null_or_missing(j: &J, k: &str) -> bool {
    j.get(k).map(|v| v.is_null()).unwrap_or(true)
}

/// Base type of a declaration; constrained data types map to their storage.
fn storage_base(b: &str) -> Option<Base> {
    Base::parse(b).or(match b {
        "simplex" | "ordered" | "positive_ordered" | "unit_vector" | "sum_to_zero_vector" => Some(Base::Vector),
        "cholesky_factor_corr" | "cholesky_factor_cov" | "corr_matrix" | "cov_matrix" | "column_stochastic_matrix"
        | "row_stochastic_matrix" | "sum_to_zero_matrix" => Some(Base::Matrix),
        _ => None,
    })
}

/// Build a value of the declared shape from scalars in Stan order.
pub fn build_val(base: Base, nadims: usize, dims: &[usize], it: &mut impl Iterator<Item = f64>) -> Result<Val, String> {
    if nadims > 0 {
        let n = dims[0];
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(build_val(base, nadims - 1, &dims[1..], it)?);
        }
        return Ok(Val::Array(v));
    }
    let mut next = || it.next().ok_or_else(|| "E_IR: not enough data values".to_string());
    Ok(match base {
        Base::Int => {
            let x = next()?;
            Val::Int(x as i64)
        }
        Base::Real => Val::real(next()?),
        Base::Vector | Base::RowVector => {
            let n = *dims.first().ok_or("E_IR: missing vector size")?;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(crate::tape::S::c(next()?));
            }
            if base == Base::Vector { Val::Vector(v) } else { Val::RowVector(v) }
        }
        Base::Matrix => {
            if dims.len() < 2 {
                return bad("missing matrix sizes");
            }
            let (r, c) = (dims[0], dims[1]);
            let mut d = Vec::with_capacity(r * c);
            for _ in 0..r * c {
                d.push(crate::tape::S::c(next()?));
            }
            Val::Matrix { r, c, d }
        }
    })
}

struct Lw {
    ops: Vec<Op>,
    nslots: usize,
    init: Vec<(usize, Val)>,
    scopes: Vec<HashMap<String, usize>>,
    /// constant pool for integers (most constants are small integers)
    ints: HashMap<i64, usize>,
}

impl Lw {
    fn slot(&mut self) -> usize {
        self.nslots += 1;
        self.nslots - 1
    }
    fn constant(&mut self, v: Val) -> usize {
        if let Val::Int(i) = v {
            if let Some(&k) = self.ints.get(&i) {
                return k;
            }
            let k = self.slot();
            self.ints.insert(i, k);
            self.init.push((k, v));
            return k;
        }
        let k = self.slot();
        self.init.push((k, v));
        k
    }
    fn op(&mut self, out: &mut Vec<Stmt>, kind: OpKind, dst: usize, args: Vec<usize>) {
        self.ops.push(Op { kind, dst, args });
        out.push(Stmt::Op(self.ops.len() - 1));
    }
    fn lookup(&self, name: &str) -> Result<usize, String> {
        for sc in self.scopes.iter().rev() {
            if let Some(&k) = sc.get(name) {
                return Ok(k);
            }
        }
        bad(format!("unknown variable '{name}'"))
    }
    fn bind(&mut self, name: &str) -> usize {
        let k = self.slot();
        self.scopes.last_mut().unwrap().insert(name.to_string(), k);
        k
    }

    fn idx_list(&mut self, idx: &[J], out: &mut Vec<Stmt>, args: &mut Vec<usize>) -> Result<Vec<IdxKind>, String> {
        let mut kinds = Vec::with_capacity(idx.len());
        for i in idx {
            match s(i, "kind")? {
                "single" => {
                    let e = i.get("e").ok_or("E_IR: index without expression")?;
                    let multi = e.get("type").and_then(|t| t.get("nd")).and_then(|v| v.as_u64()).unwrap_or(0) > 0;
                    args.push(self.expr(e, out)?);
                    kinds.push(if multi { IdxKind::Multi } else { IdxKind::Single });
                }
                "intarray" => {
                    args.push(self.expr(&i["e"], out)?);
                    kinds.push(IdxKind::Multi);
                }
                "omni" => kinds.push(IdxKind::Omni),
                "lower" => {
                    args.push(self.expr(&i["lo"], out)?);
                    kinds.push(IdxKind::Range { lo: true, hi: false });
                }
                "upper" => {
                    args.push(self.expr(&i["hi"], out)?);
                    kinds.push(IdxKind::Range { lo: false, hi: true });
                }
                "range" => {
                    args.push(self.expr(&i["lo"], out)?);
                    args.push(self.expr(&i["hi"], out)?);
                    kinds.push(IdxKind::Range { lo: true, hi: true });
                }
                k => return unsupported(format!("index kind '{k}'")),
            }
        }
        Ok(kinds)
    }

    /// Lower an expression; returns the slot holding its value.
    fn expr(&mut self, e: &J, out: &mut Vec<Stmt>) -> Result<usize, String> {
        let kind = s(e, "kind")?;
        match kind {
            "int" => {
                let v = num(&e["value"]) as i64;
                Ok(self.constant(Val::Int(v)))
            }
            "real" => {
                let v = num(&e["value"]);
                Ok(self.constant(Val::real(v)))
            }
            "var" => self.lookup(s(e, "name")?),
            "paren" => self.expr(&e["a"], out),
            "binop" => {
                let op = s(e, "op")?;
                if op == "&&" || op == "||" {
                    return self.short_circuit(op, &e["a"], &e["b"], out);
                }
                let b = BinOp::parse(op).map_or_else(|| unsupported(format!("operator '{op}'")), Ok)?;
                let a = self.expr(&e["a"], out)?;
                let c = self.expr(&e["b"], out)?;
                let d = self.slot();
                if matches!(b, BinOp::Div) {
                    let bt = &e["b"]["type"];
                    if bt.get("base").and_then(|v| v.as_str()).map(|x| x == "matrix" || x == "row_vector").unwrap_or(false) {
                        return unsupported("matrix division");
                    }
                }
                self.op(out, OpKind::Bin(b), d, vec![a, c]);
                Ok(d)
            }
            "unop" => {
                let op = s(e, "op")?;
                let a = self.expr(&e["a"], out)?;
                let d = self.slot();
                let k = match op {
                    "-" => OpKind::Neg,
                    "!" => OpKind::Not,
                    "+" => OpKind::Copy,
                    _ => return unsupported(format!("unary operator '{op}'")),
                };
                self.op(out, k, d, vec![a]);
                Ok(d)
            }
            "transpose" => {
                let a = self.expr(&e["a"], out)?;
                let d = self.slot();
                self.op(out, OpKind::Transpose, d, vec![a]);
                Ok(d)
            }
            "cond" => {
                let c = self.expr(&e["cond"], out)?;
                let d = self.slot();
                let mut then = Vec::new();
                let a = self.expr(&e["a"], &mut then)?;
                self.op(&mut then, OpKind::Copy, d, vec![a]);
                let mut els = Vec::new();
                let b = self.expr(&e["b"], &mut els)?;
                self.op(&mut els, OpKind::Copy, d, vec![b]);
                out.push(Stmt::If { cond: c, then: Box::new(Stmt::Block(then)), els: Box::new(Stmt::Block(els)) });
                Ok(d)
            }
            "index" => {
                let x = self.expr(&e["x"], out)?;
                let mut args = vec![x];
                let kinds = self.idx_list(arr(e, "idx"), out, &mut args)?;
                let d = self.slot();
                self.op(out, OpKind::Index(kinds), d, args);
                Ok(d)
            }
            "array_expr" | "rowvec_expr" => {
                let mut args = Vec::new();
                for it in arr(e, "items") {
                    args.push(self.expr(it, out)?);
                }
                let d = self.slot();
                // a row vector literal of row vectors is a matrix (handled at run time)
                let k = if kind == "array_expr" { OpKind::ArrLit } else { OpKind::RowVecLit };
                self.op(out, k, d, args);
                Ok(d)
            }
            "call" => self.call(e, out),
            k => unsupported(format!("expression kind '{k}'")),
        }
    }

    fn short_circuit(&mut self, op: &str, a: &J, b: &J, out: &mut Vec<Stmt>) -> Result<usize, String> {
        let x = self.expr(a, out)?;
        let d = self.slot();
        let one = self.constant(Val::Int(1));
        let zero = self.constant(Val::Int(0));
        let mut rest = Vec::new();
        let y = self.expr(b, &mut rest)?;
        self.op(&mut rest, OpKind::Bin(BinOp::Ne), d, vec![y, zero]);
        let mut fixed = Vec::new();
        self.op(&mut fixed, OpKind::Copy, d, vec![if op == "&&" { zero } else { one }]);
        let (then, els) = if op == "&&" { (rest, fixed) } else { (fixed, rest) };
        out.push(Stmt::If { cond: x, then: Box::new(Stmt::Block(then)), els: Box::new(Stmt::Block(els)) });
        Ok(d)
    }

    fn call(&mut self, e: &J, out: &mut Vec<Stmt>) -> Result<usize, String> {
        let name = s(e, "name")?;
        if e.get("sig").and_then(|g| g.get("user")).and_then(|u| u.as_bool()).unwrap_or(false) {
            return unsupported(format!("user defined function '{name}'"));
        }
        let ja = arr(e, "args");
        let ty = |j: &J| -> (String, u64) {
            let t = &j["type"];
            (t["base"].as_str().unwrap_or("").to_string(), t["nd"].as_u64().unwrap_or(0))
        };
        let all_of = |ok: &[&str]| ja.iter().all(|a| { let (b, nd) = ty(a); nd == 0 && ok.contains(&b.as_str()) });
        let restricted = match name {
            "log_mix" => ja.len() == 3 && all_of(&["int", "real"]),
            "append_row" => all_of(&["int", "real", "vector"]),
            "append_col" => all_of(&["int", "real", "row_vector"]),
            _ => true,
        };
        if !restricted {
            return unsupported(format!("this form of '{name}'"));
        }
        if let Some(c) = calls::constant_call(name, ja.len()) {
            return Ok(self.constant(Val::real(c)));
        }
        if name == "target" && ja.is_empty() {
            let d = self.slot();
            self.op(out, OpKind::GetTarget, d, vec![]);
            return Ok(d);
        }
        let mut args = Vec::with_capacity(ja.len());
        for a in ja {
            args.push(self.expr(a, out)?);
        }
        let d = self.slot();
        for (suffix, lu, discrete) in [("_lupdf", true, false), ("_lupmf", true, true), ("_lpdf", false, false), ("_lpmf", false, true)] {
            if let Some(dn) = name.strip_suffix(suffix) {
                if !dist::supported(dn) || dist::is_discrete(dn) != discrete {
                    return unsupported(format!("function '{name}'"));
                }
                self.op(out, OpKind::Lpdf { dist: dn.to_string(), lu, discrete }, d, args);
                return Ok(d);
            }
        }
        let k = if let Some(f) = calls::lookup(name) {
            OpKind::Call(f)
        } else if ja.len() == 1 && math::unary_fn(name).is_some() {
            OpKind::Fn1(math::unary_fn(name).unwrap())
        } else if ja.len() == 2 && math::binary_fn(name).is_some() {
            OpKind::Fn2(math::binary_fn(name).unwrap())
        } else {
            return unsupported(format!("function '{name}' with {} arguments", ja.len()));
        };
        self.op(out, k, d, args);
        Ok(d)
    }

    fn bounds(&mut self, cons: &J, out: &mut Vec<Stmt>, args: &mut Vec<usize>) -> Result<Bounds, String> {
        let mut b = Bounds::default();
        if !cons.is_object() {
            return Ok(b);
        }
        for (k, flag) in [("lower", 0), ("upper", 1), ("offset", 2), ("multiplier", 3)] {
            if let Some(x) = cons.get(k) {
                if x.is_null() {
                    continue;
                }
                args.push(self.expr(x, out)?);
                match flag {
                    0 => b.lower = true,
                    1 => b.upper = true,
                    2 => b.offset = true,
                    _ => b.multiplier = true,
                }
            }
        }
        Ok(b)
    }

    fn decl_type(&mut self, ty: &J, out: &mut Vec<Stmt>, args: &mut Vec<usize>, allow_constrained: bool) -> Result<DeclType, String> {
        let bname = s(ty, "base")?;
        let base = if allow_constrained { storage_base(bname) } else { Base::parse(bname) };
        let base = match base {
            Some(b) => b,
            None => return unsupported(format!("declaration of type '{bname}'")),
        };
        let adims = arr(ty, "adims");
        for a in adims {
            args.push(self.expr(a, out)?);
        }
        let sizes = arr(ty, "sizes");
        if sizes.len() != base.nsizes() {
            return bad(format!("type '{bname}' with {} sizes", sizes.len()));
        }
        for a in sizes {
            args.push(self.expr(a, out)?);
        }
        Ok(DeclType { base, nadims: adims.len() })
    }

    fn param(&mut self, d: &J, out: &mut Vec<Stmt>) -> Result<(), String> {
        let ty = &d["type"];
        let name = s(d, "name")?.to_string();
        let mut args = Vec::new();
        let dt = self.decl_type(ty, out, &mut args, false)?;
        let b = self.bounds(&ty["cons"], out, &mut args)?;
        let dst = self.bind(&name);
        self.op(out, OpKind::Param { ty: dt, bounds: b, name }, dst, args);
        Ok(())
    }

    /// A local or transformed parameter declaration; returns the constraint
    /// check for a transformed parameter.
    fn local(&mut self, d: &J, out: &mut Vec<Stmt>) -> Result<Option<(usize, J, String)>, String> {
        let ty = &d["type"];
        let name = s(d, "name")?.to_string();
        let mut args = Vec::new();
        let dt = self.decl_type(ty, out, &mut args, false)?;
        let dst = self.bind(&name);
        self.op(out, OpKind::Decl(dt), dst, args);
        if !null_or_missing(d, "init") {
            let v = self.expr(&d["init"], out)?;
            self.op(out, OpKind::Assign(vec![]), dst, vec![dst, v]);
        }
        let cons = &ty["cons"];
        let constrained = cons.is_object() && cons.as_object().map(|o| !o.is_empty()).unwrap_or(false);
        Ok(Some((dst, if constrained { cons.clone() } else { J::Null }, name)).filter(|_| constrained || d["block"] == "transformed parameters"))
    }

    fn stmt(&mut self, st: &J, out: &mut Vec<Stmt>) -> Result<(), String> {
        let kind = s(st, "kind")?;
        match kind {
            "decl" => {
                if let Some((_, cons, _)) = self.local(st, out)? {
                    if !cons.is_null() {
                        return unsupported("constrained local variable");
                    }
                }
            }
            "decls" => {
                for d in arr(st, "items") {
                    self.stmt(d, out)?;
                }
            }
            "block_stmt" => {
                self.scopes.push(HashMap::new());
                let mut body = Vec::new();
                for x in arr(st, "body") {
                    self.stmt(x, &mut body)?;
                }
                self.scopes.pop();
                out.push(Stmt::Block(body));
            }
            "empty" | "print" => {}
            "assign" => {
                let op = s(st, "op")?;
                let lhs = &st["lhs"];
                let (var, idx) = match s(lhs, "kind")? {
                    "var" => (lhs, &[][..]),
                    "index" => {
                        if s(&lhs["x"], "kind")? != "var" {
                            return unsupported("assignment to a nested index expression");
                        }
                        (&lhs["x"], arr(lhs, "idx"))
                    }
                    k => return unsupported(format!("assignment target '{k}'")),
                };
                let target = self.lookup(s(var, "name")?)?;
                let mut value = self.expr(&st["rhs"], out)?;
                if op != "=" {
                    let bop = op.strip_suffix('=').and_then(BinOp::parse);
                    let bop = match bop {
                        Some(b) => b,
                        None => return unsupported(format!("assignment operator '{op}'")),
                    };
                    let cur = self.expr(lhs, out)?;
                    let d = self.slot();
                    self.op(out, OpKind::Bin(bop), d, vec![cur, value]);
                    value = d;
                }
                let mut args = vec![target, value];
                let kinds = self.idx_list(idx, out, &mut args)?;
                self.op(out, OpKind::Assign(kinds), target, args);
            }
            "target_add" => {
                let v = self.expr(&st["value"], out)?;
                self.op(out, OpKind::Target, 0, vec![v]);
            }
            "jacobian_add" => {
                let v = self.expr(&st["value"], out)?;
                self.op(out, OpKind::Jacobian, 0, vec![v]);
            }
            "tilde" => {
                if !null_or_missing(st, "trunc") {
                    return unsupported("truncated sampling statement");
                }
                let dn = s(st, "dist")?;
                if !dist::supported(dn) {
                    return unsupported(format!("distribution '{dn}'"));
                }
                let mut args = vec![self.expr(&st["lhs"], out)?];
                for a in arr(st, "args") {
                    args.push(self.expr(a, out)?);
                }
                let d = self.slot();
                self.op(out, OpKind::Lpdf { dist: dn.to_string(), lu: true, discrete: dist::is_discrete(dn) }, d, args);
                self.op(out, OpKind::Target, 0, vec![d]);
            }
            "if" => {
                let c = self.expr(&st["cond"], out)?;
                let mut then = Vec::new();
                self.scoped(&st["then"], &mut then)?;
                let mut els = Vec::new();
                if !null_or_missing(st, "els") {
                    self.scoped(&st["els"], &mut els)?;
                }
                out.push(Stmt::If { cond: c, then: Box::new(Stmt::Block(then)), els: Box::new(Stmt::Block(els)) });
            }
            "for" => {
                let lo = self.expr(&st["lo"], out)?;
                let hi = self.expr(&st["hi"], out)?;
                self.scopes.push(HashMap::new());
                let var = self.bind(s(st, "var")?);
                let mut body = Vec::new();
                self.stmt(&st["body"], &mut body)?;
                self.scopes.pop();
                out.push(Stmt::For { var, lo, hi, body: Box::new(Stmt::Block(body)) });
            }
            "foreach" => {
                let c = self.expr(&st["iter"], out)?;
                self.scopes.push(HashMap::new());
                let var = self.bind(s(st, "var")?);
                let mut body = Vec::new();
                self.stmt(&st["body"], &mut body)?;
                self.scopes.pop();
                out.push(Stmt::ForEach { var, container: c, body: Box::new(Stmt::Block(body)) });
            }
            "while" => {
                let mut pre = Vec::new();
                let c = self.expr(&st["cond"], &mut pre)?;
                let mut body = Vec::new();
                self.scoped(&st["body"], &mut body)?;
                out.push(Stmt::While { pre: Box::new(Stmt::Block(pre)), cond: c, body: Box::new(Stmt::Block(body)) });
            }
            "break" => out.push(Stmt::Break),
            "continue" => out.push(Stmt::Continue),
            "reject" | "fatal_error" => {
                let mut msg = String::new();
                let mut args = Vec::new();
                for it in arr(st, "items") {
                    if it["kind"] == "string" {
                        msg.push_str(it["value"].as_str().unwrap_or(""));
                    } else {
                        msg.push_str("{}");
                        args.push(self.expr(it, out)?);
                    }
                }
                self.op(out, OpKind::Reject { msg, fatal: kind == "fatal_error" }, 0, args);
            }
            k => return unsupported(format!("statement kind '{k}'")),
        }
        Ok(())
    }

    fn scoped(&mut self, st: &J, out: &mut Vec<Stmt>) -> Result<(), String> {
        self.scopes.push(HashMap::new());
        let r = self.stmt(st, out);
        self.scopes.pop();
        r
    }
}

fn block<'a>(ir: &'a J, name: &str) -> &'a [J] {
    ir.get("blocks").and_then(|b| b.get(name)).map(|b| arr(b, "body")).unwrap_or(&[])
}

/// Lower `ir` (the JSON from `bayescomp_ir(model, json = TRUE)`) with the
/// data and transformed data values in `data`: an object mapping each name
/// to `{"dims": [...], "values": [...]}` with values in Stan order.
pub fn lower(ir: &J, data: &J, dim: usize) -> Result<Program, String> {
    let ver = s(ir, "ir_version")?;
    let major: u64 = ver.split('.').next().and_then(|x| x.parse().ok()).unwrap_or(0);
    if major != SUPPORTED_IR_MAJOR {
        return unsupported(format!("IR version {ver} (this runtime reads {SUPPORTED_IR_MAJOR}.x)"));
    }
    if !block(ir, "functions").is_empty() {
        return unsupported("functions block");
    }
    let mut lw = Lw { ops: Vec::new(), nslots: 0, init: Vec::new(), scopes: vec![HashMap::new()], ints: HashMap::new() };
    for b in ["data", "transformed data"] {
        for d in block(ir, b) {
            let items: Vec<&J> = if d["kind"] == "decls" { arr(d, "items").iter().collect() } else { vec![d] };
            for d in items {
                if d["kind"] != "decl" {
                    continue;
                }
                let name = s(d, "name")?;
                let ty = &d["type"];
                let bname = s(ty, "base")?;
                let base = match storage_base(bname) {
                    Some(b) => b,
                    None => return unsupported(format!("data of type '{bname}'")),
                };
                let nadims = arr(ty, "adims").len();
                let entry = data.get(name).ok_or_else(|| format!("E_IR: no value for data variable '{name}'"))?;
                let dims: Vec<usize> = nums(entry.get("dims")).into_iter().map(|x| x as usize).collect();
                let vals = nums(entry.get("values"));
                let mut it = vals.into_iter();
                let v = build_val(base, nadims, &dims, &mut it)?;
                let k = lw.bind(name);
                lw.init.push((k, v));
            }
        }
    }
    let mut body = Vec::new();
    for d in block(ir, "parameters") {
        let items: Vec<&J> = if d["kind"] == "decls" { arr(d, "items").iter().collect() } else { vec![d] };
        for d in items {
            lw.param(d, &mut body)?;
        }
    }
    let mut checks = Vec::new();
    for st in block(ir, "transformed parameters") {
        let items: Vec<&J> = if st["kind"] == "decls" { arr(st, "items").iter().collect() } else { vec![st] };
        for st in items {
            if st["kind"] == "decl" {
                if let Some(c) = lw.local(st, &mut body)? {
                    checks.push(c);
                }
            } else {
                lw.stmt(st, &mut body)?;
            }
        }
    }
    for (slot, cons, name) in checks {
        let mut args = vec![slot];
        let b = lw.bounds(&cons, &mut body, &mut args)?;
        lw.op(&mut body, OpKind::Check { bounds: b, name }, 0, args);
    }
    lw.scopes.push(HashMap::new());
    for st in block(ir, "model") {
        lw.stmt(st, &mut body)?;
    }
    Ok(Program { ops: lw.ops, body: Stmt::Block(body), nslots: lw.nslots, init: lw.init, dim })
}
