//! The runtime: executes kernel operations on the AD tape.
//!
//! [`Runtime::op`] runs one leaf operation; the reference interpreter
//! ([`Runtime::interpret`]) walks the statement tree, and the JIT backends
//! emit the same walk as machine code calling the `rt_*` entry points.

use crate::dist;
use crate::kernel::*;
use crate::math::{self, R};
use crate::special::{inv_logit, log1p_exp};
use crate::tape::{Tape, S};
use crate::val::Val;

pub struct Runtime {
    pub prog: Program,
    pub slots: Vec<Val>,
    pub tape: Tape,
    pub target: Vec<S>,
    pub jacobian: bool,
    pub propto: bool,
    uvars: Vec<S>,
    ucur: usize,
    pub error: Option<String>,
}

enum Raw {
    Single(i64),
    Multi(Vec<i64>),
    Range(Option<i64>, Option<i64>),
    Omni,
}

enum Sel {
    One(usize),
    Many(Vec<usize>),
}

fn resolve(r: &Raw, n: usize) -> R<Sel> {
    let chk = |i: i64| -> R<usize> {
        if i < 1 || i as usize > n {
            Err(format!("E_DOMAIN: index {i} out of range; expecting index to be between 1 and {n}"))
        } else {
            Ok((i - 1) as usize)
        }
    };
    Ok(match r {
        Raw::Single(i) => Sel::One(chk(*i)?),
        Raw::Multi(v) => Sel::Many(v.iter().map(|&i| chk(i)).collect::<R<_>>()?),
        Raw::Range(lo, hi) => {
            let (lo, hi) = (lo.unwrap_or(1), hi.unwrap_or(n as i64));
            if hi < lo {
                Sel::Many(vec![])
            } else {
                Sel::Many((lo..=hi).map(chk).collect::<R<_>>()?)
            }
        }
        Raw::Omni => Sel::Many((0..n).collect()),
    })
}

fn too_many() -> String {
    "E_INTERNAL: too many indexes for expression".into()
}

fn index_val(v: &Val, idx: &[Raw]) -> R<Val> {
    if idx.is_empty() {
        return Ok(v.clone());
    }
    match v {
        Val::Array(a) => match resolve(&idx[0], a.len())? {
            Sel::One(i) => index_val(&a[i], &idx[1..]),
            Sel::Many(is) => Ok(Val::Array(is.iter().map(|&i| index_val(&a[i], &idx[1..])).collect::<R<_>>()?)),
        },
        Val::Vector(x) | Val::RowVector(x) => {
            if idx.len() > 1 {
                return Err(too_many());
            }
            match resolve(&idx[0], x.len())? {
                Sel::One(i) => Ok(Val::Real(x[i])),
                Sel::Many(is) => {
                    let y: Vec<S> = is.iter().map(|&i| x[i]).collect();
                    Ok(if matches!(v, Val::Vector(_)) { Val::Vector(y) } else { Val::RowVector(y) })
                }
            }
        }
        Val::Matrix { r, c, d } => {
            if idx.len() > 2 {
                return Err(too_many());
            }
            let rs = resolve(&idx[0], *r)?;
            let cs = if idx.len() == 2 { Some(resolve(&idx[1], *c)?) } else { None };
            let at = |i: usize, j: usize| d[j * r + i];
            Ok(match (rs, cs) {
                (Sel::One(i), None) => Val::RowVector((0..*c).map(|j| at(i, j)).collect()),
                (Sel::One(i), Some(Sel::One(j))) => Val::Real(at(i, j)),
                (Sel::One(i), Some(Sel::Many(js))) => Val::RowVector(js.iter().map(|&j| at(i, j)).collect()),
                (Sel::Many(is), Some(Sel::One(j))) => Val::Vector(is.iter().map(|&i| at(i, j)).collect()),
                (Sel::Many(is), cs) => {
                    let js: Vec<usize> = match cs {
                        Some(Sel::Many(js)) => js,
                        _ => (0..*c).collect(),
                    };
                    let mut out = Vec::with_capacity(is.len() * js.len());
                    for &j in &js {
                        for &i in &is {
                            out.push(at(i, j));
                        }
                    }
                    Val::Matrix { r: is.len(), c: js.len(), d: out }
                }
            })
        }
        _ => Err(too_many()),
    }
}

/// Promote integers to reals where the target holds reals.
fn promote(old: &Val, new: Val) -> Val {
    match (old, new) {
        (Val::Real(_), Val::Int(i)) => Val::real(i as f64),
        (Val::Array(o), Val::Array(n)) if !o.is_empty() => {
            Val::Array(n.into_iter().map(|x| promote(&o[0], x)).collect())
        }
        (_, n) => n,
    }
}

fn same_shape(a: &Val, b: &Val) -> bool {
    match (a, b) {
        (Val::Int(_), Val::Int(_)) | (Val::Real(_), Val::Real(_)) | (Val::Real(_), Val::Int(_)) => true,
        (Val::Vector(x), Val::Vector(y)) | (Val::RowVector(x), Val::RowVector(y)) => x.len() == y.len(),
        (Val::Matrix { r, c, .. }, Val::Matrix { r: r2, c: c2, .. }) => r == r2 && c == c2,
        (Val::Array(x), Val::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_shape(p, q)),
        _ => false,
    }
}

fn scalar_of(v: &Val) -> R<S> {
    v.as_scalar().ok_or_else(|| "E_INTERNAL: expected a scalar value in assignment".to_string())
}

fn assign_val(target: &mut Val, idx: &[Raw], value: &Val) -> R<()> {
    if idx.is_empty() {
        let v = promote(target, value.clone());
        if !same_shape(target, &v) {
            return Err("E_SHAPE: size or type mismatch in assignment".into());
        }
        *target = v;
        return Ok(());
    }
    match target {
        Val::Array(a) => match resolve(&idx[0], a.len())? {
            Sel::One(i) => assign_val(&mut a[i], &idx[1..], value),
            Sel::Many(is) => {
                let vs = match value {
                    Val::Array(v) if v.len() == is.len() => v,
                    _ => return Err("E_SHAPE: size mismatch in multi index assignment".into()),
                };
                for (k, &i) in is.iter().enumerate() {
                    assign_val(&mut a[i], &idx[1..], &vs[k])?;
                }
                Ok(())
            }
        },
        Val::Vector(x) | Val::RowVector(x) => {
            if idx.len() > 1 {
                return Err(too_many());
            }
            match resolve(&idx[0], x.len())? {
                Sel::One(i) => x[i] = scalar_of(value)?,
                Sel::Many(is) => {
                    let vf = value.flat();
                    if vf.len() != is.len() {
                        return Err("E_SHAPE: size mismatch in multi index assignment".into());
                    }
                    for (k, &i) in is.iter().enumerate() {
                        x[i] = vf[k];
                    }
                }
            }
            Ok(())
        }
        Val::Matrix { r, c, d } => {
            if idx.len() > 2 {
                return Err(too_many());
            }
            let rows = match resolve(&idx[0], *r)? {
                Sel::One(i) => vec![i],
                Sel::Many(v) => v,
            };
            let cols = if idx.len() == 2 {
                match resolve(&idx[1], *c)? {
                    Sel::One(j) => vec![j],
                    Sel::Many(v) => v,
                }
            } else {
                (0..*c).collect()
            };
            let vf = value.flat();
            if vf.len() != rows.len() * cols.len() {
                return Err("E_SHAPE: size mismatch in matrix assignment".into());
            }
            for (cj, &j) in cols.iter().enumerate() {
                for (rk, &i) in rows.iter().enumerate() {
                    d[j * *r + i] = vf[cj * rows.len() + rk];
                }
            }
            Ok(())
        }
        _ => Err(too_many()),
    }
}

fn make(base: Base, adims: &[usize], esz: &[usize]) -> Val {
    if let Some((&n, rest)) = adims.split_first() {
        return Val::Array((0..n).map(|_| make(base, rest, esz)).collect());
    }
    let nan = S::c(f64::NAN);
    match base {
        Base::Int => Val::Int(i64::MIN),
        Base::Real => Val::Real(nan),
        Base::Vector => Val::Vector(vec![nan; esz[0]]),
        Base::RowVector => Val::RowVector(vec![nan; esz[0]]),
        Base::Matrix => Val::Matrix { r: esz[0], c: esz[1], d: vec![nan; esz[0] * esz[1]] },
    }
}

fn fmt_val(v: &Val) -> String {
    match v {
        Val::Int(i) => i.to_string(),
        Val::Real(s) => format!("{}", s.v),
        _ => {
            let f: Vec<String> = v.flat().iter().map(|s| format!("{}", s.v)).collect();
            format!("[{}]", f.join(", "))
        }
    }
}

pub enum Flow {
    Normal,
    Break,
    Continue,
}

impl Runtime {
    pub fn new(prog: Program) -> Runtime {
        let mut slots = vec![Val::Int(0); prog.nslots];
        for (k, v) in &prog.init {
            slots[*k] = v.clone();
        }
        Runtime { prog, slots, tape: Tape::new(), target: Vec::new(), jacobian: true, propto: false, uvars: Vec::new(), ucur: 0, error: None }
    }

    fn val(&self, k: usize) -> &Val {
        &self.slots[k]
    }

    fn int(&self, k: usize) -> R<i64> {
        self.slots[k].as_int().map_err(|_| "E_INTERNAL: expected an integer".to_string())
    }

    fn sizes(&self, args: &[usize]) -> R<Vec<usize>> {
        args.iter()
            .map(|&k| {
                let n = self.int(k)?;
                if n < 0 {
                    Err(format!("E_DOMAIN: size must be nonnegative, found {n}"))
                } else {
                    Ok(n as usize)
                }
            })
            .collect()
    }

    fn raw_idx(&self, kinds: &[IdxKind], args: &[usize]) -> R<Vec<Raw>> {
        let mut it = args.iter();
        let mut out = Vec::with_capacity(kinds.len());
        let mut next = || *it.next().expect("index argument");
        for k in kinds {
            out.push(match k {
                IdxKind::Single => Raw::Single(self.int(next())?),
                IdxKind::Multi => Raw::Multi(self.slots[next()].ints().map_err(|e| format!("E_INTERNAL: {e}"))?),
                IdxKind::Range { lo, hi } => {
                    let l = if *lo { Some(self.int(next())?) } else { None };
                    let h = if *hi { Some(self.int(next())?) } else { None };
                    Raw::Range(l, h)
                }
                IdxKind::Omni => Raw::Omni,
            });
        }
        Ok(out)
    }

    fn target_sum(&mut self) -> S {
        let v = self.target.iter().map(|s| s.v).sum();
        let ones = vec![1.0; self.target.len()];
        self.tape.node_many(v, &self.target, &ones)
    }

    /// Bound argument in slot `k` for a variable with `n` scalars; element
    /// sized bounds of an array broadcast over the array, as in the R backend.
    fn bound(&self, k: Option<usize>, n: usize) -> R<Option<Vec<S>>> {
        let k = match k {
            Some(k) => k,
            None => return Ok(None),
        };
        let f = self.slots[k].flat();
        if f.is_empty() || (f.len() != 1 && n % f.len() != 0) {
            return Err(format!("E_SHAPE: a bound has {} elements for a variable with {n}", f.len()));
        }
        Ok(Some(f))
    }

    fn param(&mut self, ty: &DeclType, b: &Bounds, args: &[usize]) -> R<Val> {
        let ns = ty.nadims + ty.base.nsizes();
        let sz = self.sizes(&args[..ns])?;
        let template = make(ty.base, &sz[..ty.nadims], &sz[ty.nadims..]);
        let n = template.size();
        if self.ucur + n > self.uvars.len() {
            return Err(format!("E_INTERNAL: the model has more than {} unconstrained parameters", self.uvars.len()));
        }
        let bargs = &args[ns..];
        let mut bi = 0;
        let mut take = |on: bool| {
            if on {
                bi += 1;
                Some(bargs[bi - 1])
            } else {
                None
            }
        };
        let (lo, hi, off, mul) = (take(b.lower), take(b.upper), take(b.offset), take(b.multiplier));
        let (lo, hi, off, mul) = (self.bound(lo, n)?, self.bound(hi, n)?, self.bound(off, n)?, self.bound(mul, n)?);
        let at = |v: &Vec<S>, i: usize| v[i % v.len()];
        let has_om = off.is_some() || mul.is_some();
        let mut xs = Vec::with_capacity(n);
        for i in 0..n {
            let u = self.uvars[self.ucur + i];
            let t = &mut self.tape;
            let (x, lj) = if has_om {
                let o = off.as_ref().map_or(S::c(0.0), |v| at(v, i));
                let m = mul.as_ref().map_or(S::c(1.0), |v| at(v, i));
                if !(m.v > 0.0) || !m.v.is_finite() {
                    return Err(format!("E_DOMAIN: multiplier is {}, but must be positive finite", m.v));
                }
                let x = t.node(o.v + m.v * u.v, &[(u, m.v), (m, u.v), (o, 1.0)]);
                let lj = t.node(m.v.ln(), &[(m, 1.0 / m.v)]);
                (x, Some(lj))
            } else {
                let l = lo.as_ref().map(|v| at(v, i));
                let h = hi.as_ref().map(|v| at(v, i));
                // -inf and +inf mean no bound; NaN and bounds that exclude every value do not
                if let Some(l) = l { if l.v.is_nan() || l.v == f64::INFINITY {
                    return Err(format!("E_DOMAIN: lower bound is {}, but must be less than infinity", l.v)); } }
                if let Some(h) = h { if h.v.is_nan() || h.v == f64::NEG_INFINITY {
                    return Err(format!("E_DOMAIN: upper bound is {}, but must be greater than negative infinity", h.v)); } }
                let l = l.filter(|s| s.v.is_finite());
                let h = h.filter(|s| s.v.is_finite());
                match (l, h) {
                    (Some(l), Some(h)) => {
                        let w = h.v - l.v;
                        if !(w > 0.0) {
                            return Err(format!("E_DOMAIN: lower bound {} is not less than upper bound {}", l.v, h.v));
                        }
                        let il = inv_logit(u.v);
                        let x = t.node(l.v + w * il, &[(u, w * il * (1.0 - il)), (l, 1.0 - il), (h, il)]);
                        let ljv = w.ln() - log1p_exp(-u.v) - log1p_exp(u.v);
                        let lj = t.node(ljv, &[(u, 1.0 - 2.0 * il), (l, -1.0 / w), (h, 1.0 / w)]);
                        (x, Some(lj))
                    }
                    (Some(l), None) => {
                        let e = u.v.exp();
                        (t.node(l.v + e, &[(u, e), (l, 1.0)]), Some(u))
                    }
                    (None, Some(h)) => {
                        let e = u.v.exp();
                        (t.node(h.v - e, &[(u, -e), (h, 1.0)]), Some(u))
                    }
                    (None, None) => (u, None),
                }
            };
            xs.push(x);
            if self.jacobian {
                if let Some(lj) = lj {
                    self.target.push(lj);
                }
            }
        }
        self.ucur += n;
        Ok(template.with_scalars(&mut xs.into_iter()))
    }

    fn check(&self, b: &Bounds, name: &str, args: &[usize]) -> R<()> {
        let x = self.slots[args[0]].flat();
        let n = x.len();
        let mut k = 1;
        let lo = if b.lower { k += 1; Some(args[k - 1]) } else { None };
        let hi = if b.upper { Some(args[k]) } else { None };
        let (lo, hi) = (self.bound(lo, n)?, self.bound(hi, n)?);
        for (i, s) in x.iter().enumerate() {
            if s.v.is_nan() {
                return Err(format!("E_DOMAIN: transformed parameters variable '{name}' is nan at element {}", i + 1));
            }
            if let Some(l) = &lo {
                let l = l[i % l.len()];
                if s.v < l.v {
                    return Err(format!("E_DOMAIN: transformed parameters variable '{name}' is {}, but must be greater than or equal to {}", s.v, l.v));
                }
            }
            if let Some(h) = &hi {
                let h = h[i % h.len()];
                if s.v > h.v {
                    return Err(format!("E_DOMAIN: transformed parameters variable '{name}' is {}, but must be less than or equal to {}", s.v, h.v));
                }
            }
        }
        Ok(())
    }

    /// Execute operation `i`.
    pub fn op(&mut self, i: usize) -> R<()> {
        // ops are immutable during a run; take a raw pointer to avoid cloning
        let op: *const Op = &self.prog.ops[i];
        let op = unsafe { &*op };
        let a = &op.args;
        let v = match &op.kind {
            OpKind::Copy => self.val(a[0]).clone(),
            OpKind::Bin(b) => math::binop(&mut self.tape, b.symbol(), &self.slots[a[0]], &self.slots[a[1]]).map_err(rt)?,
            OpKind::Neg => math::unop(&mut self.tape, "-", &self.slots[a[0]]).map_err(rt)?,
            OpKind::Not => math::unop(&mut self.tape, "!", &self.slots[a[0]]).map_err(rt)?,
            OpKind::Fn1(f) => math::map1(&mut self.tape, &self.slots[a[0]], *f),
            OpKind::Fn2(f) => math::map2(&mut self.tape, &self.slots[a[0]], &self.slots[a[1]], *f).map_err(rt)?,
            OpKind::Call(f) => {
                let args: Vec<&Val> = a.iter().map(|&k| &self.slots[k]).collect();
                f(&mut self.tape, &args).map_err(rt)?
            }
            OpKind::GetTarget => Val::Real(self.target_sum()),
            OpKind::Lpdf { dist: d, lu, discrete } => {
                let args: Vec<&Val> = a.iter().map(|&k| &self.slots[k]).collect();
                let propto = *lu && self.propto;
                let s = if *discrete { dist::lpmf(&mut self.tape, d, &args, propto) } else { dist::lpdf(&mut self.tape, d, &args, propto) };
                Val::Real(s.map_err(rt)?)
            }
            OpKind::Index(kinds) => {
                let raw = self.raw_idx(kinds, &a[1..])?;
                index_val(&self.slots[a[0]], &raw)?
            }
            OpKind::VecLit => Val::Vector(a.iter().map(|&k| scalar_of(&self.slots[k])).collect::<R<_>>()?),
            OpKind::RowVecLit => {
                if a.iter().all(|&k| self.slots[k].is_scalar()) {
                    Val::RowVector(a.iter().map(|&k| scalar_of(&self.slots[k])).collect::<R<_>>()?)
                } else {
                    let rows: Vec<Vec<S>> = a.iter().map(|&k| self.slots[k].flat()).collect();
                    let c = rows.first().map(|r| r.len()).unwrap_or(0);
                    if rows.iter().any(|r| r.len() != c) {
                        return Err("E_SHAPE: rows of a matrix literal differ in size".into());
                    }
                    let mut d = Vec::with_capacity(rows.len() * c);
                    for j in 0..c {
                        for r in &rows {
                            d.push(r[j]);
                        }
                    }
                    Val::Matrix { r: rows.len(), c, d }
                }
            }
            OpKind::ArrLit => {
                let items: Vec<Val> = a.iter().map(|&k| self.slots[k].clone()).collect();
                if items.iter().any(|x| !matches!(x, Val::Int(_))) {
                    Val::Array(items.into_iter().map(|x| if let Val::Int(i) = x { Val::real(i as f64) } else { x }).collect())
                } else {
                    Val::Array(items)
                }
            }
            OpKind::Transpose => match &self.slots[a[0]] {
                Val::Vector(x) => Val::RowVector(x.clone()),
                Val::RowVector(x) => Val::Vector(x.clone()),
                Val::Matrix { r, c, d } => {
                    let mut out = Vec::with_capacity(d.len());
                    for i in 0..*r {
                        for j in 0..*c {
                            out.push(d[j * r + i]);
                        }
                    }
                    Val::Matrix { r: *c, c: *r, d: out }
                }
                _ => return Err("E_INTERNAL: transpose of a non matrix value".into()),
            },
            OpKind::Decl(ty) => {
                let sz = self.sizes(a)?;
                make(ty.base, &sz[..ty.nadims], &sz[ty.nadims..])
            }
            OpKind::Assign(kinds) => {
                let raw = self.raw_idx(kinds, &a[2..])?;
                let mut t = std::mem::replace(&mut self.slots[a[0]], Val::Int(0));
                let r = if a[1] == a[0] {
                    // self assignment (e.g. a declaration's own slot): nothing to do
                    Ok(())
                } else {
                    assign_val(&mut t, &raw, &self.slots[a[1]])
                };
                self.slots[a[0]] = t;
                return r;
            }
            OpKind::Target => {
                let s = self.slots[a[0]].flat();
                if s.len() == 1 {
                    self.target.push(s[0]);
                } else {
                    let v = s.iter().map(|x| x.v).sum();
                    let n = self.tape.node_many(v, &s, &vec![1.0; s.len()]);
                    self.target.push(n);
                }
                return Ok(());
            }
            OpKind::Jacobian => {
                if self.jacobian {
                    let s = self.slots[a[0]].flat();
                    let v = s.iter().map(|x| x.v).sum();
                    let n = self.tape.node_many(v, &s, &vec![1.0; s.len()]);
                    self.target.push(n);
                }
                return Ok(());
            }
            OpKind::Param { ty, bounds, .. } => self.param(ty, bounds, a)?,
            OpKind::Check { bounds, name } => {
                return self.check(bounds, name, a);
            }
            OpKind::Reject { msg, fatal } => {
                let mut m = String::new();
                let mut parts = msg.split(crate::kernel::REJECT_SLOT);
                m.push_str(parts.next().unwrap_or(""));
                for (k, p) in parts.enumerate() {
                    m.push_str(&fmt_val(&self.slots[a[k]]));
                    m.push_str(p);
                }
                return Err(format!("{}: {m}", if *fatal { "E_FATAL" } else { "E_REJECT" }));
            }
        };
        self.slots[op.dst] = v;
        Ok(())
    }

    pub fn truthy(&self, k: usize) -> R<bool> {
        match &self.slots[k] {
            Val::Int(i) => Ok(*i != 0),
            Val::Real(s) => Ok(s.v != 0.0),
            _ => Err("E_INTERNAL: condition must be a scalar".into()),
        }
    }

    pub fn get_int(&self, k: usize) -> R<i64> {
        self.int(k)
    }

    pub fn set_int(&mut self, k: usize, v: i64) {
        self.slots[k] = Val::Int(v);
    }

    /// Number of iterations of a foreach loop over slot `k`.
    pub fn iter_len(&self, k: usize) -> R<usize> {
        Ok(match &self.slots[k] {
            Val::Array(a) => a.len(),
            Val::Vector(x) | Val::RowVector(x) => x.len(),
            Val::Matrix { d, .. } => d.len(),
            _ => return Err("E_INTERNAL: foreach over a scalar".into()),
        })
    }

    /// Set loop variable `var` to element `i` of container slot `k`.
    pub fn iter_set(&mut self, var: usize, k: usize, i: usize) {
        let v = match &self.slots[k] {
            Val::Array(a) => a[i].clone(),
            Val::Vector(x) | Val::RowVector(x) => Val::Real(x[i]),
            Val::Matrix { d, .. } => Val::Real(d[i]),
            _ => unreachable!(),
        };
        self.slots[var] = v;
    }

    /// Reset per evaluation state and create the independent variables.
    pub fn begin(&mut self, u: &[f64], jacobian: bool, propto: bool) {
        self.tape.clear();
        self.target.clear();
        self.jacobian = jacobian;
        self.propto = propto;
        self.uvars.clear();
        for &x in u {
            let v = self.tape.var(x);
            self.uvars.push(v);
        }
        self.ucur = 0;
        self.error = None;
    }

    /// Finish an evaluation: the log density and, if requested, its gradient.
    pub fn finish(&mut self, grad: Option<&mut [f64]>) -> R<f64> {
        if self.ucur != self.uvars.len() {
            return Err(format!("E_INTERNAL: the model has {} unconstrained parameters, but {} were supplied", self.ucur, self.uvars.len()));
        }
        let out = self.target_sum();
        if let Some(g) = grad {
            let gv = self.tape.gradient(out, self.uvars.len());
            g.copy_from_slice(&gv);
        }
        Ok(out.v)
    }

    fn run(&mut self, s: &Stmt) -> R<Flow> {
        match s {
            Stmt::Op(i) => {
                self.op(*i)?;
            }
            Stmt::Block(v) => {
                for x in v {
                    match self.run(x)? {
                        Flow::Normal => {}
                        f => return Ok(f),
                    }
                }
            }
            Stmt::For { var, lo, hi, body } => {
                let (l, h) = (self.int(*lo)?, self.int(*hi)?);
                let mut i = l;
                while i <= h {
                    self.slots[*var] = Val::Int(i);
                    if let Flow::Break = self.run(body)? {
                        break;
                    }
                    i += 1;
                }
            }
            Stmt::ForEach { var, container, body } => {
                let n = self.iter_len(*container)?;
                for i in 0..n {
                    self.iter_set(*var, *container, i);
                    if let Flow::Break = self.run(body)? {
                        break;
                    }
                }
            }
            Stmt::While { pre, cond, body } => loop {
                self.run(pre)?;
                if !self.truthy(*cond)? {
                    break;
                }
                if let Flow::Break = self.run(body)? {
                    break;
                }
            },
            Stmt::If { cond, then, els } => {
                return if self.truthy(*cond)? { self.run(then) } else { self.run(els) };
            }
            Stmt::Break => return Ok(Flow::Break),
            Stmt::Continue => return Ok(Flow::Continue),
        }
        Ok(Flow::Normal)
    }

    /// Reference interpreter: log density and gradient at `u`.
    pub fn interpret(&mut self, u: &[f64], jacobian: bool, propto: bool, grad: Option<&mut [f64]>) -> R<f64> {
        self.begin(u, jacobian, propto);
        let body: *const Stmt = &self.prog.body;
        self.run(unsafe { &*body })?;
        self.finish(grad)
    }
}

/// Errors without a code are internal errors (they abort, they do not reject).
fn rt(e: String) -> String {
    if e.starts_with("E_") { e } else { format!("E_INTERNAL: {e}") }
}
