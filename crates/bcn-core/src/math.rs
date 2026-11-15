//! Arithmetic and elementwise functions on runtime values.

use crate::special as sp;
use crate::tape::{Tape, S};
use crate::val::Val;

pub type R<T> = Result<T, String>;

/// Value and derivative of a unary function.
pub type Unary = fn(f64) -> (f64, f64);
/// Value and partial derivatives of a binary function.
pub type Binary = fn(f64, f64) -> (f64, f64, f64);

pub fn un(t: &mut Tape, x: S, f: Unary) -> S {
    let (v, d) = f(x.v);
    t.node(v, &[(x, d)])
}

pub fn bin(t: &mut Tape, a: S, b: S, f: Binary) -> S {
    let (v, da, db) = f(a.v, b.v);
    t.node(v, &[(a, da), (b, db)])
}

/// Apply a unary function elementwise.
pub fn map1(t: &mut Tape, x: &Val, f: Unary) -> Val {
    match x {
        Val::Int(i) => Val::Real(S::c(f(*i as f64).0)),
        Val::Real(s) => Val::Real(un(t, *s, f)),
        _ => {
            let v: Vec<S> = x.flat().into_iter().map(|s| un(t, s, f)).collect();
            x.with_scalars(&mut v.into_iter())
        }
    }
}

/// Apply a binary function elementwise with scalar broadcasting.
pub fn map2(t: &mut Tape, a: &Val, b: &Val, f: Binary) -> R<Val> {
    match (a.as_scalar(), b.as_scalar()) {
        (Some(x), Some(y)) => Ok(Val::Real(bin(t, x, y, f))),
        (Some(x), None) => {
            let v: Vec<S> = b.flat().into_iter().map(|y| bin(t, x, y, f)).collect();
            Ok(b.with_scalars(&mut v.into_iter()))
        }
        (None, Some(y)) => {
            let v: Vec<S> = a.flat().into_iter().map(|x| bin(t, x, y, f)).collect();
            Ok(a.with_scalars(&mut v.into_iter()))
        }
        (None, None) => {
            if a.shape() != b.shape() {
                return Err(format!("E_SHAPE: size mismatch in elementwise operation ({:?} and {:?})", a.shape(), b.shape()));
            }
            let (fa, fb) = (a.flat(), b.flat());
            let v: Vec<S> = fa.into_iter().zip(fb).map(|(x, y)| bin(t, x, y, f)).collect();
            Ok(a.with_scalars(&mut v.into_iter()))
        }
    }
}

fn add(a: f64, b: f64) -> (f64, f64, f64) {
    (a + b, 1.0, 1.0)
}
fn sub(a: f64, b: f64) -> (f64, f64, f64) {
    (a - b, 1.0, -1.0)
}
fn mul(a: f64, b: f64) -> (f64, f64, f64) {
    (a * b, b, a)
}
fn div(a: f64, b: f64) -> (f64, f64, f64) {
    (a / b, 1.0 / b, -a / (b * b))
}
fn pow(a: f64, b: f64) -> (f64, f64, f64) {
    let v = a.powf(b);
    let da = if b == 0.0 { 0.0 } else { b * a.powf(b - 1.0) };
    let db = if a > 0.0 { v * a.ln() } else { 0.0 };
    (v, da, db)
}

const OVERFLOW: &str = "E_DOMAIN: integer overflow";

/// Stan integers are 32 bit; as in the R backend, results outside
/// [-2147483647, 2147483647] are an error.
pub fn int_range(x: Option<i64>) -> R<i64> {
    match x {
        Some(v) if (-2_147_483_647..=2_147_483_647).contains(&v) => Ok(v),
        _ => Err(OVERFLOW.into()),
    }
}

fn int_binop(op: &str, a: i64, b: i64) -> R<Val> {
    if a == i64::MIN || b == i64::MIN {
        return Err("E_DOMAIN: integer variable used before it is assigned".into());
    }
    Ok(match op {
        "+" => Val::Int(int_range(a.checked_add(b))?),
        "-" => Val::Int(int_range(a.checked_sub(b))?),
        "*" => Val::Int(int_range(a.checked_mul(b))?),
        "/" | "%/%" => {
            if b == 0 {
                return Err("E_DOMAIN: integer division by zero".into());
            }
            Val::Int(a / b)
        }
        "%" => {
            if b == 0 {
                return Err("E_DOMAIN: modulus by zero".into());
            }
            Val::Int(a % b)
        }
        "^" => Val::real((a as f64).powf(b as f64)),
        _ => return cmp_binop(op, a as f64, b as f64),
    })
}

fn cmp_binop(op: &str, a: f64, b: f64) -> R<Val> {
    let r = match op {
        "<" => a < b,
        "<=" => a <= b,
        ">" => a > b,
        ">=" => a >= b,
        "==" => a == b,
        "!=" => a != b,
        "&&" => a != 0.0 && b != 0.0,
        "||" => a != 0.0 || b != 0.0,
        _ => return Err(format!("unsupported operator '{op}'")),
    };
    Ok(Val::Int(r as i64))
}

pub fn binop(t: &mut Tape, op: &str, a: &Val, b: &Val) -> R<Val> {
    if let (Val::Int(x), Val::Int(y)) = (a, b) {
        return int_binop(op, *x, *y);
    }
    if matches!(op, "<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||") {
        let (x, y) = (a.as_scalar().ok_or("comparison of non scalars")?, b.as_scalar().ok_or("comparison of non scalars")?);
        return cmp_binop(op, x.v, y.v);
    }
    match op {
        "+" => map2(t, a, b, add),
        "-" => map2(t, a, b, sub),
        ".*" => map2(t, a, b, mul),
        "./" => map2(t, a, b, div),
        "^" | ".^" => map2(t, a, b, pow),
        "/" => {
            if b.is_scalar() {
                map2(t, a, b, div)
            } else {
                Err("matrix division is not supported by the native backend".into())
            }
        }
        "*" => multiply(t, a, b),
        _ => Err(format!("unsupported operator '{op}'")),
    }
}

pub fn unop(t: &mut Tape, op: &str, a: &Val) -> R<Val> {
    match (op, a) {
        ("-", Val::Int(i)) => Ok(Val::Int(-i)),
        ("+", _) => Ok(a.clone()),
        ("-", _) => Ok(map1(t, a, |x| (-x, -1.0))),
        ("!", _) => Ok(Val::Int((a.as_scalar().ok_or("'!' needs a scalar")?.v == 0.0) as i64)),
        _ => Err(format!("unsupported unary operator '{op}'")),
    }
}

/// Dot product as one tape node.
pub fn dot(t: &mut Tape, a: &[S], b: &[S]) -> R<S> {
    if a.len() != b.len() {
        return Err(format!("E_SHAPE: size mismatch in multiplication ({} and {})", a.len(), b.len()));
    }
    let mut v = 0.0;
    let mut parents = Vec::with_capacity(2 * a.len());
    let mut partials = Vec::with_capacity(2 * a.len());
    for (x, y) in a.iter().zip(b) {
        v += x.v * y.v;
        parents.push(*x);
        partials.push(y.v);
        parents.push(*y);
        partials.push(x.v);
    }
    Ok(t.node_many(v, &parents, &partials))
}

fn col(d: &[S], r: usize, j: usize) -> &[S] {
    &d[j * r..(j + 1) * r]
}

pub fn multiply(t: &mut Tape, a: &Val, b: &Val) -> R<Val> {
    if a.is_scalar() || b.is_scalar() {
        return map2(t, a, b, mul);
    }
    match (a, b) {
        (Val::RowVector(x), Val::Vector(y)) => Ok(Val::Real(dot(t, x, y)?)),
        (Val::Vector(x), Val::RowVector(y)) => {
            let mut d = Vec::with_capacity(x.len() * y.len());
            for yj in y {
                for xi in x {
                    d.push(bin(t, *xi, *yj, mul));
                }
            }
            Ok(Val::Matrix { r: x.len(), c: y.len(), d })
        }
        (Val::Matrix { r, c, d }, Val::Vector(y)) => {
            if *c != y.len() {
                return Err("E_SHAPE: size mismatch in matrix times vector".into());
            }
            let mut out = Vec::with_capacity(*r);
            let mut row = vec![S::c(0.0); *c];
            for i in 0..*r {
                for j in 0..*c {
                    row[j] = d[j * r + i];
                }
                out.push(dot(t, &row, y)?);
            }
            Ok(Val::Vector(out))
        }
        (Val::RowVector(x), Val::Matrix { r, c, d }) => {
            if *r != x.len() {
                return Err("E_SHAPE: size mismatch in row vector times matrix".into());
            }
            let mut out = Vec::with_capacity(*c);
            for j in 0..*c {
                out.push(dot(t, x, col(d, *r, j))?);
            }
            Ok(Val::RowVector(out))
        }
        (Val::Matrix { r: ra, c: ca, d: da }, Val::Matrix { r: rb, c: cb, d: db }) => {
            if ca != rb {
                return Err("E_SHAPE: size mismatch in matrix multiplication".into());
            }
            let mut out = Vec::with_capacity(ra * cb);
            let mut row = vec![S::c(0.0); *ca];
            for j in 0..*cb {
                for i in 0..*ra {
                    for k in 0..*ca {
                        row[k] = da[k * ra + i];
                    }
                    out.push(dot(t, &row, col(db, *rb, j))?);
                }
            }
            Ok(Val::Matrix { r: *ra, c: *cb, d: out })
        }
        _ => Err("unsupported operand types for '*'".into()),
    }
}

/// Elementwise functions of one argument: value and derivative.
pub fn unary_fn(name: &str) -> Option<Unary> {
    Some(match name {
        "exp" => |x: f64| {
            let e = x.exp();
            (e, e)
        },
        "log" => |x: f64| (x.ln(), 1.0 / x),
        "sqrt" => |x: f64| {
            let s = x.sqrt();
            (s, 0.5 / s)
        },
        "square" => |x: f64| (x * x, 2.0 * x),
        "inv" => |x: f64| (1.0 / x, -1.0 / (x * x)),
        "inv_sqrt" => |x: f64| (1.0 / x.sqrt(), -0.5 / (x * x.sqrt())),
        "inv_square" => |x: f64| (1.0 / (x * x), -2.0 / (x * x * x)),
        "fabs" | "abs" => |x: f64| (x.abs(), if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
        "log1p" => |x: f64| (x.ln_1p(), 1.0 / (1.0 + x)),
        "expm1" => |x: f64| (x.exp_m1(), x.exp()),
        "log1m" => |x: f64| ((-x).ln_1p(), -1.0 / (1.0 - x)),
        "log1p_exp" => |x: f64| (sp::log1p_exp(x), sp::inv_logit(x)),
        "log1m_exp" => |x: f64| (if x > -std::f64::consts::LN_2 { (-(x.exp_m1())).ln() } else { (-x.exp()).ln_1p() }, -1.0 / (-x).exp_m1()),
        "inv_logit" | "logistic" => |x: f64| {
            let p = sp::inv_logit(x);
            (p, p * (1.0 - p))
        },
        "logit" => |x: f64| ((x / (1.0 - x)).ln(), 1.0 / (x * (1.0 - x))),
        "log_inv_logit" => |x: f64| (sp::log_inv_logit(x), 1.0 - sp::inv_logit(x)),
        "log1m_inv_logit" => |x: f64| (sp::log1m_inv_logit(x), -sp::inv_logit(x)),
        "sin" => |x: f64| (x.sin(), x.cos()),
        "cos" => |x: f64| (x.cos(), -x.sin()),
        "tan" => |x: f64| {
            let v = x.tan();
            (v, 1.0 + v * v)
        },
        "asin" => |x: f64| (x.asin(), 1.0 / (1.0 - x * x).sqrt()),
        "acos" => |x: f64| (x.acos(), -1.0 / (1.0 - x * x).sqrt()),
        "atan" => |x: f64| (x.atan(), 1.0 / (1.0 + x * x)),
        "sinh" => |x: f64| (x.sinh(), x.cosh()),
        "cosh" => |x: f64| (x.cosh(), x.sinh()),
        "tanh" => |x: f64| {
            let v = x.tanh();
            (v, 1.0 - v * v)
        },
        "asinh" => |x: f64| (x.asinh(), 1.0 / (x * x + 1.0).sqrt()),
        "acosh" => |x: f64| (x.acosh(), 1.0 / (x * x - 1.0).sqrt()),
        "atanh" => |x: f64| (x.atanh(), 1.0 / (1.0 - x * x)),
        "lgamma" => |x: f64| (sp::lgamma(x), sp::digamma(x)),
        "digamma" => |x: f64| (sp::digamma(x), trigamma(x)),
        "Phi" => |x: f64| (sp::phi(x), (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()),
        "Phi_approx" => |x: f64| {
            let z = 0.07056 * x * x * x + 1.5976 * x;
            let p = sp::inv_logit(z);
            (p, p * (1.0 - p) * (3.0 * 0.07056 * x * x + 1.5976))
        },
        "std_normal_log_qf" | "inv_Phi" => return None,
        "exp2" => |x: f64| {
            let v = x.exp2();
            (v, v * std::f64::consts::LN_2)
        },
        "log2" => |x: f64| (x.log2(), 1.0 / (x * std::f64::consts::LN_2)),
        "log10" => |x: f64| (x.log10(), 1.0 / (x * std::f64::consts::LN_10)),
        "cbrt" => |x: f64| {
            let v = x.cbrt();
            (v, 1.0 / (3.0 * v * v))
        },
        "erf" => |x: f64| (sp::erf(x), 2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp()),
        "erfc" => |x: f64| (sp::erfc(x), -2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp()),
        "floor" => |x: f64| (x.floor(), 0.0),
        "ceil" => |x: f64| (x.ceil(), 0.0),
        "round" => |x: f64| (x.round(), 0.0),
        "trunc" => |x: f64| (x.trunc(), 0.0),
        _ => return None,
    })
}

pub fn trigamma(mut x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::INFINITY;
        }
        // reflection: psi1(x) + psi1(1 - x) = pi^2 / sin(pi x)^2
        let s = (std::f64::consts::PI * x).sin();
        return std::f64::consts::PI * std::f64::consts::PI / (s * s) - trigamma(1.0 - x);
    }
    let mut r = 0.0;
    while x < 12.0 {
        r += 1.0 / (x * x);
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    r + 1.0 / x + f / 2.0 + f / x * (1.0 / 6.0 - f * (1.0 / 30.0 - f * (1.0 / 42.0 - f / 30.0)))
}

/// Elementwise functions of two arguments.
pub fn binary_fn(name: &str) -> Option<Binary> {
    Some(match name {
        "pow" => pow,
        // as C's fmin and fmax, a NaN argument gives the other one
        "fmin" => |a: f64, b: f64| if b.is_nan() || a <= b { (a, 1.0, 0.0) } else { (b, 0.0, 1.0) },
        "fmax" => |a: f64, b: f64| if b.is_nan() || a >= b { (a, 1.0, 0.0) } else { (b, 0.0, 1.0) },
        "hypot" => |a: f64, b: f64| {
            let h = a.hypot(b);
            (h, a / h, b / h)
        },
        "atan2" => |a: f64, b: f64| {
            let d = a * a + b * b;
            (a.atan2(b), b / d, -a / d)
        },
        "fdim" => |a: f64, b: f64| if a > b { (a - b, 1.0, -1.0) } else { (0.0, 0.0, 0.0) },
        "lbeta" => |a: f64, b: f64| {
            let s = sp::digamma(a + b);
            (sp::lbeta(a, b), sp::digamma(a) - s, sp::digamma(b) - s)
        },
        "lchoose" => |n: f64, k: f64| {
            (sp::lchoose(n, k), sp::digamma(n + 1.0) - sp::digamma(n - k + 1.0), -sp::digamma(k + 1.0) + sp::digamma(n - k + 1.0))
        },
        "log_diff_exp" => |a: f64, b: f64| {
            // log(exp(a) - exp(b)) = a + log1m_exp(b - a), stable for b close to a
            let d = b - a;
            if d > 0.0 || a.is_nan() || b.is_nan() { return (f64::NAN, f64::NAN, f64::NAN); }
            if b == f64::NEG_INFINITY { return (a, 1.0, 0.0); }
            let l1m = if d > -std::f64::consts::LN_2 { (-d.exp_m1()).ln() } else { (-d.exp()).ln_1p() };
            // derivatives 1 / (1 - e) and -e / (1 - e) with e = exp(d), via expm1
            let om = -d.exp_m1();
            (a + l1m, 1.0 / om, -d.exp() / om)
        },
        "log_sum_exp" => |a: f64, b: f64| {
            let m = a.max(b);
            let v = m + ((a - m).exp() + (b - m).exp()).ln();
            ((v), (a - v).exp(), (b - v).exp())
        },
        "fmod" => |a: f64, b: f64| (a % b, 1.0, -(a / b).trunc()),
        "binary_log_loss" => return None,
        _ => return None,
    })
}

/// Reductions to one scalar node.
pub fn sum(t: &mut Tape, x: &Val) -> Val {
    if let Val::Int(_) = x {
        return x.clone();
    }
    if x.is_int_like() {
        return Val::Int(x.ints().unwrap().iter().sum());
    }
    let f = x.flat();
    let v = f.iter().map(|s| s.v).sum();
    Val::Real(t.node_many(v, &f, &vec![1.0; f.len()]))
}

pub fn log_sum_exp(t: &mut Tape, x: &[S]) -> S {
    let m = x.iter().map(|s| s.v).fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return S::c(m);
    }
    let s: f64 = x.iter().map(|s| (s.v - m).exp()).sum();
    let v = m + s.ln();
    let partials: Vec<f64> = x.iter().map(|s| (s.v - v).exp()).collect();
    t.node_many(v, x, &partials)
}
