//! Built in functions other than elementwise math and densities.

use crate::math::{self, R};
use crate::tape::{Tape, S};
use crate::val::Val;

pub type CallFn = fn(&mut Tape, &[&Val]) -> R<Val>;

fn int_of(v: &Val, f: &str) -> R<i64> {
    v.as_int().map_err(|_| format!("{f}: expected an integer argument"))
}

fn size_of(v: &Val, f: &str) -> R<usize> {
    let n = int_of(v, f)?;
    if n < 0 {
        return Err(format!("E_DOMAIN: {f}: size must be nonnegative, found {n}"));
    }
    Ok(n as usize)
}

fn scalar(v: &Val, f: &str) -> R<S> {
    v.as_scalar().ok_or_else(|| format!("{f}: expected a scalar argument"))
}

fn sum(t: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(math::sum(t, a[0]))
}

fn mean(t: &mut Tape, a: &[&Val]) -> R<Val> {
    let f = a[0].flat();
    if f.is_empty() {
        return Err("E_DOMAIN: mean: argument must not be empty".into());
    }
    let n = f.len() as f64;
    let v = f.iter().map(|s| s.v).sum::<f64>() / n;
    Ok(Val::Real(t.node_many(v, &f, &vec![1.0 / n; f.len()])))
}

fn variance_core(t: &mut Tape, f: &[S], fname: &str) -> R<S> {
    if f.len() < 2 {
        return Err(format!("E_DOMAIN: {fname}: argument must have at least two elements"));
    }
    let n = f.len() as f64;
    let m = f.iter().map(|s| s.v).sum::<f64>() / n;
    let v = f.iter().map(|s| (s.v - m) * (s.v - m)).sum::<f64>() / (n - 1.0);
    let partials: Vec<f64> = f.iter().map(|s| 2.0 * (s.v - m) / (n - 1.0)).collect();
    Ok(t.node_many(v, f, &partials))
}

fn variance(t: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Real(variance_core(t, &a[0].flat(), "variance")?))
}

fn sd(t: &mut Tape, a: &[&Val]) -> R<Val> {
    let v = variance_core(t, &a[0].flat(), "sd")?;
    let s = v.v.sqrt();
    Ok(Val::Real(t.node(s, &[(v, 0.5 / s)])))
}

fn prod(t: &mut Tape, a: &[&Val]) -> R<Val> {
    if a[0].is_int_like() {
        return Ok(Val::Int(a[0].ints()?.iter().product()));
    }
    let f = a[0].flat();
    let v: f64 = f.iter().map(|s| s.v).product();
    let partials: Vec<f64> = (0..f.len())
        .map(|i| f.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, s)| s.v).product())
        .collect();
    Ok(Val::Real(t.node_many(v, &f, &partials)))
}

fn log_sum_exp(t: &mut Tape, a: &[&Val]) -> R<Val> {
    let mut f = Vec::new();
    for x in a {
        x.flatten(&mut f);
    }
    Ok(Val::Real(math::log_sum_exp(t, &f)))
}

fn dot_product(t: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Real(math::dot(t, &a[0].flat(), &a[1].flat())?))
}

fn dot_self(t: &mut Tape, a: &[&Val]) -> R<Val> {
    let f = a[0].flat();
    let v = f.iter().map(|s| s.v * s.v).sum();
    let partials: Vec<f64> = f.iter().map(|s| 2.0 * s.v).collect();
    Ok(Val::Real(t.node_many(v, &f, &partials)))
}

fn rep_vector(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let x = scalar(a[0], "rep_vector")?;
    Ok(Val::Vector(vec![x; size_of(a[1], "rep_vector")?]))
}

fn rep_row_vector(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let x = scalar(a[0], "rep_row_vector")?;
    Ok(Val::RowVector(vec![x; size_of(a[1], "rep_row_vector")?]))
}

fn rep_matrix(_: &mut Tape, a: &[&Val]) -> R<Val> {
    match a.len() {
        3 => {
            let x = scalar(a[0], "rep_matrix")?;
            let (r, c) = (size_of(a[1], "rep_matrix")?, size_of(a[2], "rep_matrix")?);
            Ok(Val::Matrix { r, c, d: vec![x; r * c] })
        }
        2 => {
            let n = size_of(a[1], "rep_matrix")?;
            match a[0] {
                Val::Vector(v) => {
                    let mut d = Vec::with_capacity(v.len() * n);
                    for _ in 0..n {
                        d.extend_from_slice(v);
                    }
                    Ok(Val::Matrix { r: v.len(), c: n, d })
                }
                Val::RowVector(v) => {
                    let mut d = Vec::with_capacity(v.len() * n);
                    for x in v {
                        d.extend(std::iter::repeat_n(*x, n));
                    }
                    Ok(Val::Matrix { r: n, c: v.len(), d })
                }
                _ => Err("rep_matrix: expected a vector or row vector".into()),
            }
        }
        _ => Err("rep_matrix: wrong number of arguments".into()),
    }
}

fn rep_array(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let mut out = a[0].clone();
    for d in a[1..].iter().rev() {
        let n = size_of(d, "rep_array")?;
        out = Val::Array(vec![out; n]);
    }
    Ok(out)
}

fn size(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Int(match a[0] {
        Val::Array(v) => v.len(),
        Val::Vector(v) | Val::RowVector(v) => v.len(),
        Val::Matrix { d, .. } => d.len(),
        _ => 1,
    } as i64))
}

fn num_elements(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Int(a[0].size() as i64))
}

fn rows(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Int(match a[0] {
        Val::Vector(v) => v.len(),
        Val::RowVector(_) => 1,
        Val::Matrix { r, .. } => *r,
        _ => return Err("rows: expected a vector or matrix".into()),
    } as i64))
}

fn cols(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Int(match a[0] {
        Val::Vector(_) => 1,
        Val::RowVector(v) => v.len(),
        Val::Matrix { c, .. } => *c,
        _ => return Err("cols: expected a vector or matrix".into()),
    } as i64))
}

fn to_vector(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Vector(a[0].flat()))
}

fn to_row_vector(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::RowVector(a[0].flat()))
}

fn minmax(t: &mut Tape, a: &[&Val], is_max: bool) -> R<Val> {
    let f = if is_max { "max" } else { "min" };
    if a.len() == 2 {
        if let (Val::Int(x), Val::Int(y)) = (a[0], a[1]) {
            return Ok(Val::Int(if is_max { *x.max(y) } else { *x.min(y) }));
        }
        let g = math::binary_fn(if is_max { "fmax" } else { "fmin" }).unwrap();
        return Ok(Val::Real(math::bin(t, scalar(a[0], f)?, scalar(a[1], f)?, g)));
    }
    if a[0].is_int_like() {
        let v = a[0].ints()?;
        if v.is_empty() {
            return Err(format!("E_DOMAIN: {f}: argument must not be empty"));
        }
        return Ok(Val::Int(if is_max { *v.iter().max().unwrap() } else { *v.iter().min().unwrap() }));
    }
    let v = a[0].flat();
    if v.is_empty() {
        return Ok(Val::real(if is_max { f64::NEG_INFINITY } else { f64::INFINITY }));
    }
    let mut best = v[0];
    for s in &v[1..] {
        if (is_max && s.v > best.v) || (!is_max && s.v < best.v) {
            best = *s;
        }
    }
    Ok(Val::Real(best))
}

fn max(t: &mut Tape, a: &[&Val]) -> R<Val> {
    minmax(t, a, true)
}

fn min(t: &mut Tape, a: &[&Val]) -> R<Val> {
    minmax(t, a, false)
}

fn fma(t: &mut Tape, a: &[&Val]) -> R<Val> {
    let (x, y, z) = (scalar(a[0], "fma")?, scalar(a[1], "fma")?, scalar(a[2], "fma")?);
    Ok(Val::Real(t.node(x.v * y.v + z.v, &[(x, y.v), (y, x.v), (z, 1.0)])))
}

fn log_mix(t: &mut Tape, a: &[&Val]) -> R<Val> {
    if a.len() != 3 || !a[0].is_scalar() {
        return Err("log_mix: only the three scalar argument form is supported by the native backend".into());
    }
    let (th, l1, l2) = (scalar(a[0], "log_mix")?, scalar(a[1], "log_mix")?, scalar(a[2], "log_mix")?);
    let (x, y) = (th.v.ln() + l1.v, (-th.v).ln_1p() + l2.v);
    let m = x.max(y);
    let v = m + ((x - m).exp() + (y - m).exp()).ln();
    let (wx, wy) = ((x - v).exp(), (y - v).exp());
    Ok(Val::Real(t.node(v, &[(th, wx / th.v - wy / (1.0 - th.v)), (l1, wx), (l2, wy)])))
}

fn softmax_core(t: &mut Tape, f: &[S], log: bool) -> Vec<S> {
    let m = f.iter().map(|s| s.v).fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = f.iter().map(|s| (s.v - m).exp()).sum();
    let lse = m + z.ln();
    let p: Vec<f64> = f.iter().map(|s| (s.v - lse).exp()).collect();
    (0..f.len())
        .map(|i| {
            let partials: Vec<f64> = if log {
                (0..f.len()).map(|j| if i == j { 1.0 - p[j] } else { -p[j] }).collect()
            } else {
                (0..f.len()).map(|j| if i == j { p[i] * (1.0 - p[i]) } else { -p[i] * p[j] }).collect()
            };
            let v = if log { f[i].v - lse } else { p[i] };
            t.node_many(v, f, &partials)
        })
        .collect()
}

fn softmax(t: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Vector(softmax_core(t, &a[0].flat(), false)))
}

fn log_softmax(t: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Vector(softmax_core(t, &a[0].flat(), true)))
}

fn cumulative_sum(t: &mut Tape, a: &[&Val]) -> R<Val> {
    if a[0].is_int_like() {
        let mut acc = 0;
        return Ok(Val::Array(a[0].ints()?.into_iter().map(|x| { acc += x; Val::Int(acc) }).collect()));
    }
    let f = a[0].flat();
    let mut out = Vec::with_capacity(f.len());
    let mut acc = S::c(0.0);
    for s in &f {
        acc = t.node(acc.v + s.v, &[(acc, 1.0), (*s, 1.0)]);
        out.push(acc);
    }
    let it = &mut out.into_iter();
    Ok(match a[0] {
        Val::Array(_) => Val::Array(it.map(Val::Real).collect()),
        other => other.with_scalars(it),
    })
}

fn slice(v: &Val, start: usize, n: usize, f: &str) -> R<Val> {
    let len = match v {
        Val::Vector(x) | Val::RowVector(x) => x.len(),
        Val::Array(x) => x.len(),
        _ => return Err(format!("{f}: expected a vector or array")),
    };
    if start + n > len {
        return Err(format!("{f}: E_DOMAIN: index out of range (size {len})"));
    }
    Ok(match v {
        Val::Vector(x) => Val::Vector(x[start..start + n].to_vec()),
        Val::RowVector(x) => Val::RowVector(x[start..start + n].to_vec()),
        Val::Array(x) => Val::Array(x[start..start + n].to_vec()),
        _ => unreachable!(),
    })
}

fn segment(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let i = int_of(a[1], "segment")?;
    if i < 1 {
        return Err("E_DOMAIN: segment: index out of range".into());
    }
    slice(a[0], (i - 1) as usize, size_of(a[2], "segment")?, "segment")
}

fn head(_: &mut Tape, a: &[&Val]) -> R<Val> {
    slice(a[0], 0, size_of(a[1], "head")?, "head")
}

fn tail(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let n = size_of(a[1], "tail")?;
    let len = match a[0] {
        Val::Vector(x) | Val::RowVector(x) => x.len(),
        Val::Array(x) => x.len(),
        _ => return Err("tail: expected a vector or array".into()),
    };
    if n > len {
        return Err("E_DOMAIN: tail: index out of range".into());
    }
    slice(a[0], len - n, n, "tail")
}

fn append_row(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let mut out = Vec::new();
    for x in a {
        match x {
            Val::Int(_) | Val::Real(_) | Val::Vector(_) => x.flatten(&mut out),
            _ => return Err("append_row: only scalars and vectors are supported by the native backend".into()),
        }
    }
    Ok(Val::Vector(out))
}

fn append_col(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let mut out = Vec::new();
    for x in a {
        match x {
            Val::Int(_) | Val::Real(_) | Val::RowVector(_) => x.flatten(&mut out),
            _ => return Err("append_col: only scalars and row vectors are supported by the native backend".into()),
        }
    }
    Ok(Val::RowVector(out))
}

fn diag_matrix(_: &mut Tape, a: &[&Val]) -> R<Val> {
    let v = a[0].flat();
    let n = v.len();
    let mut d = vec![S::c(0.0); n * n];
    for (i, s) in v.into_iter().enumerate() {
        d[i * n + i] = s;
    }
    Ok(Val::Matrix { r: n, c: n, d })
}

fn int_step(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Int((scalar(a[0], "int_step")?.v > 0.0) as i64))
}

fn step(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::real(if scalar(a[0], "step")?.v < 0.0 { 0.0 } else { 1.0 }))
}

fn abs_int(t: &mut Tape, a: &[&Val]) -> R<Val> {
    if let Val::Int(i) = a[0] {
        return Ok(Val::Int(i.abs()));
    }
    Ok(math::map1(t, a[0], math::unary_fn("abs").unwrap()))
}

fn is_inf(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Int(scalar(a[0], "is_inf")?.v.is_infinite() as i64))
}

fn is_nan(_: &mut Tape, a: &[&Val]) -> R<Val> {
    Ok(Val::Int(scalar(a[0], "is_nan")?.v.is_nan() as i64))
}

fn constant(name: &str) -> Option<f64> {
    Some(match name {
        "pi" => std::f64::consts::PI,
        "e" => std::f64::consts::E,
        "sqrt2" => std::f64::consts::SQRT_2,
        "log2" => std::f64::consts::LN_2,
        "log10" => std::f64::consts::LN_10,
        "not_a_number" => f64::NAN,
        "positive_infinity" => f64::INFINITY,
        "negative_infinity" => f64::NEG_INFINITY,
        "machine_precision" => f64::EPSILON,
        _ => return None,
    })
}

/// A constant function such as `pi()`; `log2()` and `log10()` with no
/// arguments are constants too.
pub fn constant_call(name: &str, nargs: usize) -> Option<f64> {
    if nargs == 0 { constant(name) } else { None }
}

pub fn lookup(name: &str) -> Option<CallFn> {
    Some(match name {
        "sum" => sum,
        "mean" => mean,
        "variance" => variance,
        "sd" => sd,
        "prod" => prod,
        "log_sum_exp" => log_sum_exp,
        "dot_product" => dot_product,
        "dot_self" => dot_self,
        "rep_vector" => rep_vector,
        "rep_row_vector" => rep_row_vector,
        "rep_matrix" => rep_matrix,
        "rep_array" => rep_array,
        "size" => size,
        "num_elements" => num_elements,
        "rows" => rows,
        "cols" => cols,
        "to_vector" => to_vector,
        "to_row_vector" => to_row_vector,
        "max" => max,
        "min" => min,
        "fma" => fma,
        "log_mix" => log_mix,
        "softmax" => softmax,
        "log_softmax" => log_softmax,
        "cumulative_sum" => cumulative_sum,
        "segment" => segment,
        "head" => head,
        "tail" => tail,
        "append_row" => append_row,
        "append_col" => append_col,
        "diag_matrix" => diag_matrix,
        "int_step" => int_step,
        "step" => step,
        "abs" => abs_int,
        "is_inf" => is_inf,
        "is_nan" => is_nan,
        _ => return None,
    })
}
