//! Log densities and mass functions.
//!
//! Each function follows the R backend term by term: with `propto` a term is
//! dropped when every argument it depends on is a constant, exactly as in
//! Stan. Arguments broadcast (scalars against containers of a common size).
//! The result is a single tape node whose partials are the full derivatives
//! with respect to every non constant argument.

use crate::math::R;
use crate::special::{digamma, lbeta, lchoose, lgamma, log1p_exp};
use crate::tape::{Tape, S};
use crate::val::Val;

const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_8;

fn zmul(a: f64, b: f64) -> f64 {
    if a == 0.0 { 0.0 } else { a * b }
}

struct Args {
    x: Vec<Vec<S>>,
    n: usize,
    c: Vec<bool>,
}

impl Args {
    fn new(name: &str, args: &[&Val], keep: &[bool]) -> R<Args> {
        let x: Vec<Vec<S>> = args.iter().map(|a| a.flat()).collect();
        let mut n = 1usize;
        let mut any_container = false;
        for (a, v) in args.iter().zip(&x) {
            if !a.is_scalar() {
                if any_container && v.len() != n {
                    return Err(format!("E_SHAPE: {name}: size mismatch between arguments ({} and {})", n, v.len()));
                }
                if !any_container {
                    n = v.len();
                    any_container = true;
                }
            }
        }
        let c = args.iter().enumerate().map(|(k, a)| a.is_const() && !keep.get(k).copied().unwrap_or(false)).collect();
        Ok(Args { x, n, c })
    }
    fn get(&self, k: usize, i: usize) -> f64 {
        let v = &self.x[k];
        if v.len() == 1 { v[0].v } else { v[i].v }
    }
    /// Is a term depending on these arguments included?
    fn inc(&self, propto: bool, deps: &[usize]) -> bool {
        !propto || deps.iter().any(|&k| !self.c[k])
    }
}

/// Accumulates partials per argument (scalar arguments collapse to one).
struct Partials {
    d: Vec<Vec<f64>>,
}

impl Partials {
    fn new(a: &Args) -> Partials {
        Partials { d: a.x.iter().map(|v| vec![0.0; v.len()]).collect() }
    }
    fn add(&mut self, k: usize, i: usize, v: f64) {
        let d = &mut self.d[k];
        if d.len() == 1 { d[0] += v } else { d[i] += v }
    }
    fn node(self, t: &mut Tape, a: &Args, value: f64) -> S {
        let mut parents = Vec::new();
        let mut partials = Vec::new();
        for (k, xs) in a.x.iter().enumerate() {
            if a.c[k] {
                continue;
            }
            parents.extend_from_slice(xs);
            partials.extend_from_slice(&self.d[k]);
        }
        t.node_many(value, &parents, &partials)
    }
}

#[derive(Clone, Copy)]
enum Chk {
    NotNan,
    Finite,
    PosFin,
    Nonneg,
    Unit,
    Int01,
}

/// Argument checks of each distribution, in the R backend's order. They run
/// over every element before anything else, so an empty variate does not
/// hide invalid parameters.
fn arg_checks(dist: &str) -> &'static [(usize, Chk, &'static str)] {
    use Chk::*;
    const RV: &str = "Random variable";
    const LOC: &str = "Location parameter";
    const SC: &str = "Scale parameter";
    match dist {
        "normal" | "cauchy" => &[(0, NotNan, RV), (1, Finite, LOC), (2, PosFin, SC)],
        "std_normal" => &[(0, NotNan, RV)],
        "student_t" => &[(0, NotNan, RV), (1, PosFin, "Degrees of freedom parameter"), (2, Finite, LOC), (3, PosFin, SC)],
        "double_exponential" | "logistic" => &[(0, Finite, RV), (1, Finite, LOC), (2, PosFin, SC)],
        "exponential" => &[(0, Nonneg, RV), (1, PosFin, "Inverse scale parameter")],
        "gamma" => &[(0, NotNan, RV), (1, PosFin, "Shape parameter"), (2, PosFin, "Inverse scale parameter")],
        "inv_gamma" => &[(0, NotNan, RV), (1, PosFin, "Shape parameter"), (2, PosFin, SC)],
        "lognormal" => &[(0, Nonneg, RV), (1, Finite, LOC), (2, PosFin, SC)],
        "beta" => &[(1, PosFin, "First shape parameter"), (2, PosFin, "Second shape parameter"), (0, Unit, RV)],
        "uniform" => &[(0, NotNan, RV), (1, Finite, "Lower bound parameter"), (2, Finite, "Upper bound parameter")],
        "weibull" => &[(0, Finite, RV), (1, PosFin, "Shape parameter"), (2, PosFin, SC)],
        "chi_square" => &[(0, Nonneg, RV), (1, PosFin, "Degrees of freedom parameter")],
        "bernoulli" => &[(0, Int01, "n"), (1, Unit, "Probability parameter")],
        "bernoulli_logit" => &[(0, Int01, "n"), (1, NotNan, "Logit transformed probability parameter")],
        "binomial" => &[(1, Nonneg, "Population size parameter"), (2, Unit, "Probability parameter")],
        "binomial_logit" => &[(2, Finite, "Probability parameter")],
        "poisson" => &[(0, Nonneg, RV), (1, Nonneg, "Rate parameter"), (1, Finite, "Rate parameter")],
        "poisson_log" => &[(0, Nonneg, RV), (1, NotNan, "Log rate parameter")],
        "neg_binomial_2" => &[(0, Nonneg, "Failures variable"), (1, PosFin, LOC), (2, PosFin, "Precision parameter")],
        "neg_binomial_2_log" => &[(0, Nonneg, "Failures variable"), (1, Finite, "Log location parameter"), (2, PosFin, "Precision parameter")],
        _ => &[],
    }
}

fn check_args(dist: &str, f: &str, a: &Args) -> R<()> {
    for &(k, c, what) in arg_checks(dist) {
        let Some(xs) = a.x.get(k) else { continue };
        for s in xs {
            let v = s.v;
            let ok = match c {
                Chk::NotNan => !v.is_nan(),
                Chk::Finite => v.is_finite(),
                Chk::PosFin => v > 0.0 && v.is_finite(),
                Chk::Nonneg => v >= 0.0,
                Chk::Unit | Chk::Int01 => (0.0..=1.0).contains(&v),
            };
            if !ok {
                let need = match c {
                    Chk::NotNan => "must not be nan",
                    Chk::Finite => "must be finite",
                    Chk::PosFin => "must be positive and finite",
                    Chk::Nonneg => "must be nonnegative",
                    Chk::Unit | Chk::Int01 => "must be in the interval [0, 1]",
                };
                return Err(format!("E_DOMAIN: {f}: {what} is {v}, but {need}"));
            }
        }
    }
    Ok(())
}

/// x * log(y) with 0 * log(0) = 0, and its partials in x and y.
fn lmultiply(x: f64, y: f64) -> (f64, f64, f64) {
    if x == 0.0 && y == 0.0 { (0.0, 0.0, 0.0) } else { (x * y.ln(), y.ln(), x / y) }
}

fn positive(name: &str, what: &str, v: f64) -> R<()> {
    if !(v > 0.0) || !v.is_finite() {
        return Err(format!("E_DOMAIN: {name}: {what} is {v}, but must be positive finite!"));
    }
    Ok(())
}

fn not_nan(name: &str, what: &str, v: f64) -> R<()> {
    if v.is_nan() {
        return Err(format!("E_DOMAIN: {name}: {what} is nan, but must not be nan!"));
    }
    Ok(())
}

/// Log density of `dist` (name without suffix) at `args` (the variate
/// first).
pub fn lpdf(t: &mut Tape, dist: &str, args: &[&Val], keep: &[bool], propto: bool) -> R<S> {
    let f = format!("{dist}_lpdf");
    let f = f.as_str();
    let a = Args::new(f, args, keep)?;
    let n = a.n;
    check_args(dist, f, &a)?;
    if a.x.iter().any(|v| v.is_empty()) {
        return Ok(S::c(0.0));
    }
    let mut p = Partials::new(&a);
    let mut lp = 0.0;
    macro_rules! need {
        ($k:expr) => {
            if a.x.len() <= $k {
                return Err(format!("E_INTERNAL: {f}: wrong number of arguments"));
            }
        };
    }
    match dist {
        "normal" => {
            need!(2);
            let (i_c, i_s, i_q) = (a.inc(propto, &[]), a.inc(propto, &[2]), a.inc(propto, &[0, 1, 2]));
            for i in 0..n {
                let (y, mu, s) = (a.get(0, i), a.get(1, i), a.get(2, i));
                not_nan(f, "Random variable", y)?;
                positive(f, "Scale parameter", s)?;
                let z = (y - mu) / s;
                if i_c { lp -= HALF_LOG_TWO_PI }
                if i_s { lp -= s.ln() }
                if i_q { lp -= 0.5 * z * z }
                p.add(0, i, -z / s);
                p.add(1, i, z / s);
                p.add(2, i, -1.0 / s + z * z / s);
            }
        }
        "std_normal" => {
            let (i_c, i_q) = (a.inc(propto, &[]), a.inc(propto, &[0]));
            for i in 0..n {
                let y = a.get(0, i);
                not_nan(f, "Random variable", y)?;
                if i_c { lp -= HALF_LOG_TWO_PI }
                if i_q { lp -= 0.5 * y * y }
                p.add(0, i, -y);
            }
        }
        "student_t" => {
            need!(3);
            let (i_nu, i_c, i_s, i_q) = (a.inc(propto, &[1]), a.inc(propto, &[]), a.inc(propto, &[3]), a.inc(propto, &[0, 1, 2, 3]));
            for i in 0..n {
                let (y, nu, mu, s) = (a.get(0, i), a.get(1, i), a.get(2, i), a.get(3, i));
                positive(f, "Degrees of freedom parameter", nu)?;
                positive(f, "Scale parameter", s)?;
                let z = (y - mu) / s;
                let r = z * z / nu;
                if i_nu { lp += lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * nu.ln() }
                if i_c { lp -= 0.5 * std::f64::consts::PI.ln() }
                if i_s { lp -= s.ln() }
                if i_q { lp -= (nu + 1.0) / 2.0 * r.ln_1p() }
                let dy = -(nu + 1.0) * z / (s * nu * (1.0 + r));
                p.add(0, i, dy);
                p.add(1, i, 0.5 * digamma((nu + 1.0) / 2.0) - 0.5 * digamma(nu / 2.0) - 0.5 / nu - 0.5 * r.ln_1p()
                    + (nu + 1.0) / 2.0 * r / (nu * (1.0 + r)));
                p.add(2, i, -dy);
                p.add(3, i, -1.0 / s + (nu + 1.0) * r / (s * (1.0 + r)));
            }
        }
        "cauchy" => {
            need!(2);
            let (i_c, i_s, i_q) = (a.inc(propto, &[]), a.inc(propto, &[2]), a.inc(propto, &[0, 1, 2]));
            for i in 0..n {
                let (y, mu, s) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "Scale parameter", s)?;
                let z = (y - mu) / s;
                if i_c { lp -= std::f64::consts::PI.ln() }
                if i_s { lp -= s.ln() }
                if i_q { lp -= (z * z).ln_1p() }
                let dy = -2.0 * z / (s * (1.0 + z * z));
                p.add(0, i, dy);
                p.add(1, i, -dy);
                p.add(2, i, -1.0 / s + 2.0 * z * z / (s * (1.0 + z * z)));
            }
        }
        "double_exponential" => {
            need!(2);
            let (i_c, i_s, i_q) = (a.inc(propto, &[]), a.inc(propto, &[2]), a.inc(propto, &[0, 1, 2]));
            for i in 0..n {
                let (y, mu, s) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "Scale parameter", s)?;
                let d = y - mu;
                if i_c { lp -= std::f64::consts::LN_2 }
                if i_s { lp -= s.ln() }
                if i_q { lp -= d.abs() / s }
                let sg = if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 };
                p.add(0, i, -sg / s);
                p.add(1, i, sg / s);
                p.add(2, i, -1.0 / s + d.abs() / (s * s));
            }
        }
        "logistic" => {
            need!(2);
            let (i_s, i_q) = (a.inc(propto, &[2]), a.inc(propto, &[0, 1, 2]));
            for i in 0..n {
                let (y, mu, s) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "Scale parameter", s)?;
                let z = (y - mu) / s;
                if i_s { lp -= s.ln() }
                if i_q { lp -= z + 2.0 * log1p_exp(-z) }
                // d/dz of -(z + 2 log1p(exp(-z))) = -1 + 2 / (1 + exp(z))
                let dz = -1.0 + 2.0 * crate::special::inv_logit(-z);
                p.add(0, i, dz / s);
                p.add(1, i, -dz / s);
                p.add(2, i, -1.0 / s - dz * z / s);
            }
        }
        "exponential" => {
            need!(1);
            let (i_b, i_q) = (a.inc(propto, &[1]), a.inc(propto, &[0, 1]));
            for i in 0..n {
                let (y, b) = (a.get(0, i), a.get(1, i));
                if i_b { lp += b.ln() }
                if i_q { lp -= b * y }
                p.add(0, i, -b);
                p.add(1, i, 1.0 / b - y);
            }
        }
        "gamma" => {
            need!(2);
            let (i_a, i_ab, i_ya, i_yb) = (a.inc(propto, &[1]), a.inc(propto, &[1, 2]), a.inc(propto, &[0, 1]), a.inc(propto, &[0, 2]));
            for i in 0..n {
                let (y, al, b) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "Shape parameter", al)?;
                positive(f, "Inverse scale parameter", b)?;
                if y < 0.0 {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                let ly = y.ln();
                if i_a { lp -= lgamma(al) }
                if i_ab { lp += al * b.ln() }
                if i_ya { lp += zmul(al - 1.0, ly) }
                if i_yb { lp -= b * y }
                p.add(0, i, zmul(al - 1.0, 1.0 / y) - b);
                p.add(1, i, -digamma(al) + b.ln() + ly);
                p.add(2, i, al / b - y);
            }
        }
        "inv_gamma" => {
            need!(2);
            let (i_a, i_ab, i_ya, i_yb) = (a.inc(propto, &[1]), a.inc(propto, &[1, 2]), a.inc(propto, &[0, 1]), a.inc(propto, &[0, 2]));
            for i in 0..n {
                let (y, al, b) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "Shape parameter", al)?;
                positive(f, "Scale parameter", b)?;
                if y <= 0.0 {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                let ly = y.ln();
                if i_a { lp -= lgamma(al) }
                if i_ab { lp += al * b.ln() }
                if i_ya { lp -= (al + 1.0) * ly }
                if i_yb { lp -= b / y }
                p.add(0, i, -(al + 1.0) / y + b / (y * y));
                p.add(1, i, -digamma(al) + b.ln() - ly);
                p.add(2, i, al / b - 1.0 / y);
            }
        }
        "lognormal" => {
            need!(2);
            // Stan Math keeps the constant whenever any argument is a parameter
            let (i_c, i_s, i_y, i_q) = (a.inc(propto, &[0, 1, 2]), a.inc(propto, &[2]), a.inc(propto, &[0]), a.inc(propto, &[0, 1, 2]));
            for i in 0..n {
                let (y, mu, s) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "Scale parameter", s)?;
                if y < 0.0 {
                    return Err(format!("E_DOMAIN: {f}: Random variable is {y}, but must be nonnegative!"));
                }
                if y == 0.0 {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                let ly = y.ln();
                let z = (ly - mu) / s;
                if i_c { lp -= HALF_LOG_TWO_PI }
                if i_s { lp -= s.ln() }
                if i_y { lp -= ly }
                if i_q { lp -= 0.5 * z * z }
                p.add(0, i, -1.0 / y - z / (s * y));
                p.add(1, i, z / s);
                p.add(2, i, -1.0 / s + z * z / s);
            }
        }
        "beta" => {
            need!(2);
            let (i_ab, i_ta, i_tb) = (a.inc(propto, &[1, 2]), a.inc(propto, &[0, 1]), a.inc(propto, &[0, 2]));
            for i in 0..n {
                let (th, al, be) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "First shape parameter", al)?;
                positive(f, "Second shape parameter", be)?;
                if !(0.0..=1.0).contains(&th) {
                    return Err(format!("E_DOMAIN: {f}: Random variable is {th}, but must be in the interval [0, 1]"));
                }
                let (lt, l1t) = (th.ln(), (-th).ln_1p());
                if i_ab { lp -= lbeta(al, be) }
                if i_ta { lp += zmul(al - 1.0, lt) }
                if i_tb { lp += zmul(be - 1.0, l1t) }
                let dab = digamma(al + be);
                p.add(0, i, zmul(al - 1.0, 1.0 / th) - zmul(be - 1.0, 1.0 / (1.0 - th)));
                p.add(1, i, -digamma(al) + dab + lt);
                p.add(2, i, -digamma(be) + dab + l1t);
            }
        }
        "uniform" => {
            need!(2);
            let i_ab = a.inc(propto, &[1, 2]);
            for i in 0..n {
                let (y, lo, hi) = (a.get(0, i), a.get(1, i), a.get(2, i));
                if hi <= lo {
                    return Err(format!("E_DOMAIN: {f}: Upper bound parameter must be greater than the lower bound"));
                }
                if y < lo || y > hi {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                if i_ab { lp -= (hi - lo).ln() }
                p.add(1, i, 1.0 / (hi - lo));
                p.add(2, i, -1.0 / (hi - lo));
            }
        }
        "weibull" => {
            need!(2);
            let (i_a, i_ya, i_as, i_q) = (a.inc(propto, &[1]), a.inc(propto, &[0, 1]), a.inc(propto, &[1, 2]), a.inc(propto, &[0, 1, 2]));
            for i in 0..n {
                let (y, al, s) = (a.get(0, i), a.get(1, i), a.get(2, i));
                positive(f, "Shape parameter", al)?;
                positive(f, "Scale parameter", s)?;
                if y < 0.0 {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                let w = (y / s).powf(al);
                let (lm, lm_x, lm_y) = lmultiply(al - 1.0, y);
                if i_a { lp += al.ln() }
                if i_ya { lp += lm }
                if i_as { lp -= al * s.ln() }
                if i_q { lp -= w }
                p.add(0, i, lm_y - al * (y / s).powf(al - 1.0) / s);
                p.add(1, i, 1.0 / al + lm_x - s.ln() - if w == 0.0 { 0.0 } else { w * (y / s).ln() });
                p.add(2, i, -al / s + al * w / s);
            }
        }
        "chi_square" => {
            need!(1);
            let (i_nu, i_ynu, i_y) = (a.inc(propto, &[1]), a.inc(propto, &[0, 1]), a.inc(propto, &[0]));
            for i in 0..n {
                let (y, nu) = (a.get(0, i), a.get(1, i));
                positive(f, "Degrees of freedom parameter", nu)?;
                if y < 0.0 {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                if i_nu { lp -= lgamma(nu / 2.0) + nu / 2.0 * std::f64::consts::LN_2 }
                let (lm, lm_x, lm_y) = lmultiply(nu / 2.0 - 1.0, y);
                if i_ynu { lp += lm }
                if i_y { lp -= y / 2.0 }
                p.add(0, i, lm_y - 0.5);
                p.add(1, i, -0.5 * digamma(nu / 2.0) - 0.5 * std::f64::consts::LN_2 + 0.5 * lm_x);
            }
        }
        _ => return Err(format!("E_UNSUPPORTED: distribution '{dist}' is not supported by the native backend")),
    }
    Ok(p.node(t, &a, lp))
}

fn int_arg(name: &str, v: &Val) -> R<Vec<i64>> {
    v.ints().map_err(|_| format!("E_INTERNAL: {name}: expected integer arguments"))
}

/// Log probability mass of `dist` at integer outcomes `args[0]`.
pub fn lpmf(t: &mut Tape, dist: &str, args: &[&Val], keep: &[bool], propto: bool) -> R<S> {
    let f = format!("{dist}_lpmf");
    let f = f.as_str();
    let nn_all = int_arg(f, args[0])?;
    // discrete trials (binomial) are integer arguments too
    let a = Args::new(f, args, keep)?;
    let n = a.n;
    check_args(dist, f, &a)?;
    if a.x.iter().any(|v| v.is_empty()) {
        return Ok(S::c(0.0));
    }
    let ni = |i: usize| -> f64 { if nn_all.len() == 1 { nn_all[0] as f64 } else { nn_all[i] as f64 } };
    let mut p = Partials::new(&a);
    let mut lp = 0.0;
    match dist {
        "bernoulli" => {
            let i_t = a.inc(propto, &[1]);
            for i in 0..n {
                let (k, th) = (ni(i), a.get(1, i));
                if !(0.0..=1.0).contains(&th) {
                    return Err(format!("E_DOMAIN: {f}: Probability parameter is {th}, but must be in the interval [0, 1]"));
                }
                if !(k == 0.0 || k == 1.0) {
                    return Err(format!("E_DOMAIN: {f}: n is {k}, but must be in the interval [0, 1]"));
                }
                if i_t { lp += zmul(k, th.ln()) + zmul(1.0 - k, (-th).ln_1p()) }
                p.add(1, i, zmul(k, 1.0 / th) - zmul(1.0 - k, 1.0 / (1.0 - th)));
            }
        }
        "bernoulli_logit" => {
            let i_t = a.inc(propto, &[1]);
            for i in 0..n {
                let (k, al) = (ni(i), a.get(1, i));
                if !(k == 0.0 || k == 1.0) {
                    return Err(format!("E_DOMAIN: {f}: n is {k}, but must be in the interval [0, 1]"));
                }
                let sg = 2.0 * k - 1.0;
                let x = sg * al;
                if i_t { lp += -log1p_exp(-x) }
                p.add(1, i, sg * crate::special::inv_logit(-x));
            }
        }
        "binomial" | "binomial_logit" => {
            let trials = int_arg(f, args[1])?;
            let i_c = a.inc(propto, &[]);
            let i_t = a.inc(propto, &[2]);
            for i in 0..n {
                let k = ni(i);
                let nt = if trials.len() == 1 { trials[0] as f64 } else { trials[i] as f64 };
                if k < 0.0 || k > nt {
                    return Err(format!("E_DOMAIN: {f}: Successes variable must be in [0, N]"));
                }
                let x = a.get(2, i);
                if i_c { lp += lchoose(nt, k) }
                if dist == "binomial" {
                    if !(0.0..=1.0).contains(&x) {
                        return Err(format!("E_DOMAIN: {f}: Probability parameter is {x}, but must be in the interval [0, 1]"));
                    }
                    if i_t { lp += zmul(k, x.ln()) + zmul(nt - k, (-x).ln_1p()) }
                    p.add(2, i, zmul(k, 1.0 / x) - zmul(nt - k, 1.0 / (1.0 - x)));
                } else {
                    if i_t { lp += k * (-log1p_exp(-x)) + (nt - k) * (-log1p_exp(x)) }
                    p.add(2, i, k - nt * crate::special::inv_logit(x));
                }
            }
        }
        "poisson" => {
            let i_c = a.inc(propto, &[]);
            let i_l = a.inc(propto, &[1]);
            for i in 0..n {
                let (k, l) = (ni(i), a.get(1, i));
                if l.is_infinite() {
                    return Err(format!("E_DOMAIN: {f}: Rate parameter is inf, but must be finite"));
                }
                if l < 0.0 {
                    return Err(format!("E_DOMAIN: {f}: Rate parameter is {l}, but must be nonnegative!"));
                }
                if l == 0.0 && k > 0.0 {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                if i_c { lp -= lgamma(k + 1.0) }
                if i_l { lp += zmul(k, l.ln()) - l }
                p.add(1, i, zmul(k, 1.0 / l) - 1.0);
            }
        }
        "poisson_log" => {
            let i_c = a.inc(propto, &[]);
            let i_l = a.inc(propto, &[1]);
            for i in 0..n {
                let (k, al) = (ni(i), a.get(1, i));
                if al == f64::INFINITY || (al == f64::NEG_INFINITY && k != 0.0) {
                    return Ok(S::c(f64::NEG_INFINITY));
                }
                let e = al.exp();
                if i_c { lp -= lgamma(k + 1.0) }
                if i_l { lp += zmul(k, al) - e }
                p.add(1, i, k - e);
            }
        }
        "neg_binomial_2" | "neg_binomial_2_log" => {
            let i_phi = a.inc(propto, &[2]);
            let i_c = a.inc(propto, &[]);
            let i_mp = a.inc(propto, &[1, 2]);
            for i in 0..n {
                let (k, m, ph) = (ni(i), a.get(1, i), a.get(2, i));
                positive(f, "Precision parameter", ph)?;
                // lgamma(k + phi) - lgamma(phi) = lgamma(k) - lbeta(k, phi), stable for large phi
                if i_phi && k > 0.0 { lp += lgamma(k) - lbeta(k, ph) }
                if i_c { lp -= lgamma(k + 1.0) }
                let dratio = if k <= 100.0 { (0..k as i64).map(|j| 1.0 / (ph + j as f64)).sum() } else { digamma(k + ph) - digamma(ph) };
                if dist == "neg_binomial_2" {
                    let lmp = (m + ph).ln();
                    let l1 = (m / ph).ln_1p();
                    if i_mp { lp += zmul(k, m.ln()) - k * lmp - ph * l1 }
                    p.add(1, i, zmul(k, 1.0 / m) - (k + ph) / (m + ph));
                    p.add(2, i, dratio - l1 + (m - k) / (m + ph));
                } else {
                    let lphi = ph.ln();
                    let mx = m.max(lphi);
                    let lse = mx + ((m - mx).exp() + (lphi - mx).exp()).ln();
                    let l1 = (m - lphi).exp().ln_1p();
                    if i_mp { lp += k * (m - lse) - ph * l1 }
                    let w = (m - lse).exp(); // mu / (mu + phi)
                    p.add(1, i, k - (k + ph) * w);
                    p.add(2, i, dratio - l1 + w - k / ph * (1.0 - w));
                }
            }
        }
        _ => return Err(format!("E_UNSUPPORTED: distribution '{dist}' is not supported by the native backend")),
    }
    Ok(p.node(t, &a, lp))
}

/// True if the distribution is discrete.
pub fn is_discrete(dist: &str) -> bool {
    matches!(
        dist,
        "bernoulli" | "bernoulli_logit" | "binomial" | "binomial_logit" | "poisson" | "poisson_log" | "neg_binomial_2" | "neg_binomial_2_log"
    )
}

pub fn supported(dist: &str) -> bool {
    is_discrete(dist)
        || matches!(
            dist,
            "normal" | "std_normal" | "student_t" | "cauchy" | "double_exponential" | "logistic" | "exponential" | "gamma"
                | "inv_gamma" | "lognormal" | "beta" | "uniform" | "weibull" | "chi_square"
        )
}
