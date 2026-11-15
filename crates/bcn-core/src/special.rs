//! Special functions on f64.

use std::f64::consts::PI;

const LANCZOS_G: f64 = 7.0;
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// log |Gamma(x)|.
pub fn lgamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x <= 0.0 && x == x.floor() {
        // poles at zero and the negative integers
        return f64::INFINITY;
    }
    if x < 0.5 {
        // reflection
        let s = (PI * x).sin();
        if s == 0.0 {
            return f64::INFINITY;
        }
        return (PI / s.abs()).ln() - lgamma(1.0 - x);
    }
    if x > 1e15 {
        return x * (x.ln() - 1.0);
    }
    // Stirling series for large x is more accurate than Lanczos there
    if x >= 15.0 {
        let z = 1.0 / (x * x);
        let series = (1.0 / 12.0 - z * (1.0 / 360.0 - z * (1.0 / 1260.0 - z * (1.0 / 1680.0 - z / 1188.0)))) / x;
        return (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + series;
    }
    let x = x - 1.0;
    let mut a = LANCZOS[0];
    let t = x + LANCZOS_G + 0.5;
    for (i, c) in LANCZOS.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Digamma function.
pub fn digamma(mut x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut r = 0.0;
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // reflection
        r = -PI / (PI * x).tan();
        x = 1.0 - x;
    }
    while x < 12.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    r + x.ln() - 0.5 / x
        - f * (1.0 / 12.0 - f * (1.0 / 120.0 - f * (1.0 / 252.0 - f * (1.0 / 240.0 - f * (1.0 / 132.0)))))
}

/// lgamma(x) minus its Stirling approximation, for x >= 10.
fn lgamma_stirling_diff(x: f64) -> f64 {
    const C: [f64; 6] = [1.0 / 12.0, -1.0 / 360.0, 1.0 / 1260.0, -1.0 / 1680.0, 1.0 / 1188.0, -691.0 / 360_360.0];
    let m = 1.0 / x;
    let m2 = m * m;
    let mut r = C[5];
    for c in C[..5].iter().rev() {
        r = r * m2 + c;
    }
    r * m
}

/// log Beta(a, b), without the cancellation of the lgamma difference for
/// large arguments (Stan's algorithm).
pub fn lbeta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let (x, y) = if a < b { (a, b) } else { (b, a) };
    if x == 0.0 {
        return f64::INFINITY;
    }
    if y.is_infinite() {
        return f64::NEG_INFINITY;
    }
    if y < 10.0 {
        return lgamma(x) + lgamma(y) - lgamma(x + y);
    }
    let xy = x + y;
    if x < 10.0 {
        let st = (y - 0.5) * (-x / xy).ln_1p() + x * (1.0 - xy.ln());
        return st + lgamma(x) + lgamma_stirling_diff(y) - lgamma_stirling_diff(xy);
    }
    let sd = lgamma_stirling_diff(x) + lgamma_stirling_diff(y) - lgamma_stirling_diff(xy);
    let st = (x - 0.5) * (x / xy).ln() + y * (-x / xy).ln_1p() + 0.918_938_533_204_672_8 - 0.5 * y.ln();
    st + sd
}

pub fn lchoose(n: f64, k: f64) -> f64 {
    lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)
}

pub fn log1p_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

pub fn inv_logit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// log(inv_logit(x)).
pub fn log_inv_logit(x: f64) -> f64 {
    -log1p_exp(-x)
}

/// log(1 - inv_logit(x)).
pub fn log1m_inv_logit(x: f64) -> f64 {
    -log1p_exp(x)
}

/// Complementary error function: power series for small x, Lentz continued
/// fraction otherwise.
pub fn erfc(x: f64) -> f64 {
    // Use the relation with the regularized incomplete gamma function.
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x < 2.0 {
        return 1.0 - erf_series(x);
    }
    erfc_cf(x)
}

/// Error function, from the series where 1 - erfc would lose relative precision.
pub fn erf(x: f64) -> f64 {
    if x.abs() < 2.0 { erf_series(x) } else { 1.0 - erfc(x) }
}

fn erf_series(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) * sum_n (-1)^n x^(2n+1) / (n! (2n+1)), for |x| < 2,
    // where 100 terms reach full precision
    if x == 0.0 {
        return x;
    }
    let mut sum = x;
    let mut term = x;
    let x2 = x * x;
    for k in 1..100 {
        let n = k as f64;
        term *= -x2 / n;
        let add = term / (2.0 * n + 1.0);
        let next = sum + add;
        if next == sum {
            break;
        }
        sum = next;
    }
    2.0 / PI.sqrt() * sum
}

fn erfc_cf(x: f64) -> f64 {
    // Lentz continued fraction for erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
    let tiny = 1e-300;
    let mut f = x;
    if f == 0.0 {
        f = tiny;
    }
    let mut c = f;
    let mut d = 0.0;
    for i in 1..500 {
        let a = i as f64 / 2.0;
        d = x + a * d;
        if d == 0.0 {
            d = tiny;
        }
        c = x + a / c;
        if c == 0.0 {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x * x).exp() / PI.sqrt() / f
}

/// Standard normal CDF.
pub fn phi(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        assert!((lgamma(0.5) - 0.572_364_942_924_700_1).abs() < 1e-14);
        assert!((lgamma(10.0) - 12.801_827_480_081_469).abs() < 1e-13);
        assert!((lgamma(20.5) - 40.831_500_974_530_798).abs() < 1e-12);
        assert!((digamma(1.0) + 0.577_215_664_901_532_9).abs() < 1e-14);
        assert!((digamma(0.3) + 3.502_524_222_200_133).abs() < 1e-13);
        assert!((phi(-1.0) - 0.158_655_253_931_457_05).abs() < 1e-15);
        assert!((phi(2.5) - 0.993_790_334_674_223_7).abs() < 1e-15);
    }

    fn lbeta_ref() -> f64 {
        // log Beta(n, n) ~ log(2 sqrt(pi)) - (2n - 0.5) log 2 - 0.5 log n, n = 1e16
        (2.0 * std::f64::consts::PI.sqrt()).ln() - (2e16 - 0.5) * std::f64::consts::LN_2 - 0.5 * 1e16f64.ln()
    }

    #[test]
    fn edges() {
        assert_eq!(erfc(0.0), 1.0);
        assert!((erf(1e-10) / 1.128_379_167_095_512_6e-10 - 1.0).abs() < 1e-15);
        assert_eq!(phi(0.0), 0.5);
        assert_eq!(phi(f64::INFINITY), 1.0);
        assert_eq!(phi(f64::NEG_INFINITY), 0.0);
        assert!(erfc(f64::NAN).is_nan());
        assert!((erfc(1e-300) - 1.0).abs() < 1e-15);
        assert_eq!(lgamma(-1.0), f64::INFINITY);
        assert!((lbeta(1e16, 1e16) - lbeta_ref()).abs() < 1e-6 * lbeta_ref().abs());
        assert!((lbeta(12.5, 30.0) - (-25.908_683_991_310_08)).abs() < 1e-10);
        assert!((lbeta(3.0, 50.0) - (-11.101_945_176_173_95)).abs() < 1e-10);
        assert_eq!(lgamma(0.0), f64::INFINITY);
        assert_eq!(lgamma(-1e20), f64::INFINITY);
        assert!((lgamma(-0.5) - 1.265_512_123_484_645_4).abs() < 1e-13);
        assert!(digamma(-1e20).is_nan());
        assert_eq!(crate::math::trigamma(-1e20), f64::INFINITY);
        assert!((crate::math::trigamma(-0.5) - 8.934_802_200_544_704).abs() < 1e-11);
    }
}
