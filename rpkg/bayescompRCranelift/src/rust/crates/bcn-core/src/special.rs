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
    if x.is_nan() {
        return f64::NAN;
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

pub fn lbeta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
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
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 {
        return 1.0 - erf_series(x);
    }
    erfc_cf(x)
}

fn erf_series(x: f64) -> f64 {
    // erf(x) = 2/sqrt(pi) * sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    let mut sum = x;
    let mut term = x;
    let x2 = x * x;
    let mut n = 0.0;
    loop {
        n += 1.0;
        term *= -x2 / n;
        let add = term / (2.0 * n + 1.0);
        sum += add;
        if add.abs() < 1e-17 * sum.abs() {
            break;
        }
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

/// log of the standard normal CDF.
pub fn log_phi(x: f64) -> f64 {
    if x < -5.0 {
        // log(erfc(t)/2) with erfc computed through its continued fraction
        let t = -x / std::f64::consts::SQRT_2;
        -t * t - 0.5 * PI.ln() - (2.0 * erfc_cf_denominator(t)).ln()
    } else {
        phi(x).ln()
    }
}

fn erfc_cf_denominator(x: f64) -> f64 {
    // erfc(x) = exp(-x^2)/sqrt(pi) / D(x); returns D(x)
    (-x * x).exp() / PI.sqrt() / erfc_cf(x)
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
        assert!((log_phi(-10.0) + 53.231_285_150_512_04).abs() < 1e-9);
    }
}
