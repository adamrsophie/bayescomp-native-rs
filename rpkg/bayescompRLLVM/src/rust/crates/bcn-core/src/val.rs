//! Runtime values. Matrices are column major, as in Stan and R.

use crate::tape::S;

#[derive(Clone, Debug)]
pub enum Val {
    Int(i64),
    Real(S),
    Vector(Vec<S>),
    RowVector(Vec<S>),
    Matrix { r: usize, c: usize, d: Vec<S> },
    Array(Vec<Val>),
}

impl Val {
    pub fn real(v: f64) -> Val {
        Val::Real(S::c(v))
    }

    pub fn as_int(&self) -> Result<i64, String> {
        match self {
            Val::Int(i) => Ok(*i),
            _ => Err("expected an integer".into()),
        }
    }

    /// The value as a scalar, promoting integers.
    pub fn as_scalar(&self) -> Option<S> {
        match self {
            Val::Int(i) => Some(S::c(*i as f64)),
            Val::Real(s) => Some(*s),
            _ => None,
        }
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, Val::Int(_) | Val::Real(_))
    }

    /// All scalars in Stan's order (column major, arrays outermost first).
    pub fn flatten(&self, out: &mut Vec<S>) {
        match self {
            Val::Int(i) => out.push(S::c(*i as f64)),
            Val::Real(s) => out.push(*s),
            Val::Vector(v) | Val::RowVector(v) => out.extend_from_slice(v),
            Val::Matrix { d, .. } => out.extend_from_slice(d),
            Val::Array(a) => {
                for x in a {
                    x.flatten(out)
                }
            }
        }
    }

    pub fn flat(&self) -> Vec<S> {
        let mut v = Vec::new();
        self.flatten(&mut v);
        v
    }

    /// Number of scalars.
    pub fn size(&self) -> usize {
        match self {
            Val::Int(_) | Val::Real(_) => 1,
            Val::Vector(v) | Val::RowVector(v) => v.len(),
            Val::Matrix { d, .. } => d.len(),
            Val::Array(a) => a.iter().map(|x| x.size()).sum(),
        }
    }

    /// Same container shape with new scalars (in flatten order).
    pub fn with_scalars(&self, it: &mut impl Iterator<Item = S>) -> Val {
        match self {
            Val::Int(_) | Val::Real(_) => Val::Real(it.next().unwrap()),
            Val::Vector(v) => Val::Vector(v.iter().map(|_| it.next().unwrap()).collect()),
            Val::RowVector(v) => Val::RowVector(v.iter().map(|_| it.next().unwrap()).collect()),
            Val::Matrix { r, c, d } => Val::Matrix { r: *r, c: *c, d: d.iter().map(|_| it.next().unwrap()).collect() },
            Val::Array(a) => Val::Array(a.iter().map(|x| x.with_scalars(it)).collect()),
        }
    }

    /// True if every scalar is a constant (data).
    pub fn is_const(&self) -> bool {
        match self {
            Val::Int(_) => true,
            Val::Real(s) => s.is_const(),
            Val::Vector(v) | Val::RowVector(v) => v.iter().all(|s| s.is_const()),
            Val::Matrix { d, .. } => d.iter().all(|s| s.is_const()),
            Val::Array(a) => a.iter().all(|x| x.is_const()),
        }
    }

    /// Integer contents of an integer or integer array (for indexing and
    /// discrete distributions).
    pub fn ints(&self) -> Result<Vec<i64>, String> {
        match self {
            Val::Int(i) => Ok(vec![*i]),
            Val::Array(a) => {
                let mut out = Vec::with_capacity(a.len());
                for x in a {
                    out.extend(x.ints()?);
                }
                Ok(out)
            }
            _ => Err("expected integers".into()),
        }
    }

    pub fn is_int_like(&self) -> bool {
        match self {
            Val::Int(_) => true,
            Val::Array(a) => a.iter().all(|x| x.is_int_like()),
            _ => false,
        }
    }
}
