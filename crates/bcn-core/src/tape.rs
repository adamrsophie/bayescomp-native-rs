//! Reverse mode automatic differentiation on a flat tape.
//!
//! Every scalar is an [`S`]: a value and the index of its tape node, or
//! [`CONST`] for values that do not depend on the parameters. A node stores
//! its parents and the partial derivatives with respect to them, so the
//! backward sweep is a single reverse loop over flat arrays.

/// Node index of constants.
pub const CONST: u32 = u32::MAX;

/// A scalar: value and tape node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct S {
    pub v: f64,
    pub id: u32,
}

impl S {
    pub fn c(v: f64) -> S {
        S { v, id: CONST }
    }
    pub fn is_const(&self) -> bool {
        self.id == CONST
    }
}

/// The tape.
#[derive(Default)]
pub struct Tape {
    val: Vec<f64>,
    adj: Vec<f64>,
    start: Vec<u32>,
    parent: Vec<u32>,
    partial: Vec<f64>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape { start: vec![0], ..Default::default() }
    }

    pub fn clear(&mut self) {
        self.val.clear();
        self.adj.clear();
        self.start.clear();
        self.start.push(0);
        self.parent.clear();
        self.partial.clear();
    }

    pub fn len(&self) -> usize {
        self.val.len()
    }

    pub fn is_empty(&self) -> bool {
        self.val.is_empty()
    }

    /// A new independent variable.
    pub fn var(&mut self, v: f64) -> S {
        let id = self.val.len() as u32;
        self.val.push(v);
        self.start.push(self.parent.len() as u32);
        S { v, id }
    }

    /// A node with value `v` and the given (parent, partial) pairs; constant
    /// parents are skipped, and a node with no variable parent is a constant.
    pub fn node(&mut self, v: f64, edges: &[(S, f64)]) -> S {
        if edges.iter().all(|(p, _)| p.is_const()) {
            return S::c(v);
        }
        let id = self.val.len() as u32;
        self.val.push(v);
        for (p, d) in edges {
            if !p.is_const() {
                self.parent.push(p.id);
                self.partial.push(*d);
            }
        }
        self.start.push(self.parent.len() as u32);
        S { v, id }
    }

    /// A node with many parents given as parallel slices.
    pub fn node_many(&mut self, v: f64, parents: &[S], partials: &[f64]) -> S {
        debug_assert_eq!(parents.len(), partials.len());
        if parents.iter().all(|p| p.is_const()) {
            return S::c(v);
        }
        let id = self.val.len() as u32;
        self.val.push(v);
        for (p, d) in parents.iter().zip(partials) {
            if !p.is_const() {
                self.parent.push(p.id);
                self.partial.push(*d);
            }
        }
        self.start.push(self.parent.len() as u32);
        S { v, id }
    }

    /// Gradient of `out` with respect to the first `n` variables (the
    /// independent variables are created first).
    pub fn gradient(&mut self, out: S, n: usize) -> Vec<f64> {
        let mut g = vec![0.0; n];
        if out.is_const() {
            return g;
        }
        self.adj.clear();
        self.adj.resize(self.val.len(), 0.0);
        self.adj[out.id as usize] = 1.0;
        for i in (0..=out.id as usize).rev() {
            let a = self.adj[i];
            if a == 0.0 {
                continue;
            }
            let (s, e) = (self.start[i] as usize, self.start[i + 1] as usize);
            for k in s..e {
                self.adj[self.parent[k] as usize] += a * self.partial[k];
            }
        }
        g.copy_from_slice(&self.adj[..n]);
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_rule() {
        let mut t = Tape::new();
        let x = t.var(3.0);
        let y = t.var(4.0);
        let z = t.node(x.v * y.v, &[(x, y.v), (y, x.v)]);
        let w = t.node(z.v + x.v, &[(z, 1.0), (x, 1.0)]);
        assert_eq!(t.gradient(w, 2), vec![5.0, 3.0]);
    }

    #[test]
    fn constants_are_skipped() {
        let mut t = Tape::new();
        let c = t.node(2.0, &[(S::c(1.0), 1.0)]);
        assert!(c.is_const());
    }
}
