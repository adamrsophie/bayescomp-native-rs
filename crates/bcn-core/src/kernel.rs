//! The kernel: what the IR is lowered to and what the backends compile.
//!
//! Values live in numbered slots. Leaf work is an [`Op`] (all arithmetic,
//! calls, indexing, assignment, parameter transforms, target increments),
//! which the runtime executes by index. Control flow is a tree of [`Stmt`];
//! the code generators turn it into native branches and loops, so the only
//! calls left in the machine code are the leaf operations.

/// Marks where a reject or fatal_error argument goes in its message; Stan
/// string literals cannot contain this character.
pub const REJECT_SLOT: char = '\u{0}';

use crate::math::{Binary, Unary};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    EMul,
    EDiv,
    Pow,
    Mod,
    IntDiv,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl BinOp {
    pub fn parse(s: &str) -> Option<BinOp> {
        Some(match s {
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            "*" => BinOp::Mul,
            "/" => BinOp::Div,
            ".*" => BinOp::EMul,
            "./" => BinOp::EDiv,
            "^" | ".^" => BinOp::Pow,
            "%" => BinOp::Mod,
            "%/%" => BinOp::IntDiv,
            "<" => BinOp::Lt,
            "<=" => BinOp::Le,
            ">" => BinOp::Gt,
            ">=" => BinOp::Ge,
            "==" => BinOp::Eq,
            "!=" => BinOp::Ne,
            _ => return None,
        })
    }
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::EMul => ".*",
            BinOp::EDiv => "./",
            BinOp::Pow => "^",
            BinOp::Mod => "%",
            BinOp::IntDiv => "%/%",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IdxKind {
    /// one integer index
    Single,
    /// an integer array index
    Multi,
    /// `lo:hi`, either side optional
    Range { lo: bool, hi: bool },
    /// `:`
    Omni,
}

/// The shape of a declared variable (for locals and parameters).
#[derive(Clone, Debug)]
pub struct DeclType {
    pub base: Base,
    /// number of array dimensions; the sizes come first among the op args
    pub nadims: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    Int,
    Real,
    Vector,
    RowVector,
    Matrix,
}

impl Base {
    pub fn parse(s: &str) -> Option<Base> {
        Some(match s {
            "int" => Base::Int,
            "real" => Base::Real,
            "vector" => Base::Vector,
            "row_vector" => Base::RowVector,
            "matrix" => Base::Matrix,
            _ => return None,
        })
    }
    /// number of size arguments of the element type
    pub fn nsizes(self) -> usize {
        match self {
            Base::Int | Base::Real => 0,
            Base::Vector | Base::RowVector => 1,
            Base::Matrix => 2,
        }
    }
}

/// Bounds of a parameter or checked variable (which optional args exist).
#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds {
    pub lower: bool,
    pub upper: bool,
    pub offset: bool,
    pub multiplier: bool,
}

#[derive(Clone)]
pub enum OpKind {
    Copy,
    Bin(BinOp),
    Neg,
    Not,
    Fn1(Unary),
    Fn2(Binary),
    Call(crate::calls::CallFn),
    GetTarget,
    /// log density: distribution name, `lupdf` style (drops constants when
    /// the program runs with propto), discrete
    /// `keep[k]`: argument k is not data only, so its terms stay under propto
    /// even when its value happens to be constant (as in Stan, which decides
    /// from the static type)
    Lpdf { dist: String, lu: bool, discrete: bool, keep: Vec<bool> },
    Index(Vec<IdxKind>),
    RowVecLit,
    ArrLit,
    Transpose,
    /// declare: element type and array rank; args are the sizes
    Decl(DeclType),
    /// args: [target, value, index args...]; assigns into the target slot
    Assign(Vec<IdxKind>),
    /// args: [value]
    Target,
    /// args: [value]; only when the Jacobian is requested
    Jacobian,
    /// args: [sizes..., bound args...]; dst receives the constrained value
    Param { ty: DeclType, bounds: Bounds, name: String },
    /// args: [value, bound args...]
    Check { bounds: Bounds, name: String },
    /// message pieces separated by REJECT_SLOT, one per printed argument
    Reject { msg: String, fatal: bool },
}

#[derive(Clone)]
pub struct Op {
    pub kind: OpKind,
    pub dst: usize,
    pub args: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Op(usize),
    Block(Vec<Stmt>),
    /// `for (var in lo:hi)`; lo and hi are int slots computed before
    For { var: usize, lo: usize, hi: usize, body: Box<Stmt> },
    /// `for (var in container)`
    ForEach { var: usize, container: usize, body: Box<Stmt> },
    /// `while (cond)`: `pre` recomputes the condition slot
    While { pre: Box<Stmt>, cond: usize, body: Box<Stmt> },
    If { cond: usize, then: Box<Stmt>, els: Box<Stmt> },
    Break,
    Continue,
}

/// A lowered program.
#[derive(Clone)]
pub struct Program {
    pub ops: Vec<Op>,
    pub body: Stmt,
    pub nslots: usize,
    /// initial slot values: constants and data
    pub init: Vec<(usize, crate::val::Val)>,
    /// number of unconstrained parameters expected (checked at run time)
    pub dim: usize,
}
