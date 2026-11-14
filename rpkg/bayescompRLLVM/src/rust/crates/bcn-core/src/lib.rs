//! Shared core of the bayescompR native backends.
//!
//! The IR produced by bayescompR is lowered ([`lower`]) to a kernel (leaf
//! operations over value slots plus a statement tree, see [`kernel`]). The
//! runtime ([`exec`]) executes leaf operations on a reverse mode AD tape; the
//! Cranelift and LLVM backends compile the statement tree to machine code
//! that calls the runtime for each leaf operation.

pub mod calls;
pub mod dist;
pub mod exec;
pub mod kernel;
pub mod lower;
pub mod math;
pub mod special;
pub mod tape;
pub mod val;
