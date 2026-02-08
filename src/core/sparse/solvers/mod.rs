//! Sparse direct and iterative solvers.

pub mod simplicial_llt;
pub mod sparse_lu;
pub mod iterative_solver_base;
pub mod conjugate_gradient;
pub mod bicgstab;
pub mod incomplete_lut;
pub mod incomplete_cholesky;
pub mod sparse_qr;

pub use simplicial_llt::SimplicialLLT;
pub use sparse_lu::SparseLU;
pub use iterative_solver_base::{Preconditioner, IdentityPreconditioner, DiagonalPreconditioner};
pub use conjugate_gradient::ConjugateGradient;
pub use bicgstab::BiCGSTAB;
pub use incomplete_lut::IncompleteLUT;
pub use incomplete_cholesky::IncompleteCholesky;
pub use sparse_qr::SparseQR;

pub mod gmres;
pub use gmres::GMRES;

pub mod simplicial_ldlt;
pub use simplicial_ldlt::SimplicialLDLT;
