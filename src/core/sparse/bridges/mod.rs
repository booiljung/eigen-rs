//! Optional bridges to high-performance external sparse solvers.

#[cfg(feature = "suitesparse")]
pub mod suitesparse;

#[cfg(feature = "mkl")]
pub mod mkl;
