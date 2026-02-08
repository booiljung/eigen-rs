//! Sparse Linear Algebra module.

pub mod sparse_matrix;
pub mod iterators;
pub mod solvers;
pub mod ordering;
pub mod cuda_storage;
pub mod cuda_ops;
pub mod ops;
pub mod bridges;

pub use sparse_matrix::{SparseMatrix, Triplet, StorageOrder};
pub use iterators::InnerIterator;
pub use ordering::{Ordering, Permutation, NaturalOrdering};
#[cfg(feature = "cuda")]
pub use cuda_storage::CudaSparseStorage;
