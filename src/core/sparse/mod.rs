//! Sparse Linear Algebra module.

pub mod bridges;
pub mod cuda_ops;
pub mod cuda_storage;
pub mod iterators;
pub mod ops;
pub mod ordering;
pub mod solvers;
pub mod sparse_matrix;

#[cfg(feature = "cuda")]
pub use cuda_storage::CudaSparseStorage;
pub use iterators::InnerIterator;
pub use ordering::{NaturalOrdering, Ordering, Permutation};
pub use sparse_matrix::{SparseMatrix, StorageOrder, Triplet};
