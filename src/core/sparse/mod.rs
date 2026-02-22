//! Sparse Linear Algebra module.
//!
//! This module provides a comprehensive suite of tools for constructing, manipulating,
//! and solving mathematical systems involving sparse matrices using Compressed Storage formats
//! (both CSR and CSC). It includes:
//! - **Core Structures**: `SparseMatrix` for expression management and efficient arithmetic.
//! - **Iterative Solvers**: `ConjugateGradient` and `BiCGSTAB` with preconditioning support.
//! - **Direct Solvers**: `SimplicialLLT`, `SimplicialLDLT`, `SparseLU`, and `SparseQR`.
//! - **Hardware Acceleration**: Transparent bridging to `cuSPARSE` for GPU-accelerated operations.

pub mod bridges;
pub mod cuda_ops;
pub mod cuda_storage;
pub mod iterators;
pub mod ops;
pub mod ordering;
pub mod solvers;
pub mod sparse_matrix;

#[cfg(feature = "cuda")]
pub mod cuda_sparse_bridge;
#[cfg(feature = "cuda")]
pub use cuda_storage::CudaSparseStorage;
pub use iterators::InnerIterator;
pub use ordering::{NaturalOrdering, Ordering, Permutation};
pub use sparse_matrix::{SparseMatrix, StorageOrder, Triplet};
