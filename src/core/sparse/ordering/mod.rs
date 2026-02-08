//! Infrastructure for sparse matrix ordering and permutations.

pub mod colamd;
pub use colamd::COLAMD;

use crate::core::scalar::Scalar;
use crate::core::sparse::sparse_matrix::SparseMatrix;

/// Trait for sparse matrix ordering strategies.
pub trait Ordering {
    /// Compute the permutation for the given matrix.
    fn compute<T: Scalar>(&self, matrix: &SparseMatrix<T>) -> Permutation;
}

/// Represents a permutation of indices.
pub struct Permutation {
    indices: Vec<usize>,
    inv_indices: Vec<usize>,
}

impl Permutation {
    pub fn new(indices: Vec<usize>) -> Self {
        let n = indices.len();
        let mut inv_indices = vec![0; n];
        for (i, &p) in indices.iter().enumerate() {
            inv_indices[p] = i;
        }
        Self {
            indices,
            inv_indices,
        }
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn inverse_indices(&self) -> &[usize] {
        &self.inv_indices
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Identity ordering (no permutation).
pub struct NaturalOrdering;

impl Ordering for NaturalOrdering {
    fn compute<T: Scalar>(&self, matrix: &SparseMatrix<T>) -> Permutation {
        let n = matrix.rows();
        Permutation::new((0..n).collect())
    }
}
