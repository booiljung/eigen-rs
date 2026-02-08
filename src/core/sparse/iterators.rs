//! Iterators for sparse matrices.

use crate::core::scalar::Scalar;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};

/// An iterator over the non-zero elements of a row (for CSR) or column (for CSC).
pub struct InnerIterator<'a, T: Scalar> {
    matrix: &'a SparseMatrix<T>,
    outer: usize,
    current_idx: usize,
    end_idx: usize,
}

impl<'a, T: Scalar> InnerIterator<'a, T> {
    pub fn new(matrix: &'a SparseMatrix<T>, outer: usize) -> Self {
        let start = matrix.outer_starts()[outer];
        let end = matrix.outer_starts()[outer + 1];
        Self {
            matrix,
            outer,
            current_idx: start,
            end_idx: end,
        }
    }

    /// Returns true if the iterator points to a valid element.
    pub fn is_valid(&self) -> bool {
        self.current_idx < self.end_idx
    }

    /// Moves the iterator to the next non-zero element.
    pub fn next(&mut self) {
        self.current_idx += 1;
    }

    /// Returns the value of the current element.
    pub fn value(&self) -> T {
        self.matrix.values()[self.current_idx]
    }

    /// Returns the inner index (row for CSC, col for CSR) of the current element.
    pub fn index(&self) -> usize {
        self.matrix.inner_indices()[self.current_idx]
    }

    /// Returns the row index of the current element.
    pub fn row(&self) -> usize {
        if self.matrix.order() == StorageOrder::RowMajor {
            self.outer
        } else {
            self.index()
        }
    }

    /// Returns the column index of the current element.
    pub fn col(&self) -> usize {
        if self.matrix.order() == StorageOrder::ColMajor {
            self.outer
        } else {
            self.index()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sparse::sparse_matrix::Triplet;

    #[test]
    fn test_sparse_inner_iterator() {
        let mut m = SparseMatrix::<f64>::new(3, 3, StorageOrder::RowMajor);
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 2, 4.0),
            Triplet::new(2, 0, 2.0),
            Triplet::new(2, 2, 5.0),
        ];
        m.set_from_triplets(triplets);

        // Row 0
        let mut it0 = InnerIterator::new(&m, 0);
        assert!(it0.is_valid());
        assert_eq!(it0.value(), 1.0);
        assert_eq!(it0.col(), 0);
        it0.next();
        assert!(it0.is_valid());
        assert_eq!(it0.value(), 4.0);
        assert_eq!(it0.col(), 2);
        it0.next();
        assert!(!it0.is_valid());

        // Row 1 (Empty)
        let it1 = InnerIterator::new(&m, 1);
        assert!(!it1.is_valid());

        // Row 2
        let mut it2 = InnerIterator::new(&m, 2);
        assert!(it2.is_valid());
        assert_eq!(it2.value(), 2.0);
        assert_eq!(it2.col(), 0);
        it2.next();
        assert!(it2.is_valid());
        assert_eq!(it2.value(), 5.0);
        assert_eq!(it2.col(), 2);
        it2.next();
        assert!(!it2.is_valid());
    }
}
