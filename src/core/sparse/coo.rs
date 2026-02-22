use crate::core::scalar::Scalar;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder, Triplet};

/// Coordinate (COO) format sparse matrix.
///
/// COO is the simplest sparse format, storing a list of (row, col, value)
/// triplets. It is typically used for incrementally building sparse matrices
/// before converting them to a more efficient computational format like CSR or CSC.
#[derive(Clone, Debug)]
pub struct CooMatrix<T: Scalar> {
    rows: usize,
    cols: usize,
    triplets: Vec<Triplet<T>>,
}

impl<T: Scalar> CooMatrix<T> {
    /// Creates a new empty COO matrix with specified dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            triplets: Vec::new(),
        }
    }

    /// Creates a new COO matrix with pre-allocated capacity for triplets.
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols,
            triplets: Vec::with_capacity(capacity),
        }
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the number of non-zero elements currently stored.
    pub fn non_zeros(&self) -> usize {
        self.triplets.len()
    }

    /// Adds a new triplet (row, col, value) to the matrix.
    ///
    /// Note: Duplicate entries for the same (row, col) pair are naturally allowed
    /// in COO format and will be summed together during conversion to Compressed formats.
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows, "Row index out of bounds");
        assert!(col < self.cols, "Column index out of bounds");
        self.triplets.push(Triplet::new(row, col, value));
    }

    /// Returns a slice of the internal triplets.
    pub fn triplets(&self) -> &[Triplet<T>] {
        &self.triplets
    }

    /// Converts the COO matrix into a Compressed Sparse Row (CSR) or
    /// Compressed Sparse Column (CSC) matrix.
    ///
    /// During conversion, duplicate indices will be summed.
    pub fn to_sparse_matrix(&self, order: StorageOrder) -> SparseMatrix<T> {
        let mut mat = SparseMatrix::new(self.rows, self.cols, order);
        // We clone the triplets since set_from_triplets consumes them
        mat.set_from_triplets(self.triplets.clone());
        mat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_creation_and_conversion() {
        let mut coo = CooMatrix::<f64>::new(3, 4);
        
        coo.push(0, 0, 1.0);
        coo.push(0, 2, 2.0);
        coo.push(1, 1, 3.0);
        coo.push(2, 0, 4.0);
        coo.push(2, 3, 5.0);
        // Duplicate entry should sum
        coo.push(0, 0, 0.5);

        assert_eq!(coo.rows(), 3);
        assert_eq!(coo.cols(), 4);
        assert_eq!(coo.non_zeros(), 6);

        // Convert to CSR
        let csr = coo.to_sparse_matrix(StorageOrder::RowMajor);
        assert_eq!(csr.outer_size(), 3); 
        assert_eq!(csr.non_zeros(), 5); // 0,0 was summed (6 -> 5 elements)
        
        let vals = csr.values();
        assert_eq!(vals[0], 1.5); // (0, 0)
        assert_eq!(vals[1], 2.0); // (0, 2)
        assert_eq!(vals[2], 3.0); // (1, 1)
        assert_eq!(vals[3], 4.0); // (2, 0)
        assert_eq!(vals[4], 5.0); // (2, 3)
    }
}
