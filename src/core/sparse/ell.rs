use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::Storage;

/// ELL (ELLPACK) format sparse matrix.
///
/// ELL format is heavily optimized for vector (SIMD/GPU) architectures.
/// It assumes a maximum number of non-zeros per row (`max_nnz_per_row` or `K`),
/// and pads shorter rows with explicit zeros (or dummy structures).
/// By storing the data column-major, hardware threads can perform
/// perfectly coalesced memory accesses across matrix rows.
#[derive(Clone, Debug)]
pub struct EllMatrix<T: Scalar> {
    rows: usize,
    cols: usize,
    /// Maximum number of non-zeros found in any single row.
    max_nnz_per_row: usize,
    /// The values array. Size is `rows * max_nnz_per_row`.
    /// Stored Column-Major: `values[i + r * max_nnz_per_row]` where `i` is the
    /// local non-zero index (0..K) and `r` is the row.
    values: Vec<T>,
    /// The column indices for each value. Size is `rows * max_nnz_per_row`.
    /// Padded entries generally duplicate the last valid column to avoid OOB reads.
    indices: Vec<usize>,
}

impl<T: Scalar> EllMatrix<T> {
    /// Constructs a new EllMatrix from raw components.
    pub fn from_raw(
        rows: usize,
        cols: usize,
        max_nnz_per_row: usize,
        values: Vec<T>,
        indices: Vec<usize>,
    ) -> Result<Self, String> {
        let expected_len = rows * max_nnz_per_row;
        if values.len() != expected_len {
            return Err(format!("values length must be rows * max_nnz = {}", expected_len));
        }
        if indices.len() != expected_len {
            return Err(format!("indices length must be rows * max_nnz = {}", expected_len));
        }

        Ok(Self {
            rows,
            cols,
            max_nnz_per_row,
            values,
            indices,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn max_nnz_per_row(&self) -> usize {
        self.max_nnz_per_row
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Performs Sparse Matrix-Vector multiplication (SpMV) using ELL format.
    /// `y = A * x`
    ///
    /// The structure favors having the outer loop over `k` (the maximum non-zeros per row)
    /// and the inner loop over `r` (the row index) to allow vectorizers to load
    /// contiguous `r`-elements if arrays are padded correctly.
    pub fn mul_dense_vector<S: Storage<T>, R_STORE: Storage<T>>(
        &self,
        x: &Matrix<T, S>,
        y: &mut Matrix<T, R_STORE>,
    ) -> Result<(), String> {
        if x.cols() != 1 || y.cols() != 1 {
            return Err("ELL mul_dense_vector only supports Matrix vectors (cols=1)".to_string());
        }
        if x.rows() != self.cols {
            return Err("Incompatible dimensions: x.rows != Ell.cols()".to_string());
        }
        if y.rows() != self.rows {
            return Err("Incompatible dimensions: y.rows != Ell.rows()".to_string());
        }

        let x_ptr = x.storage().data().as_ptr();
        let y_ptr = y.storage_mut().data_mut().as_mut_ptr();

        // Zero the output vector
        for i in 0..self.rows {
            unsafe { *y_ptr.add(i) = T::zero(); }
        }

        let val_ptr = self.values.as_ptr();
        let idx_ptr = self.indices.as_ptr();
        let rows = self.rows;

        // Optimized traversal: Outer loop over K, inner loop over Rows.
        // This is perfectly suited for vectorization/GPU threading as 
        // `r` accesses are completely contiguous.
        for k in 0..self.max_nnz_per_row {
            for r in 0..rows {
                unsafe {
                    let offset = k * rows + r; // Column-major layout 
                    let val = *val_ptr.add(offset);
                    // Padding is either true 0.0 or duplicate indices that don't harm summation
                    // if val is 0.0.
                    let col = *idx_ptr.add(offset);
                    let rx = *x_ptr.add(col);
                    *y_ptr.add(r) += val * rx;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_ell_creation_and_spmv() {
        // Create a 3x3 matrix where max_nnz_per_row (K) = 2.
        // A = 
        // [ 1  0  2 ] -> Row 0: NNZ=2
        // [ 0  3  0 ] -> Row 1: NNZ=1 (needs padding)
        // [ 4  5  0 ] -> Row 2: NNZ=2
        
        let rows = 3;
        let cols = 3;
        let k = 2; // max_nnz_per_row
        
        // Stored Column-Major by K (e.g. k=0 contiguous, then k=1 contiguous)
        // k=0 elements: (Row 0: 1), (Row 1: 3), (Row 2: 4)
        // k=1 elements: (Row 0: 2), (Row 1: 0(pad)), (Row 2: 5)
        let values = vec![
            1.0, 3.0, 4.0, // k=0 strip
            2.0, 0.0, 5.0, // k=1 strip (row 1 is padded with 0.0)
        ];
        
        let indices = vec![
            0, 1, 0, // k=0 column indices
            2, 1, 1, // k=1 column indices (row 1 padded col doesn't matter since val is 0.0, use 1)
        ];

        let ell = EllMatrix::<f64>::from_raw(rows, cols, k, values, indices).unwrap();

        assert_eq!(ell.rows(), 3);
        assert_eq!(ell.max_nnz_per_row(), 2);

        // x = [1, 2, 3]^T
        let x = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let mut y = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(3, 1).unwrap();

        ell.mul_dense_vector(&x, &mut y).unwrap();

        // Check result:
        // y[0] = 1*1 + 0*2 + 2*3 = 1 + 6 = 7
        // y[1] = 0*1 + 3*2 + 0*3 = 6
        // y[2] = 4*1 + 5*2 + 0*3 = 4 + 10 = 14

        assert_eq!(*y.get(0, 0).unwrap(), 7.0);
        assert_eq!(*y.get(1, 0).unwrap(), 6.0);
        assert_eq!(*y.get(2, 0).unwrap(), 14.0);
    }
}
