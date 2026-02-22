use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::Storage;

/// Block Sparse Row (BSR) format sparse matrix.
///
/// BSR stores non-zero elements as dense blocks of size `R` x `C`.
/// It is highly efficient for matrices arising from PDE discretizations
/// where degrees of freedom naturally group together (e.g., 2D/3D momentum components),
/// improving cache locality and vectorization throughput compared to scalar CSR.
#[derive(Clone, Debug)]
pub struct BsrMatrix<T: Scalar, const R: usize, const C: usize> {
    block_rows: usize,
    block_cols: usize,
    /// Flat array of blocks. Length will be `nnz_blocks * R * C`.
    /// Each block is stored in column-major order to align with the rest of eigen-rs.
    values: Vec<T>,
    /// Column indices of each block. Length is `nnz_blocks`.
    inner_indices: Vec<usize>,
    /// Row pointers marking the start of each block row. Length is `block_rows + 1`.
    outer_starts: Vec<usize>,
}

impl<T: Scalar, const R: usize, const C: usize> BsrMatrix<T, R, C> {
    /// Creates a new BsrMatrix from raw components.
    pub fn from_raw(
        block_rows: usize,
        block_cols: usize,
        values: Vec<T>,
        inner_indices: Vec<usize>,
        outer_starts: Vec<usize>,
    ) -> Result<Self, String> {
        if outer_starts.len() != block_rows + 1 {
            return Err("outer_starts length must be block_rows + 1".to_string());
        }
        if outer_starts[0] != 0 {
            return Err("outer_starts must begin with 0".to_string());
        }
        let nnz_blocks = *outer_starts.last().unwrap();
        if inner_indices.len() != nnz_blocks {
            return Err("inner_indices length must match nnz_blocks".to_string());
        }
        if values.len() != nnz_blocks * R * C {
            return Err("values length must match nnz_blocks * R * C".to_string());
        }

        Ok(Self {
            block_rows,
            block_cols,
            values,
            inner_indices,
            outer_starts,
        })
    }

    pub fn block_rows(&self) -> usize {
        self.block_rows
    }

    pub fn block_cols(&self) -> usize {
        self.block_cols
    }

    pub fn rows(&self) -> usize {
        self.block_rows * R
    }

    pub fn cols(&self) -> usize {
        self.block_cols * C
    }

    pub fn nnz_blocks(&self) -> usize {
        self.inner_indices.len()
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn inner_indices(&self) -> &[usize] {
        &self.inner_indices
    }

    pub fn outer_starts(&self) -> &[usize] {
        &self.outer_starts
    }

    /// Performs Sparse Matrix-Vector multiplication (SpMV) for the BSR format.
    /// `y = A * x` where `A` is this BSR matrix.
    pub fn mul_dense_vector<S: Storage<T>, R_STORE: Storage<T>>(
        &self,
        x: &Matrix<T, S>,
        y: &mut Matrix<T, R_STORE>,
    ) -> Result<(), String> {
        if x.cols() != 1 || y.cols() != 1 {
            return Err("BSR mul_dense_vector only supports Matrix vectors (cols=1)".to_string());
        }
        if x.rows() != self.cols() {
            return Err("Incompatible dimensions: x.rows != Bsr.cols()".to_string());
        }
        if y.rows() != self.rows() {
            return Err("Incompatible dimensions: y.rows != Bsr.rows()".to_string());
        }

        let x_ptr = x.storage().data().as_ptr();
        let y_ptr = y.storage_mut().data_mut().as_mut_ptr();
        
        // Zero the output vector first
        for i in 0..y.rows() {
            unsafe { *y_ptr.add(i) = T::zero(); }
        }

        let block_size = R * C;

        for br in 0..self.block_rows {
            let start = self.outer_starts[br];
            let end = self.outer_starts[br + 1];

            for idx in start..end {
                let bc = self.inner_indices[idx];
                
                // Block offset in the values array
                let val_offset = idx * block_size;
                
                // Vector offsets
                let row_offset = br * R;
                let col_offset = bc * C;

                // Multiply the R x C dense block with the C x 1 vector segment
                for c in 0..C {
                    let rx = unsafe { *x_ptr.add(col_offset + c) };
                    for r in 0..R {
                        unsafe {
                            let val = *self.values.as_ptr().add(val_offset + c * R + r);
                            *y_ptr.add(row_offset + r) += val * rx;
                        }
                    }
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
    fn test_bsr_creation_and_spmv() {
        // Create a 4x4 matrix using 2x2 blocks
        // Block Layout:
        // [ B0  0  ]
        // [ 0   B1 ]
        //
        // B0 = [1 2; 3 4]
        // B1 = [5 6; 7 8]

        let values = vec![
            1.0, 3.0, 2.0, 4.0, // B0 (Column-major)
            5.0, 7.0, 6.0, 8.0, // B1 (Column-major)
        ];
        let inner_indices = vec![0, 1];
        let outer_starts = vec![0, 1, 2];

        let bsr = BsrMatrix::<f64, 2, 2>::from_raw(2, 2, values, inner_indices, outer_starts).unwrap();

        assert_eq!(bsr.rows(), 4);
        assert_eq!(bsr.cols(), 4);

        // x = [1, 1, 1, 1]^T
        let x = Matrix::<f64, DynamicStorage<f64>>::from_vec(4, 1, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let mut y = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(4, 1).unwrap();

        bsr.mul_dense_vector(&x, &mut y).unwrap();

        // Check result:
        // y[0] = B0_row0 * [1, 1] = 1*1 + 2*1 = 3
        // y[1] = B0_row1 * [1, 1] = 3*1 + 4*1 = 7
        // y[2] = B1_row0 * [1, 1] = 5*1 + 6*1 = 11
        // y[3] = B1_row1 * [1, 1] = 7*1 + 8*1 = 15

        assert_eq!(*y.get(0, 0).unwrap(), 3.0);
        assert_eq!(*y.get(1, 0).unwrap(), 7.0);
        assert_eq!(*y.get(2, 0).unwrap(), 11.0);
        assert_eq!(*y.get(3, 0).unwrap(), 15.0);
    }
}
