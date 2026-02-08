//! Sparse matrix implementation using Compressed Storage formats (CSR/CSC).

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::storage::{DynamicStorage, Storage};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Storage order for the sparse matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageOrder {
    RowMajor, // Compressed Sparse Row (CSR)
    ColMajor, // Compressed Sparse Column (CSC)
}

/// A triplet (row, col, value) used for efficient sparse matrix construction.
#[derive(Debug, Clone)]
pub struct Triplet<T: Scalar> {
    pub row: usize,
    pub col: usize,
    pub value: T,
}

impl<T: Scalar> Triplet<T> {
    pub fn new(row: usize, col: usize, value: T) -> Self {
        Self { row, col, value }
    }
}

/// Sparse matrix in compressed storage format.
#[derive(Clone)]
pub struct SparseMatrix<T: Scalar> {
    rows: usize,
    cols: usize,
    values: Vec<T>,
    inner_indices: Vec<usize>,
    outer_starts: Vec<usize>,
    order: StorageOrder,
}

impl<T: Scalar> SparseMatrix<T> {
    /// Creates a new empty sparse matrix of given dimensions.
    pub fn new(rows: usize, cols: usize, order: StorageOrder) -> Self {
        let outer_size = match order {
            StorageOrder::RowMajor => rows,
            StorageOrder::ColMajor => cols,
        };
        Self {
            rows,
            cols,
            values: Vec::new(),
            inner_indices: Vec::new(),
            outer_starts: vec![0; outer_size + 1],
            order,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn non_zeros(&self) -> usize {
        self.values.len()
    }
    pub fn order(&self) -> StorageOrder {
        self.order
    }
    pub fn outer_size(&self) -> usize {
        match self.order {
            StorageOrder::RowMajor => self.rows,
            StorageOrder::ColMajor => self.cols,
        }
    }

    /// Sets the matrix content from a list of triplets.
    /// This is the most efficient way to build a sparse matrix.
    pub fn set_from_triplets(&mut self, mut triplets: Vec<Triplet<T>>) {
        if triplets.is_empty() {
            self.values.clear();
            self.inner_indices.clear();
            let outer_size = if self.order == StorageOrder::RowMajor {
                self.rows
            } else {
                self.cols
            };
            self.outer_starts = vec![0; outer_size + 1];
            return;
        }

        // 1. Sort triplets based on storage order
        match self.order {
            StorageOrder::RowMajor => {
                triplets.sort_by(|a, b| a.row.cmp(&b.row).then(a.col.cmp(&b.col)));
            }
            StorageOrder::ColMajor => {
                triplets.sort_by(|a, b| a.col.cmp(&b.col).then(a.row.cmp(&b.row)));
            }
        }

        // 2. Sum duplicate entries
        let mut unique_triplets: Vec<Triplet<T>> = Vec::with_capacity(triplets.len());
        if !triplets.is_empty() {
            let mut current = triplets[0].clone();
            for next in triplets.into_iter().skip(1) {
                if next.row == current.row && next.col == current.col {
                    current.value += next.value;
                } else {
                    unique_triplets.push(current);
                    current = next;
                }
            }
            unique_triplets.push(current);
        }

        // 3. Populate compressed structure
        let outer_limit = if self.order == StorageOrder::RowMajor {
            self.rows
        } else {
            self.cols
        };
        self.values = Vec::with_capacity(unique_triplets.len());
        self.inner_indices = Vec::with_capacity(unique_triplets.len());
        self.outer_starts = vec![0; outer_limit + 1];

        let mut current_outer = 0;
        for t in unique_triplets {
            let (outer, inner) = if self.order == StorageOrder::RowMajor {
                (t.row, t.col)
            } else {
                (t.col, t.row)
            };

            while current_outer < outer {
                current_outer += 1;
                self.outer_starts[current_outer] = self.values.len();
            }

            self.values.push(t.value);
            self.inner_indices.push(inner);
        }

        while current_outer < outer_limit {
            current_outer += 1;
            self.outer_starts[current_outer] = self.values.len();
        }
    }

    /// Returns the raw values.
    pub fn value_ptr(&self) -> *const T {
        self.values.as_ptr()
    }
    /// Returns the inner indices.
    pub fn inner_index_ptr(&self) -> *const usize {
        self.inner_indices.as_ptr()
    }
    /// Returns the outer starts.
    pub fn outer_start_ptr(&self) -> *const usize {
        self.outer_starts.as_ptr()
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

    /// Returns the transpose of the sparse matrix.
    /// If the matrix is RowMajor (CSR), the transpose will be ColMajor (CSC) with the same data, and vice-versa.
    /// To keep the same storage order, use `transpose_inplace_reorder`.
    pub fn transpose(&self) -> Self {
        let new_order = match self.order {
            StorageOrder::RowMajor => StorageOrder::ColMajor,
            StorageOrder::ColMajor => StorageOrder::RowMajor,
        };
        Self {
            rows: self.cols,
            cols: self.rows,
            values: self.values.clone(),
            inner_indices: self.inner_indices.clone(),
            outer_starts: self.outer_starts.clone(),
            order: new_order,
        }
    }

    /// Returns a transpose that preserves the current storage order.
    /// This requires a physical reordering of elements.
    pub fn transpose_reordered(&self) -> Self {
        let mut triplets = Vec::with_capacity(self.values.len());
        for i in 0..self.outer_starts.len() - 1 {
            let start = self.outer_starts[i];
            let end = self.outer_starts[i + 1];
            for k in start..end {
                let inner = self.inner_indices[k];
                let val = self.values[k];
                if self.order == StorageOrder::RowMajor {
                    // Current is (row: i, col: inner) -> Transpose is (row: inner, col: i)
                    triplets.push(Triplet::new(inner, i, val));
                } else {
                    // Current is (col: i, row: inner) -> Transpose is (col: inner, row: i)
                    triplets.push(Triplet::new(i, inner, val));
                }
            }
        }
        let mut res = Self::new(self.cols, self.rows, self.order);
        res.set_from_triplets(triplets);
        res
    }

    /// Scales all elements by a scalar factor.
    pub fn scale(&mut self, factor: T) {
        for v in self.values.iter_mut() {
            *v *= factor;
        }
    }

    pub(crate) fn from_raw(
        rows: usize,
        cols: usize,
        values: Vec<T>,
        inner_indices: Vec<usize>,
        outer_starts: Vec<usize>,
        order: StorageOrder,
    ) -> Self {
        Self {
            rows,
            cols,
            values,
            inner_indices,
            outer_starts,
            order,
        }
    }

    /// Multiplies this sparse matrix by a dense matrix.
    pub fn mul_dense<S: Storage<T>>(
        &self,
        rhs: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if self.cols != rhs.rows() {
            return Err("Incompatible dimensions for sparse-dense product".to_string());
        }
        let mut res = Matrix::<T, DynamicStorage<T>>::new_dynamic(self.rows, rhs.cols())?;

        #[cfg(feature = "parallel")]
        {
            if self.non_zeros() > 50000 || (self.rows * rhs.cols() > 10000) {
                return self.par_mul_dense(rhs);
            }
        }

        match self.order {
            StorageOrder::RowMajor => {
                for i in 0..self.rows {
                    let mut it = InnerIterator::new(self, i);
                    while it.is_valid() {
                        let val = it.value();
                        let col = it.index();
                        for j in 0..rhs.cols() {
                            *res.get_mut(i, j).unwrap() =
                                *res.get(i, j).unwrap() + val * (*rhs.get(col, j).unwrap());
                        }
                        it.next();
                    }
                }
            }
            StorageOrder::ColMajor => {
                for j in 0..self.cols {
                    let mut it = InnerIterator::new(self, j);
                    while it.is_valid() {
                        let val = it.value();
                        let row = it.index();
                        for k in 0..rhs.cols() {
                            *res.get_mut(row, k).unwrap() =
                                *res.get(row, k).unwrap() + val * (*rhs.get(j, k).unwrap());
                        }
                        it.next();
                    }
                }
            }
        }
        Ok(res)
    }

    /// Parallel multiplication by a dense matrix.
    #[cfg(feature = "parallel")]
    pub fn par_mul_dense<S: Storage<T> + Sync>(
        &self,
        rhs: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String>
    where
        T: Scalar + Send + Sync + 'static,
    {
        if self.cols != rhs.rows() {
            return Err("Incompatible dimensions for parallel sparse-dense product".to_string());
        }
        let mut res = Matrix::<T, DynamicStorage<T>>::new_dynamic(self.rows, rhs.cols())?;
        let res_rows = self.rows;
        let res_cols = rhs.cols();

        // Get raw pointer for parallel updates
        let res_ptr_val = res.storage_mut().data_mut().as_mut_ptr() as usize;

        match self.order {
            StorageOrder::RowMajor => {
                // CSR: Parallelize over rows i
                (0..self.rows).into_par_iter().for_each(move |i| {
                    let mut it = InnerIterator::new(self, i);
                    let res_ptr = res_ptr_val as *mut T;
                    while it.is_valid() {
                        let val = it.value();
                        let col = it.index();
                        for j in 0..res_cols {
                            unsafe {
                                let target = res_ptr.add(j * res_rows + i);
                                *target += val * (*rhs.get(col, j).unwrap());
                            }
                        }
                        it.next();
                    }
                });
            }
            StorageOrder::ColMajor => {
                // CSC: Parallelize over result columns j
                (0..rhs.cols()).into_par_iter().for_each(move |j| {
                    let res_ptr = res_ptr_val as *mut T;
                    for k in 0..self.cols {
                        let mut it = InnerIterator::new(self, k);
                        while it.is_valid() {
                            let val = it.value();
                            let row = it.index();
                            unsafe {
                                let target = res_ptr.add(j * res_rows + row);
                                *target += val * (*rhs.get(k, j).unwrap());
                            }
                            it.next();
                        }
                    }
                });
            }
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_triplets_csc() {
        let mut m = SparseMatrix::<f64>::new(3, 3, StorageOrder::ColMajor);
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(2, 0, 2.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(0, 2, 4.0),
            Triplet::new(2, 2, 5.0),
            Triplet::new(1, 1, 1.0), // Duplicate (1,1) -> 3.0 + 1.0 = 4.0
        ];
        m.set_from_triplets(triplets);

        assert_eq!(m.non_zeros(), 5);
        assert_eq!(m.values(), &[1.0, 2.0, 4.0, 4.0, 5.0]);
        assert_eq!(m.inner_indices(), &[0, 2, 1, 0, 2]);
        assert_eq!(m.outer_starts(), &[0, 2, 3, 5]);
    }

    #[test]
    fn test_sparse_matrix_triplets_csr() {
        let mut m = SparseMatrix::<f64>::new(3, 3, StorageOrder::RowMajor);
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 2, 4.0),
            Triplet::new(1, 1, 4.0),
            Triplet::new(2, 0, 2.0),
            Triplet::new(2, 2, 5.0),
        ];
        m.set_from_triplets(triplets);

        assert_eq!(m.non_zeros(), 5);
        assert_eq!(m.values(), &[1.0, 4.0, 4.0, 2.0, 5.0]);
        assert_eq!(m.inner_indices(), &[0, 2, 1, 0, 2]);
        assert_eq!(m.outer_starts(), &[0, 2, 3, 5]);
    }

    #[test]
    fn test_sparse_dense_mul() {
        let mut m = SparseMatrix::<f64>::new(2, 3, StorageOrder::RowMajor);
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 2, 2.0),
            Triplet::new(1, 1, 3.0),
        ];
        m.set_from_triplets(triplets);

        let mut dense = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(3, 2).unwrap();
        // [1 2; 3 4; 5 6]
        let dense_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for i in 0..3 {
            for j in 0..2 {
                *dense.get_mut(i, j).unwrap() = dense_data[i * 2 + j];
            }
        }

        let res = m.mul_dense(&dense).unwrap();
        // Row 0: 1*[1 2] + 2*[5 6] = [1+10 2+12] = [11 14]
        // Row 1: 3*[3 4] = [9 12]
        assert_eq!(*res.get(0, 0).unwrap(), 11.0);
        assert_eq!(*res.get(0, 1).unwrap(), 14.0);
        assert_eq!(*res.get(1, 0).unwrap(), 9.0);
        assert_eq!(*res.get(1, 1).unwrap(), 12.0);
    }

    #[test]
    fn test_sparse_dense_mul_csc() {
        let mut m = SparseMatrix::<f64>::new(2, 3, StorageOrder::ColMajor);
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 2, 2.0),
            Triplet::new(1, 1, 3.0),
        ];
        m.set_from_triplets(triplets);

        let mut dense = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(3, 2).unwrap();
        let dense_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for i in 0..3 {
            for j in 0..2 {
                *dense.get_mut(i, j).unwrap() = dense_data[i * 2 + j];
            }
        }

        let res = m.mul_dense(&dense).unwrap();
        assert_eq!(*res.get(0, 0).unwrap(), 11.0);
        assert_eq!(*res.get(0, 1).unwrap(), 14.0);
        assert_eq!(*res.get(1, 0).unwrap(), 9.0);
        assert_eq!(*res.get(1, 1).unwrap(), 12.0);
    }

    #[test]
    fn test_sparse_transpose() {
        let mut m = SparseMatrix::<f64>::new(2, 3, StorageOrder::RowMajor);
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 2, 2.0),
            Triplet::new(1, 1, 3.0),
        ];
        m.set_from_triplets(triplets);

        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.order(), StorageOrder::ColMajor);
        assert_eq!(t.values(), m.values());

        let t_re = m.transpose_reordered();
        assert_eq!(t_re.rows(), 3);
        assert_eq!(t_re.cols(), 2);
        assert_eq!(t_re.order(), StorageOrder::RowMajor);
        // Transpose of RowMajor(2x3) -> RowMajor(3x2)
        // (0,0)->1.0, (2,0)->2.0, (1,1)->3.0
        assert_eq!(t_re.non_zeros(), 3);
    }

    #[test]
    fn test_sparse_addition() {
        let mut a = SparseMatrix::<f64>::new(2, 2, StorageOrder::RowMajor);
        a.set_from_triplets(vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 2.0)]);

        let mut b = SparseMatrix::<f64>::new(2, 2, StorageOrder::RowMajor);
        b.set_from_triplets(vec![Triplet::new(0, 0, 10.0), Triplet::new(0, 1, 5.0)]);

        let res = (&a + &b).unwrap();
        assert_eq!(res.non_zeros(), 3);
        // (0,0) -> 1+10=11, (0,1) -> 5, (1,1) -> 2
        assert_eq!(res.values(), &[11.0, 5.0, 2.0]);
    }

    #[test]
    fn test_sparse_sparse_mul() {
        let mut a = SparseMatrix::<f64>::new(2, 3, StorageOrder::RowMajor);
        a.set_from_triplets(vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 2, 3.0)]);

        let mut b = SparseMatrix::<f64>::new(3, 2, StorageOrder::RowMajor);
        b.set_from_triplets(vec![
            Triplet::new(0, 0, 10.0),
            Triplet::new(0, 1, 5.0),
            Triplet::new(2, 0, 1.0),
        ]);

        let res = (&a * &b).unwrap();
        assert_eq!(res.rows(), 2);
        assert_eq!(res.cols(), 2);
        assert_eq!(res.non_zeros(), 2);
        assert_eq!(res.values(), &[13.0, 5.0]);
        assert_eq!(res.inner_indices(), &[0, 1]);
    }
}
