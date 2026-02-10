//! Incomplete Cholesky (IC) preconditioner.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::solvers::iterative_solver_base::Preconditioner;
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::storage::{DynamicStorage, Storage};

/// Incomplete Cholesky (IC) preconditioner.
pub struct IncompleteCholesky<T: Scalar> {
    l: SparseMatrix<T>,
    is_initialized: bool,
}

impl<T: Scalar> Default for IncompleteCholesky<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> IncompleteCholesky<T> {
    pub fn new() -> Self {
        Self {
            l: SparseMatrix::new(0, 0, crate::core::sparse::StorageOrder::ColMajor),
            is_initialized: false,
        }
    }
}

impl<T: Scalar> Preconditioner<T> for IncompleteCholesky<T> {
    fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let n = matrix.rows();
        self.l = SparseMatrix::new(n, n, crate::core::sparse::StorageOrder::ColMajor);

        // IC(0) - Sparsity(L) = Sparsity(lower(A))
        // We need to keep track of the sparsity pattern.
        let mut pattern = vec![Vec::new(); n];
        for (j, pat) in pattern.iter_mut().enumerate().take(n) {
            let mut it = InnerIterator::new(matrix, j);
            while it.is_valid() {
                let i = it.row();
                if i >= j {
                    pat.push(i);
                }
                it.next();
            }
        }

        let mut l_values = vec![Vec::new(); n];
        let mut workspace = vec![T::default(); n];

        for j in 0..n {
            // Load column j of A into workspace
            let mut it = InnerIterator::new(matrix, j);
            while it.is_valid() {
                let i = it.row();
                if i >= j {
                    workspace[i] = it.value();
                }
                it.next();
            }

            // Update column j using previous columns k < j
            for k in 0..j {
                // Find L_jk
                let mut l_jk = T::default();
                // We need efficient access to L_jk.
                // Since l_values[k] contains rows starting from k, we can search for j.
                // In IC(0), if (j, k) is in the pattern, it exists.

                // Let's optimize: we only care about k such that L_jk is in the pattern of A.
                // But Cholesky is more complex. Let's find l_jk in l_values[k].
                // The rows in l_values[k] correspond to pattern[k].
                if let Some(pos) = pattern[k].iter().position(|&r| r == j) {
                    l_jk = l_values[k][pos];
                }

                if l_jk != T::default() {
                    // For each i >= j such that (i, j) is in the pattern AND (i, k) is in the pattern
                    // In IC(0), we only update positions that are in the pattern of A.
                    for (idx_k, &i) in pattern[k].iter().enumerate() {
                        if i >= j {
                            // Check if i is in pattern[j]
                            if pattern[j].contains(&i) {
                                let l_ik = l_values[k][idx_k];
                                workspace[i] -= l_jk * l_ik;
                            }
                        }
                    }
                }
            }

            // Calculate L_jj and column j
            let l_jj_sq = workspace[j];
            if l_jj_sq.to_f64() <= 1e-18 {
                // Not positive definite or too small, fallback/fail
                return Err(format!(
                    "IC(0) failed at column {}: diagonal element {} is not positive",
                    j,
                    l_jj_sq.to_f64()
                ));
            }
            let l_jj = l_jj_sq.sqrt();

            for &i in &pattern[j] {
                let val = if i == j { l_jj } else { workspace[i] / l_jj };
                l_values[j].push(val);
                workspace[i] = T::default(); // Reset workspace
            }
        }

        // Convert l_values and pattern to SparseMatrix
        let mut triplets = Vec::new();
        for j in 0..n {
            for (idx, &i) in pattern[j].iter().enumerate() {
                triplets.push(crate::core::sparse::Triplet::new(i, j, l_values[j][idx]));
            }
        }
        self.l.set_from_triplets(triplets);
        self.is_initialized = true;
        Ok(())
    }

    fn solve<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_initialized {
            return Err("Preconditioner not initialized".to_string());
        }

        let n = b.rows();
        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;

        for k in 0..b.cols() {
            // Forward solve L*y = b
            let mut y = vec![T::default(); n];
            for (i, val) in y.iter_mut().enumerate().take(n) {
                *val = *b.get(i, k).unwrap();
            }

            for j in 0..n {
                let mut it = InnerIterator::new(&self.l, j);
                let mut diag = T::from_f64(1.0);
                while it.is_valid() {
                    if it.row() == j {
                        diag = it.value();
                        break;
                    }
                    it.next();
                }

                if diag.abs().to_f64() > 1e-18 {
                    y[j] /= diag;
                }

                let val_j = y[j];
                let mut it = InnerIterator::new(&self.l, j);
                while it.is_valid() {
                    let r = it.row();
                    if r > j {
                        y[r] -= val_j * it.value();
                    }
                    it.next();
                }
            }

            // Backward solve L^T * x = y
            for j in (0..n).rev() {
                // For L^T, row j is column j of L? No.
                // Row j of L^T is column j of L (elements L_ij for i >= j).
                // x_j = (y_j - sum_{i > j} L_ij x_i) / L_jj
                // This is a dot product of column j of L with x[j..n].
                let mut sum = T::default();
                let mut it = InnerIterator::new(&self.l, j);
                let mut diag = T::from_f64(1.0);
                while it.is_valid() {
                    let i = it.row();
                    if i == j {
                        diag = it.value();
                    } else if i > j {
                        sum += it.value() * y[i];
                    }
                    it.next();
                }
                y[j] -= sum;
                if diag.abs().to_f64() > 1e-18 {
                    y[j] /= diag;
                }
            }

            for (i, val) in y.iter().enumerate().take(n) {
                *x.get_mut(i, k).unwrap() = *val;
            }
        }

        Ok(x)
    }
}
