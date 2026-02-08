//! Incomplete LU with Thresholding (ILUT) preconditioner.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::solvers::iterative_solver_base::Preconditioner;
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::storage::{DynamicStorage, Storage};

/// Incomplete LU with Thresholding (ILUT) preconditioner.
pub struct IncompleteLUT<T: Scalar> {
    l: SparseMatrix<T>,
    u: SparseMatrix<T>,
    fill_factor: f64,
    drop_tol: f64,
    is_initialized: bool,
}

impl<T: Scalar> Default for IncompleteLUT<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> IncompleteLUT<T> {
    pub fn new() -> Self {
        Self {
            l: SparseMatrix::new(0, 0, crate::core::sparse::StorageOrder::ColMajor),
            u: SparseMatrix::new(0, 0, crate::core::sparse::StorageOrder::ColMajor),
            fill_factor: 10.0,
            drop_tol: 1e-4,
            is_initialized: false,
        }
    }

    pub fn set_fill_factor(&mut self, factor: f64) {
        self.fill_factor = factor;
    }

    pub fn set_drop_tolerance(&mut self, tol: f64) {
        self.drop_tol = tol;
    }
}

impl<T: Scalar> Preconditioner<T> for IncompleteLUT<T> {
    fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let n = matrix.rows();
        self.l = SparseMatrix::new(n, n, crate::core::sparse::StorageOrder::ColMajor);
        self.u = SparseMatrix::new(n, n, crate::core::sparse::StorageOrder::ColMajor);

        // ILU(0) - Sparsity pattern of (L+U) = Sparsity pattern of A
        let mut pattern_l = vec![Vec::new(); n];
        let mut pattern_u = vec![Vec::new(); n];
        for j in 0..n {
            let mut it = InnerIterator::new(matrix, j);
            while it.is_valid() {
                let i = it.row();
                if i < j {
                    pattern_u[j].push(i);
                } else if i == j {
                    pattern_u[j].push(i);
                } else {
                    pattern_l[j].push(i);
                }
                it.next();
            }
        }

        let mut l_values = vec![Vec::new(); n];
        let mut u_values = vec![Vec::new(); n];
        let mut workspace = vec![T::default(); n];

        for j in 0..n {
            // Load column j of A into workspace
            let mut it = InnerIterator::new(matrix, j);
            while it.is_valid() {
                workspace[it.row()] = it.value();
                it.next();
            }

            // Update column j using previous columns k < j
            // We need to find k such that U_kj != 0
            for k in 0..j {
                // Find U_kj
                let mut u_kj = T::default();
                if let Some(_pos) = pattern_u[j].iter().position(|&r| r == k) {
                    // Wait, this is tricky. U_kj is in column j, but we are computing column j.
                    // Actually, at this point, we've computed U_kj if it's in the workspace.
                    u_kj = workspace[k];
                }

                if u_kj != T::default() {
                    // For each i > k such that L_ik != 0
                    // And we only update if (i, j) is in pattern
                    // L_values[k] contains L_ik for i in pattern_l[k]
                    for (idx_k, &i) in pattern_l[k].iter().enumerate() {
                        if i >= j {
                            // Check if i is in pattern_l[j] or pattern_u[j]
                            // For simplicity, let's just use a fast way to check pattern
                            // or just skip if not in pattern.
                            let in_pattern = if i == j {
                                true
                            } else if i > j {
                                pattern_l[j].contains(&i)
                            } else {
                                pattern_u[j].contains(&i)
                            };

                            if in_pattern {
                                let l_ik = l_values[k][idx_k];
                                workspace[i] -= l_ik * u_kj;
                            }
                        } else {
                            // i < j, check in pattern_u[j]
                            if pattern_u[j].contains(&i) {
                                let l_ik = l_values[k][idx_k];
                                workspace[i] -= l_ik * u_kj;
                            }
                        }
                    }
                }
            }

            // Finalize column j
            // U_jj = workspace[j]
            let u_jj = workspace[j];
            if u_jj.abs().to_f64() < 1e-16 {
                workspace[j] = T::from_f64(1.0);
            }
            let u_jj_safe = workspace[j];

            for &i in &pattern_u[j] {
                u_values[j].push(workspace[i]);
                workspace[i] = T::default();
            }

            for &i in &pattern_l[j] {
                l_values[j].push(workspace[i] / u_jj_safe);
                workspace[i] = T::default();
            }
        }

        // Convert to SparseMatrix
        let mut l_triplets = Vec::new();
        let mut u_triplets = Vec::new();
        for j in 0..n {
            l_triplets.push(crate::core::sparse::Triplet::new(j, j, T::from_f64(1.0)));
            for (idx, &i) in pattern_l[j].iter().enumerate() {
                l_triplets.push(crate::core::sparse::Triplet::new(i, j, l_values[j][idx]));
            }
            for (idx, &i) in pattern_u[j].iter().enumerate() {
                u_triplets.push(crate::core::sparse::Triplet::new(i, j, u_values[j][idx]));
            }
        }

        self.l.set_from_triplets(l_triplets);
        self.u.set_from_triplets(u_triplets);
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
            // Forward substitution for L (unit diagonal)
            let mut y = vec![T::default(); n];
            for i in 0..n {
                y[i] = *b.get(i, k).unwrap();
            }

            for j in 0..n {
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

            // Backward substitution for U (CSC)
            // Column-based backward solve:
            // For j from n-1 down to 0:
            //   x_j = y_j / U_jj
            //   For i < j: y_i = y_i - U_ij * x_j
            for j in (0..n).rev() {
                let mut it = InnerIterator::new(&self.u, j);
                let mut diag_val = T::from_f64(1.0);
                while it.is_valid() {
                    if it.row() == j {
                        diag_val = it.value();
                        break;
                    }
                    it.next();
                }

                if diag_val.abs().to_f64() > 1e-18 {
                    y[j] /= diag_val;
                }

                let val_j = y[j];
                let mut it = InnerIterator::new(&self.u, j);
                while it.is_valid() {
                    let r = it.row();
                    if r < j {
                        y[r] -= val_j * it.value();
                    }
                    it.next();
                }
            }

            for i in 0..n {
                *x.get_mut(i, k).unwrap() = y[i];
            }
        }

        Ok(x)
    }
}
