//! Simplicial Sparse LU factorization for non-symmetric sparse matrices.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};
use crate::core::storage::{DynamicStorage, Storage};

/// Simplicial Sparse LU factorization of a general sparse matrix.
/// A = P * L * U * Q where P and Q are permutation matrices.
/// This initial implementation uses partial pivoting (P * A = L * U).
pub struct SparseLU<T: Scalar> {
    l: SparseMatrix<T>,
    u: SparseMatrix<T>,
    p: Vec<usize>, // Row permutation: p[i] = row index in A that is now at row i in LU
    p_inv: Vec<usize>,
    is_factorized: bool,
}

impl<T: Scalar> Default for SparseLU<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> SparseLU<T> {
    /// Creates a new uninitialized solver.
    pub fn new() -> Self {
        Self {
            l: SparseMatrix::new(0, 0, StorageOrder::ColMajor),
            u: SparseMatrix::new(0, 0, StorageOrder::ColMajor),
            p: Vec::new(),
            p_inv: Vec::new(),
            is_factorized: false,
        }
    }

    /// Computes the LU factorization of the given matrix.
    pub fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        self.analyze_pattern(matrix)?;
        self.factorize(matrix)
    }

    /// Symbolic factorization: prepares the structures.
    pub fn analyze_pattern(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        if matrix.rows() != matrix.cols() {
            return Err("Matrix must be square for LU factorization".to_string());
        }
        let n = matrix.rows();
        self.p = (0..n).collect();
        self.p_inv = (0..n).collect();
        self.l = SparseMatrix::new(n, n, StorageOrder::ColMajor);
        self.u = SparseMatrix::new(n, n, StorageOrder::ColMajor);
        self.is_factorized = false;
        Ok(())
    }

    /// Numerical factorization: computes L and U with partial pivoting.
    pub fn factorize(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let n = matrix.rows();

        // Simplicial LU often uses a column-by-column approach.
        // For column j, we solve L * u_j = a_j (where a_j is the j-th column of permuted A).
        // Then we find the pivot in the remaining part of u_j to define L(i, j) entries.

        let mut l_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut u_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];

        // Workspace for current column
        let mut dense_col = vec![T::default(); n];
        let mut p = (0..n).collect::<Vec<usize>>(); // Current row permutation

        for j in 0..n {
            // 1. Initialize dense_col with the j-th column of A
            let mut it = InnerIterator::new(matrix, j);
            while it.is_valid() {
                dense_col[it.row()] = it.value();
                it.next();
            }

            // 2. Solve L * u_j = a_j for the part above diagonal (u_j elements)
            for k in 0..j {
                let pivot_row = p[k];
                let l_kj_val = dense_col[pivot_row]; // This is effectively U(k, j)

                if l_kj_val != T::default() {
                    for &(row, l_val) in &l_cols[k] {
                        if row > k {
                            dense_col[p[row]] -= l_val * l_kj_val;
                        }
                    }
                    u_cols[j].push((k, l_kj_val));
                }
            }

            // 3. Partial Pivoting: Find max in dense_col[p[j..n]]
            let mut pivot_idx = j;
            let mut max_abs = -1.0;

            for i in j..n {
                let row = p[i];
                let val_abs_f64 = dense_col[row].abs().to_f64();
                if val_abs_f64 > max_abs {
                    max_abs = val_abs_f64;
                    pivot_idx = i;
                }
            }

            if max_abs < 1e-14 {
                // TODO: epsilon
                return Err("Matrix is singular".to_string());
            }

            // Swap rows in p
            p.swap(j, pivot_idx);
            let j_row = p[j];

            // 4. Finalize U(j, j) and L(:, j)
            let u_jj = dense_col[j_row];
            u_cols[j].push((j, u_jj));

            l_cols[j].push((j, T::from_f64(1.0))); // Unit diagonal for L
            for i in (j + 1)..n {
                let curr_row = p[i];
                let l_ij = dense_col[curr_row] / u_jj;
                if l_ij != T::default() {
                    l_cols[j].push((i, l_ij));
                }
                dense_col[curr_row] = T::default(); // Reset workspace
            }
            dense_col[j_row] = T::default(); // Reset
                                             // Also reset upper part just in case
            for k in 0..j {
                dense_col[p[k]] = T::default();
            }
        }

        // Store results
        self.p = p.clone();
        self.p_inv = vec![0; n];
        for (i, &val) in p.iter().enumerate() {
            self.p_inv[val] = i;
        }

        let mut l_triplets = Vec::new();
        let mut u_triplets = Vec::new();
        for j in 0..n {
            for &(i_idx, val) in &l_cols[j] {
                l_triplets.push(crate::core::sparse::sparse_matrix::Triplet::new(
                    i_idx, j, val,
                ));
            }
            for &(i_idx, val) in &u_cols[j] {
                u_triplets.push(crate::core::sparse::sparse_matrix::Triplet::new(
                    i_idx, j, val,
                ));
            }
        }

        self.l.set_from_triplets(l_triplets);
        self.u.set_from_triplets(u_triplets);
        self.is_factorized = true;
        Ok(())
    }

    /// Solves Ax = b.
    pub fn solve<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_factorized {
            return Err("Solver is not factorized".to_string());
        }
        let n = self.l.rows();
        if b.rows() != n {
            return Err("Incompatible dimensions".to_string());
        }

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;

        for k in 0..b.cols() {
            let mut sol = vec![T::default(); n];
            // 1. Permute b: b' = P * b
            for i in 0..n {
                sol[i] = *b.get(self.p[i], k).unwrap();
            }

            // 2. Forward substitution L * y = b'
            for j in 0..n {
                let mut it = InnerIterator::new(&self.l, j);
                while it.is_valid() {
                    let r = it.row();
                    if r > j {
                        sol[r] = sol[r] - sol[j] * it.value();
                    }
                    it.next();
                }
            }

            // 3. Backward substitution U * x = y
            for j in (0..n).rev() {
                let mut it = InnerIterator::new(&self.u, j);
                let mut u_jj = T::default();
                let mut found_u_jj = false;
                while it.is_valid() {
                    if it.row() == j {
                        u_jj = it.value();
                        found_u_jj = true;
                        break;
                    }
                    it.next();
                }

                if !found_u_jj {
                    return Err("Missing diagonal in U".to_string());
                }

                let x_j = sol[j] / u_jj;
                sol[j] = x_j;

                // Column-wise backward update
                let mut it2 = InnerIterator::new(&self.u, j);
                while it2.is_valid() {
                    let r = it2.row();
                    if r < j {
                        sol[r] -= x_j * it2.value();
                    }
                    it2.next();
                }
            }

            for i in 0..n {
                *x.get_mut(i, k).unwrap() = sol[i];
            }
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sparse::Triplet;

    #[test]
    fn test_sparse_lu_basic() {
        // A = [ 1  2  0 ]
        //     [ 0  3  4 ]
        //     [ 5  0  6 ]
        let mut a = SparseMatrix::<f64>::new(3, 3, StorageOrder::ColMajor);
        a.set_from_triplets(vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(1, 2, 4.0),
            Triplet::new(2, 0, 5.0),
            Triplet::new(2, 2, 6.0),
        ]);

        let mut lu = SparseLU::new();
        lu.compute(&a).unwrap();

        let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![5.0, 11.0, 17.0]).unwrap();
        let _x = lu.solve(&b).unwrap();

        // Check Ax = b
        // 1*x0 + 2*x1 = 5
        // 3*x1 + 4*x2 = 11
        // 5*x0 + 6*x2 = 17
        // Solution is x = [1, 2, 2]
        // 1 + 4 = 5 (ok)
        // 6 + 8 = 14? Wait. 11. 3*2 + 4*x2 = 11 => 6 + 4*x2 = 11 => 4*x2 = 5 => x2 = 1.25
        // 5*1 + 6*1.25 = 5 + 7.5 = 12.5? No.

        // Let's pick b for x = [1, 1, 1]
        // b = [3, 7, 11]
        let b2 = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![3.0, 7.0, 11.0]).unwrap();
        let x2 = lu.solve(&b2).unwrap();

        assert!((x2.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x2.get(1, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x2.get(2, 0).unwrap() - 1.0).abs() < 1e-10);
    }
}
