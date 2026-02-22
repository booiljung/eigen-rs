use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::ordering::{Ordering, Permutation, COLAMD};
use crate::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder};
use crate::core::storage::{DynamicStorage, Storage};

/// Simplicial Cholesky (LLT) factorization of a sparse symmetric positive-definite matrix.
pub struct SimplicialLLT<T: Scalar> {
    l: SparseMatrix<T>,
    p: Option<Permutation>,
    is_factorized: bool,
}

impl<T: Scalar + std::cmp::PartialOrd> Default for SimplicialLLT<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + std::cmp::PartialOrd> SimplicialLLT<T> {
    /// Creates a new uninitialized solver.
    pub fn new() -> Self {
        Self {
            l: SparseMatrix::new(0, 0, StorageOrder::ColMajor),
            p: None,
            is_factorized: false,
        }
    }

    /// Computes the factorization of the given matrix using COLAMD ordering.
    pub fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        self.compute_with_ordering(matrix, &COLAMD)
    }

    /// Computes the factorization using a specific ordering strategy.
    pub fn compute_with_ordering<O: Ordering>(&mut self, matrix: &SparseMatrix<T>, ordering: &O) -> Result<(), String> {
        self.analyze_pattern_with_ordering(matrix, ordering)?;
        self.factorize(matrix)
    }

    /// Symbolic factorization: analyzes the sparsity pattern using COLAMD.
    pub fn analyze_pattern(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        self.analyze_pattern_with_ordering(matrix, &COLAMD)
    }

    /// Symbolic factorization: analyzes the sparsity pattern using a specific ordering.
    pub fn analyze_pattern_with_ordering<O: Ordering>(&mut self, matrix: &SparseMatrix<T>, ordering: &O) -> Result<(), String> {
        if matrix.rows() != matrix.cols() {
            return Err("Matrix must be square for Cholesky factorization".to_string());
        }

        let p = ordering.compute(matrix);

        // Store inverse permutation for convenience if needed,
        // though our Permutation struct has inv_indices.
        // We clone it for now to keep ownership simple.
        // Actually Permutation struct holds both indices and inv_indices.

        self.p = Some(p);
        self.is_factorized = false;
        Ok(())
    }

    /// Numerical factorization: computes the actual LLT decomposition.
    pub fn factorize(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let n = matrix.rows();

        // 1. Permute Matrix: A_prime = P * A * P^T
        let a_prime = if let Some(ref p) = self.p {
            p.permute_symmetric(matrix)
        } else {
            matrix.clone()
        };

        // We'll store columns of L as they are computed.
        let mut l_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut l_dense = vec![T::default(); n]; // Temporary dense column

        for j in 0..n {
            // 1. Initialize dense column with A's j-th column (lower part)
            let mut it = InnerIterator::new(&a_prime, j);
            while it.is_valid() {
                let r = it.row();
                // We only care about lower triangular part
                // Note: permute_symmetric creates a full matrix if input was full.
                if r >= j {
                    l_dense[r] = it.value();
                }
                it.next();
            }

            // 2. Update with previous columns of L
            for (_k, l_col_k) in l_cols.iter().enumerate().take(j) {
                // Find L(j,k)
                let mut l_jk = T::default();
                for &(r, val) in l_col_k {
                    if r == j {
                        l_jk = val;
                        break;
                    }
                }

                if l_jk != T::default() {
                    // Update current dense column with col k
                    for &(r, val) in l_col_k {
                        if r >= j {
                            l_dense[r] -= l_jk * val;
                        }
                    }
                }
            }

            // 3. Finalize column j
            let l_jj_sq = l_dense[j];
            if l_jj_sq <= T::default() {
                return Err("Matrix is not positive definite".to_string());
            }
            let l_jj = l_jj_sq.sqrt();
            l_dense[j] = l_jj;

            l_cols[j].push((j, l_jj));
            for (i, val) in l_dense.iter_mut().enumerate().take(n).skip(j + 1) {
                *val /= l_jj;
                if *val != T::default() {
                    l_cols[j].push((i, *val));
                }
                *val = T::default(); // Reset for next use
            }
            l_dense[j] = T::default(); // Reset
        }

        // Convert l_cols to SparseMatrix
        let mut l_triplets = Vec::new();
        for (j, l_col_j) in l_cols.iter().enumerate().take(n) {
            for &(i, val) in l_col_j {
                l_triplets.push(crate::core::sparse::sparse_matrix::Triplet::new(i, j, val));
            }
        }

        self.l = SparseMatrix::new(n, n, StorageOrder::ColMajor);
        self.l.set_from_triplets(l_triplets);
        self.is_factorized = true;
        Ok(())
    }

    /// Solves the linear system Ax = b.
    pub fn solve<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_factorized {
            return Err("Solver is not factorized".to_string());
        }
        let n = self.l.rows();
        if b.rows() != n {
            return Err("Incompatible dimensions for solve".to_string());
        }

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;

        let p = self.p.as_ref().unwrap(); // Should exist if factorized
        let _p_indices = p.indices(); // new = old[p[i]]? Check Permutation def.
        let p_inv = p.inverse_indices(); // new -> old map?
                                         // Wait, Permutation::permute_symmetric used inv_indices to map A.
                                         // P maps: new_vector[inv_indices[i]] = old_vector[i] (scatter)
                                         // or new_vector[i] = old_vector[indices[i]] (gather)

        // Let's rely on logic:
        // A x = b
        // P A P^T (P x) = P b
        // A' y = c
        // where y = P x, c = P b.

        // 1. Compute c = P b
        // c[i] = b[indices[i]] (Gather)
        // Check `permute_symmetric` used `inv_indices`.
        // If `permute_symmetric` maps A_{r,c} to A_{inv[r], inv[c]},
        // then it effectively renames index k to inv[k].
        // This corresponds to y[inv[k]] = x[k].
        // So y[new_idx] = x[old_idx].
        // Where new_idx = inv[old_idx].
        // So c[inv[k]] = b[k]. (Scatter)

        // Let's verify `Permutation::new` again.
        // indices[i] = p. inv_indices[p] = i.
        // indices maps: new_pos -> old_pos.
        // inv_indices maps: old_pos -> new_pos.

        // So `permute_symmetric` used `inv_indices` to map row/col (old) to new_row/new_col (new).
        // Correct.
        // So to compute c = P b, we want c to be the "new" vector.
        // c[new_idx] = b[old_idx] => c[inv_indices[k]] = b[k].

        for k in 0..b.cols() {
            let mut c = vec![T::default(); n];
            // 1. Permute b -> c
            for i in 0..n {
                let new_idx = p_inv[i];
                c[new_idx] = *b.get(i, k).unwrap();
            }

            // 2. Solve L L^T y = c
            // Forward substitution L * z = c
            for j in 0..n {
                let mut it = InnerIterator::new(&self.l, j);
                if it.is_valid() && it.row() == j {
                    let l_jj = it.value();
                    c[j] /= l_jj;
                    it.next();
                } else {
                    return Err("Missing diagonal element in L".to_string());
                }

                let s_j = c[j];
                while it.is_valid() {
                    let r = it.row();
                    c[r] -= s_j * it.value();
                    it.next();
                }
            }

            // Backward substitution L^T * y = z
            for j in (0..n).rev() {
                let mut it = InnerIterator::new(&self.l, j);
                let mut l_jj = T::default();
                if it.is_valid() && it.row() == j {
                    l_jj = it.value();
                    it.next();
                }

                let mut sum = T::default();
                while it.is_valid() {
                    sum += it.value() * c[it.row()];
                    it.next();
                }
                c[j] = (c[j] - sum) / l_jj;
            }

            // 3. Permute y -> x
            // y is "new" vector. x is "old".
            // x = P^T y.
            // x[old_idx] = y[new_idx] => x[i] = y[inv_indices[i]]

            for i in 0..n {
                let new_idx = p_inv[i];
                *x.get_mut(i, k).unwrap() = c[new_idx];
            }
        }

        Ok(x)
    }

    pub fn matrix_l(&self) -> &SparseMatrix<T> {
        &self.l
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::sparse::sparse_matrix::Triplet;

    #[test]
    fn test_sparse_llt_basic() {
        // A = [ 2 -1  0 ]
        //     [-1  2 -1 ]
        //     [ 0 -1  2 ]
        // This is a common SPD matrix (Poisson 1D).
        let mut a = SparseMatrix::<f64>::new(3, 3, StorageOrder::ColMajor);
        a.set_from_triplets(vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, 2.0),
            Triplet::new(1, 2, -1.0),
            Triplet::new(2, 1, -1.0),
            Triplet::new(2, 2, 2.0),
        ]);

        let mut llt = SimplicialLLT::new();
        llt.compute(&a).unwrap();

        let l = llt.matrix_l();
        assert_eq!(l.rows(), 3);
        assert_eq!(l.cols(), 3);

        let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![1.0, 0.0, 1.0]).unwrap();
        let x = llt.solve(&b).unwrap();

        assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x.get(1, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((x.get(2, 0).unwrap() - 1.0).abs() < 1e-10);
    }
}
