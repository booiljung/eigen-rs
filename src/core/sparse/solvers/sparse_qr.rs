//! Sparse QR decomposition using Householder transformations.

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::sparse::iterators::InnerIterator;
use crate::core::sparse::ordering::{NaturalOrdering, Ordering, Permutation};
use crate::core::sparse::sparse_matrix::SparseMatrix;
use crate::core::storage::{DynamicStorage, Storage};

/// Sparse QR decomposition.
pub struct SparseQR<T: Scalar, O: Ordering = NaturalOrdering> {
    ordering: O,
    r: SparseMatrix<T>,
    p: Permutation,
    h_coeffs: Vec<T>,
    v_matrix: SparseMatrix<T>, // Stores Householder vectors v efficiently
    is_initialized: bool,
}

impl<T: Scalar> Default for SparseQR<T, NaturalOrdering> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> SparseQR<T, NaturalOrdering> {
    pub fn new() -> Self {
        Self {
            ordering: NaturalOrdering,
            r: SparseMatrix::new(0, 0, crate::core::sparse::StorageOrder::ColMajor),
            p: Permutation::new(Vec::new()),
            h_coeffs: Vec::new(),
            v_matrix: SparseMatrix::new(0, 0, crate::core::sparse::StorageOrder::ColMajor),
            is_initialized: false,
        }
    }
}

impl<T: Scalar, O: Ordering> SparseQR<T, O> {
    pub fn with_ordering(ordering: O) -> Self {
        Self {
            ordering,
            r: SparseMatrix::new(0, 0, crate::core::sparse::StorageOrder::ColMajor),
            p: Permutation::new(Vec::new()),
            h_coeffs: Vec::new(),
            v_matrix: SparseMatrix::new(0, 0, crate::core::sparse::StorageOrder::ColMajor),
            is_initialized: false,
        }
    }

    pub fn compute(&mut self, matrix: &SparseMatrix<T>) -> Result<(), String> {
        let m = matrix.rows();
        let n = matrix.cols();
        let size = std::cmp::min(m, n);

        self.p = self.ordering.compute(matrix);

        // Workspace
        let mut workspace = vec![T::zero(); m];

        // Storage for V columns to allow efficient "apply previous"
        // v_cols[j] stores pairs (row, val) for v_j where row > j. v_j[j] is implicitly 1.
        let mut v_cols: Vec<Vec<(usize, T)>> = Vec::with_capacity(size);

        self.h_coeffs = vec![T::zero(); size];
        let mut r_triplets = Vec::new();
        let mut v_triplets = Vec::new();

        for k in 0..n {
            // 1. Load Column input
            let pk = self.p.indices()[k];
            workspace.fill(T::zero()); // O(m)

            let mut it = InnerIterator::new(matrix, pk);
            while it.is_valid() {
                let row = it.row();
                if row >= m {
                    break;
                }
                workspace[row] = it.value();
                it.next();
            }

            // 2. Apply previous H_0 ... H_{j-1} where j < k (and j < size)
            // w = (I - tau v v') w = w - tau v (v' w)
            let effective_h = std::cmp::min(k, size);
            for j in 0..effective_h {
                let tau = self.h_coeffs[j];
                if tau == T::zero() {
                    continue;
                }

                // Dot product v_j . workspace
                // v_j has 1 at index j, and stored tail.
                let mut dot = workspace[j]; // v_j[j] * w[j] = 1 * w[j]

                for &(r_idx, v_val) in &v_cols[j] {
                    dot += v_val * workspace[r_idx];
                }

                // Update workspace
                // w = w - factor * v
                let factor = tau * dot;
                workspace[j] -= factor;
                for &(r_idx, v_val) in &v_cols[j] {
                    workspace[r_idx] -= factor * v_val;
                }
            }

            // 3. Compute new Householder H_k reflection for current column
            if k < size {
                // The vector to reflect is workspace[k..m]
                let mut norm_sq = T::zero();
                for i in k..m {
                    norm_sq += workspace[i] * workspace[i];
                }

                // Store R[k,k] (diagonal) and upper part R[0..k, k]
                // R[i, k] for i < k comes from workspace[i] (which is finished processing)
                for i in 0..k {
                    let val = workspace[i];
                    if val != T::zero() {
                        r_triplets.push(crate::core::sparse::Triplet::new(i, k, val));
                    }
                }

                // Compute Householder
                let norm = norm_sq.sqrt();
                if norm != T::zero() {
                    let v0 = workspace[k];
                    // Stable sign choice
                    let sigma = if v0 >= T::zero() {
                        norm
                    } else {
                        T::zero() - norm
                    };
                    let v0_new = v0 + sigma;
                    let tau = v0_new / sigma;
                    self.h_coeffs[k] = tau;

                    // Store R[k,k] = -sigma (result of reflection)
                    r_triplets.push(crate::core::sparse::Triplet::new(k, k, T::zero() - sigma));

                    // Compute v vector: v = x / v0_new. v[k]=1 (implicit).
                    // Store tail v[k+1..m]
                    let inv_v0 = v0_new.recip();
                    let mut current_v_col = Vec::new();

                    for i in k + 1..m {
                        let val = workspace[i];
                        let v_val = val * inv_v0;
                        if v_val != T::zero() {
                            current_v_col.push((i, v_val));
                            v_triplets.push(crate::core::sparse::Triplet::new(i, k, v_val));
                        }
                        // R[i,k] is 0 implicitly
                    }
                    v_cols.push(current_v_col);
                } else {
                    // Norm is zero, column is zero. H_k = I, tau = 0.
                    // R[k, k] = 0
                    v_cols.push(Vec::new());
                    // R entries above are already pushed.
                }
            } else {
                // Rectangular wide case (k >= size)
                // Just store the column into R (it is not annihilated)
                for i in 0..size {
                    // Only up to size rows in R? Usually R is m x n but zero below diagonal?
                    // In Thin QR (m >= n), R is n x n.
                    // In Wide QR (m < n), R is m x n?
                    // Indeed, R corresponds to Q^T A.
                    // Q is m x m. Q^T A is m x n.
                    // We only Householder-ized 'size' columns.
                    // If m < n (Wide), size = m. We effectively processed all rows.
                    // workspace[0..m] are valid entries of R column k.
                    let val = workspace[i];
                    if val != T::zero() {
                        r_triplets.push(crate::core::sparse::Triplet::new(i, k, val));
                    }
                }
                // Remaining part workspace[size..m] should be zero if m <= n?
                // If m < n, size=m. index i goes to m. Correct.
            }
        }

        self.r = SparseMatrix::new(m, n, crate::core::sparse::StorageOrder::ColMajor);
        self.r.set_from_triplets(r_triplets);

        self.v_matrix = SparseMatrix::new(m, size, crate::core::sparse::StorageOrder::ColMajor);
        self.v_matrix.set_from_triplets(v_triplets);

        self.is_initialized = true;
        Ok(())
    }

    pub fn solve<S: Storage<T>>(
        &self,
        b: &Matrix<T, S>,
    ) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        if !self.is_initialized {
            return Err("SparseQR not initialized".to_string());
        }

        let m = self.r.rows();
        let n = self.r.cols();
        let size = self.h_coeffs.len();

        if b.rows() != m {
            return Err("Dimension mismatch in SparseQR solve".to_string());
        }

        let mut x = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, b.cols())?;
        x.assign(b)?;

        // 1. Apple Q^T to b
        for k in 0..size {
            let tau = self.h_coeffs[k];
            if tau != T::default() {
                for col in 0..b.cols() {
                    // dot = v^T * x_col
                    let mut dot = *x.get(k, col).unwrap(); // v[k]=1
                                                           // Iterate v from v_matrix column k (which stores indices > k)
                                                           // InnerIterator for sparse column
                    let mut it = InnerIterator::new(&self.v_matrix, k);
                    while it.is_valid() {
                        let row = it.row();
                        dot += it.value() * *x.get(row, col).unwrap();
                        it.next();
                    }

                    let factor = tau * dot;
                    *x.get_mut(k, col).unwrap() -= factor;

                    let mut it = InnerIterator::new(&self.v_matrix, k);
                    while it.is_valid() {
                        let row = it.row();
                        *x.get_mut(row, col).unwrap() -= factor * it.value();
                        it.next();
                    }
                }
            }
        }

        // 2. Back substitution with R (upper triangular)
        // x now contains Q^T * b. We solve R * y = (Q^T * b)[0..n]
        // Result will be permuted: P^T * final_x = y  => final_x = P * y

        // Truncate/Resize x if needed or just use top n rows?
        // We will store result in separate matrix `y` size n x RHS
        let mut y = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;

        for k in 0..b.cols() {
            // Back subst
            // Copy x[0..n] to y column k

            // Copy x[0..n] to y column k
            for i in 0..n {
                *y.get_mut(i, k).unwrap() = *x.get(i, k).unwrap();
            }

            // Perform column-based back-substitution on y
            for j in (0..n).rev() {
                let mut diag = T::default();
                let mut it = InnerIterator::new(&self.r, j);

                // Find diagonal and update others
                // To do this strictly:
                // 1. Find diagonal R[j,j]
                // 2. Scale y[j]
                // 3. Update y[row] for row < j

                // We need to iterate the column j.
                // We can collect the updates.
                let mut col_vals = Vec::new();
                while it.is_valid() {
                    col_vals.push((it.row(), it.value()));
                    it.next();
                }

                // Find diagonal
                // As R is upper triangular, diagonal is usually the last entry if sorted?
                // Or we search.
                for (row, val) in &col_vals {
                    if *row == j {
                        diag = *val;
                        break;
                    }
                }

                if diag.abs().to_f64() < 1e-18 {
                    // Singular or rank deficient
                    // For least squares in rank deficient, we should set x[j] = 0 or similar?
                    // For now, error or 0.
                    *y.get_mut(j, k).unwrap() = T::default();
                } else {
                    *y.get_mut(j, k).unwrap() = *y.get(j, k).unwrap() / diag;
                }

                let final_yj = *y.get(j, k).unwrap();

                // Update residuals
                for (row, val) in &col_vals {
                    if *row < j {
                        *y.get_mut(*row, k).unwrap() -= *val * final_yj;
                    }
                }
            }
        }

        // 3. Apply Permutation P^-1 (since A*P = Q*R => A = Q*R*P^T => Ax=b => Q*R*P^T*x = b => R(P^T x) = Q^T b)
        // Let z = P^T x. We solved for z (stored in y). x = P z.

        let mut final_res = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, b.cols())?;
        let p_indices = self.p.indices(); // p maps i -> p[i].
                                          // P z means vector w where w[p[i]] = z[i]?
                                          // Or w[i] = z[p[i]]?
                                          // Definition: A_p = A * P. P is permutation matrix.
                                          // x = P * z.
                                          // If P corresponds to indices `p_indices`, usually P * e_i = e_{p[i]}.
                                          // So (P z)[k] = z[j] where k = p[j].
                                          // So final_res[p[i]] = y[i]

        for i in 0..n {
            let target_row = p_indices[i];
            for col in 0..b.cols() {
                *final_res.get_mut(target_row, col).unwrap() = *y.get(i, col).unwrap();
            }
        }

        Ok(final_res)
    }
}
