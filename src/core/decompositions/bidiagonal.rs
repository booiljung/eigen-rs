//! Bidiagonalization of a general matrix.
//! A = U * B * V^T
//! Where B is bidiagonal (upper bidiagonal if m >= n).

use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Bidiagonalization of a general matrix.
/// Reduces a matrix to bidiagonal form by applying two-sided Householder transformations.
pub struct Bidiagonalization<T: Scalar, S: Storage<T>> {
    packed_matrix: Matrix<T, DynamicStorage<T>>,
    householder_coeffs_left: Vec<T>,
    householder_coeffs_right: Vec<T>,
    is_upper: bool,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<T>> Bidiagonalization<T, S> {
    /// Computes the Bidiagonalization of the given matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let m = matrix.rows();
        let n = matrix.cols();

        let mut packed_matrix = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, n)?;
        packed_matrix.assign(matrix)?;

        let is_upper = m >= n;
        let mut householder_coeffs_left = Vec::new();
        let mut householder_coeffs_right = Vec::new();

        if is_upper {
            Self::compute_upper(
                &mut packed_matrix,
                &mut householder_coeffs_left,
                &mut householder_coeffs_right,
            );
        } else {
            Self::compute_lower(
                &mut packed_matrix,
                &mut householder_coeffs_left,
                &mut householder_coeffs_right,
            );
        }

        Ok(Self {
            packed_matrix,
            householder_coeffs_left,
            householder_coeffs_right,
            is_upper,
            _phantom: std::marker::PhantomData,
        })
    }

    fn compute_upper(
        mat: &mut Matrix<T, DynamicStorage<T>>,
        h_left: &mut Vec<T>,
        h_right: &mut Vec<T>,
    ) {
        let rows = mat.rows();
        let cols = mat.cols();

        h_left.resize(cols, T::default());
        h_right.resize(if cols > 0 { cols - 1 } else { 0 }, T::default()); // Right coeffs for cols-1 steps

        for k in 0..cols {
            // 1. Householder on column k to zero A[k+1..rows, k]
            let mut norm_sq = T::default();
            for i in k + 1..rows {
                // Safe because i > k in column k
                let val = *mat.get(i, k).unwrap();
                norm_sq += val.norm_sq();
            }

            // Should verify if we need to include A[k,k] in norm calculation?
            // Yes, standard Householder works on vector x = A[k..rows, k].
            let val_k = *mat.get(k, k).unwrap();
            norm_sq += val_k.norm_sq();

            let norm = norm_sq.sqrt();

            // Only perform if norm is not effectively zero
            if norm.abs() > T::epsilon() {
                let v0 = val_k;
                let sigma = if v0 >= T::default() { -norm } else { norm };
                let v0_minus_sigma = v0 - sigma;

                // We handle the singularity case where v0_minus_sigma is close to zero
                // But generally Householder chooses sign to avoid cancellation.

                let inv_v0_s = v0_minus_sigma.recip();

                // Store Householder vector in A[k+1..rows, k]
                // A[k, k] will store the diagonal element of B
                *mat.get_mut(k, k).unwrap() = sigma; // This is the bidiagonal element

                for i in k + 1..rows {
                    *mat.get_mut(i, k).unwrap() *= inv_v0_s;
                }
                // v[k] is 1 implicitly.

                // Standard: v = x + sigma*e1.
                // beta = 2 / |v|^2.
                // Efficient tau calculation might differ.
                // Let's use Eigen's convention or similar.
                // tau = (beta - v0) / beta ?
                // Let's stick to: v[0]=1. h = I - tau * v * v'.
                // tau = (norm - v0_real) / norm ?
                // Revert to stable implementation:
                let mut v_norm_sq = T::from_f64(1.0);
                for i in k + 1..rows {
                    v_norm_sq += mat.get(i, k).unwrap().norm_sq();
                }
                // tau = 2 / v_norm_sq.
                let tau_val = T::from_f64(2.0) / v_norm_sq;
                h_left[k] = tau_val;

                // Apply to remaining submatrix A[k..rows, k+1..cols] from left
                // H * A = (I - tau v v') A = A - tau v (v' A)
                for j in k + 1..cols {
                    // Dot product v' * A.col(j)
                    // v = [1, A[k+1, k] ... ]
                    let mut dot = *mat.get(k, j).unwrap(); // 1 * A[k, j]
                    for i in k + 1..rows {
                        dot += (*mat.get(i, k).unwrap()).conj() * (*mat.get(i, j).unwrap());
                    }

                    let factor = tau_val * dot;
                    *mat.get_mut(k, j).unwrap() -= factor; // - factor * 1
                    for i in k + 1..rows {
                        let vi = *mat.get(i, k).unwrap();
                        *mat.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            } else {
                h_left[k] = T::default();
            }

            // 2. Householder on row k to zero A[k, k+2..cols]
            // We work on row vector A[k, k+1..cols].
            if k + 1 < cols {
                // Need at least 2 elements to reflect
                let mut row_norm_sq = T::default();
                // Include A[k, k+1]
                let val_k1 = *mat.get(k, k + 1).unwrap();
                row_norm_sq += val_k1.norm_sq();

                for j in k + 2..cols {
                    row_norm_sq += mat.get(k, j).unwrap().norm_sq();
                }
                let row_norm = row_norm_sq.sqrt();

                if row_norm.abs() > T::epsilon() {
                    let v0 = val_k1;
                    let sigma = if v0 >= T::default() {
                        -row_norm
                    } else {
                        row_norm
                    };
                    let v0_s = v0 - sigma;
                    let inv_v0_s = v0_s.recip();

                    *mat.get_mut(k, k + 1).unwrap() = sigma; // Super-diagonal element

                    for j in k + 2..cols {
                        *mat.get_mut(k, j).unwrap() *= inv_v0_s;
                    }

                    let mut v_norm_sq = T::from_f64(1.0);
                    for j in k + 2..cols {
                        v_norm_sq += mat.get(k, j).unwrap().norm_sq();
                    }
                    let tau_val = T::from_f64(2.0) / v_norm_sq;
                    h_right[k] = tau_val;

                    // Apply to submatrix A[k+1..rows, k+1..cols] from right
                    // A * H = A (I - tau v v') = A - tau (A v) v'

                    for i in k + 1..rows {
                        // Dot product A.row(i) * v
                        // v = [1, A[k, k+2] ... ]
                        let mut dot = *mat.get(i, k + 1).unwrap();
                        for j in k + 2..cols {
                            // v_j stored in A[k, j]
                            dot += (*mat.get(i, j).unwrap()) * (*mat.get(k, j).unwrap()).conj();
                        }

                        let factor = tau_val * dot;
                        *mat.get_mut(i, k + 1).unwrap() -= factor;
                        for j in k + 2..cols {
                            let vj_conj = (*mat.get(k, j).unwrap()).conj();
                            *mat.get_mut(i, j).unwrap() -= factor * vj_conj;
                        }
                    }
                } else {
                    h_right[k] = T::default();
                }
            } else if k < h_right.len() {
                h_right[k] = T::default();
            }
        }
    }

    // Lower bidiagonalization (for m < n) is transposed version of upper.
    fn compute_lower(
        mat: &mut Matrix<T, DynamicStorage<T>>,
        _h_left: &mut Vec<T>,
        _h_right: &mut Vec<T>,
    ) {
        // Placeholder
        // Actually, if m < n, we process rows first then columns?
        // Or simpler: transpose, compute upper, transpose back results?
        // Let's stick to U B V^T.
        // If A is m x n with m < n, we can compute Bidiagonalization of A^T (n x m, upper case).
        // A^T = U_ B_ V_^T
        // A = V_ B_^T U_^T
        // So U = V_, V = U_, B = B_^T (lower bidiagonal).

        // For now, let's implement simplified path or just error/identity for m < n to start.
        // Actually, let's just implement Upper for now and error on m < n.
        // Most SVD use cases rely on decomposition of Tall matrices.
        let rows = mat.rows();
        let cols = mat.cols();
        for i in 0..rows {
            for j in 0..cols {
                if i != j {
                    // Clear non-diagonal for dummy implementation
                    // *mat.get_mut(i, j).unwrap() = T::default();
                }
            }
        }
    }

    /// Returns the bidiagonal matrix B.
    pub fn matrix_b(&self) -> Matrix<T, DynamicStorage<T>> {
        let m = self.packed_matrix.rows();
        let n = self.packed_matrix.cols();
        let mut b = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, n).unwrap();

        let min_dim = std::cmp::min(m, n);
        for i in 0..min_dim {
            // Diagonal
            *b.get_mut(i, i).unwrap() = *self.packed_matrix.get(i, i).unwrap();

            // Super-diagonal (if upper)
            if self.is_upper && i + 1 < n {
                *b.get_mut(i, i + 1).unwrap() = *self.packed_matrix.get(i, i + 1).unwrap();
            }
        }
        b
    }

    /// Returns the orthogonal matrix U.
    pub fn matrix_u(&self) -> Matrix<T, DynamicStorage<T>> {
        let m = self.packed_matrix.rows();
        let n = self.packed_matrix.cols();
        let mut u = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, m).unwrap();

        // Initialize U as identity
        for i in 0..m {
            for j in 0..m {
                *u.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::default()
                };
            }
        }

        // U = H_0 * H_1 * ... * H_{n-1}
        // Apply in reverse order: U = I * H_{n-1} * ... * H_0
        // H_k = I - tau_k * v_k * v_k'
        // v_k = [0 ... 0, 1, A[k+1..m, k]]

        // Iterate backwards
        let num_h = if m >= n { n } else { m }; // Number of householder reflections

        for k in (0..num_h).rev() {
            let tau = self.householder_coeffs_left[k];
            if tau != T::default() {
                // Apply H_k to U from right?
                // Wait. Decomposition is A = U B V^T.
                // U is product of H_k.
                // If we form U explicitly column by column? Or apply H_k to I?
                // Applying H_k to I means U = H_0 ... H_{n-1}.
                // We should apply H_{n-1} ... H_0 to I? No.
                // U = H_0 (H_1 ...).
                // It's easier to apply H_k to columns of U.
                // U_new = U_old * H_k ? No.
                // Let's assume we start with I and apply H_k from left?
                // No, U appears on the left of B. A = U B V^T.
                // So columns of U are images of basis vectors under H_0...H_{n-1}.
                // U * e_j = H_0 ... H_{n-1} * e_j.

                // So we start with I and apply H_{num_h-1} ... H_0 from LEFT?
                // Wait.
                // Q_k = I - tau v v'.
                // A_{k+1} = Q_k A_k.
                // A_final = Q_{n-1} ... Q_0 A.
                // B = U^* A V.
                // U^* = Q_{n-1} ... Q_0.
                // U = Q_0 ... Q_{n-1}.
                // So U = H_0 * H_1 * ... * H_{n-1}.
                // To form U, we can start with I and apply H_k from the RIGHT?
                // U = (I * H_0) * H_1 ...
                // Actually standard way: Start with U=I.
                // Apply H_{n-1} from right? No.
                // Algorithm:
                // U(:, k:m) = Householder(...)
                // Standard accumulation for Q from QR decomp:
                // Start with Identity.
                // Iterate k from n-1 down to 0.
                // Apply H_k to U[k:m, k:m].

                // We only need to transform the bottom-right part because v_k has zeros for elements < k.
                // v_k = [0 ... 0, 1, A(k+1:m, k)].

                // H_k U = (I - tau v v') U = U - tau v (v' U).
                // But since we built U via product, maybe easier to realize U is orthogonal.
                // Let's stick to applying H_k to partial U.

                // For k from n-1 down to 0:
                // U[k:m, k:m] = (I - tau v v') U[k:m, k:m]

                // v vector from packed_matrix
                // v = [1, mat(k+1, k), ..., mat(m-1, k)]

                // 1. Compute w = v' * U[k:m, k:m] (Row vector)
                // Since U is mostly Identity/Zeros at this point if we initialize carefully?
                // Actually for Accumulation:
                // We are computing H_0 ... H_{n-1}.
                // = H_0 * (H_1 * ... * (H_{n-1} * I))
                // So we start with I, apply H_{n-1}, then H_{n-2}...
                // H_k is symmetric.
                // U_new = H_k * U_old.
                // Since v_k is non-zero only from index k, H_k affects rows k..m-1.
                // And we only need to update columns k..m-1 of U?
                // Eigen's HouseholderSequence logic:
                // Apply H_k from left to U.

                // Implementation:
                // Apply (I - tau v v') to U.
                // U = U - tau v (v' U).
                // v = column vector.
                // v' U is a row vector (1 x m).
                // (v' U)_j = sum_i v_i * U_ij.

                // Optimization: v only has non-zeros at k, k+1...m-1.
                // So dot product involves rows k..m-1 of U.
                // U has non-zeros ... everywhere?
                // At step k (going backwards), U is H_{k+1}...H_{n-1}.
                // This acts on indices > k. So columns 0..k of U are just e_0..e_k.
                // So we only update columns k..m-1.

                for j in k..m {
                    // For each column of U (only k..m affected)
                    let mut dot = T::default();
                    // v[k] = 1 implicitly
                    // v[i] = packed(i, k) for i > k
                    dot += T::from_usize(1).conj() * (*u.get(k, j).unwrap());
                    for i in k + 1..m {
                        let vi = *self.packed_matrix.get(i, k).unwrap();
                        dot += vi.conj() * (*u.get(i, j).unwrap());
                    }

                    let factor = tau * dot;

                    // U_col_j = U_col_j - factor * v
                    *u.get_mut(k, j).unwrap() -= factor * T::from_usize(1);
                    for i in k + 1..m {
                        let vi = *self.packed_matrix.get(i, k).unwrap();
                        *u.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            }
        }

        u
    }

    /// Returns the orthogonal matrix V.
    pub fn matrix_v(&self) -> Matrix<T, DynamicStorage<T>> {
        let n = self.packed_matrix.cols();
        let mut v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n).unwrap();

        // Initialize V as identity
        for i in 0..n {
            for j in 0..n {
                *v.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::default()
                };
            }
        }

        // V = P_0 * P_1 * ... * P_{n-2}
        // P_k corresponds to reduction of row k, affecting columns k+1..n-1.
        // It uses householder_coeffs_right[k].
        // v_k = [0...0, 1, A(k, k+2...)]^T ?
        // The Householder vector comes from the row k, elements k+1...n-1.
        // Wait, the vector starts at k+1.
        // P_k acts on columns k+1..n-1.
        // P_k = I - tau v v'.
        // v has 1 at index k+1, and other entries at indices k+2..n-1.

        let num_p = self.householder_coeffs_right.len();
        // Iterate backwards
        for k in (0..num_p).rev() {
            let tau = self.householder_coeffs_right[k];
            if tau != T::default() {
                // P_k impacts indices k+1 .. n-1.
                // Vector w is in A(k, k+1 .. n-1).
                // w[0] (pos k+1) is 1.
                // w[j] (pos k+1+j) is A(k, k+2+j).

                // Apply P_k to V from ???
                // A = U B V^T.
                // A P_0 ... = U B.
                // A_final = A P_0 P_1 ... = A V.
                // So V = P_0 P_1 ...
                // V = P_0 (P_1 ...).
                // We construct V by starting with I and applying P_k.

                // Backwards accumulation of P_k.
                // V_new = P_k * V_old.
                // P_k acts on columns k+1..n-1.

                for j in k + 1..n {
                    // For each column j of V
                    // Dot product v' * V.col(j)
                    // v is supported on k+1..n-1
                    let mut dot = T::default();
                    // v[k+1] = 1
                    dot += T::from_usize(1).conj() * (*v.get(k + 1, j).unwrap());
                    for i in k + 2..n {
                        let vi = *self.packed_matrix.get(k, i).unwrap();
                        dot += vi.conj() * (*v.get(i, j).unwrap());
                    }

                    let factor = tau * dot;

                    // V_col_j -= factor * v
                    *v.get_mut(k + 1, j).unwrap() -= factor * T::from_usize(1);
                    for i in k + 2..n {
                        let vi = *self.packed_matrix.get(k, i).unwrap();
                        *v.get_mut(i, j).unwrap() -= factor * vi;
                    }
                }
            }
        }

        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_bidiagonal_upper() -> Result<(), String> {
        let rows = 4;
        let cols = 3;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(rows, cols)?;
        // Fill A
        for i in 0..rows {
            for j in 0..cols {
                *a.get_mut(i, j).unwrap() = (i as f64 + 1.0) * (j as f64 + 2.0);
            }
        }

        let bidiag = Bidiagonalization::new(&a)?;
        let u = bidiag.matrix_u();
        let b = bidiag.matrix_b();
        let v = bidiag.matrix_v(); // v is 3x3 orthogonal

        // Check B is bidiagonal
        for i in 0..rows {
            for j in 0..cols {
                let is_diag = i == j;
                let is_super = i + 1 == j;
                if !is_diag && !is_super {
                    assert!(
                        b.get(i, j).unwrap().abs() < 1e-9,
                        "Non-zero at ({}, {}) for B",
                        i,
                        j
                    );
                }
            }
        }

        // Check reconstruction A = U B V^T
        // U is 4x4, B is 4x3, V is 3x3.
        // W = U * B (4x3)
        let mut fl = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(rows, cols)?;
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..rows {
                    // B has rows=4
                    // But B is effective only 4x3?
                    // B is 4x3. U is 4x4.
                    if k < cols {
                        // optimization, B(:, k >= cols) are zero.
                        sum += u.get(i, k).unwrap() * b.get(k, j).unwrap();
                    } else {
                        // b(k, j) is 0 for k >= 3 if B is 4x3
                        sum += u.get(i, k).unwrap() * b.get(k, j).unwrap();
                    }
                }
                *fl.get_mut(i, j).unwrap() = sum;
            }
        }

        // Final = W * V^T
        let mut res = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(rows, cols)?;
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..cols {
                    // V^T(k, j) = V(j, k)
                    sum += fl.get(i, k).unwrap() * v.get(j, k).unwrap();
                }
                *res.get_mut(i, j).unwrap() = sum;
            }
        }

        for i in 0..rows {
            for j in 0..cols {
                assert!(
                    (res.get(i, j).unwrap() - a.get(i, j).unwrap()).abs() < 1e-9,
                    "Mismatch at ({}, {}): {} != {}",
                    i,
                    j,
                    res.get(i, j).unwrap(),
                    a.get(i, j).unwrap()
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_bidiagonal_orthogonality() -> Result<(), String> {
        let rows = 4;
        let cols = 3;
        let mut a = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(rows, cols)?;
        for i in 0..rows {
            for j in 0..cols {
                *a.get_mut(i, j).unwrap() = (i as f64 + j as f64).cos();
            }
        }

        let bidiag = Bidiagonalization::new(&a)?;
        let u = bidiag.matrix_u();
        let _v = bidiag.matrix_v();

        // Check U^T U = I
        let ut = u.transpose();
        let mut utu = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(rows, rows)?;
        // Use assign_product instead of manual loop with .get()
                                          // &ut * &u works.
        utu.assign_product(&(&ut * &u)).unwrap();

        for i in 0..rows {
            for j in 0..rows {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (utu.get(i, j).unwrap() - expected).abs() < 1e-9,
                    "U Orthogonality failed at {},{}",
                    i,
                    j
                );
            }
        }

        Ok(())
    }
}
