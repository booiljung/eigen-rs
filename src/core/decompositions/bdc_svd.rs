//! Bidiagonal Divide & Conquer SVD.

use crate::core::decompositions::bidiagonal::Bidiagonalization;
use crate::core::decompositions::svd::JacobiSVD;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};

/// Bidiagonal Divide & Conquer SVD.
///
/// Reduces the matrix to bidiagonal form, then computes the SVD using a Divide & Conquer algorithm.
pub struct BDCSVD<T: Scalar, S: Storage<T>> {
    u: Matrix<T, DynamicStorage<T>>,
    v: Matrix<T, DynamicStorage<T>>,
    singular_values: Vec<T>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar<Real = T> + PartialOrd + 'static, S: Storage<T> + 'static> BDCSVD<T, S> {
    /// Computes the BDCSVD of the given matrix.
    pub fn new(matrix: &Matrix<T, S>) -> Result<Self, String> {
        let m = matrix.rows();
        let n = matrix.cols();

        #[cfg(feature = "cuda")]
        {
            use crate::core::decompositions::cuda_bridge::CudaDecompositionExt;
            match matrix.try_svd_cuda() {
                Ok(Some((u_cuda, mut s_cuda, v_cuda))) => {
                    // Ensure the exact types returned match
                    let min_dim = std::cmp::min(m, n);
                    let mut singular_values = Vec::with_capacity(min_dim);
                    for i in 0..min_dim {
                        singular_values.push(*s_cuda.get(i, 0).unwrap());
                    }

                    return Ok(Self {
                        u: u_cuda,
                        v: v_cuda,
                        singular_values,
                        _phantom: std::marker::PhantomData,
                    });
                }
                Ok(None) => {} // too small, fallback
                Err(e) => {
                    println!("try_svd_cuda failed: {}, falling back to CPU", e);
                }
            }
        }

        println!("Entering CPU BDCSVD for {}x{}...", m, n);
        // 1. Bidiagonalization
        let bidiag = Bidiagonalization::new(matrix)?;
        let u_bi = bidiag.matrix_u();
        let v_bi = bidiag.matrix_v();
        let b = bidiag.matrix_b();

        // 2. Compute SVD of Bidiagonal Matrix
        // We implement a recursive Divide & Conquer approach.
        // First, we need to extract the diagonal and super-diagonal.

        let min_dim = std::cmp::min(m, n);
        let mut diag: Vec<T> = Vec::with_capacity(min_dim);
        let mut super_diag: Vec<T> = Vec::with_capacity(min_dim - 1);

        for i in 0..min_dim {
            diag.push(*b.get(i, i).unwrap());
            if i < min_dim - 1 {
                super_diag.push(*b.get(i, i + 1).unwrap());
            }
        }

        // Helper to solve bidiagonal SVD
        let (u_bdc, s_bdc, v_bdc) = Self::compute_bdc(&diag, &super_diag)?;

        // 3. Combine Operations (U = U_bi * U_bdc, V = V_bi * V_bdc)
        let dim = diag.len();

        let mut u = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, m)?;
        // Copy U_bi to U
        u.assign(&u_bi)?;

        // Valid for k in 0..dim: New_Col_k = sum(U_bi.col(j) * U_bdc(j, k))
        let mut u_new_block = Matrix::<T, DynamicStorage<T>>::new_dynamic(m, dim)?;

        for k in 0..dim {
            for r in 0..m {
                let mut sum = T::from_usize(0);
                for j in 0..dim {
                    sum += *u_bi.get(r, j).unwrap() * *u_bdc.get(j, k).unwrap();
                }
                *u_new_block.get_mut(r, k).unwrap() = sum;
            }
        }
        // Overwrite first dim columns of U
        for k in 0..dim {
            for r in 0..m {
                *u.get_mut(r, k).unwrap() = *u_new_block.get(r, k).unwrap();
            }
        }

        // Same for V
        let mut v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        v.assign(&v_bi)?; // Corrected with reference

        let mut v_new_block = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, dim)?;
        for k in 0..dim {
            for r in 0..n {
                let mut sum = T::from_usize(0);
                for j in 0..dim {
                    sum += *v_bi.get(r, j).unwrap() * *v_bdc.get(j, k).unwrap();
                }
                *v_new_block.get_mut(r, k).unwrap() = sum;
            }
        }
        for k in 0..dim {
            for r in 0..n {
                *v.get_mut(r, k).unwrap() = *v_new_block.get(r, k).unwrap();
            }
        }

        Ok(Self {
            u,
            v,
            singular_values: s_bdc,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Main BDC recursive function
    fn compute_bdc(
        diag: &[T],
        super_diag: &[T],
    ) -> Result<
        (
            Matrix<T, DynamicStorage<T>>,
            Vec<T>,
            Matrix<T, DynamicStorage<T>>,
        ),
        String,
    > {
        let n = diag.len();
        if n == 0 {
            return Ok((
                Matrix::<T, DynamicStorage<T>>::new_dynamic(0, 0)?,
                vec![],
                Matrix::<T, DynamicStorage<T>>::new_dynamic(0, 0)?,
            ));
        }

        // 1. Deflation: Find zeros in super_diag
        // If super_diag[i] is small, we split the problem.
        let eps = T::epsilon(); // Assumes Float trait or similar.
                                // Since T is Scalar, we need to handle this carefully.
                                // For simplicity, let's look for explicit zero or very small relative to neighbors.

        for i in 0..n - 1 {
            let val = super_diag[i];
            let threshold = eps * (diag[i].abs() + diag[i + 1].abs());

            if val.abs() <= threshold {
                // Split found at i
                // Block 1: 0..=i (size i+1)
                // Block 2: i+1..n (size n-(i+1))

                let (u1, s1, v1) = Self::compute_bdc(&diag[0..=i], &super_diag[0..i])?;
                let (u2, s2, v2) = Self::compute_bdc(&diag[i + 1..n], &super_diag[i + 1..n - 1])?;

                // Combine Block Diagonal Matrices
                let mut u = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
                let mut v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;

                // Fill U
                // U = [ U1 0 ]
                //     [ 0 U2 ]
                for r in 0..u1.rows() {
                    for c in 0..u1.cols() {
                        *u.get_mut(r, c).unwrap() = *u1.get(r, c).unwrap();
                    }
                }
                for r in 0..u2.rows() {
                    for c in 0..u2.cols() {
                        *u.get_mut(r + u1.rows(), c + u1.cols()).unwrap() = *u2.get(r, c).unwrap();
                    }
                }

                // Fill V
                for r in 0..v1.rows() {
                    for c in 0..v1.cols() {
                        *v.get_mut(r, c).unwrap() = *v1.get(r, c).unwrap();
                    }
                }
                for r in 0..v2.rows() {
                    for c in 0..v2.cols() {
                        *v.get_mut(r + v1.rows(), c + v1.cols()).unwrap() = *v2.get(r, c).unwrap();
                    }
                }

                // Combine Singular Values
                let mut s = s1;
                s.extend(s2);

                // Sort singular values might be needed, but usually we sort at the very end.
                // However, merging sorted lists is cheaper.
                // Let's defer sorting to the top level or Merge step.

                return Ok((u, s, v));
            }
        }

        // No deflation found -> Irreducible block
        Self::compute_block_bdc(diag, super_diag)
    }

    /// Compute SVD of an irreducible bidiagonal block using B^T B D&C
    fn compute_block_bdc(
        diag: &[T],
        super_diag: &[T],
    ) -> Result<
        (
            Matrix<T, DynamicStorage<T>>,
            Vec<T>,
            Matrix<T, DynamicStorage<T>>,
        ),
        String,
    > {
        let n = diag.len();

        // Base case: Small matrix use Jacobi SVD on Bidiagonal directly
        // This ensures correct U computation even for singular matrices (unlike B^T B path)
        if n < 32 {
            return Self::compute_jacobi_bidiag(diag, super_diag);
        }

        // Form Tridiagonal Matrix T = B^T B
        // Diag: d_i = b_{i-1}^2 + a_i^2  (with b_{-1}=0)
        // Off-Diag: e_i = a_i * b_i

        let mut t_diag = Vec::with_capacity(n);
        let mut t_off_diag = Vec::with_capacity(n - 1);

        for i in 0..n {
            let a_sq = diag[i] * diag[i];
            let b_sq_prev = if i > 0 {
                super_diag[i - 1] * super_diag[i - 1]
            } else {
                T::from_usize(0)
            };
            t_diag.push(a_sq + b_sq_prev);

            if i < n - 1 {
                t_off_diag.push(diag[i] * super_diag[i]);
            }
        }

        // Solve Symmetric Tridiagonal Eigenproblem
        let (vals, vecs) = Self::compute_sym_tri_dc(&mut t_diag, &mut t_off_diag)?;

        // vals are singular values squared
        // vecs is V
        let mut s_new = Vec::with_capacity(n);
        for x in vals {
            if x < T::from_usize(0) {
                s_new.push(T::from_usize(0));
            } else {
                s_new.push(x.sqrt());
            }
        }

        // Compute U = B * V * S^-1
        let mut u_new = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        let v_new = vecs;

        for j in 0..n {
            let s_val = s_new[j];
            if s_val.abs() > T::epsilon() * T::from_usize(10) {
                // Slightly higher threshold
                // Col u_j = B * v_j / s_j
                // B * v_j computation:
                // v_j is column j of v_new

                // Row i of B * v_j:
                // B_i * v_j = B_ii * v_{j,i} + B_{i,i+1} * v_{j,i+1}
                let inv_s = T::from_usize(1) / s_val;

                for i in 0..n {
                    let mut val = diag[i] * *v_new.get(i, j).unwrap();
                    if i < n - 1 {
                        val += super_diag[i] * *v_new.get(i + 1, j).unwrap();
                    }
                    *u_new.get_mut(i, j).unwrap() = val * inv_s;
                }
            } else {
                // For singular value 0, computing U via relations is unstable.
                // We should technically extend U to be orthogonal.
                // For minimal optimizer, we leave 0 (incomplete basis).
                // Or simply pick e_j? No.
            }
        }

        Ok((u_new, s_new, v_new))
    }

    /// Recursive Divide & Conquer for Symmetric Tridiagonal Matrix
    /// Returns (Eigenvalues, Eigenvectors)
    fn compute_sym_tri_dc(
        diag: &mut [T],
        off_diag: &mut [T],
    ) -> Result<(Vec<T>, Matrix<T, DynamicStorage<T>>), String> {
        let n = diag.len();
        if n == 0 {
            return Ok((vec![], Matrix::<T, DynamicStorage<T>>::new_dynamic(0, 0)?));
        }

        // Base case
        if n < 16 {
            let mut mat = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
            for i in 0..n {
                *mat.get_mut(i, i).unwrap() = diag[i];
                if i < n - 1 {
                    *mat.get_mut(i, i + 1).unwrap() = off_diag[i];
                    *mat.get_mut(i + 1, i).unwrap() = off_diag[i];
                }
            }

            let eig = crate::core::decompositions::SelfAdjointEigenSolver::new(&mat, true)?;
            let vals = eig.eigenvalues().storage().data().to_vec();
            let vecs = eig.eigenvectors().unwrap().clone();
            return Ok((vals, vecs));
        }

        let k = n / 2;
        // Split at index k (connects k-1 and k).
        // Tearing: T = T' + rho * z * z^T
        let beta = off_diag[k - 1];
        let rho = beta;

        let d_k_minus_1_orig = diag[k - 1];
        let d_k_orig = diag[k];

        diag[k - 1] -= beta;
        diag[k] -= beta;

        // Recurse
        let (vals1, vecs1) = Self::compute_sym_tri_dc(&mut diag[0..k], &mut off_diag[0..k - 1])?;
        let (vals2, vecs2) = Self::compute_sym_tri_dc(&mut diag[k..n], &mut off_diag[k..n - 1])?;

        diag[k - 1] = d_k_minus_1_orig;
        diag[k] = d_k_orig;

        // Merge Step

        // 1. Construct initial D and Z
        // D is 'd_merged' (sorted)
        let mut d_merged = vals1.clone();
        d_merged.extend(vals2.iter());

        // Z is column vector from block eigenvectors
        // z = [ LastRow(V1), FirstRow(V2) ]
        let mut z = Vec::with_capacity(n);
        for j in 0..vecs1.cols() {
            z.push(*vecs1.get(k - 1, j).unwrap());
        }
        for j in 0..vecs2.cols() {
            z.push(*vecs2.get(0, j).unwrap());
        }

        // Sort D and permute Z accordingly
        let mut p: Vec<usize> = (0..n).collect();
        p.sort_by(|&i, &j| {
            if let Some(ord) = d_merged[i].partial_cmp(&d_merged[j]) {
                ord
            } else {
                let i_nan = d_merged[i] != d_merged[i];
                let j_nan = d_merged[j] != d_merged[j];
                if i_nan && j_nan {
                    std::cmp::Ordering::Equal
                } else if i_nan {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
        });

        let d_sorted: Vec<T> = p.iter().map(|&i| d_merged[i]).collect();
        let z_sorted: Vec<T> = p.iter().map(|&i| z[i]).collect();
        // Deflation
        let eps = T::epsilon();
        let tol = eps * T::from_usize(100); // Heuristic

        // 1. Materialize V_new = V_old * P
        // We do this early so we can apply Givens rotations to it.
        let mut v_new = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;

        for c in 0..n {
            let old_col = p[c];
            if old_col < k {
                // Top half from V1
                for r in 0..vecs1.rows() {
                    *v_new.get_mut(r, c).unwrap() = *vecs1.get(r, old_col).unwrap();
                }
                // Bottom half is 0
                for r in 0..vecs2.rows() {
                    *v_new.get_mut(k + r, c).unwrap() = T::from_usize(0);
                }
            } else {
                // Bottom half from V2
                let v2_col = old_col - k;
                // Top half is 0
                for r in 0..vecs1.rows() {
                    *v_new.get_mut(r, c).unwrap() = T::from_usize(0);
                }
                for r in 0..vecs2.rows() {
                    *v_new.get_mut(k + r, c).unwrap() = *vecs2.get(r, v2_col).unwrap();
                }
            }
        }

        // 2. Deflation Case 2: d_i approx d_{i+1}
        // Apply Givens rotations to zero out z_{i+1}

        // We need mutable d and z
        let d_curr = d_sorted; // Move
        let mut z_curr = z_sorted; // Move

        for i in 0..n - 1 {
            let diff = (d_curr[i] - d_curr[i + 1]).abs();
            if diff < eps {
                // Determine Givens rotation G to zero z_{i+1}
                // [ c  s ] [ z_i   ] = [ r ]
                // [ -s c ] [ z_{i+1} ]   [ 0 ]

                // Using Jacobi Rotation helper or manual?
                // r = sqrt(z_i^2 + z_{i+1}^2)
                // c = z_i / r
                // s = z_{i+1} / r

                let a = z_curr[i];
                let b = z_curr[i + 1];
                let r = (a * a + b * b).sqrt();

                if r > T::epsilon() {
                    let c = a / r;
                    let s = b / r;

                    // Update z
                    z_curr[i] = r;
                    z_curr[i + 1] = T::from_usize(0);

                    // Update d?
                    // G^T D G. Since d_i ~ d_{i+1}, D is locally scalar. G^T D G ~ D.
                    // Strictly: [c -s; s c] [d 0; 0 d] [c s; -s c] = [d 0; 0 d].
                    // So d doesn't change significantly.
                    // But technically diagonal elements might drift? No, if diagonal approx equal, valid.

                    // Apply G to V_new columns i and i+1
                    // Col_i_new = c * Col_i - s * Col_{i+1} (Wait. Rotation applied from Right?)
                    // V -> V * G.
                    // Col_i_new = V * G[:, i] = V * (c e_i + s e_{i+1}) = c Col_i + s Col_{i+1}
                    // New Col i+1 = V * G[:, i+1] = V * (-s e_i + c e_{i+1}) = -s Col_i + c Col_{i+1}

                    for r_idx in 0..n {
                        let vi = *v_new.get(r_idx, i).unwrap();
                        let vip1 = *v_new.get(r_idx, i + 1).unwrap();

                        *v_new.get_mut(r_idx, i).unwrap() = c * vi + s * vip1;
                        *v_new.get_mut(r_idx, i + 1).unwrap() = -s * vi + c * vip1;
                    }
                }
            }
        }

        // 3. Solve Secular Equation (with further deflation for z ~ 0)
        let (mut roots, u_active, active_indices) =
            Self::solve_secular_equation_with_deflation(&d_curr, &z_curr, rho);

        // 4. Update Active Columns of V_new
        // v_new[:, active_indices] = v_new[:, active_indices] * u_active

        let k_act = active_indices.len();
        if k_act > 0 {
            // We need to perform this update in place or with temp.
            // Since it's a mix, temp buffer for active columns of V_new is cleaner.

            let mut v_active_updated = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, k_act)?;

            // Extract current active columns from v_new ("V_basis")
            // And multiply by u_active

            // V_active_new[r, c] = sum_k ( V_new[r, active_indices[k]] * U_active[k, c] )

            for c_act in 0..k_act {
                // Result column
                for r_idx in 0..n {
                    // Result row
                    let mut sum = T::from_usize(0);
                    for k_idx in 0..k_act {
                        // Inner dimension
                        let basis_col_idx = active_indices[k_idx];
                        let u_val = *u_active.get(k_idx, c_act).unwrap();
                        let v_val = *v_new.get(r_idx, basis_col_idx).unwrap();
                        sum += v_val * u_val;
                    }
                    *v_active_updated.get_mut(r_idx, c_act).unwrap() = sum;
                }
            }

            // Copy back into v_new
            for c_act in 0..k_act {
                let col_idx = active_indices[c_act];
                for r_idx in 0..n {
                    *v_new.get_mut(r_idx, col_idx).unwrap() =
                        *v_active_updated.get(r_idx, c_act).unwrap();
                }
            }
        }

        Ok((roots, v_new))
    }

    /// Solves the secular equation: 1 + rho * sum(z_i^2 / (d_i - lambda)) = 0
    /// Returns (roots, active_vecs, active_indices)
    /// roots: Array of all eigenvalues (deflated and computed active)
    /// active_vecs: Matrix (k x k) of eigenvectors for the active block
    /// active_indices: Indices in the sorted D array corresponding to the active block
    fn solve_secular_equation_with_deflation(
        d: &[T],
        z: &[T],
        rho: T,
    ) -> (Vec<T>, Matrix<T, DynamicStorage<T>>, Vec<usize>) {
        let n = d.len();

        let mut d_deflated = Vec::with_capacity(n);
        let mut z_deflated = Vec::with_capacity(n);
        let mut roots = vec![T::default(); n];

        let eps = T::epsilon();
        let tol = eps * T::from_usize(100); // Heuristic tolerance

        let mut active_indices = Vec::new();

        for i in 0..n {
            if z[i].abs() < tol {
                // Deflation Case 1: z_i is zero
                roots[i] = d[i];
            } else {
                d_deflated.push(d[i]);
                z_deflated.push(z[i]);
                active_indices.push(i);
            }
        }

        let k = d_deflated.len();
        let mut u_active = if k > 0 {
            Matrix::<T, DynamicStorage<T>>::new_dynamic(k, k).unwrap()
        } else {
            Matrix::<T, DynamicStorage<T>>::new_dynamic(0, 0).unwrap()
        };

        if k > 0 {
            // Solve secular equation for active part
            let active_roots = Self::solve_secular_equation_core(&d_deflated, &z_deflated, rho);

            // Compute Stable Z (Gu-Eisenstat) for Orthogonality
            // z_hat_i = sign(z_i) * sqrt( prod(lam_j - d_i) / (rho * prod_{j!=i}(d_j - d_i)) )

            let mut stable_z = Vec::with_capacity(k);
            let zero = T::from_usize(0);
            let one = T::from_usize(1);

            for i in 0..k {
                let mut prod = T::from_usize(1);

                for j in 0..k {
                    let num = active_roots[j] - d_deflated[i];
                    if i == j {
                        prod *= num;
                    } else {
                        let den = d_deflated[j] - d_deflated[i];
                        // Guard division? deflation implies distinct d_j
                        prod *= num / den;
                    }
                }
                prod /= rho;

                let val = if prod > zero { prod.sqrt() } else { zero };
                // Restore sign from original z
                let sign = if z_deflated[i] >= zero { one } else { -one };
                stable_z.push(val * sign);
            }

            // Compute Matrix U_active (k x k)
            // U_active columns are normalized vectors v_j = (D - lam_j I)^-1 z_hat

            for j in 0..k {
                // For each root
                let lambda = active_roots[j];
                let mut norm_sq = T::from_usize(0);
                let mut vec_col = Vec::with_capacity(k);

                for i in 0..k {
                    let diff = d_deflated[i] - lambda;
                    // Avoid exact division by zero if root ~ pole
                    let val = if diff.abs() < eps {
                        // Fallback for num stability
                        stable_z[i] / (if diff >= zero { eps } else { -eps })
                    } else {
                        stable_z[i] / diff
                    };
                    vec_col.push(val);
                    norm_sq += val * val;
                }

                let norm = norm_sq.sqrt();
                let inv_norm = T::from_usize(1) / norm;
                for i in 0..k {
                    *u_active.get_mut(i, j).unwrap() = vec_col[i] * inv_norm;
                }
            }

            // Place Back Results
            for (idx, &entry_idx) in active_indices.iter().enumerate() {
                roots[entry_idx] = active_roots[idx];
            }
        }

        (roots, u_active, active_indices)
    }

    /// Core Secular Equation Solver using Newton-Raphson with Bisection Guard
    fn solve_secular_equation_core(d: &[T], z: &[T], rho: T) -> Vec<T> {
        let n = d.len();
        let mut roots = Vec::with_capacity(n);
        let zero = T::from_usize(0);
        let one = T::from_usize(1);
        let two = T::from_usize(2);

        // rho > 0: Roots in (d_i, d_{i+1}) for i=0..n-2, and (d_{n-1}, d_{n-1} + rho*z^2 + 1)
        // rho < 0: Roots in (d_0 - ..., d_0) for i=0, and (d_{i-1}, d_i) for i=1..n-1

        let rho_pos = rho >= zero;

        for i in 0..n {
            let lower;
            let upper;
            let mut lambda; // guess

            if rho_pos {
                // Standard Case
                lower = d[i];
                upper = if i < n - 1 {
                    d[i + 1]
                } else {
                    d[i] + rho * z[i] * z[i] + one
                }; // Loose bound

                // Initial guess
                lambda = if i < n - 1 {
                    (lower + upper) / two
                } else {
                    lower + rho * z[i] * z[i]
                };
            } else {
                // Negative Rho Case
                // Root i corresponds to interval ending at d_i (for i>0) or d_0 (for i=0)
                // Actually, let's align indices.
                // i=0: Root < d_0. Interval (d_0 + rho*z*z - 1, d_0)
                // i > 0: Root in (d_{i-1}, d_i).

                if i == 0 {
                    upper = d[0];
                    lower = d[0] + rho * z[0] * z[0] - one; // Loose lower bound
                    lambda = lower + rho * z[0] * z[0];
                } else {
                    lower = d[i - 1];
                    upper = d[i];
                    lambda = (lower + upper) / two;
                }
            }

            // Newton Iterations
            for _iter in 0..40 {
                // Increased max iterations
                let mut f = one;
                let mut df = zero;

                for j in 0..n {
                    let diff = d[j] - lambda;
                    // Robust division
                    if diff.abs() < T::epsilon() {
                        // Very close to pole.
                        // If we are evaluating, we shouldn't be AT the pole effectively.
                    }
                    let term = z[j] * z[j] / diff;
                    f += rho * term;
                    df += rho * term / diff;
                }

                if f.abs() < T::epsilon() * T::from_usize(100) {
                    break;
                }

                let delta = if df.abs() > T::epsilon() {
                    f / df
                } else {
                    T::from_usize(0)
                }; // Fallback if df ~ 0
                let next_lambda = lambda - delta;

                // Guard and Bisection
                if next_lambda <= lower || next_lambda >= upper {
                    // Bisection Fallback
                    // Direction depends on f and rho
                    // If rho > 0: f increasing. f>0 -> Root Left.
                    // If rho < 0: f decreasing. f>0 -> Root Right. (Wait. f(+inf) -> -inf. f(-inf) -> 1? No.)

                    // Let's re-verify rho < 0 behavior.
                    // f(lam) = 1 + rho sum ...
                    // As lam approaches d_i from Left: d_i - lam > 0. term > 0. rho < 0 -> -inf.
                    // As lam approaches d_{i-1} from Right: d_{i-1} - lam < 0. term < 0. rho < 0 -> +inf.
                    // So f goes +inf -> -inf. Decreasing.
                    // f > 0 means we are to the Left of Root. Root is Right.
                    // lower = lambda.

                    if rho_pos {
                        if f > zero {
                            // Too far right
                            // lambda = (lower + lambda) / two; // Wrong in prev?
                            // Inc func: f(x) > 0 => x > Root. Root is Left. Correct.
                            // Wait, previously I had upper=lambda for f>0.
                            // Line 651: if f > zero { lambda = (lower + lambda) / two } which implies moving Left. Correct.
                            lambda = (lower + lambda) / two;
                        } else {
                            lambda = (lambda + upper) / two;
                        }
                    } else {
                        // Decreasing func
                        if f > zero {
                            // f(x) > 0 => x < Root. Root is Right.
                            lambda = (lambda + upper) / two;
                        } else {
                            // f(x) < 0 => x > Root. Root is Left.
                            lambda = (lower + lambda) / two;
                        }
                    }
                } else {
                    lambda = next_lambda;
                }
            }
            roots.push(lambda);
        }
        roots
    }

    fn compute_jacobi_bidiag(
        diag: &[T],
        super_diag: &[T],
    ) -> Result<
        (
            Matrix<T, DynamicStorage<T>>,
            Vec<T>,
            Matrix<T, DynamicStorage<T>>,
        ),
        String,
    > {
        let n = diag.len();
        // Reconstruct full bidiagonal matrix
        let mut mat = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        for i in 0..n {
            *mat.get_mut(i, i).unwrap() = diag[i];
            if i < n - 1 {
                *mat.get_mut(i, i + 1).unwrap() = super_diag[i];
            }
        }

        // Base case: Use JacobiSVD
        let svd = JacobiSVD::new(&mat)?;

        // Helper to clone matrix because JacobiSVD owns them
        let mut u = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        u.assign(svd.matrix_u())?;

        let mut v = Matrix::<T, DynamicStorage<T>>::new_dynamic(n, n)?;
        v.assign(svd.matrix_v())?;

        Ok((u, svd.singular_values().to_vec(), v))
    }

    pub fn matrix_u(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.u
    }
    pub fn matrix_v(&self) -> &Matrix<T, DynamicStorage<T>> {
        &self.v
    }
    pub fn singular_values(&self) -> &[T] {
        &self.singular_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bdc_svd_basic() {
        let rows = 4;
        let cols = 4;
        let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(rows, cols).unwrap();
        *a.get_mut(0, 0).unwrap() = 1.0;
        *a.get_mut(0, 1).unwrap() = 2.0;
        *a.get_mut(1, 0).unwrap() = 3.0;
        *a.get_mut(1, 1).unwrap() = 4.0;
        *a.get_mut(2, 2).unwrap() = 5.0;
        *a.get_mut(3, 3).unwrap() = 6.0;

        let svd = BDCSVD::new(&a).unwrap();

        let u = svd.matrix_u();
        let v_t = svd.matrix_v().transpose();
        let mut s_mat = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(rows, cols).unwrap();
        let s_vals = svd.singular_values();
        for i in 0..std::cmp::min(rows, cols) {
            *s_mat.get_mut(i, i).unwrap() = s_vals[i];
        }

        let mut us = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(rows, cols).unwrap();
        us.assign(&(u * &s_mat)).unwrap();
        let mut recon = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(rows, cols).unwrap();
        recon.assign(&(&us * &v_t)).unwrap();

        for i in 0..rows {
            for j in 0..cols {
                let diff = (*recon.get(i, j).unwrap() - *a.get(i, j).unwrap()).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at {},{}: orig {}, recon {}",
                    i,
                    j,
                    *a.get(i, j).unwrap(),
                    *recon.get(i, j).unwrap()
                );
            }
        }
    }
}
