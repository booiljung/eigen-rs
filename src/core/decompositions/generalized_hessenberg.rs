//! Generalized Hessenberg-Triangular (GHT) reduction of a pair of complex matrices (A, B).
//! Finds unitary matrices Q and Z such that H = Q * A * Z is upper Hessenberg
//! and R = Q * B * Z is upper triangular.

use crate::core::complex::Complex;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{DynamicStorage, Storage};
// use crate::core::decompositions::HouseholderQR; // Removed

/// Generalized Hessenberg-Triangular reduction.
pub struct GeneralizedHessenbergTriangular<T: Scalar, S: Storage<Complex<T>>> {
    h: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    r: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    q: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    z: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<Complex<T>>> GeneralizedHessenbergTriangular<T, S> {
    /// Computes the GHT reduction of the given pair of square complex matrices (A, B).
    pub fn new(a: &Matrix<Complex<T>, S>, b: &Matrix<Complex<T>, S>) -> Result<Self, String> {
        let n = a.rows();
        if n != a.cols() || n != b.rows() || n != b.cols() {
            return Err("GHT reduction requires square matrices of the same size".to_string());
        }

        // Initial reduction: mat_a = a, mat_b = b
        let mut mat_a = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;
        mat_a.assign(a)?;
        let mut mat_b = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;
        mat_b.assign(b)?;
        let mut mat_q = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;
        for i in 0..n {
            *mat_q.get_mut(i, i).unwrap() = Complex::from_f64(1.0);
        }
        let mut mat_z = Matrix::<Complex<T>, DynamicStorage<Complex<T>>>::new_dynamic(n, n)?;
        for i in 0..n {
            *mat_z.get_mut(i, i).unwrap() = Complex::from_f64(1.0);
        }

        Self::compute_inplace(&mut mat_a, &mut mat_b, &mut mat_q, &mut mat_z)?;

        Ok(Self {
            h: mat_a,
            r: mat_b,
            q: mat_q,
            z: mat_z,
            _phantom: std::marker::PhantomData,
        })
    }

    fn compute_inplace(
        a: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        b: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        q: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        z: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    ) -> Result<(), String> {
        let n = a.rows();

        // 1. Initial reduction of B to upper triangular form
        // We can use Householder reflections on B and apply them to A and Q
        for i in 0..n {
            // Householder reflection to zero B[i+1:n, i]
            // This is standard QR, but we apply it to A as well (left multiplication)
            let (tau, v, sigma) = Self::make_householder(b, i, i, n);
            if tau != Complex::default() {
                Self::apply_householder_left(b, i, n, i, n, tau, &v);
                *b.get_mut(i, i).unwrap() = Complex::default() - sigma;
                for k in i + 1..n {
                    *b.get_mut(k, i).unwrap() = Complex::default();
                }

                Self::apply_householder_left(a, i, n, 0, n, tau, &v);
                Self::apply_householder_left_to_q(q, i, n, tau, &v);
            }
        }

        // 2. Reduce A to Hessenberg form while preserving B's triangularity
        // We zero A[i+2:n, i] for i from 0 to n-3
        for i in 0..n.saturating_sub(2) {
            for j in (i + 2..n).rev() {
                // Zero A[j, i] using a Givens rotation G on rows (j-1, j)
                let (c, s) =
                    Self::givens_rotation(*a.get(j - 1, i).unwrap(), *a.get(j, i).unwrap());

                // Apply G to A (left)
                Self::apply_givens_left(a, j - 1, j, i, n, c, s);
                *a.get_mut(j, i).unwrap() = Complex::default();

                // Apply G to B (left). This creates a bulge at B[j, j-1]
                Self::apply_givens_left(b, j - 1, j, j - 1, n, c, s);

                // Apply G to Q (left)
                Self::apply_givens_left(q, j - 1, j, 0, n, c, s);

                // Now zero the bulge in B[j, j-1] using a Givens rotation G_z on columns (j-1, j)
                // This rotation is on the right.
                let val_j_jm1 = *b.get(j, j - 1).unwrap();
                let (cz, sz) = Self::givens_rotation(*b.get(j, j).unwrap(), -val_j_jm1);

                // Apply G_z to B (right): B = B * G
                Self::apply_givens_right(b, 0, j + 1, j - 1, j, cz, sz);
                *b.get_mut(j, j - 1).unwrap() = Complex::default();

                // Apply G_z to A (right)
                Self::apply_givens_right(a, 0, n, j - 1, j, cz, sz);

                // Apply G_z to Z (right)
                Self::apply_givens_right(z, 0, n, j - 1, j, cz, sz);
            }
        }

        Ok(())
    }

    fn givens_rotation(a: Complex<T>, b: Complex<T>) -> (T, Complex<T>) {
        if b == Complex::default() {
            return (T::from_f64(1.0), Complex::default());
        }
        let norm = (a.norm_sq() + b.norm_sq()).sqrt();
        let c = a.norm() / norm;
        let s = (a / Complex::new(a.norm(), T::default())).conj()
            * (b / Complex::new(norm, T::default()));
        (c, s)
    }

    fn apply_givens_left(
        m: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        i: usize,
        j: usize,
        col_start: usize,
        col_end: usize,
        c: T,
        s: Complex<T>,
    ) {
        let cc = Complex::new(c, T::default());
        for k in col_start..col_end {
            let v1 = *m.get(i, k).unwrap();
            let v2 = *m.get(j, k).unwrap();
            *m.get_mut(i, k).unwrap() = cc * v1 + s.conj() * v2;
            *m.get_mut(j, k).unwrap() = -s * v1 + cc * v2;
        }
    }

    fn apply_givens_right(
        m: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        row_start: usize,
        row_end: usize,
        i: usize,
        j: usize,
        c: T,
        s: Complex<T>,
    ) {
        let cc = Complex::new(c, T::default());
        for k in row_start..row_end {
            let v1 = *m.get(k, i).unwrap();
            let v2 = *m.get(k, j).unwrap();
            // Right multiplication by G = [ [c, -s*], [s, c] ]
            *m.get_mut(k, i).unwrap() = cc * v1 + s * v2;
            *m.get_mut(k, j).unwrap() = -s.conj() * v1 + cc * v2;
        }
    }

    // Helper for initial Householder (manual implementation to avoid type issues)
    fn make_householder(
        m: &Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        col: usize,
        row_start: usize,
        row_end: usize,
    ) -> (Complex<T>, Vec<Complex<T>>, Complex<T>) {
        let x1 = *m.get(row_start, col).unwrap();
        let mut norm_sq = T::default();
        for k in row_start..row_end {
            norm_sq += m.get(k, col).unwrap().norm_sq();
        }
        let norm = norm_sq.sqrt();
        if norm == T::default() {
            return (Complex::default(), vec![], Complex::default());
        }

        let sigma = if x1.norm() == T::default() {
            Complex::new(norm, T::default())
        } else {
            x1 / Complex::new(x1.norm(), T::default()) * Complex::new(norm, T::default())
        };

        let v1 = x1 + sigma;
        let tau = v1.conj() / sigma.conj();
        let mut v = vec![Complex::from_f64(1.0)];
        for k in row_start + 1..row_end {
            v.push(*m.get(k, col).unwrap() / v1);
        }
        (tau, v, sigma)
    }

    fn apply_householder_left(
        m: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        r_start: usize,
        _r_end: usize,
        c_start: usize,
        c_end: usize,
        tau: Complex<T>,
        v: &[Complex<T>],
    ) {
        for j in c_start..c_end {
            let mut dot = *m.get(r_start, j).unwrap();
            for (k, vk) in v.iter().enumerate().skip(1) {
                dot += vk.conj() * (*m.get(r_start + k, j).unwrap());
            }
            let factor = tau * dot;
            *m.get_mut(r_start, j).unwrap() -= factor;
            for (k, vk) in v.iter().enumerate().skip(1) {
                *m.get_mut(r_start + k, j).unwrap() -= factor * *vk;
            }
        }
    }

    fn apply_householder_left_to_q(
        q: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
        r_start: usize,
        _r_end: usize,
        tau: Complex<T>,
        v: &[Complex<T>],
    ) {
        let n = q.rows();
        // Q = H * Q => q[r_start:r_end, :] = (I - tau v v*) q[r_start:r_end, :]
        // Wait, normally we store Q such that A = Q H Z*.
        // Initial reduction: B = Q_b R => Q_b* B = R. So Q = Q_b*.
        // Reflection is H_i. Q_new = H_i * Q_old.
        for j in 0..n {
            let mut dot = *q.get(r_start, j).unwrap();
            for (k, vk) in v.iter().enumerate().skip(1) {
                dot += vk.conj() * (*q.get(r_start + k, j).unwrap());
            }
            let factor = tau * dot;
            *q.get_mut(r_start, j).unwrap() -= factor;
            for (k, vk) in v.iter().enumerate().skip(1) {
                *q.get_mut(r_start + k, j).unwrap() -= factor * *vk;
            }
        }
    }

    pub fn matrix_h(&self) -> &Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        &self.h
    }
    pub fn matrix_r(&self) -> &Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        &self.r
    }
    pub fn matrix_q(&self) -> &Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        &self.q
    }
    pub fn matrix_z(&self) -> &Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        &self.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_ght_basic() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        let mut b = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;

        let data_a = [
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.5),
            Complex::new(3.0, 2.0),
            Complex::new(1.0, -1.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 2.0),
        ];
        let data_b = [
            Complex::new(5.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(4.0, 2.0),
            Complex::new(2.0, 1.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.0),
            Complex::new(3.0, -1.0),
        ];

        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data_a[i * n + j];
                *b.get_mut(i, j).unwrap() = data_b[i * n + j];
            }
        }

        let ght = GeneralizedHessenbergTriangular::new(&a, &b)?;
        let h = ght.matrix_h();
        let r = ght.matrix_r();
        let q = ght.matrix_q();
        let z = ght.matrix_z();

        // 1. Unitarity of Q: Q*Q* = I (or Q* Q = I, here q is applied from left as Q*A)
        // Wait, current implementation: Q = H_n ... H_1 I. So Q* A = H.
        // Thus A = Q H Z*. Actually, let's just check A * Z = Q^* * H?
        // No, if Q* A Z = H, then A Z = Q H.
        // Similarly, B Z = Q R.

        // Let's verify A * Z = Q^* * H.
        // Wait, my apply_householder_left_to_q does Q = H * Q.
        // So Q_final = H_last * ... * H_first * I.
        // Thus Q * A_initial * Z = H_final.
        // A_initial * Z = Q^* * H_final.

        for i in 0..n {
            for j in 0..n {
                let mut az = Complex::default();
                let mut qh = Complex::default();
                for k in 0..n {
                    az += (*a.get(i, k).unwrap()) * (*z.get(k, j).unwrap());
                    qh += q.get(k, i).unwrap().conj() * (*h.get(k, j).unwrap());
                }
                assert!(
                    (az.re - qh.re).abs() < 1e-10,
                    "A parity failure at ({}, {})",
                    i,
                    j
                );
                assert!(
                    (az.im - qh.im).abs() < 1e-10,
                    "A parity failure at ({}, {})",
                    i,
                    j
                );
            }
        }

        for i in 0..n {
            for j in 0..n {
                let mut bz = Complex::default();
                let mut qr = Complex::default();
                for k in 0..n {
                    bz += (*b.get(i, k).unwrap()) * (*z.get(k, j).unwrap());
                    qr += q.get(k, i).unwrap().conj() * (*r.get(k, j).unwrap());
                }
                assert!(
                    (bz.re - qr.re).abs() < 1e-10,
                    "B parity failure at ({}, {})",
                    i,
                    j
                );
                assert!(
                    (bz.im - qr.im).abs() < 1e-10,
                    "B parity failure at ({}, {})",
                    i,
                    j
                );
            }
        }

        // 2. Form checks
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    assert!(
                        h.get(i, j).unwrap().norm_sq() < 1e-20,
                        "H is not Hessenberg at ({}, {})",
                        i,
                        j
                    );
                }
                if i > j {
                    assert!(
                        r.get(i, j).unwrap().norm_sq() < 1e-20,
                        "R is not triangular at ({}, {})",
                        i,
                        j
                    );
                }
            }
        }

        Ok(())
    }
}
