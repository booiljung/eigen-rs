//! Schur decomposition of a square complex matrix.
//! A = U * T * U^*

use crate::core::matrix::Matrix;
use crate::core::storage::{Storage, DynamicStorage};
use crate::core::scalar::Scalar;
use crate::core::complex::Complex;
use crate::core::decompositions::HessenbergDecomposition;

/// Schur decomposition of a square complex matrix.
/// For complex matrices, the result T is always strictly upper triangular.
pub struct ComplexSchur<T: Scalar, S: Storage<Complex<T>>> {
    t: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    u: Matrix<Complex<T>, DynamicStorage<Complex<T>>>,
    _phantom: std::marker::PhantomData<S>,
}

impl<T: Scalar, S: Storage<Complex<T>>> ComplexSchur<T, S> {
    /// Computes the Schur decomposition of the given square complex matrix.
    pub fn new(matrix: &Matrix<Complex<T>, S>) -> Result<Self, String> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err("Complex Schur decomposition requires a square matrix".to_string());
        }

        // 1. Initial Hessenberg reduction: A = U H U^*
        let hessenberg = HessenbergDecomposition::new(matrix)?;
        let mut h = hessenberg.matrix_h();
        let mut u = hessenberg.matrix_q();

        if n > 1 {
            Self::compute_inplace(&mut h, &mut u)?;
        }

        Ok(Self {
            t: h,
            u,
            _phantom: std::marker::PhantomData,
        })
    }

    fn compute_inplace(h: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>, u: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>) -> Result<(), String> {
        let n = h.rows();
        let max_iter = 40 * n;
        let mut iter = 0;
        let mut high = n - 1;

        let eps = Complex::<T>::epsilon();

        while high > 0 {
            if iter > max_iter {
                return Err("ComplexSchur failed to converge".to_string());
            }

            // 1. Deflation: Check for small sub-diagonal elements
            let mut i = high;
            while i > 0 {
                let h_i_im1 = h.get(i, i - 1).unwrap().norm_sq();
                let h_im1_im1 = h.get(i - 1, i - 1).unwrap().norm_sq();
                let h_i_i = h.get(i, i).unwrap().norm_sq();
                if h_i_im1 <= eps.re * (h_im1_im1 + h_i_i) {
                    *h.get_mut(i, i - 1).unwrap() = Complex::default();
                    break;
                }
                i -= 1;
            }

            if i == high {
                // Found a 1x1 block at the end
                high = high.saturating_sub(1);
                iter = 0;
            } else {
                // 2. Complex QR step on the sub-matrix [i..high+1, i..high+1]
                let start = i;
                let end = high;

                // Wilkinson shift from the bottom 2x2 block
                let lambda = Self::wilkinson_shift(h, end);
                Self::qr_step(h, u, start, end, lambda);
                iter += 1;
            }
        }

        Ok(())
    }

    fn wilkinson_shift(h: &Matrix<Complex<T>, DynamicStorage<Complex<T>>>, end: usize) -> Complex<T> {
        let d1 = *h.get(end - 1, end - 1).unwrap();
        let d2 = *h.get(end, end).unwrap();
        let h12 = *h.get(end - 1, end).unwrap();
        let h21 = *h.get(end, end - 1).unwrap();

        let tr = d1 + d2;
        let det = d1 * d2 - h12 * h21;
        // solve s^2 - tr*s + det = 0
        let disc = tr * tr - Complex::from_f64(4.0) * det;
        let s_disc = disc.sqrt();
        let lambda1 = (tr + s_disc) / Complex::from_f64(2.0);
        let lambda2 = (tr - s_disc) / Complex::from_f64(2.0);

        // Choose the one closer to d2
        if (lambda1 - d2).norm_sq() < (lambda2 - d2).norm_sq() {
            lambda1
        } else {
            lambda2
        }
    }

    fn qr_step(h: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>, u: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>, start: usize, end: usize, shift: Complex<T>) {
        let n = h.rows();
        
        // Initial reflection to create bulge
        let x = *h.get(start, start).unwrap() - shift;
        let y = *h.get(start + 1, start).unwrap();
        let (tau, v) = Self::make_householder(&[x, y]);

        // Apply Householder reflection P = I - tau v v^*
        // Left, Right (Hermitian conjugate), and U (Hermitian conjugate)
        Self::apply_householder_left(h, start, start + 1, start, n, tau, &v);
        Self::apply_householder_right(h, 0, n, start, start + 1, tau.conj(), &v);
        Self::apply_householder_right(u, 0, n, start, start + 1, tau.conj(), &v);

        // Bulge chasing
        for k in start..end - 1 {
            let x = *h.get(k + 1, k).unwrap();
            let y = *h.get(k + 2, k).unwrap();
            let (tau, v, sigma) = Self::make_householder_with_sigma(&[x, y]); // Need sigma here

            Self::apply_householder_left(h, k + 1, k + 2, k, n, tau, &v);
            *h.get_mut(k + 1, k).unwrap() = Complex::default() - sigma;
            *h.get_mut(k + 2, k).unwrap() = Complex::default();
            
            Self::apply_householder_right(h, 0, n, k + 1, k + 2, tau.conj(), &v);
            Self::apply_householder_right(u, 0, n, k + 1, k + 2, tau.conj(), &v);
        }
    }

    fn make_householder_with_sigma(v: &[Complex<T>]) -> (Complex<T>, Vec<Complex<T>>, Complex<T>) {
        let x1 = v[0];
        let norm = (v[0].norm_sq() + v[1].norm_sq()).sqrt();
        if norm == T::default() {
            return (Complex::default(), vec![Complex::from_f64(1.0), Complex::default()], Complex::default());
        }
        
        let sigma = if x1.re == T::default() && x1.im == T::default() {
            Complex::new(norm, T::default())
        } else {
            let abs_x1 = x1.norm();
            x1 / Complex::new(abs_x1, T::default()) * Complex::new(norm, T::default())
        };

        let v1 = x1 + sigma;
        let tau = v1.conj() / sigma.conj();
        let h_v = vec![Complex::from_f64(1.0), v[1] / v1];
        
        (tau, h_v, sigma)
    }

    fn make_householder(v: &[Complex<T>]) -> (Complex<T>, Vec<Complex<T>>) {
        let (tau, h_v, _) = Self::make_householder_with_sigma(v);
        (tau, h_v)
    }

    fn apply_householder_left(h: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>, r1: usize, r2: usize, col_start: usize, col_end: usize, tau: Complex<T>, v: &[Complex<T>]) {
        for j in col_start..col_end {
            let dot = *h.get(r1, j).unwrap() + v[1].conj() * (*h.get(r2, j).unwrap());
            let factor = tau * dot;
            *h.get_mut(r1, j).unwrap() -= factor;
            *h.get_mut(r2, j).unwrap() -= factor * v[1];
        }
    }

    fn apply_householder_right(h: &mut Matrix<Complex<T>, DynamicStorage<Complex<T>>>, row_start: usize, row_end: usize, c1: usize, c2: usize, tau: Complex<T>, v: &[Complex<T>]) {
        for i in row_start..row_end {
            let dot = *h.get(i, c1).unwrap() + (*h.get(i, c2).unwrap()) * v[1];
            let factor = tau * dot;
            *h.get_mut(i, c1).unwrap() -= factor;
            *h.get_mut(i, c2).unwrap() -= factor * v[1].conj();
        }
    }

    pub fn matrix_t(&self) -> &Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        &self.t
    }

    pub fn matrix_u(&self) -> &Matrix<Complex<T>, DynamicStorage<Complex<T>>> {
        &self.u
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::storage::DynamicStorage;

    #[test]
    fn test_complex_schur_basic() -> Result<(), String> {
        let n = 3;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        // Non-symmetric complex matrix
        let data = [
            Complex::new(1.0, 2.0), Complex::new(2.0, -1.0), Complex::new(0.0, 1.0),
            Complex::new(1.0, 1.0), Complex::new(4.0, 0.0),  Complex::new(2.0, 3.0),
            Complex::new(0.0, 0.0), Complex::new(1.0, 2.0),  Complex::new(5.0, -2.0),
        ];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let schur = ComplexSchur::new(&a)?;
        
        // Diagnostic: Verify Hessenberg first
        let hess = HessenbergDecomposition::new(&a)?;
        let hh = hess.matrix_h();
        let hq = hess.matrix_q();
        
        // 0. Check H is upper Hessenberg
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    assert!(hh.get(i, j).unwrap().norm_sq() < 1e-15, "H is not upper Hessenberg at ({}, {})", i, j);
                }
            }
        }

        // 1. Check hq is unitary
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::new(0.0, 0.0);
                for k in 0..n {
                    sum += hq.get(k, i).unwrap().conj() * (*hq.get(k, j).unwrap());
                }
                let expected = if i == j { Complex::new(1.0, 0.0) } else { Complex::default() };
                assert!((sum.re - expected.re).abs() < 1e-10, "HQ Unitarity mismatch at ({}, {})", i, j);
                assert!((sum.im - expected.im).abs() < 1e-10, "HQ Unitarity mismatch at ({}, {})", i, j);
            }
        }

        // 2. Check A * hq = hq * hh
        for i in 0..n {
            for j in 0..n {
                let mut ahq = Complex::new(0.0, 0.0);
                let mut hqh = Complex::new(0.0, 0.0);
                for k in 0..n {
                    ahq += (*a.get(i, k).unwrap()) * (*hq.get(k, j).unwrap());
                    hqh += (*hq.get(i, k).unwrap()) * (*hh.get(k, j).unwrap());
                }
                assert!((ahq.re - hqh.re).abs() < 1e-10, "Hessenberg Real mismatch at ({}, {}): {} vs {}", i, j, ahq.re, hqh.re);
                assert!((ahq.im - hqh.im).abs() < 1e-10, "Hessenberg Imag mismatch at ({}, {}): {} vs {}", i, j, ahq.im, hqh.im);
            }
        }

        let t = schur.matrix_t();
        let u = schur.matrix_u();

        // 1. Parity check A * U = U * T
        for i in 0..n {
            for j in 0..n {
                let mut au = Complex::new(0.0, 0.0);
                let mut ut = Complex::new(0.0, 0.0);
                for k in 0..n {
                    au += (*a.get(i, k).unwrap()) * (*u.get(k, j).unwrap());
                    ut += (*u.get(i, k).unwrap()) * (*t.get(k, j).unwrap());
                }
                let diff_re = (au.re - ut.re).abs();
                let diff_im = (au.im - ut.im).abs();
                if diff_re > 1e-8 || diff_im > 1e-8 {
                    eprintln!("Mismatch at ({}, {}): A*U = {}, U*T = {}, diff = ({}, {})", i, j, au, ut, diff_re, diff_im);
                }
                assert!(diff_re < 1e-8, "Real mismatch at ({}, {})", i, j);
                assert!(diff_im < 1e-8, "Imag mismatch at ({}, {})", i, j);
            }
        }

        // 2. Unitarity check U^* * U = I
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::new(0.0, 0.0);
                for k in 0..n {
                    sum += u.get(k, i).unwrap().conj() * (*u.get(k, j).unwrap());
                }
                let expected = if i == j { Complex::new(1.0, 0.0) } else { Complex::default() };
                assert!((sum.re - expected.re).abs() < 1e-10);
                assert!((sum.im - expected.im).abs() < 1e-10);
            }
        }

        // 3. Triangularity check
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    assert!(t.get(i, j).unwrap().norm_sq() < 1e-15);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_complex_schur_2x2() -> Result<(), String> {
        let n = 2;
        let mut a = Matrix::<Complex<f64>, DynamicStorage<Complex<f64>>>::new_dynamic(n, n)?;
        let data = [
            Complex::new(1.0, 2.0), Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0), Complex::new(7.0, 8.0),
        ];
        for i in 0..n {
            for j in 0..n {
                *a.get_mut(i, j).unwrap() = data[i * n + j];
            }
        }

        let schur = ComplexSchur::new(&a)?;
        let t = schur.matrix_t();
        let u = schur.matrix_u();

        // Parity
        for i in 0..n {
            for j in 0..n {
                let mut au = Complex::new(0.0, 0.0);
                let mut ut = Complex::new(0.0, 0.0);
                for k in 0..n {
                    au += (*a.get(i, k).unwrap()) * (*u.get(k, j).unwrap());
                    ut += (*u.get(i, k).unwrap()) * (*t.get(k, j).unwrap());
                }
                assert!((au.re - ut.re).abs() < 1e-9, "Mismatch at ({}, {})", i, j);
                assert!((au.im - ut.im).abs() < 1e-9, "Mismatch at ({}, {})", i, j);
            }
        }
        Ok(())
    }
}
