use crate::core::complex::Complex;
use crate::core::decompositions::EigenSolver;
use crate::core::matrix::MatrixX;
use crate::core::scalar::Scalar;
use alloc::vec::Vec;

pub mod jenkins_traub;
pub use jenkins_traub::JenkinsTraubSolver;

/// Evaluates a polynomial using Horner's method.
/// The polynomial is represented by coefficients `poly` in ascending order of power.
/// P(x) = poly[0] + poly[1]*x + ... + poly[n]*x^n
pub fn poly_eval<T: Scalar>(poly: &[T], x: T) -> T {
    let mut grid = T::from_usize(0);
    for &coeff in poly.iter().rev() {
        grid = grid * x + coeff;
    }
    grid
}

/// Helper struct for finding roots of a polynomial.
/// Roots are found by computing eigenvalues of the Companion Matrix.
pub struct PolynomialSolver<T: Scalar> {
    roots: Vec<Complex<T>>,
}

impl<T: Scalar<Real = T> + PartialOrd> Default for PolynomialSolver<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar<Real = T> + PartialOrd> PolynomialSolver<T> {
    /// Creates a new solver instance.
    pub fn new() -> Self {
        Self { roots: Vec::new() }
    }

    /// Computes roots of the polynomial.
    /// `poly`: coefficients in ascending order (c0 + c1*x + ... + cn*x^n).
    pub fn compute(&mut self, poly: &[T]) -> Result<(), String> {
        if poly.is_empty() {
            return Err("Empty polynomial".to_string());
        }

        // Find the effective degree (ignore trailing zeros)
        let mut n = poly.len() - 1;
        while n > 0 && poly[n] == T::from_usize(0) {
            n -= 1;
        }

        if n == 0 {
            // Constant polynomial: no roots
            self.roots.clear();
            return Ok(());
        }

        let leading = poly[n];

        if leading == T::from_usize(0) {
            // Should not happen due to loop above, but if n=0 and poly[0]=0, logic holds.
            return Ok(());
        }

        // Companion Matrix logic
        // For P(x) = c_0 + ... + c_n x^n
        // Monic: x^n + (c_{n-1}/c_n)x^{n-1} + ... + (c_0/c_n)
        // Let a_i = c_i / c_n
        // Companion M (size n x n):
        // [ 0   0   ...   0   -a_0 ]
        // [ 1   0   ...   0   -a_1 ]
        // [ 0   1   ...   0   -a_2 ]
        // [ ... ... ... ...   ...  ]
        // [ 0   0   ...   1   -a_{n-1} ]

        let mut companion = MatrixX::<T>::new_dynamic(n, n).map_err(|e| e.to_string())?;
        companion.set_zero();

        for (i, coeff) in poly.iter().enumerate().take(n) {
            // Subdiagonal ones
            if i > 0 {
                *companion.get_mut(i, i - 1).unwrap() = T::from_usize(1);
            }

            // Last column: -c_i / leading
            let c = *coeff;
            let val = -(c / leading);
            *companion.get_mut(i, n - 1).unwrap() = val;
        }

        // Solve eigenvalues
        let solver = EigenSolver::new(&companion, false)?;
        self.roots = solver.eigenvalues().to_vec();

        Ok(())
    }

    /// Returns the computed roots.
    pub fn roots(&self) -> &[Complex<T>] {
        &self.roots
    }
}
