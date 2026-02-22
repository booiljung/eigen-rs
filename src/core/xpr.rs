//! Expression traits for eigen-rs.
//! Mirrors Eigen's XprKind and base expression types.

use crate::core::scalar::Scalar;

/// Trait representing a generic matrix expression.
/// All expressions and concrete matrices implement this trait.
pub trait MatrixXpr<T: Scalar>: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    /// Optional: Returns a raw pointer to the storage if contiguous.
    /// Used for BLAS/GEMM optimizations.
    fn as_ptr(&self) -> Option<*const T> {
        None
    }

    /// Optional: Returns (row_stride, col_stride).
    /// Used for BLAS/GEMM optimizations.
    fn strides(&self) -> Option<(isize, isize)> {
        None
    }

    /// Evaluates the expression at a specific coordinate.
    /// This is the heart of lazy evaluation.
    fn eval(&self, row: usize, col: usize) -> T;

    /// Evaluates a packet of coefficients starting at (row, col).
    /// Default implementation uses scalar eval in a loop (slow fallback).
    fn packet_eval<P: crate::core::arch::Packet<T>>(&self, row: usize, col: usize) -> P
    where
        T: Scalar,
    {
        let mut data = [T::default(); 16]; // Max packet size for now (e.g. AVX512/AMX future proofing)
        let size = P::SIZE;
        for (i, val) in data.iter_mut().enumerate().take(size) {
            // Default fallback assumes ColMajor (standard for this crate)
            *val = self.eval(row + i, col);
        }
        unsafe { P::load(data.as_ptr()) }
    }

    /// Returns true if the expression allows linear access (1D indexing).
    fn has_linear_access(&self) -> bool {
        false
    }

    /// Evaluates the expression at linear index `i`.
    /// User should verify `has_linear_access()` before calling this.
    fn eval_linear(&self, _i: usize) -> T {
        panic!("Linear evaluation not supported for this expression");
    }

    /// Evaluates a packet at linear index `i`.
    fn packet_eval_linear<P: crate::core::arch::Packet<T>>(&self, _i: usize) -> P
    where
        T: Scalar,
    {
        panic!("Linear packet evaluation not supported for this expression");
    }

    /// Optimized linear evaluation directly to a destination buffer.
    /// Returns true if the evaluation was performed using a fast path.
    fn try_eval_to<P: crate::core::arch::Packet<T>>(&self, _dest: *mut T, _size: usize) -> bool {
        false
    }

    /// Returns the total size of the expression.
    fn size(&self) -> usize {
        self.rows() * self.cols()
    }

    /// Returns a lazy transpose expression.
    fn transpose(&self) -> crate::core::ops::TransposeOp<'_, T, Self>
    where
        Self: Sized,
        T: crate::core::scalar::Scalar,
    {
        crate::core::ops::TransposeOp::new(self)
    }

    /// Returns the sum of all elements.
    fn sum(&self) -> T
    where
        T: crate::core::scalar::Scalar + Send + Sync,
    {
        #[cfg(feature = "parallel")]
        {
            if self.size() > 10000 {
                use rayon::prelude::*;
                let rows = self.rows();
                return (0..self.cols())
                    .into_par_iter()
                    .map(|c| {
                        let mut s = T::default();
                        for r in 0..rows {
                            s += self.eval(r, c);
                        }
                        s
                    })
                    .reduce(|| T::default(), |a, b| a + b);
            }
        }

        let mut sum = T::default();
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                sum += self.eval(r, c);
            }
        }
        sum
    }

    /// Returns the minimum element.
    fn min(&self) -> T
    where
        T: crate::core::scalar::Scalar + Send + Sync + core::cmp::PartialOrd,
    {
        if self.size() == 0 {
            return T::default();
        }

        #[cfg(feature = "parallel")]
        {
            if self.size() > 10000 {
                use rayon::prelude::*;
                let rows = self.rows();
                return (0..self.cols())
                    .into_par_iter()
                    .map(|c| {
                        let mut m = self.eval(0, c);
                        for r in 0..rows {
                            let val = self.eval(r, c);
                            if val < m {
                                m = val;
                            }
                        }
                        m
                    })
                    .reduce(|| self.eval(0, 0), |a, b| if a < b { a } else { b });
            }
        }

        let mut m = self.eval(0, 0);
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                let val = self.eval(r, c);
                if val < m {
                    m = val;
                }
            }
        }
        m
    }

    /// Returns the maximum element.
    fn max(&self) -> T
    where
        T: crate::core::scalar::Scalar + Send + Sync + core::cmp::PartialOrd,
    {
        if self.size() == 0 {
            return T::default();
        }

        #[cfg(feature = "parallel")]
        {
            if self.size() > 10000 {
                use rayon::prelude::*;
                let rows = self.rows();
                return (0..self.cols())
                    .into_par_iter()
                    .map(|c| {
                        let mut m = self.eval(0, c);
                        for r in 0..rows {
                            let val = self.eval(r, c);
                            if val > m {
                                m = val;
                            }
                        }
                        m
                    })
                    .reduce(|| self.eval(0, 0), |a, b| if a > b { a } else { b });
            }
        }

        let mut m = self.eval(0, 0);
        for c in 0..self.cols() {
            for r in 0..self.rows() {
                let val = self.eval(r, c);
                if val > m {
                    m = val;
                }
            }
        }
        m
    }

    /// Returns the mean of all elements.
    fn mean(&self) -> T
    where
        T: crate::core::scalar::Scalar,
    {
        let s = self.size();
        if s == 0 {
            return T::default();
        }
        self.sum() / T::from_usize(s)
    }

    /// Returns the determinant of the matrix.
    /// Currently only supports square matrices up to 3x3.
    fn determinant(&self) -> T
    where
        T: crate::core::scalar::Scalar,
    {
        if self.rows() != self.cols() {
            panic!(
                "Determinant only defined for square matrices ({}x{})",
                self.rows(),
                self.cols()
            );
        }

        match self.rows() {
            0 => T::from_usize(1),
            1 => self.eval(0, 0),
            2 => self.eval(0, 0) * self.eval(1, 1) - self.eval(1, 0) * self.eval(0, 1),
            3 => {
                let m00 = self.eval(0, 0);
                let m01 = self.eval(0, 1);
                let m02 = self.eval(0, 2);
                let m10 = self.eval(1, 0);
                let m11 = self.eval(1, 1);
                let m12 = self.eval(1, 2);
                let m20 = self.eval(2, 0);
                let m21 = self.eval(2, 1);
                let m22 = self.eval(2, 2);

                m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
                    + m02 * (m10 * m21 - m11 * m20)
            }
            _ => {
                // For N > 3, use LU decomposition.
                // This requires evaluating the current expression into a matrix first if it's not already one.
                // For simplicity, we assume this is called on a Matrix or we'd need a way to evaluate into a temp.
                // If it's a MatrixXpr, we'd ideally eval into a DynamicMatrix.
                // For now, let's keep it simple: if it's a large matrix, we expect it to be a Matrix type or we've evaluated it.
                // However, the trait method is &self.
                // Let's implement a helper that evals to DynamicStorage if needed.
                match self.rows() {
                    n if n > 3 => {
                        // We can't easily call partial_piv_lu here because MatrixXpr doesn't guarantee it's a Matrix.
                        // For now, we'll keep the panic but update the message to note we need an eval-then-LU path.
                        panic!("Determinant for N > 3 requires LU decomposition. Please evaluate the expression into a Matrix first and call partial_piv_lu().determinant().");
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Returns a lazy block expression.
    fn block(
        &self,
        start_row: usize,
        start_col: usize,
        rows: usize,
        cols: usize,
    ) -> crate::core::ops::BlockOp<'_, T, Self>
    where
        Self: Sized,
        T: crate::core::scalar::Scalar,
    {
        crate::core::ops::BlockOp::new(self, start_row, start_col, rows, cols)
            .expect("Block out of bounds")
    }

    /// Returns a lazy row expression.
    fn row(&self, i: usize) -> crate::core::ops::BlockOp<'_, T, Self>
    where
        Self: Sized,
        T: crate::core::scalar::Scalar,
    {
        self.block(i, 0, 1, self.cols())
    }

    /// Returns a lazy column expression.
    fn col(&self, j: usize) -> crate::core::ops::BlockOp<'_, T, Self>
    where
        Self: Sized,
        T: crate::core::scalar::Scalar,
    {
        self.block(0, j, self.rows(), 1)
    }
}
