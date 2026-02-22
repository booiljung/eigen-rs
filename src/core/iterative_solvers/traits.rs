use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::Storage;

/// Common interface for iterative solvers.
/// M: The type of the matrix A (can be Dense or Sparse).
pub trait IterativeSolver<T: Scalar, S: Storage<T>> {
    type MatrixType;

    /// Initializes the solver with the matrix A.
    fn compute(&mut self, matrix: &Self::MatrixType);

    /// Computes the solution x of Ax = b using the current decomposition.
    fn solve<S2>(&self, b: &Matrix<T, S2>) -> Matrix<T, S>
    where
        S2: Storage<T>;

    /// Computes the solution x of Ax = b using the current decomposition and an initial guess x0.
    fn solve_with_guess<S2, S3>(&self, b: &Matrix<T, S2>, x0: &Matrix<T, S3>) -> Matrix<T, S>
    where
        S2: Storage<T>,
        S3: Storage<T>;

    /// Returns the number of iterations performed in the last solve.
    fn iterations(&self) -> usize;

    /// Returns the estimated error of the last solve.
    fn error(&self) -> T;
}
