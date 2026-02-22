//! Optimization module for non-linear solvers.

use crate::core::matrix::MatrixX;
use crate::core::scalar::Scalar;

pub mod auto_tune;
pub mod autodiff;
pub mod hybrid;
pub mod levenberg_marquardt;
pub mod numerical_diff;

/// Trait representing a non-linear function for optimization.
///
/// Users must implement this to define the residuals and the Jacobian.
pub trait Functor<T: Scalar> {
    /// Number of parameters (variables to optimize).
    fn inputs(&self) -> usize;

    /// Number of residuals (equations).
    fn values(&self) -> usize;

    /// Evaluates the function at point `x`, storing the residuals in `fvec`.
    fn operator(&self, x: &MatrixX<T>, fvec: &mut MatrixX<T>) -> Result<(), String>;

    /// Evaluates the Jacobian at point `x`, storing it in `fjac`.
    ///
    /// Jacobian J_ij = df_i / dx_j
    fn jacobian(&self, x: &MatrixX<T>, fjac: &mut MatrixX<T>) -> Result<(), String>;
}

/// Status of the optimization solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Status {
    Running,
    Converged,
    MaxIterationsReached,
    SingularJacobian,
    ParameterError,
}
