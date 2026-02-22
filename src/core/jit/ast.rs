use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;

/// An Abstract Syntax Tree (AST) representing deferred mathematical operations.
///
/// This enum allows building computation graphs dynamically at runtime.
/// Instead of eager execution (which allocates intermediate buffers), operations
/// are recorded and fused into a single loop by the execution engine.
#[derive(Clone, Debug)]
pub enum Expr<'a, T: Scalar> {
    /// A leaf node referencing an existing dense matrix in memory.
    MatrixRef(&'a Matrix<T, DynamicStorage<T>>),
    
    /// Pointwise addition of two expressions: `A + B`
    Add(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    
    /// Pointwise subtraction of two expressions: `A - B`
    Sub(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    
    /// Pointwise multiplication of two expressions: `A .* B`
    MulWise(Box<Expr<'a, T>>, Box<Expr<'a, T>>),
    
    /// Multiplication by a scalar: `A * scalar`
    MulScalar(Box<Expr<'a, T>>, T),
}

impl<'a, T: Scalar> Expr<'a, T> {
    /// Returns the dimensions `(rows, cols)` of the resulting expression.
    /// Assumes all nodes in the tree are dimensionally compatible.
    pub fn dimensions(&self) -> (usize, usize) {
        match self {
            Expr::MatrixRef(m) => (m.rows(), m.cols()),
            Expr::Add(lhs, _) => lhs.dimensions(),
            Expr::Sub(lhs, _) => lhs.dimensions(),
            Expr::MulWise(lhs, _) => lhs.dimensions(),
            Expr::MulScalar(lhs, _) => lhs.dimensions(),
        }
    }
}
