use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::DynamicStorage;
use crate::core::jit::ast::Expr;

/// JIT Execution Engine.
///
/// This engine compiles the AST expression tree into a single fused evaluation loop.
/// This perfectly mimics JIT-kernel compilation by completely removing intermediate 
/// buffer allocations, evaluating complex equations directly into the destination tensor.
impl<'a, T: Scalar> Expr<'a, T> {
    /// Evaluates the expression tree scalar-by-scalar at a given `(row, col)`.
    /// This function is recursive but aggressive inlining typically flattens it.
    #[inline(always)]
    fn eval_at(&self, r: usize, c: usize) -> T {
        match self {
            Expr::MatrixRef(m) => unsafe { *m.get_unchecked(r, c) },
            Expr::Add(lhs, rhs) => lhs.eval_at(r, c) + rhs.eval_at(r, c),
            Expr::Sub(lhs, rhs) => lhs.eval_at(r, c) - rhs.eval_at(r, c),
            Expr::MulWise(lhs, rhs) => lhs.eval_at(r, c) * rhs.eval_at(r, c),
            Expr::MulScalar(lhs, scalar) => lhs.eval_at(r, c) * *scalar,
        }
    }

    /// Executes the AST computation graph directly into the given destination matrix.
    /// This performs "Loop Fusion", executing the entire tree in exactly one memory pass.
    pub fn execute_into(&self, out: &mut Matrix<T, DynamicStorage<T>>) -> Result<(), String> {
        let (expr_rows, expr_cols) = self.dimensions();
        
        if out.rows() != expr_rows || out.cols() != expr_cols {
            return Err("Destination matrix dimension mismatch against AST expression".to_string());
        }

        // Fused Evaluation Loop: Iterate exactly once over the destination layout
        for c in 0..expr_cols {
            for r in 0..expr_rows {
                let computed_val = self.eval_at(r, c);
                // Writing to the destination avoiding intermediate allocations completely
                unsafe {
                    *out.get_unchecked_mut(r, c) = computed_val;
                }
            }
        }

        Ok(())
    }

    /// Convenience generic method allocating and executing the AST graph.
    pub fn execute(&self) -> Matrix<T, DynamicStorage<T>> {
        let (rows, cols) = self.dimensions();
        let mut out = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols).unwrap();
        // Ignoring error bounds mathematically guaranteed by initial dimensions extraction
        let _ = self.execute_into(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_fused_execution() {
        // Build matrices
        let a = Matrix::<f64, DynamicStorage<f64>>::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(2, 2, vec![2.0, 2.0, 2.0, 2.0]).unwrap();
        let c = Matrix::<f64, DynamicStorage<f64>>::from_vec(2, 2, vec![3.0, 3.0, 3.0, 3.0]).unwrap();

        // Create AST: Equation: D = (A + B) * 10.0 - C
        // Mathematically: ((1 + 2) * 10) - 3 = 27
        let expr = Expr::Sub(
            Box::new(Expr::MulScalar(
                Box::new(Expr::Add(
                    Box::new(Expr::MatrixRef(&a)),
                    Box::new(Expr::MatrixRef(&b))
                )),
                10.0
            )),
            Box::new(Expr::MatrixRef(&c))
        );

        // Execute JIT Fused Loop
        let mut d = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(2, 2).unwrap();
        expr.execute_into(&mut d).unwrap();
        
        // Assert correct fused operation
        assert_eq!(*d.get(0, 0).unwrap(), 27.0);
        assert_eq!(*d.get(1, 1).unwrap(), 27.0);
    }
}
