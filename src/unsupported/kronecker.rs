

use crate::core::scalar::Scalar;

use crate::core::xpr::MatrixXpr;

/// Expression representing the Kronecker product of two matrices.
///
/// The Kronecker product A (x) B is a block matrix where each element a_ij is replaced by a_ij * B.
/// The resulting matrix has dimensions (A.rows * B.rows) x (A.cols * B.cols).
pub struct KroneckerProduct<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    lhs: &'a L,
    rhs: &'a R,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, L, R> KroneckerProduct<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    pub fn new(lhs: &'a L, rhs: &'a R) -> Self {
        Self {
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, L, R> MatrixXpr<T> for KroneckerProduct<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    fn rows(&self) -> usize {
        self.lhs.rows() * self.rhs.rows()
    }

    fn cols(&self) -> usize {
        self.lhs.cols() * self.rhs.cols()
    }

    fn eval(&self, row: usize, col: usize) -> T {
        // Map global (row, col) to block coordinates
        // Block index (i, j) corresponding to element a_ij in LHS
        // Local index (p, q) corresponding to element b_pq in RHS
        // row = i * rhs.rows() + p
        // col = j * rhs.cols() + q

        // Therefore:
        // i = row / rhs.rows()
        // p = row % rhs.rows()
        // j = col / rhs.cols()
        // q = col % rhs.cols()

        let rhs_rows = self.rhs.rows();
        let rhs_cols = self.rhs.cols();

        let i = row / rhs_rows;
        let p = row % rhs_rows;
        let j = col / rhs_cols;
        let q = col % rhs_cols;

        self.lhs.eval(i, j) * self.rhs.eval(p, q)
    }

    // Packet evaluation is hard because memory access is strided/discontinuous for RHS blocks.
    // We explicitly don't implement optimized packet_eval here, falling back to scalar.
}

impl<'a, T, L, R> crate::core::cuda::CudaDispatcher<T> for KroneckerProduct<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    // Default implementation returns false (fallback to CPU)
}

/// Computes the Kronecker product of two matrices.
pub fn kronecker_product<'a, T, L, R>(lhs: &'a L, rhs: &'a R) -> KroneckerProduct<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    KroneckerProduct::new(lhs, rhs)
}
