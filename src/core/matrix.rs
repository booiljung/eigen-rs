use crate::core::decompositions::PartialPivLU;
use crate::core::ops::{CwiseAddOp, CwiseScalarMulOp, CwiseSubOp, Product};
use crate::core::scalar::Scalar;
pub use crate::core::storage::{DynamicStorage, FixedStorage, Storage};
use crate::core::xpr::MatrixXpr;
use std::ops::{Add, Mul, Sub};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// The main Matrix structure for eigen-rs.
/// Generic over Scalar type T and Storage type S.
#[derive(Debug)]
pub struct Matrix<T, S: Storage<T>> {
    storage: S,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, S: Storage<T> + Clone> Clone for Matrix<T, S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, S: Storage<T> + Copy> Copy for Matrix<T, S> {}

impl<T: PartialEq + Scalar, S1: Storage<T>, S2: Storage<T>> PartialEq<Matrix<T, S2>>
    for Matrix<T, S1>
{
    fn eq(&self, other: &Matrix<T, S2>) -> bool {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            return false;
        }
        for j in 0..self.cols() {
            for i in 0..self.rows() {
                if self.get(i, j) != other.get(i, j) {
                    return false;
                }
            }
        }
        true
    }
}

impl<T, S: Storage<T>> crate::core::cuda::CudaDispatcher<T> for Matrix<T, S>
where
    T: Scalar,
    S: Storage<T>,
{
    fn as_cuda_storage(&self) -> Option<&crate::core::storage::cuda::CudaStorage<T>> {
        self.storage().as_cuda_storage()
    }
}

impl<T, S: Storage<T>> MatrixXpr<T> for Matrix<T, S>
where
    T: Scalar,
{
    fn rows(&self) -> usize {
        self.storage.rows()
    }
    fn cols(&self) -> usize {
        self.storage.cols()
    }

    fn eval(&self, row: usize, col: usize) -> T {
        // Column-major: index = col * rows + row
        self.storage.data()[col * self.rows() + row]
    }

    fn packet_eval<P: crate::core::arch::Packet<T>>(&self, row: usize, col: usize) -> P
    where
        T: Scalar,
    {
        unsafe { P::load(self.storage.get_ptr(row, col)) }
    }
}

impl<T: Scalar, S: Storage<T>> Matrix<T, S> {
    pub fn rows(&self) -> usize {
        self.storage.rows()
    }
    pub fn cols(&self) -> usize {
        self.storage.cols()
    }
    pub fn size(&self) -> usize {
        self.rows() * self.cols()
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows() && col < self.cols() {
            let rows = self.rows();
            let index = col * rows + row;
            Some(&self.storage.data()[index])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows() && col < self.cols() {
            let rows = self.rows();
            let index = col * rows + row;
            Some(&mut self.storage.data_mut()[index])
        } else {
            None
        }
    }

    /// Computes the Partial Pivoting LU decomposition of the matrix.
    pub fn partial_piv_lu(&self) -> Result<PartialPivLU<T, S>, String> {
        PartialPivLU::new(self)
    }

    /// Computes the Householder QR decomposition of the matrix.
    pub fn householder_qr(
        &self,
    ) -> Result<crate::core::decompositions::HouseholderQR<T, S>, String> {
        crate::core::decompositions::HouseholderQR::new(self)
    }

    /// Computes the LLT decomposition of the matrix.
    pub fn llt(&self) -> Result<crate::core::decompositions::LLT<T, S>, String>
    where
        T: Scalar + 'static,
        S: Storage<T> + 'static,
    {
        crate::core::decompositions::LLT::new(self)
    }

    /// Computes the LDLT decomposition of the matrix.
    pub fn ldlt(&self) -> Result<crate::core::decompositions::LDLT<T, S>, String>
    where
        T: Scalar + 'static,
        S: Storage<T> + 'static,
    {
        crate::core::decompositions::LDLT::new(self)
    }

    /// Computes the Jacobi SVD decomposition of the matrix.
    pub fn jacobi_svd(&self) -> Result<crate::core::decompositions::JacobiSVD<T, S>, String>
    where
        T: Scalar + 'static,
        S: Storage<T> + 'static,
    {
        crate::core::decompositions::JacobiSVD::new(self)
    }

    /// Computes the inverse of the matrix.
    pub fn inverse(&self) -> Result<Matrix<T, DynamicStorage<T>>, String> {
        let rows = self.rows();
        let cols = self.cols();
        if rows != cols {
            return Err("Inverse only defined for square matrices".to_string());
        }

        match rows {
            2 => {
                let det = self.determinant();
                if det == T::from_usize(0) {
                    return Err("Matrix is singular".to_string());
                }
                let inv_det = T::from_usize(1) / det;
                let mut res = Matrix::<T, DynamicStorage<T>>::new_dynamic(2, 2)?;
                *res.get_mut(0, 0).unwrap() = *self.get(1, 1).unwrap() * inv_det;
                *res.get_mut(0, 1).unwrap() =
                    (T::from_usize(0) - *self.get(0, 1).unwrap()) * inv_det;
                *res.get_mut(1, 0).unwrap() =
                    (T::from_usize(0) - *self.get(1, 0).unwrap()) * inv_det;
                *res.get_mut(1, 1).unwrap() = *self.get(0, 0).unwrap() * inv_det;
                Ok(res)
            }
            _ => {
                // For N > 2 (or N != 2), use LU solve against identity
                let lu = self.partial_piv_lu()?;
                let mut ident = Matrix::<T, DynamicStorage<T>>::new_dynamic(rows, cols)?;
                for i in 0..rows {
                    for j in 0..cols {
                        *ident.get_mut(i, j).unwrap() = if i == j {
                            T::from_usize(1)
                        } else {
                            T::from_usize(0)
                        };
                    }
                }
                lu.solve(&ident)
            }
        }
    }

    /// Assigns from another expression (evaluates the expression).
    pub fn assign<X: MatrixXpr<T>>(&mut self, xpr: &X) -> Result<(), String>
    where
        T: Scalar + 'static,
    {
        if self.rows() != xpr.rows() || self.cols() != xpr.cols() {
            return Err("Dimension mismatch in assignment".to_string());
        }

        // Try CUDA acceleration
        if xpr.try_assign_cuda(self)? {
            return Ok(());
        }

        #[cfg(feature = "parallel")]
        {
            if self.size() > 10000 {
                return self.par_assign(xpr);
            }
        }

        let rows = self.rows();
        let cols = self.cols();

        for c in 0..cols {
            for r in 0..rows {
                let val = xpr.eval(r, c);
                if let Some(mut_ref) = self.get_mut(r, c) {
                    *mut_ref = val;
                }
            }
        }
        Ok(())
    }

    /// Parallel version of assign.
    #[cfg(feature = "parallel")]
    pub fn par_assign<X: MatrixXpr<T> + Sync>(&mut self, xpr: &X) -> Result<(), String>
    where
        T: Scalar + Send + Sync + 'static,
    {
        if self.rows() != xpr.rows() || self.cols() != xpr.cols() {
            return Err("Dimension mismatch in parallel assignment".to_string());
        }

        let rows = self.rows();
        let cols = self.cols();

        // We need to get a raw pointer to the storage to safely mutate in parallel.
        // This is safe because each thread will access a distinct column.
        let data_ptr = self.storage_mut().data_mut().as_mut_ptr() as usize;

        (0..cols).into_par_iter().for_each(move |c| {
            let base_idx = c * rows;
            unsafe {
                let ptr = data_ptr as *mut T;
                for r in 0..rows {
                    let val = xpr.eval(r, c);
                    *ptr.add(base_idx + r) = val;
                }
            }
        });

        Ok(())
    }

    /// Specialized addition for CUDA.
    #[cfg(feature = "cuda")]
    pub fn assign_add_cuda<S1, S2>(
        &mut self,
        lhs: &Matrix<T, S1>,
        rhs: &Matrix<T, S2>,
    ) -> Result<(), String>
    where
        T: Scalar + 'static,
        S1: Storage<T>,
        S2: Storage<T>,
    {
        use crate::core::cuda::get_cuda_context;
        use crate::core::storage::CudaStorage;

        if std::any::TypeId::of::<S>() != std::any::TypeId::of::<CudaStorage<T>>()
            || std::any::TypeId::of::<S1>() != std::any::TypeId::of::<CudaStorage<T>>()
            || std::any::TypeId::of::<S2>() != std::any::TypeId::of::<CudaStorage<T>>()
        {
            return Err("CUDA addition requires all matrices to have CudaStorage".to_string());
        }

        // We know they are CudaStorage now.
        let ctx = get_cuda_context()?;
        let n = self.size() as i32;

        // Safely get pointers since we checked TypeId
        // This is still unsafe because get_ptr is on Storage trait but we are casting conceptually.
        let a_ptr = lhs.storage().get_ptr(0, 0);
        let b_ptr = rhs.storage().get_ptr(0, 0);
        let c_ptr = self.storage_mut().get_ptr(0, 0) as *mut T;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            unsafe {
                ctx.launch_add_f32(
                    a_ptr as *const f32,
                    b_ptr as *const f32,
                    c_ptr as *mut f32,
                    n,
                )?;
            }
            Ok(())
        } else {
            Err("CUDA addition only implemented for f32 for now".to_string())
        }
    }

    /// Specialized subtraction for CUDA.
    #[cfg(feature = "cuda")]
    pub fn assign_sub_cuda<S1, S2>(
        &mut self,
        lhs: &Matrix<T, S1>,
        rhs: &Matrix<T, S2>,
    ) -> Result<(), String>
    where
        T: Scalar + 'static,
        S1: Storage<T>,
        S2: Storage<T>,
    {
        use crate::core::cuda::get_cuda_context;
        use crate::core::storage::cuda::CudaStorage;

        if std::any::TypeId::of::<S>() != std::any::TypeId::of::<CudaStorage<T>>()
            || std::any::TypeId::of::<S1>() != std::any::TypeId::of::<CudaStorage<T>>()
            || std::any::TypeId::of::<S2>() != std::any::TypeId::of::<CudaStorage<T>>()
        {
            return Err("CUDA subtraction requires all matrices to have CudaStorage".to_string());
        }

        let ctx = get_cuda_context()?;
        let n = self.size() as i32;
        let a_ptr = lhs.storage().get_ptr(0, 0);
        let b_ptr = rhs.storage().get_ptr(0, 0);
        let c_ptr = self.storage_mut().get_ptr(0, 0) as *mut T;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            unsafe {
                ctx.launch_sub_f32(
                    a_ptr as *const f32,
                    b_ptr as *const f32,
                    c_ptr as *mut f32,
                    n,
                )?;
            }
            Ok(())
        } else {
            Err("CUDA subtraction only implemented for f32".to_string())
        }
    }

    /// Specialized scalar multiplication for CUDA.
    #[cfg(feature = "cuda")]
    pub fn assign_scalar_mul_cuda<S1>(
        &mut self,
        lhs: &Matrix<T, S1>,
        scalar: T,
    ) -> Result<(), String>
    where
        T: Scalar + 'static,
        S1: Storage<T>,
    {
        use crate::core::cuda::get_cuda_context;
        use crate::core::storage::cuda::CudaStorage;

        if std::any::TypeId::of::<S>() != std::any::TypeId::of::<CudaStorage<T>>()
            || std::any::TypeId::of::<S1>() != std::any::TypeId::of::<CudaStorage<T>>()
        {
            return Err("CUDA scalar mul requires all matrices to have CudaStorage".to_string());
        }

        let ctx = get_cuda_context()?;
        let n = self.size() as i32;
        let a_ptr = lhs.storage().get_ptr(0, 0);
        let c_ptr = self.storage_mut().get_ptr(0, 0) as *mut T;

        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // Safety: We checked that T is f32
            let s_f32: f32 = unsafe { *(&scalar as *const T as *const f32) };
            unsafe {
                ctx.launch_scalar_mul_f32(a_ptr as *const f32, s_f32, c_ptr as *mut f32, n)?;
            }
            Ok(())
        } else {
            Err("CUDA scalar mul only implemented for f32".to_string())
        }
    }

    pub fn transpose(&self) -> crate::core::ops::TransposeOp<'_, T, Self>
    where
        Self: Sized + 'static,
    {
        crate::core::ops::TransposeOp::new(self)
    }

    pub fn block(
        &self,
        start_row: usize,
        start_col: usize,
        rows: usize,
        cols: usize,
    ) -> crate::core::ops::BlockOp<'_, T, Self>
    where
        Self: Sized + 'static,
    {
        crate::core::ops::BlockOp::new(self, start_row, start_col, rows, cols)
            .expect("Block out of bounds")
    }

    pub fn row(&self, i: usize) -> crate::core::ops::BlockOp<'_, T, Self>
    where
        Self: Sized + 'static,
    {
        self.block(i, 0, 1, self.cols())
    }

    pub fn col(&self, j: usize) -> crate::core::ops::BlockOp<'_, T, Self>
    where
        Self: Sized + 'static,
    {
        self.block(0, j, self.rows(), 1)
    }

    /// Specialized assignment for matrix products.
    pub fn assign_product<'a, L, R>(
        &mut self,
        product: &crate::core::ops::Product<'a, T, L, R>,
    ) -> Result<(), String>
    where
        T: Scalar,
        L: crate::core::xpr::MatrixXpr<T>,
        R: crate::core::xpr::MatrixXpr<T>,
    {
        crate::core::ops::gemm::gemm_cm_unoptimized_xpr(product.lhs(), product.rhs(), self)
    }

    /// Computes the eigenvalues and eigenvectors of a self-adjoint matrix.
    pub fn self_adjoint_eigen_solver(
        &self,
        compute_eigenvectors: bool,
    ) -> Result<crate::core::decompositions::SelfAdjointEigenSolver<T, S>, String>
    where
        Self: Sized + 'static,
    {
        crate::core::decompositions::SelfAdjointEigenSolver::new(self, compute_eigenvectors)
    }

    /// Returns the eigenvalues of a self-adjoint matrix.
    pub fn eigenvalues(&self) -> Result<Matrix<T, DynamicStorage<T>>, String>
    where
        Self: Sized + 'static,
    {
        let solver = crate::core::decompositions::SelfAdjointEigenSolver::new(self, false)?;
        Ok(solver.eigenvalues().clone())
    }

    /// LU decomposition using LAPACK.
    #[cfg(feature = "lapack")]
    pub fn lu_lapack(&self) -> Result<crate::core::decompositions::lapack::LapackLU<T>, String>
    where
        T: Scalar + 'static,
    {
        let mut lu = crate::core::decompositions::lapack::LapackLU::new(self.rows(), self.cols());
        lu.compute(self)?;
        Ok(lu)
    }

    /// QR decomposition using LAPACK.
    #[cfg(feature = "lapack")]
    pub fn qr_lapack(&self) -> Result<crate::core::decompositions::lapack::LapackQR<T>, String>
    where
        T: Scalar + 'static,
    {
        let mut qr = crate::core::decompositions::lapack::LapackQR::new(self.rows(), self.cols());
        qr.compute(self)?;
        Ok(qr)
    }

    /// Computes the dot product of this vector with another vector.
    pub fn dot<S2: Storage<T>>(&self, other: &Matrix<T, S2>) -> T {
        assert_eq!(
            self.size(),
            other.size(),
            "Dot product requires vectors of equal size"
        );
        let mut sum = T::default();
        // Simple dot product for now
        for i in 0..self.size() {
            sum += self.storage.data()[i] * other.storage().data()[i];
        }
        sum
    }

    /// Computes the squared norm of the vector.
    pub fn squared_norm(&self) -> T {
        self.dot(self)
    }

    /// Computes the norm of the vector.
    pub fn norm(&self) -> T {
        self.squared_norm().sqrt()
    }

    /// Normalizes the vector in-place.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n != T::default() {
            let inv_n = T::from_usize(1) / n;
            for i in 0..self.size() {
                self.storage.data_mut()[i] *= inv_n;
            }
        }
    }

    /// Returns a normalized copy of the vector.
    pub fn normalized(&self) -> Matrix<T, DynamicStorage<T>> {
        let mut res =
            Matrix::<T, DynamicStorage<T>>::new_dynamic(self.rows(), self.cols()).unwrap();
        // Copy data
        for i in 0..self.size() {
            *res.storage.data_mut().get_mut(i).unwrap() = self.storage.data()[i];
        }
        res.normalize();
        res
    }

    /// Scales all elements by a scalar factor in-place.
    pub fn scale(&mut self, factor: T) {
        let size = self.size();
        let data = self.storage_mut().data_mut();
        for i in 0..size {
            data[i] *= factor;
        }
    }

    pub fn set_zero(&mut self) {
        let size = self.size();
        let data = self.storage_mut().data_mut();
        for i in 0..size {
            data[i] = T::default();
        }
    }

    pub fn set_constant(&mut self, val: T) {
        let size = self.size();
        let data = self.storage_mut().data_mut();
        for i in 0..size {
            data[i] = val;
        }
    }

    pub fn set_identity(&mut self) {
        self.set_zero();
        let rows = self.rows();
        let cols = self.cols();
        let n = std::cmp::min(rows, cols);
        for i in 0..n {
            *self.get_mut(i, i).unwrap() = T::from_f64(1.0);
        }
    }
}

/// Specialization for 3D vectors to support cross product.
impl<T: Scalar> Matrix<T, FixedStorage<T, 3, 1, 3>> {
    pub fn cross<S2: Storage<T>>(&self, other: &Matrix<T, S2>) -> Self {
        assert_eq!(other.rows(), 3);
        assert_eq!(other.cols(), 1);

        let a1 = *self.get(0, 0).unwrap();
        let a2 = *self.get(1, 0).unwrap();
        let a3 = *self.get(2, 0).unwrap();

        let b1 = *other.get(0, 0).unwrap();
        let b2 = *other.get(1, 0).unwrap();
        let b3 = *other.get(2, 0).unwrap();

        let mut res = Self::new_fixed();
        *res.get_mut(0, 0).unwrap() = a2 * b3 - a3 * b2;
        *res.get_mut(1, 0).unwrap() = a3 * b1 - a1 * b3;
        *res.get_mut(2, 0).unwrap() = a1 * b2 - a2 * b1;

        res
    }
}

impl<'a, T, S1, S2> Add<&'a Matrix<T, S2>> for &'a Matrix<T, S1>
where
    T: Scalar + 'static,
    S1: Storage<T>,
    S2: Storage<T>,
{
    type Output = CwiseAddOp<'a, T, Matrix<T, S1>, Matrix<T, S2>>;

    fn add(self, rhs: &'a Matrix<T, S2>) -> Self::Output {
        CwiseAddOp::new(self, rhs).expect("Dimension mismatch in Matrix addition")
    }
}

impl<'a, T, S1, S2> Sub<&'a Matrix<T, S2>> for &'a Matrix<T, S1>
where
    T: Scalar + 'static,
    S1: Storage<T>,
    S2: Storage<T>,
{
    type Output = CwiseSubOp<'a, T, Matrix<T, S1>, Matrix<T, S2>>;

    fn sub(self, rhs: &'a Matrix<T, S2>) -> Self::Output {
        CwiseSubOp::new(self, rhs).expect("Dimension mismatch in Matrix subtraction")
    }
}

impl<'a, T, S> Mul<T> for &'a Matrix<T, S>
where
    T: Scalar + 'static,
    S: Storage<T>,
{
    type Output = CwiseScalarMulOp<'a, T, Matrix<T, S>>;

    fn mul(self, rhs: T) -> Self::Output {
        CwiseScalarMulOp::new(self, rhs)
    }
}

impl<'a, T, S1, S2> Mul<&'a Matrix<T, S2>> for &'a Matrix<T, S1>
where
    T: Scalar + 'static,
    S1: Storage<T>,
    S2: Storage<T>,
{
    type Output = Product<'a, T, Matrix<T, S1>, Matrix<T, S2>>;

    fn mul(self, rhs: &'a Matrix<T, S2>) -> Self::Output {
        Product::new(self, rhs).expect("Dimension mismatch in Matrix multiplication")
    }
}

// Support &BlockOp * &Matrix
impl<'a, T, X, S> Mul<&'a Matrix<T, S>> for &'a crate::core::ops::BlockOp<'a, T, X>
where
    T: Scalar + 'static,
    X: crate::core::xpr::MatrixXpr<T> + 'static,
    S: Storage<T>,
{
    type Output = Product<'a, T, crate::core::ops::BlockOp<'a, T, X>, Matrix<T, S>>;

    fn mul(self, rhs: &'a Matrix<T, S>) -> Self::Output {
        Product::new(self, rhs).expect("Dimension mismatch in Block * Matrix multiplication")
    }
}

// Support &TransposeOp * &Matrix
impl<'a, T, X, S> Mul<&'a Matrix<T, S>> for &'a crate::core::ops::TransposeOp<'a, T, X>
where
    T: Scalar + 'static,
    X: crate::core::xpr::MatrixXpr<T> + 'static,
    S: Storage<T>,
{
    type Output = Product<'a, T, crate::core::ops::TransposeOp<'a, T, X>, Matrix<T, S>>;

    fn mul(self, rhs: &'a Matrix<T, S>) -> Self::Output {
        Product::new(self, rhs).expect("Dimension mismatch in Transpose * Matrix multiplication")
    }
}

// Support &Matrix * &BlockOp
impl<'a, T, S, X> Mul<&'a crate::core::ops::BlockOp<'a, T, X>> for &'a Matrix<T, S>
where
    T: Scalar + 'static,
    S: Storage<T>,
    X: crate::core::xpr::MatrixXpr<T> + 'static,
{
    type Output = Product<'a, T, Matrix<T, S>, crate::core::ops::BlockOp<'a, T, X>>;

    fn mul(self, rhs: &'a crate::core::ops::BlockOp<'a, T, X>) -> Self::Output {
        Product::new(self, rhs).expect("Dimension mismatch in Matrix * Block multiplication")
    }
}

// Support &Matrix * &TransposeOp
impl<'a, T, S, X> Mul<&'a crate::core::ops::TransposeOp<'a, T, X>> for &'a Matrix<T, S>
where
    T: Scalar + 'static,
    S: Storage<T>,
    X: crate::core::xpr::MatrixXpr<T> + 'static,
{
    type Output = Product<'a, T, Matrix<T, S>, crate::core::ops::TransposeOp<'a, T, X>>;

    fn mul(self, rhs: &'a crate::core::ops::TransposeOp<'a, T, X>) -> Self::Output {
        Product::new(self, rhs).expect("Dimension mismatch in Matrix * Transpose multiplication")
    }
}

/// Trigonometric and Exponential functions
impl<'a, T, S> Matrix<T, S>
where
    T: Scalar + 'static,
    S: Storage<T>,
{
    pub fn sin(&self) -> crate::core::ops::CwiseUnaryOp<'_, T, Self, crate::core::ops::ScalarSin> {
        crate::core::ops::CwiseUnaryOp::new(self, crate::core::ops::ScalarSin)
    }

    pub fn cos(&self) -> crate::core::ops::CwiseUnaryOp<'_, T, Self, crate::core::ops::ScalarCos> {
        crate::core::ops::CwiseUnaryOp::new(self, crate::core::ops::ScalarCos)
    }

    pub fn exp(&self) -> crate::core::ops::CwiseUnaryOp<'_, T, Self, crate::core::ops::ScalarExp> {
        crate::core::ops::CwiseUnaryOp::new(self, crate::core::ops::ScalarExp)
    }

    pub fn ln(&self) -> crate::core::ops::CwiseUnaryOp<'_, T, Self, crate::core::ops::ScalarLog> {
        crate::core::ops::CwiseUnaryOp::new(self, crate::core::ops::ScalarLog)
    }
}

/// Specialization for Fixed-size matrices.
impl<T: Scalar, const R: usize, const C: usize, const S: usize>
    Matrix<T, FixedStorage<T, R, C, S>>
{
    pub fn new_fixed() -> Self {
        Self {
            storage: FixedStorage::new(R, C).unwrap(),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn from_array(data: [T; S]) -> Self {
        Self {
            storage: FixedStorage::from_array(data),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn zeros() -> Self {
        Self::new_fixed()
    }
}

/// Specialization for Dynamic-size matrices.
impl<T: Scalar> Matrix<T, DynamicStorage<T>> {
    pub fn new_dynamic(rows: usize, cols: usize) -> Result<Self, String> {
        Ok(Self {
            storage: DynamicStorage::new(rows, cols)?,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn from_vec(rows: usize, cols: usize, vec: Vec<T>) -> Result<Self, String> {
        Ok(Self {
            storage: DynamicStorage::from_vec(rows, cols, vec)?,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn identity(rows: usize, cols: usize) -> Self {
        let mut m = Self::new_dynamic(rows, cols).expect("Failed to allocate identity matrix");
        for i in 0..rows {
            for j in 0..cols {
                let val = if i == j {
                    T::from_usize(1)
                } else {
                    T::from_usize(0)
                };
                *m.get_mut(i, j).unwrap() = val;
            }
        }
        m
    }
}

/// Specialization for CUDA matrices.
#[cfg(feature = "cuda")]
impl<T: Scalar> Matrix<T, crate::core::storage::cuda::CudaStorage<T>> {
    pub fn new_dynamic(rows: usize, cols: usize) -> Result<Self, String> {
        Ok(Self {
            storage: crate::core::storage::cuda::CudaStorage::new(rows, cols)?,
            _phantom: std::marker::PhantomData,
        })
    }
}

// Convenience Type Aliases
pub type Matrix2<T> = Matrix<T, FixedStorage<T, 2, 2, 4>>;
pub type Matrix3<T> = Matrix<T, FixedStorage<T, 3, 3, 9>>;
pub type Matrix4<T> = Matrix<T, FixedStorage<T, 4, 4, 16>>;
pub type Vector3<T> = Matrix<T, FixedStorage<T, 3, 1, 3>>;
pub type Vector4<T> = Matrix<T, FixedStorage<T, 4, 1, 4>>;
pub type MatrixX<T> = Matrix<T, DynamicStorage<T>>;
pub type VectorX<T> = Matrix<T, DynamicStorage<T>>; // dynamic vector
pub type Map<'a, T> = Matrix<T, crate::core::storage::MapStorage<'a, T>>;

impl<'a, T: Sync> Map<'a, T> {
    pub unsafe fn new(ptr: *mut T, rows: usize, cols: usize) -> Self {
        Self {
            storage: unsafe { crate::core::storage::MapStorage::new(ptr, rows, cols) },
            _phantom: std::marker::PhantomData,
        }
    }

    pub unsafe fn new_with_stride(ptr: *mut T, rows: usize, cols: usize, stride: usize) -> Self {
        Self {
            storage: unsafe {
                crate::core::storage::MapStorage::new_with_stride(ptr, rows, cols, stride)
            },
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_fixed_creation() {
        let mut m = Matrix2::<f32>::new_fixed();
        *m.get_mut(0, 0).unwrap() = 1.0;
        *m.get_mut(1, 1).unwrap() = 2.0;
        assert_eq!(*m.get(0, 0).unwrap(), 1.0);
        assert_eq!(*m.get(1, 1).unwrap(), 2.0);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
    }

    #[test]
    fn test_matrix_dynamic_creation() {
        let mut m = MatrixX::<f32>::new_dynamic(3, 3).unwrap();
        *m.get_mut(0, 0).unwrap() = 1.0;
        *m.get_mut(2, 2).unwrap() = 5.0;
        assert_eq!(*m.get(0, 0).unwrap(), 1.0);
        assert_eq!(*m.get(2, 2).unwrap(), 5.0);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 3);
    }
}
