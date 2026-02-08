//! Expression templates for Tensors.

use crate::core::scalar::Scalar;
use crate::core::tensor::Tensor;

/// Trait for N-dimensional tensor expressions.
pub trait TensorXpr<T: Scalar, const RANK: usize> {
    /// Returns the dimensions of the tensor expression.
    fn dims(&self) -> [usize; RANK];
    
    /// Evaluates the expression at the given indices.
    fn eval(&self, indices: [usize; RANK]) -> T;

    /// Returns the total number of elements.
    fn size(&self) -> usize {
        self.dims().iter().product()
    }
}

// Implement TensorXpr for Tensor
impl<T: Scalar, const RANK: usize> TensorXpr<T, RANK> for Tensor<T, RANK> {
    fn dims(&self) -> [usize; RANK] {
        self.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        *self.get(indices).unwrap_or(&T::default())
    }
}

// Implement TensorXpr for &Tensor (Essential for lazy evaluation)
impl<T: Scalar, const RANK: usize> TensorXpr<T, RANK> for &Tensor<T, RANK> {
    fn dims(&self) -> [usize; RANK] {
        (*self).dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        *self.get(indices).unwrap_or(&T::default())
    }
}

/// Lazy coefficient-wise addition.
/// Holds L and R by value to allow moving temporary expressions.
pub struct CwiseTensorAddOp<T: Scalar, const RANK: usize, L, R> 
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    lhs: L,
    rhs: R,
    dims: [usize; RANK],
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const RANK: usize, L, R> CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    pub fn new(lhs: L, rhs: R) -> Result<Self, String> {
        if lhs.dims() != rhs.dims() {
            return Err("Dimension mismatch in Tensor addition".to_string());
        }
        Ok(Self {
            dims: lhs.dims(),
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T: Scalar, const RANK: usize, L, R> TensorXpr<T, RANK> for CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.dims
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.lhs.eval(indices) + self.rhs.eval(indices)
    }
}

/// Lazy coefficient-wise subtraction.
pub struct CwiseTensorSubOp<T: Scalar, const RANK: usize, L, R> 
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    lhs: L,
    rhs: R,
    dims: [usize; RANK],
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const RANK: usize, L, R> CwiseTensorSubOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    pub fn new(lhs: L, rhs: R) -> Result<Self, String> {
        if lhs.dims() != rhs.dims() {
            return Err("Dimension mismatch in Tensor subtraction".to_string());
        }
        Ok(Self {
            dims: lhs.dims(),
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T: Scalar, const RANK: usize, L, R> TensorXpr<T, RANK> for CwiseTensorSubOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.dims
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.lhs.eval(indices) - self.rhs.eval(indices)
    }
}

/// Lazy scalar multiplication (broadcasting).
pub struct CwiseTensorScalarMulOp<T: Scalar, const RANK: usize, X> 
where
    X: TensorXpr<T, RANK>,
{
    xpr: X,
    scalar: T,
}

impl<T: Scalar, const RANK: usize, X> CwiseTensorScalarMulOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    pub fn new(xpr: X, scalar: T) -> Self {
        Self { xpr, scalar }
    }
}

impl<T: Scalar, const RANK: usize, X> TensorXpr<T, RANK> for CwiseTensorScalarMulOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.xpr.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.xpr.eval(indices) * self.scalar
    }
}

/// Lazy scalar addition (broadcasting).
pub struct CwiseTensorScalarAddOp<T: Scalar, const RANK: usize, X> 
where
    X: TensorXpr<T, RANK>,
{
    xpr: X,
    scalar: T,
}

impl<T: Scalar, const RANK: usize, X> CwiseTensorScalarAddOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    pub fn new(xpr: X, scalar: T) -> Self {
        Self { xpr, scalar }
    }
}

impl<T: Scalar, const RANK: usize, X> TensorXpr<T, RANK> for CwiseTensorScalarAddOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.xpr.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.xpr.eval(indices) + self.scalar
    }
}

/// Lazy scalar subtraction (broadcasting).
pub struct CwiseTensorScalarSubOp<T: Scalar, const RANK: usize, X> 
where
    X: TensorXpr<T, RANK>,
{
    xpr: X,
    scalar: T,
}

impl<T: Scalar, const RANK: usize, X> CwiseTensorScalarSubOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    pub fn new(xpr: X, scalar: T) -> Self {
        Self { xpr, scalar }
    }
}

impl<T: Scalar, const RANK: usize, X> TensorXpr<T, RANK> for CwiseTensorScalarSubOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.xpr.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.xpr.eval(indices) - self.scalar
    }
}
