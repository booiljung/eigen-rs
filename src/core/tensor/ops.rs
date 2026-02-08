use crate::core::scalar::Scalar;
use crate::core::tensor::xpr::{
    CwiseTensorAddOp, CwiseTensorScalarAddOp, CwiseTensorScalarMulOp, CwiseTensorScalarSubOp,
    CwiseTensorSubOp, TensorXpr,
};
use crate::core::tensor::Tensor;
use std::ops::{Add, Mul, Sub};

// --- Add ---

// &Tensor + &Tensor
impl<'a, 'b, T: Scalar, const RANK: usize> Add<&'b Tensor<T, RANK>> for &'a Tensor<T, RANK> {
    type Output = CwiseTensorAddOp<T, RANK, &'a Tensor<T, RANK>, &'b Tensor<T, RANK>>;
    fn add(self, rhs: &'b Tensor<T, RANK>) -> Self::Output {
        CwiseTensorAddOp::new(self, rhs).expect("Dimension mismatch")
    }
}

// &Tensor + T (Broadcasting)
impl<'a, T: Scalar, const RANK: usize> Add<T> for &'a Tensor<T, RANK> {
    type Output = CwiseTensorScalarAddOp<T, RANK, &'a Tensor<T, RANK>>;
    fn add(self, rhs: T) -> Self::Output {
        CwiseTensorScalarAddOp::new(self, rhs)
    }
}

// (Add Xpr) + &Tensor
impl<'a, T: Scalar, const RANK: usize, L, R> Add<&'a Tensor<T, RANK>>
    for CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    type Output = CwiseTensorAddOp<T, RANK, CwiseTensorAddOp<T, RANK, L, R>, &'a Tensor<T, RANK>>;
    fn add(self, rhs: &'a Tensor<T, RANK>) -> Self::Output {
        CwiseTensorAddOp::new(self, rhs).expect("Dimension mismatch")
    }
}

// (Add Xpr) + T (Broadcasting)
impl<T: Scalar, const RANK: usize, L, R> Add<T> for CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    type Output = CwiseTensorScalarAddOp<T, RANK, CwiseTensorAddOp<T, RANK, L, R>>;
    fn add(self, rhs: T) -> Self::Output {
        CwiseTensorScalarAddOp::new(self, rhs)
    }
}

// --- Sub ---

// &Tensor - &Tensor
impl<'a, 'b, T: Scalar, const RANK: usize> Sub<&'b Tensor<T, RANK>> for &'a Tensor<T, RANK> {
    type Output = CwiseTensorSubOp<T, RANK, &'a Tensor<T, RANK>, &'b Tensor<T, RANK>>;
    fn sub(self, rhs: &'b Tensor<T, RANK>) -> Self::Output {
        CwiseTensorSubOp::new(self, rhs).expect("Dimension mismatch")
    }
}

// &Tensor - T (Broadcasting)
impl<'a, T: Scalar, const RANK: usize> Sub<T> for &'a Tensor<T, RANK> {
    type Output = CwiseTensorScalarSubOp<T, RANK, &'a Tensor<T, RANK>>;
    fn sub(self, rhs: T) -> Self::Output {
        CwiseTensorScalarSubOp::new(self, rhs)
    }
}

// (Sub Xpr) - &Tensor
impl<'a, T: Scalar, const RANK: usize, L, R> Sub<&'a Tensor<T, RANK>>
    for CwiseTensorSubOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    type Output = CwiseTensorSubOp<T, RANK, CwiseTensorSubOp<T, RANK, L, R>, &'a Tensor<T, RANK>>;
    fn sub(self, rhs: &'a Tensor<T, RANK>) -> Self::Output {
        CwiseTensorSubOp::new(self, rhs).expect("Dimension mismatch")
    }
}

// --- Mul (Scalar) ---

// &Tensor * T
impl<'a, T: Scalar, const RANK: usize> Mul<T> for &'a Tensor<T, RANK> {
    type Output = CwiseTensorScalarMulOp<T, RANK, &'a Tensor<T, RANK>>;
    fn mul(self, rhs: T) -> Self::Output {
        CwiseTensorScalarMulOp::new(self, rhs)
    }
}

// (Add Xpr) * T
impl<T: Scalar, const RANK: usize, L, R> Mul<T> for CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    type Output = CwiseTensorScalarMulOp<T, RANK, CwiseTensorAddOp<T, RANK, L, R>>;
    fn mul(self, rhs: T) -> Self::Output {
        CwiseTensorScalarMulOp::new(self, rhs)
    }
}

// (Sub Xpr) * T
impl<T: Scalar, const RANK: usize, L, R> Mul<T> for CwiseTensorSubOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    type Output = CwiseTensorScalarMulOp<T, RANK, CwiseTensorSubOp<T, RANK, L, R>>;
    fn mul(self, rhs: T) -> Self::Output {
        CwiseTensorScalarMulOp::new(self, rhs)
    }
}
