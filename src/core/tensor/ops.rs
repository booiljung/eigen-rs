use super::{
    xpr::{
        CwiseTensorAddOp, CwiseTensorScalarAddOp, CwiseTensorScalarMulOp, CwiseTensorScalarSubOp,
        CwiseTensorSubOp, TensorXpr,
    },
    Tensor,
};
use crate::core::scalar::Scalar;
use std::ops::{Add, Mul, Sub};

// --- Add ---

use crate::core::tensor::device::Device;

// &Tensor + &Tensor
impl<'a, 'b, T: Scalar, const RANK: usize, D: Device> Add<&'b Tensor<T, RANK, D>>
    for &'a Tensor<T, RANK, D>
{
    type Output = CwiseTensorAddOp<T, RANK, &'a Tensor<T, RANK, D>, &'b Tensor<T, RANK, D>>;
    fn add(self, rhs: &'b Tensor<T, RANK, D>) -> Self::Output {
        CwiseTensorAddOp::new(self, rhs).expect("Dimension mismatch")
    }
}

// &Tensor + T (Broadcasting)
impl<'a, T: Scalar, const RANK: usize, D: Device> Add<T> for &'a Tensor<T, RANK, D> {
    type Output = CwiseTensorScalarAddOp<T, RANK, &'a Tensor<T, RANK, D>>;
    fn add(self, rhs: T) -> Self::Output {
        CwiseTensorScalarAddOp::new(self, rhs)
    }
}

// (Add Xpr) + &Tensor
impl<'a, T: Scalar, const RANK: usize, L, R, D: Device> Add<&'a Tensor<T, RANK, D>>
    for CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    type Output =
        CwiseTensorAddOp<T, RANK, CwiseTensorAddOp<T, RANK, L, R>, &'a Tensor<T, RANK, D>>;
    fn add(self, rhs: &'a Tensor<T, RANK, D>) -> Self::Output {
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
impl<'a, 'b, T: Scalar, const RANK: usize, D: Device> Sub<&'b Tensor<T, RANK, D>>
    for &'a Tensor<T, RANK, D>
{
    type Output = CwiseTensorSubOp<T, RANK, &'a Tensor<T, RANK, D>, &'b Tensor<T, RANK, D>>;
    fn sub(self, rhs: &'b Tensor<T, RANK, D>) -> Self::Output {
        CwiseTensorSubOp::new(self, rhs).expect("Dimension mismatch")
    }
}

// &Tensor - T (Broadcasting)
impl<'a, T: Scalar, const RANK: usize, D: Device> Sub<T> for &'a Tensor<T, RANK, D> {
    type Output = CwiseTensorScalarSubOp<T, RANK, &'a Tensor<T, RANK, D>>;
    fn sub(self, rhs: T) -> Self::Output {
        CwiseTensorScalarSubOp::new(self, rhs)
    }
}

// (Sub Xpr) - &Tensor
impl<'a, T: Scalar, const RANK: usize, L, R, D: Device> Sub<&'a Tensor<T, RANK, D>>
    for CwiseTensorSubOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    type Output =
        CwiseTensorSubOp<T, RANK, CwiseTensorSubOp<T, RANK, L, R>, &'a Tensor<T, RANK, D>>;
    fn sub(self, rhs: &'a Tensor<T, RANK, D>) -> Self::Output {
        CwiseTensorSubOp::new(self, rhs).expect("Dimension mismatch")
    }
}

// --- Mul (Scalar) ---

// &Tensor * T
impl<'a, T: Scalar, const RANK: usize, D: Device> Mul<T> for &'a Tensor<T, RANK, D> {
    type Output = CwiseTensorScalarMulOp<T, RANK, &'a Tensor<T, RANK, D>>;
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
