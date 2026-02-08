use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::FixedStorage;

use std::ops::Mul;

/// Represents a non-uniform scaling transformation.
#[derive(Clone, Debug, PartialEq)]
pub struct Scaling<T: Scalar, const RANK: usize> {
    pub coeffs: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
}

impl<T: Scalar, const RANK: usize> Scaling<T, RANK> {
    pub fn new(coeffs: Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> Self {
        Self { coeffs }
    }

    pub fn uniform(s: T) -> Self {
        let mut coeffs = Matrix::zeros();
        for i in 0..RANK {
            *coeffs.get_mut(i, 0).unwrap() = s;
        }
        Self { coeffs }
    }

    pub fn identity() -> Self {
        Self::uniform(T::from_f64(1.0))
    }

    pub fn inverse(&self) -> Self {
        let mut inv = self.coeffs;
        for i in 0..RANK {
            let val = *self.coeffs.get(i, 0).unwrap();
            *inv.get_mut(i, 0).unwrap() = T::from_f64(1.0) / val;
        }
        Self { coeffs: inv }
    }
}

// Scaling * Vector (Apply scaling)
impl<'a, T: Scalar, const RANK: usize> Mul<&'a Matrix<T, FixedStorage<T, RANK, 1, RANK>>>
    for &'a Scaling<T, RANK>
{
    type Output = Matrix<T, FixedStorage<T, RANK, 1, RANK>>;

    fn mul(self, rhs: &'a Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> Self::Output {
        let mut res = *rhs;
        for i in 0..RANK {
            *res.get_mut(i, 0).unwrap() *= *self.coeffs.get(i, 0).unwrap();
        }
        res
    }
}

// Scaling * Scaling (Composition)
impl<'a, T: Scalar, const RANK: usize> Mul<&'a Scaling<T, RANK>> for &'a Scaling<T, RANK> {
    type Output = Scaling<T, RANK>;

    fn mul(self, rhs: &'a Scaling<T, RANK>) -> Self::Output {
        let mut res = self.coeffs;
        for i in 0..RANK {
            *res.get_mut(i, 0).unwrap() *= *rhs.coeffs.get(i, 0).unwrap();
        }
        Scaling { coeffs: res }
    }
}
