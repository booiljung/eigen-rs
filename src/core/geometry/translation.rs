use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::FixedStorage;

use std::ops::Mul;

/// Represents a translation transformation.
#[derive(Clone, Debug, PartialEq)]
pub struct Translation<T: Scalar, const RANK: usize> {
    pub vector: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
}

impl<T: Scalar<Real = T> + PartialOrd, const RANK: usize> Translation<T, RANK> {
    pub fn new(vector: Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> Self {
        Self { vector }
    }

    pub fn identity() -> Self {
        Self {
            vector: Matrix::zeros(),
        }
    }

    pub fn inverse(&self) -> Self {
        let mut inv = self.vector;
        for i in 0..RANK {
            *inv.get_mut(i, 0).unwrap() = -*self.vector.get(i, 0).unwrap();
        }
        Self { vector: inv }
    }

    pub fn vector(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.vector
    }
}

// Translation * Vector (Apply translation)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a, T: Scalar, const RANK: usize> Mul<&'a Matrix<T, FixedStorage<T, RANK, 1, RANK>>>
    for &'a Translation<T, RANK>
{
    type Output = Matrix<T, FixedStorage<T, RANK, 1, RANK>>;

    fn mul(self, rhs: &'a Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> Self::Output {
        // T * v = v + t
        let mut res = *rhs;
        for i in 0..RANK {
            *res.get_mut(i, 0).unwrap() += *self.vector.get(i, 0).unwrap();
        }
        res
    }
}

// Translation * Translation (Composition)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<'a, T: Scalar, const RANK: usize> Mul<&'a Translation<T, RANK>> for &'a Translation<T, RANK> {
    type Output = Translation<T, RANK>;

    fn mul(self, rhs: &'a Translation<T, RANK>) -> Self::Output {
        // T1 * T2 = Translation(v1 + v2)
        let mut res = self.vector;
        for i in 0..RANK {
            *res.get_mut(i, 0).unwrap() += *rhs.vector.get(i, 0).unwrap();
        }
        Translation { vector: res }
    }
}
