use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::FixedStorage;

/// A line defined by an origin and a direction vector.
#[derive(Clone, Debug, PartialEq)]
pub struct ParametrizedLine<T: Scalar, const RANK: usize> {
    origin: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    direction: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
}

impl<T: Scalar, const RANK: usize> ParametrizedLine<T, RANK> {
    pub fn new(
        origin: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
        direction: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    ) -> Self {
        // Typically direction is normalized, but not strictly required for point_at.
        // For distance, it should be normalized.
        Self { origin, direction }
    }

    pub fn point_at(&self, t: T) -> Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        let mut p = self.origin;
        for i in 0..RANK {
            *p.get_mut(i, 0).unwrap() += *self.direction.get(i, 0).unwrap() * t;
        }
        p
    }

    // Squared distance
    pub fn squared_distance(&self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> T {
        let mut diff = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();
        for i in 0..RANK {
            *diff.get_mut(i, 0).unwrap() =
                *point.get(i, 0).unwrap() - *self.origin.get(i, 0).unwrap();
        }

        // t = direction . diff / direction . direction
        let mut num = T::default();
        let mut den = T::default();

        for i in 0..RANK {
            let d_i = *self.direction.get(i, 0).unwrap();
            num += d_i * *diff.get(i, 0).unwrap();
            den += d_i * d_i;
        }

        let t = num / den;
        let proj_point = self.point_at(t);

        let mut dist_sq = T::default();
        for i in 0..RANK {
            let delta = *point.get(i, 0).unwrap() - *proj_point.get(i, 0).unwrap();
            dist_sq += delta * delta;
        }
        dist_sq
    }

    pub fn distance(&self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> T {
        self.squared_distance(point).sqrt()
    }

    pub fn projection(
        &self,
        point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    ) -> Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        let mut diff = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();
        for i in 0..RANK {
            *diff.get_mut(i, 0).unwrap() =
                *point.get(i, 0).unwrap() - *self.origin.get(i, 0).unwrap();
        }

        let mut num = T::default();
        let mut den = T::default();

        for i in 0..RANK {
            let d_i = *self.direction.get(i, 0).unwrap();
            num += d_i * *diff.get(i, 0).unwrap();
            den += d_i * d_i;
        }

        let t = num / den;
        self.point_at(t)
    }

    pub fn origin(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.origin
    }

    pub fn direction(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.direction
    }
}
